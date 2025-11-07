"""PaliGemma/Gemma3 feature extractor."""

import re

import torch
from transformers import AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig

from .base import BaseFeatureExtractor, TapOutput

try:
    from transformers import PaliGemmaForConditionalGeneration as ModelClass
except Exception:
    try:
        from transformers import AutoModelForVision2Seq as ModelClass  # type: ignore[assignment]
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelClass  # type: ignore[assignment]


class GemmaFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for PaliGemma/Gemma3 models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        int8: bool = False,
        use_fast_processor: bool = True,
        llm_layers: str | list[int] = "all",
    ) -> None:
        """
        Initialize Gemma feature extractor.

        Args:
            model_id: HuggingFace model ID
            device: Device to use
            int8: Use 8-bit quantization
            use_fast_processor: Use fast processor
            llm_layers: LLM layers to tap ('all' or list of indices)
        """
        super().__init__(device)

        # Load model
        quant: BitsAndBytesConfig | None = (
            BitsAndBytesConfig(load_in_8bit=True) if int8 and torch.cuda.is_available() else None  # type: ignore[no-untyped-call]
        )
        self.model = ModelClass.from_pretrained(  # type: ignore[no-untyped-call]
            model_id,
            quantization_config=quant,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()  # type: ignore[no-untyped-call]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_id, use_fast=use_fast_processor
        )

        # Setup hooks
        self.llm_layers = llm_layers
        self._tap = TapOutput()
        self._last_bs: int | None = None
        self._last_attn: torch.Tensor | None = None

        self._capture_enabled = True
        self._in_gen = False
        self._gen_sum: dict[str, torch.Tensor] = {}
        self._gen_cnt: dict[str, int] = {}

        # Store number of layers for dynamic access
        self._num_llm_layers: int = 0

        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Setup forward hooks for feature extraction."""
        # Vision tower hooks (v_enc)
        vision_tower = self._get_first_module(
            [
                "model.vision_tower",
                "vision_tower",
                "model.vision_model",
                "vision_model",
            ]
        )
        if vision_tower is not None:

            def vision_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
                if not self._capture_enabled:
                    return
                y = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(y, torch.Tensor):
                    return

                # Vision output: (B, num_patches, hidden_dim)
                if y.ndim >= 2:
                    pooled = y.mean(dim=1)
                else:
                    pooled = y.flatten(start_dim=0)

                self._tap.v_enc = pooled.detach().to("cpu")

            vision_tower.register_forward_hook(vision_hook)

        # Multi-modal projector hooks (v_proj)
        projector = self._get_first_module(
            [
                "model.multi_modal_projector",
                "multi_modal_projector",
                "model.mm_projector",
                "mm_projector",
            ]
        )
        if projector is not None:

            def projector_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
                if not self._capture_enabled:
                    return
                y = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(y, torch.Tensor):
                    return

                # Projector output: (B, num_patches, llm_hidden_dim)
                if y.ndim >= 2:
                    pooled = y.mean(dim=1)
                else:
                    pooled = y.flatten(start_dim=0)

                self._tap.v_proj = pooled.detach().to("cpu")

            projector.register_forward_hook(projector_hook)

        # LLM layer hooks
        self._llm_hooks = []
        if self.llm_layers is not None and (self.llm_layers == "all" or len(self.llm_layers) > 0):
            lm_layers = self._get_first_module(
                [
                    "model.language_model.model.layers",  # Gemma style
                    "language_model.model.layers",
                    "model.language_model.layers",
                    "language_model.layers",
                ]
            )
            if lm_layers is None:
                error_msg = "language_model.(model.)layers not found"
                raise ValueError(error_msg)

            num_layers = len(lm_layers)  # type: ignore[arg-type]
            self._num_llm_layers = num_layers  # Store for later use
            want: list[int] = list(range(num_layers)) if self.llm_layers == "all" else self.llm_layers  # type: ignore[assignment]

            for idx in want:
                if not (0 <= idx < num_layers):
                    continue
                layer = lm_layers[idx]  # type: ignore[index]
                tag = f"l{idx:02d}"

                def make_hook(tag):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
                    def _hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
                        if not self._capture_enabled:
                            return
                        y = output[0] if isinstance(output, (tuple, list)) else output
                        if not isinstance(y, torch.Tensor):
                            return

                        if self._in_gen:
                            # During generation: accumulate last token
                            pooled = y[:, -1, :]
                            cpu = pooled.detach().to("cpu")
                            if tag not in self._gen_sum:
                                self._gen_sum[tag] = torch.zeros_like(cpu)
                                self._gen_cnt[tag] = 0
                            self._gen_sum[tag] += cpu
                            self._gen_cnt[tag] += 1
                        else:
                            # Before generation: last valid token
                            if self._last_attn is None:
                                pooled = y[:, -1, :]
                            else:
                                idxs = self._last_attn.sum(dim=1) - 1
                                ar = torch.arange(y.shape[0], device=y.device)
                                pooled = y[ar, idxs, :]
                            self._tap.layers[tag] = pooled.detach().to("cpu")

                    return _hook

                self._llm_hooks.append(layer.register_forward_hook(make_hook(tag)))  # type: ignore[no-untyped-call]

    def _get_module(self, root: torch.nn.Module, dotted: str) -> torch.nn.Module | None:
        """
        Get nested module by dotted path.

        Args:
            root: Root module
            dotted: Dotted path (e.g., 'model.vision_tower')

        Returns:
            Module if found, None otherwise
        """
        cur = root
        for p in dotted.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    def _get_first_module(self, candidates: list[str]) -> torch.nn.Module | None:
        """
        Get first available module from candidate paths.

        Args:
            candidates: List of dotted paths to try

        Returns:
            First found module, None if none found
        """
        for path in candidates:
            m = self._get_module(self.model, path)
            if m is not None:
                return m
        return None

    @torch.no_grad()
    def forward(
        self,
        images: list | None = None,
        texts: list[str] | None = None,
        *,
        use_image: bool = True,
        decode: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        generation_kwargs: dict | None = None,
    ) -> TapOutput:
        """Extract features with or without images."""
        if texts is None:
            texts = [""] * (len(images) if images else 1)

        # PaliGemma uses simple image+text concatenation
        # No need for chat template in most cases
        prompts = list(texts)

        # Process inputs
        if use_image:
            if images is None:
                error_msg = "use_image=True requires images"
                raise ValueError(error_msg)
            # Replicate images to match text batch size if needed
            if len(images) < len(prompts):
                # Repeat the last image to match batch size
                images = list(images) + [images[-1]] * (len(prompts) - len(images))
            batch = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        else:
            # Text only
            batch = self.processor(text=prompts, return_tensors="pt", padding=True)

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Extract features (v_enc/v_proj/layers)
        self._tap = TapOutput()
        self._last_bs = len(images) if images is not None else len(texts)
        self._last_attn = batch.get("attention_mask")
        self._capture_enabled = True
        _ = self.model(**batch, output_hidden_states=False, return_dict=True)

        # Generate if needed
        if decode:
            tok = self.processor.tokenizer
            eos_id = tok.eos_token_id
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

            gen_kwargs = dict(generation_kwargs or {})
            # Handle temperature
            if "temperature" in gen_kwargs and (gen_kwargs["temperature"] is None or gen_kwargs["temperature"] <= 0):
                gen_kwargs.pop("temperature", None)

            _do_sample = bool(do_sample) or ("temperature" in gen_kwargs and gen_kwargs["temperature"] > 0)
            if not _do_sample:
                for k in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "typical_p",
                    "epsilon_cutoff",
                    "eta_cutoff",
                ):
                    gen_kwargs.pop(k, None)

            self._in_gen = True
            self._gen_sum, self._gen_cnt = {}, {}

            gen = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=_do_sample,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

            # Decode new tokens only
            prompt_lens = batch["attention_mask"].sum(dim=1)
            outs, parsed = [], []
            sequences = gen.sequences  # type: ignore[union-attr]
            for i in range(sequences.size(0)):
                start = int(prompt_lens[i].item())
                new_tokens = sequences[i, start:]
                text = tok.decode(new_tokens, skip_special_tokens=True)
                outs.append(text)

                # Try to parse bracketed format first: {answer}
                m = re.search(r"\{([^}]+)\}", text)
                if m:
                    parsed.append(m.group(1).strip())
                else:
                    # Fallback: use the first word/token (for non-bracketed format)
                    cleaned = text.strip()
                    parsed.append(cleaned.split()[0] if cleaned else None)

            self._tap.gen_texts = outs
            self._tap.gen_parsed = parsed

            # Average output features
            for tag, s in self._gen_sum.items():
                cnt = max(1, self._gen_cnt.get(tag, 1))
                avg = s / cnt
                self._tap.layers[f"{tag}_outavg"] = avg

            self._in_gen = False
            self._gen_sum, self._gen_cnt = {}, {}

        return self._tap

    def get_tap_points(self) -> list[str]:
        """
        Get available tap point names.

        Returns:
            List of tap point names
        """
        points = ["v_enc", "v_proj"]
        if self.llm_layers == "all":
            # Use dynamically detected number of layers
            points += [f"l{i:02d}" for i in range(self._num_llm_layers)]
        elif isinstance(self.llm_layers, list):
            points += [f"l{i:02d}" for i in self.llm_layers]
        return points
