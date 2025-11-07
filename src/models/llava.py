"""LLaVA-1.5 feature extractor."""

import re

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig

from .base import BaseFeatureExtractor, TapOutput


class LlavaFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for LLaVA-1.5 models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        int8: bool = False,
        use_fast_processor: bool = True,
        llm_layers: str | list[int] = "all",
    ) -> None:
        """
        Initialize LLaVA feature extractor.

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
        self.model = LlavaForConditionalGeneration.from_pretrained(  # type: ignore[no-untyped-call]
            model_id,
            quantization_config=quant,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        # Multi-modal projector hooks (v_enc/v_proj)
        projector = self._get_first_module(
            [
                "model.multi_modal_projector",
                "model.mm_projector",
                "multi_modal_projector",
                "mm_projector",
            ]
        )
        if projector is None:
            error_msg = "multi_modal_projector (or mm_projector) not found"
            raise ValueError(error_msg)

        def projector_pre_hook(module, inputs) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
            if not self._capture_enabled:
                return
            x = inputs[0]
            x = x[0] if isinstance(x, (tuple, list)) else x
            if not isinstance(x, torch.Tensor):
                return

            # x: (B, T_v, D_v) or similar
            if x.ndim >= 2:
                pooled = x.mean(dim=1)
            else:
                pooled = x.flatten(start_dim=0)

            self._tap.v_enc = pooled.detach().to("cpu")

        def projector_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
            if not self._capture_enabled:
                return
            y = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(y, torch.Tensor):
                return

            # y: (B, T_v, D_text) - language side embeddings
            if y.ndim >= 2:
                pooled = y.mean(dim=1)
            else:
                pooled = y.flatten(start_dim=0)

            self._tap.v_proj = pooled.detach().to("cpu")

        projector.register_forward_pre_hook(projector_pre_hook)
        projector.register_forward_hook(projector_hook)

        # LLM layer hooks
        self._llm_hooks = []
        if self.llm_layers is not None and (self.llm_layers == "all" or len(self.llm_layers) > 0):
            lm_layers = self._get_first_module(
                [
                    "model.language_model.model.layers",  # LlamaForCausalLM.model.layers
                    "model.language_model.layers",  # Some implementations
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
            dotted: Dotted path (e.g., 'model.multi_modal_projector')

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
        extract_logits: bool = False,
        options: list[list[str]] | None = None,
    ) -> TapOutput:
        """Extract features with or without images."""
        if texts is None:
            texts = [""] * (len(images) if images else 1)

        # Build chat template based on use_image
        if use_image:
            if images is None:
                error_msg = "use_image=True requires images"
                raise ValueError(error_msg)
            msgs = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": t},
                        ],
                    }
                ]
                for img, t in zip(images, texts, strict=False)
            ]
        else:
            # Text only
            msgs = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in texts]

        templated = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        if use_image:
            batch = self.processor(text=templated, images=images, return_tensors="pt", padding=True)
        else:
            batch = self.processor(text=templated, return_tensors="pt", padding=True)
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
            pad_id = eos_id if eos_id is not None else tok.pad_token_id

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

        # Extract logits if requested
        if extract_logits and options is not None:
            self._extract_choice_logits(batch, options)

        return self._tap

    def _extract_choice_logits(self, batch: dict, options: list[list[str]]) -> None:  # noqa: ARG002
        """
        Extract logits for choice tokens from all layers.

        Args:
            batch: Input batch dictionary (unused, kept for consistency)
            options: List of option lists for each sample (B, num_choices)
        """
        tok = self.processor.tokenizer
        lm_head = self.model.get_output_embeddings()

        if lm_head is None:
            return

        # Tokenize all options to get first token IDs
        max_choices = max(len(opts) for opts in options)
        choice_token_ids = []

        for sample_options in options:
            sample_tokens = []
            for opt in sample_options:
                # Tokenize without special tokens and get first token
                token_ids = tok.encode(opt, add_special_tokens=False)
                if len(token_ids) > 0:
                    sample_tokens.append(token_ids[0])
                else:
                    # Fallback: use unknown token
                    sample_tokens.append(tok.unk_token_id if tok.unk_token_id is not None else 0)
            choice_token_ids.append(sample_tokens)

        self._tap.choice_token_ids = choice_token_ids

        # Extract logits from each layer
        for layer_name, hidden_states in self._tap.layers.items():
            if hidden_states is None or hidden_states.ndim != 2:
                continue

            # Apply lm_head: (B, hidden_size) -> (B, vocab_size)
            with torch.no_grad():
                hidden_states_gpu = hidden_states.to(self.device)
                vocab_logits = lm_head(hidden_states_gpu)  # (B, vocab_size)

                # Store full logits (optional, can be large)
                self._tap.logits[layer_name] = vocab_logits.detach().cpu()

                # Extract choice logits
                choice_logits_batch = []
                for i, sample_token_ids in enumerate(choice_token_ids):
                    sample_logits = vocab_logits[i, sample_token_ids]  # (num_choices,)
                    choice_logits_batch.append(sample_logits)

                # Pad to max_choices and stack
                choice_logits_padded = []
                for cl in choice_logits_batch:
                    if len(cl) < max_choices:
                        # Pad with very negative values
                        padding = torch.full(
                            (max_choices - len(cl),),
                            float("-inf"),
                            dtype=cl.dtype,
                            device=cl.device,
                        )
                        padded_cl = torch.cat([cl, padding])
                    else:
                        padded_cl = cl
                    choice_logits_padded.append(padded_cl)

                self._tap.choice_logits[layer_name] = torch.stack(choice_logits_padded).detach().cpu()

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
