"""InternVL feature extractor."""

import re
from typing import TYPE_CHECKING

import torch
import torchvision as tv
from transformers import AutoModel, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from .base import BaseFeatureExtractor, TapOutput

if TYPE_CHECKING:
    from torch._tensor import Tensor
    from torch.nn.modules.module import Module


class InternVLFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for InternVL models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        int8: bool = False,
        use_fast_processor: bool = True,
        llm_layers: str | list[int] = "all",
    ) -> None:
        """
        Initialize InternVL feature extractor.

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
        self.model = AutoModel.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_id,
            quantization_config=quant,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()  # type: ignore[no-untyped-call]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            pretrained_model_name_or_path=model_id, trust_remote_code=True, use_fast=use_fast_processor
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
        # Vision encoder output hook (v_enc)
        vision_model: Module | None = self._get_first_module(
            candidates=[
                "model.vision_model",
                "vision_model",
            ]
        )
        if vision_model is not None:

            def vision_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
                if not self._capture_enabled:
                    return
                # InternVL vision output is typically (B, num_patches, hidden_dim)
                y = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(y, torch.Tensor):
                    return

                pooled: Tensor
                if y.ndim >= 2:
                    pooled = y.mean(dim=1)
                else:
                    pooled = y.flatten(start_dim=0)

                self._tap.v_enc = pooled.detach().to("cpu")

            vision_model.register_forward_hook(hook=vision_hook)

        # MLP projector hooks (v_proj)
        projector: Module | None = self._get_first_module(
            candidates=[
                "model.mlp1",
                "mlp1",
                "model.mlp_projector",
                "mlp_projector",
            ]
        )
        if projector is not None:

            def projector_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
                if not self._capture_enabled:
                    return
                y = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(y, torch.Tensor):
                    return

                # y: (B, num_patches, llm_hidden_dim) - projected to language space
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
                    "model.language_model.model.layers",  # InternLM style
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
            dotted: Dotted path (e.g., 'model.vision_model')

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

        # Build prompts based on use_image
        if use_image:
            if images is None:
                error_msg = "use_image=True requires images"
                raise ValueError(error_msg)
            # InternVL uses <image> placeholder in prompts
            prompts = [f"<image>\n{t}" if t else "<image>" for t in texts]
        else:
            # Text only
            prompts = list(texts)

        # Convert images to pixel_values
        if use_image and images is not None:
            # Use model's built-in image loading if available
            pixel_values = []
            for img in images:
                if hasattr(self.model, "load_image"):
                    pv = self.model.load_image(img, max_num=12)  # type: ignore[attr-defined]
                    # load_image returns a tensor that may already be batched
                    # We need to handle it properly
                    if isinstance(pv, torch.Tensor):
                        pixel_values.append(pv)
                    else:
                        pixel_values.append(pv)
                else:
                    transform = tv.transforms.Compose(
                        [
                            tv.transforms.Resize((448, 448)),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
                    pv = transform(img).unsqueeze(0)
                    pixel_values.append(pv)

            # Stack pixel values - for InternVL, each load_image returns individual tensor
            if len(pixel_values) > 0:
                # Convert to model's dtype (usually bfloat16)
                model_dtype = next(self.model.parameters()).dtype
                pixel_values_list = [pv.to(device=self.device, dtype=model_dtype) for pv in pixel_values]
                pixel_values_tensor = pixel_values_list
            else:
                pixel_values_tensor = None
        else:
            pixel_values_tensor = None

        # Tokenize text
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Add pixel values if using images
        if pixel_values_tensor is not None:
            # InternVL expects pixel_values as a list of tensors (one per image)
            inputs["pixel_values"] = pixel_values_tensor
            # Add image_flags to indicate which samples have images
            batch_size = len(prompts)
            inputs["image_flags"] = torch.tensor([1] * batch_size, dtype=torch.long).to(self.device)
        else:
            # No images - set image_flags to 0
            batch_size = len(prompts)
            inputs["image_flags"] = torch.tensor([0] * batch_size, dtype=torch.long).to(self.device)

        # Extract features (v_enc/v_proj/layers)
        self._tap = TapOutput()
        self._last_bs = len(images) if images is not None else len(texts)
        self._last_attn = inputs.get("attention_mask")
        self._capture_enabled = True
        _ = self.model(**inputs, output_hidden_states=False, return_dict=True)

        # Generate if needed
        if decode:
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

            # InternVL generation
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id

            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=_do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

            # Decode new tokens only
            prompt_lens = inputs["attention_mask"].sum(dim=1)
            outs, parsed = [], []
            sequences = gen.sequences  # type: ignore[union-attr]
            for i in range(sequences.size(0)):
                start = int(prompt_lens[i].item())
                new_tokens = sequences[i, start:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outs.append(text)
                m = re.search(r"\{([^}]+)\}", text)
                parsed.append(m.group(1).strip() if m else None)

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
