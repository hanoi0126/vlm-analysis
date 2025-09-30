"""Qwen2.5-VL feature extractor."""

import re
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from .base import BaseFeatureExtractor, TapOutput

try:
    from transformers import AutoModelForImageTextToText as ModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as ModelClass


class QwenVLFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Qwen2.5-VL models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        int8: bool = False,
        use_fast_processor: bool = True,
        llm_layers: Union[str, List[int]] = "all",
    ) -> None:
        """
        Initialize Qwen VL feature extractor.

        Args:
            model_id: HuggingFace model ID
            device: Device to use
            int8: Use 8-bit quantization
            use_fast_processor: Use fast processor
            llm_layers: LLM layers to tap ('all' or list of indices)
        """
        super().__init__(device)

        # Load model
        quant = (
            BitsAndBytesConfig(load_in_8bit=True)
            if int8 and torch.cuda.is_available()
            else None
        )
        self.model = ModelClass.from_pretrained(
            model_id,
            quantization_config=quant,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id, use_fast=use_fast_processor
        )

        # Setup hooks
        self.llm_layers = llm_layers
        self._tap = TapOutput()
        self._last_bs: Optional[int] = None
        self._last_attn: Optional[torch.Tensor] = None

        self._capture_enabled = True
        self._in_gen = False
        self._gen_sum: Dict[str, torch.Tensor] = {}
        self._gen_cnt: Dict[str, int] = {}

        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Setup forward hooks for feature extraction."""
        # Visual encoder hooks (pre/post)
        merger = self._get_module(self.model, "model.visual.merger")
        if merger is None:
            raise ValueError("visual.merger not found")

        def merger_pre_hook(module, inputs):
            if not self._capture_enabled:
                return
            x = inputs[0]
            x = x[0] if isinstance(x, (tuple, list)) else x
            if not isinstance(x, torch.Tensor):
                return

            if x.ndim == 3:
                pooled = x.mean(dim=1)
            elif x.ndim == 2:
                B = self._last_bs or x.shape[0]
                T = x.shape[0] // B
                pooled = x.view(B, T, -1).mean(dim=1)
            else:
                pooled = x.flatten(start_dim=1)

            self._tap.pre = pooled.detach().to("cpu")

        def merger_hook(module, inputs, output):
            if not self._capture_enabled:
                return
            y = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(y, torch.Tensor):
                return

            if y.ndim == 3:
                pooled = y.mean(dim=1)
            elif y.ndim == 2:
                B = self._last_bs or y.shape[0]
                T = y.shape[0] // B
                pooled = y.view(B, T, -1).mean(dim=1)
            else:
                pooled = y.flatten(start_dim=1)

            self._tap.post = pooled.detach().to("cpu")

        merger.register_forward_pre_hook(merger_pre_hook)
        merger.register_forward_hook(merger_hook)

        # LLM layer hooks
        self._llm_hooks = []
        if self.llm_layers is not None and (
            self.llm_layers == "all" or len(self.llm_layers) > 0
        ):
            lm_layers = self._get_module(self.model, "model.language_model.layers")
            if lm_layers is None:
                raise ValueError("language_model.layers not found")

            num_layers = len(lm_layers)
            want = (
                list(range(num_layers)) if self.llm_layers == "all" else self.llm_layers
            )

            for idx in want:
                if not (0 <= idx < num_layers):
                    continue
                layer = lm_layers[idx]
                tag = f"l{idx:02d}"

                def make_hook(tag):
                    def _hook(module, inputs, output):
                        if not self._capture_enabled:
                            return
                        Y = output[0] if isinstance(output, (tuple, list)) else output
                        if not isinstance(Y, torch.Tensor):
                            return

                        if self._in_gen:
                            # During generation: accumulate last token
                            pooled = Y[:, -1, :]
                            cpu = pooled.detach().to("cpu")
                            if tag not in self._gen_sum:
                                self._gen_sum[tag] = torch.zeros_like(cpu)
                                self._gen_cnt[tag] = 0
                            self._gen_sum[tag] += cpu
                            self._gen_cnt[tag] += 1
                        else:
                            # Before generation: last valid token
                            if self._last_attn is None:
                                pooled = Y[:, -1, :]
                            else:
                                idxs = self._last_attn.sum(dim=1) - 1
                                ar = torch.arange(Y.shape[0], device=Y.device)
                                pooled = Y[ar, idxs, :]
                            self._tap.layers[tag] = pooled.detach().to("cpu")

                    return _hook

                self._llm_hooks.append(layer.register_forward_hook(make_hook(tag)))

    def _get_module(
        self, root: torch.nn.Module, dotted: str
    ) -> Optional[torch.nn.Module]:
        """
        Get nested module by dotted path.

        Args:
            root: Root module
            dotted: Dotted path (e.g., 'model.visual.merger')

        Returns:
            Module if found, None otherwise
        """
        cur = root
        for p in dotted.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    @torch.no_grad()
    def forward(
        self,
        images: List,
        texts: Optional[List[str]] = None,
        *,
        decode: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        generation_kwargs: Optional[Dict] = None,
    ) -> TapOutput:
        """
        Extract features from images and texts.

        Args:
            images: List of PIL images
            texts: List of text prompts
            decode: Whether to decode generated text
            max_new_tokens: Max new tokens for generation
            do_sample: Use sampling
            generation_kwargs: Additional generation kwargs

        Returns:
            TapOutput with extracted features
        """
        if texts is None:
            texts = [""] * len(images)

        # Prepare input using chat template
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
            for img, t in zip(images, texts)
        ]
        templated = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in msgs
        ]
        batch = self.processor(
            text=templated, images=images, return_tensors="pt", padding=True
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Extract features (pre/post/layers)
        self._tap = TapOutput()
        self._last_bs = len(images)
        self._last_attn = batch.get("attention_mask", None)
        self._capture_enabled = True
        _ = self.model(**batch, output_hidden_states=False, return_dict=True)

        # Generate if needed
        if decode:
            tok = self.processor.tokenizer
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
            pad_eos = im_end_id if im_end_id is not None else tok.eos_token_id

            gen_kwargs = dict(generation_kwargs or {})
            # Handle temperature
            if "temperature" in gen_kwargs and (
                gen_kwargs["temperature"] is None or gen_kwargs["temperature"] <= 0
            ):
                gen_kwargs.pop("temperature", None)

            _do_sample = bool(do_sample) or (
                "temperature" in gen_kwargs and gen_kwargs["temperature"] > 0
            )
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
                pad_token_id=pad_eos,
                eos_token_id=pad_eos,
                return_dict_in_generate=True,
                **gen_kwargs,
            )

            # Decode new tokens only
            prompt_lens = batch["attention_mask"].sum(dim=1)
            outs, parsed = [], []
            for i in range(gen.sequences.size(0)):
                start = int(prompt_lens[i].item())
                new_tokens = gen.sequences[i, start:]
                text = tok.decode(new_tokens, skip_special_tokens=True)
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

    def get_tap_points(self) -> List[str]:
        """
        Get available tap point names.

        Returns:
            List of tap point names
        """
        points = ["pre", "post"]
        if self.llm_layers == "all":
            # Assuming 36 layers for Qwen2.5-VL-3B
            points += [f"l{i:02d}" for i in range(36)]
            points += [f"l{i:02d}_outavg" for i in range(36)]
        elif isinstance(self.llm_layers, list):
            points += [f"l{i:02d}" for i in self.llm_layers]
            points += [f"l{i:02d}_outavg" for i in self.llm_layers]
        return points
