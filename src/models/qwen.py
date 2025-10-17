"""Qwen2.5-VL feature extractor."""

import re

import torch
from transformers import AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig

from .base import BaseFeatureExtractor, TapOutput

try:
    from transformers import AutoModelForImageTextToText as ModelClass
except Exception:
    from transformers import AutoModelForVision2Seq as ModelClass  # type: ignore[assignment]


class QwenVLFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for Qwen2.5-VL models."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        int8: bool = False,
        use_fast_processor: bool = True,
        llm_layers: str | list[int] = "all",
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
        quant: BitsAndBytesConfig | None = (
            BitsAndBytesConfig(load_in_8bit=True) if int8 and torch.cuda.is_available() else None  # type: ignore[no-untyped-call]
        )
        self.model = ModelClass.from_pretrained(  # type: ignore[no-untyped-call]
            model_id,
            quantization_config=quant,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        # Visual encoder hooks (v_enc/v_proj)
        merger = self._get_module(self.model, "model.visual.merger")
        if merger is None:
            error_msg = "visual.merger not found"
            raise ValueError(error_msg)

        def merger_pre_hook(module, inputs) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
            if not self._capture_enabled:
                return
            x = inputs[0]
            x = x[0] if isinstance(x, (tuple, list)) else x
            if not isinstance(x, torch.Tensor):
                return

            ndim_3d = 3
            ndim_2d = 2
            if x.ndim == ndim_3d:
                pooled = x.mean(dim=1)
            elif x.ndim == ndim_2d:
                batch_size = self._last_bs or x.shape[0]
                seq_len = x.shape[0] // batch_size
                pooled = x.view(batch_size, seq_len, -1).mean(dim=1)
            else:
                pooled = x.flatten(start_dim=1)

            self._tap.v_enc = pooled.detach().to("cpu")

        def merger_hook(module, inputs, output) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, ARG001
            if not self._capture_enabled:
                return
            y = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(y, torch.Tensor):
                return

            ndim_3d = 3
            ndim_2d = 2
            if y.ndim == ndim_3d:
                pooled = y.mean(dim=1)
            elif y.ndim == ndim_2d:
                batch_size = self._last_bs or y.shape[0]
                seq_len = y.shape[0] // batch_size
                pooled = y.view(batch_size, seq_len, -1).mean(dim=1)
            else:
                pooled = y.flatten(start_dim=1)

            self._tap.v_proj = pooled.detach().to("cpu")

        merger.register_forward_pre_hook(merger_pre_hook)
        merger.register_forward_hook(merger_hook)

        # LLM layer hooks
        self._llm_hooks = []
        if self.llm_layers is not None and (self.llm_layers == "all" or len(self.llm_layers) > 0):
            lm_layers = self._get_module(self.model, "model.language_model.layers")
            if lm_layers is None:
                error_msg = "language_model.layers not found"
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
        images: list | None = None,  # Optional に変更
        texts: list[str] | None = None,
        *,
        use_image: bool = True,  # 新規パラメータ
        decode: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        generation_kwargs: dict | None = None,
        extract_logits: bool = False,  # 新規: logit抽出
        options: list[list[str]] | None = None,  # 新規: 選択肢リスト
    ) -> TapOutput:
        """Extract features with or without images."""
        if texts is None:
            texts = [""] * (len(images) if images else 1)

        # Chat template の構築を use_image で分岐
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
            # テキストのみ
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
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
            pad_eos = im_end_id if im_end_id is not None else tok.eos_token_id

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
                pad_token_id=pad_eos,
                eos_token_id=pad_eos,
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
        lm_head = self.model.get_output_embeddings()  # type: ignore[no-untyped-call]

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
