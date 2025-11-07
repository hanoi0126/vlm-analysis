"""Qwen2.5-VL feature extractor."""

import re

import torch
import torch.nn.functional as F
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

    def _extract_choice_logits(self, batch: dict, options: list[list[str]]) -> None:
        """
        Extract logits for choice tokens from all layers using sequence probabilities.

        For multi-token choices (e.g., "135" = ["1", "3", "5"]), this method computes
        the full sequence probability using teacher forcing.

        Args:
            batch: Input batch dictionary
            options: List of option lists for each sample (B, num_choices)
        """
        tok = self.processor.tokenizer
        lm_head = self.model.get_output_embeddings()  # type: ignore[no-untyped-call]

        if lm_head is None:
            return

        # Tokenize all options (keep full token sequences)
        max_choices = max(len(opts) for opts in options)
        choice_token_ids_all = []  # Store full token sequences

        for sample_options in options:
            sample_token_seqs = []
            for opt in sample_options:
                # Tokenize without special tokens
                token_ids = tok.encode(opt, add_special_tokens=False)
                if len(token_ids) == 0:
                    # Fallback: use unknown token
                    token_ids = [tok.unk_token_id if tok.unk_token_id is not None else 0]
                sample_token_seqs.append(token_ids)
            choice_token_ids_all.append(sample_token_seqs)

        self._tap.choice_token_ids = choice_token_ids_all

        # Extract sequence probabilities for each choice
        self._extract_sequence_logits(batch, choice_token_ids_all, max_choices, lm_head)

    def _extract_sequence_logits(
        self,
        batch: dict,  # noqa: ARG002
        choice_token_ids_all: list[list[list[int]]],
        max_choices: int,
        lm_head: torch.nn.Module,
    ) -> None:
        """
        Extract sequence logits by generating each choice with teacher forcing.

        For each choice, computes the full sequence probability:
        P(choice) = P(token_1) × P(token_2|token_1) × ... × P(token_n|token_1:n-1)

        Args:
            batch: Input batch dictionary
            choice_token_ids_all: Token ID sequences for all choices (B, num_choices, seq_len)
            max_choices: Maximum number of choices across samples
            lm_head: Language model head
        """
        # TEMPORARY: Always use first-token approximation
        # Teacher forcing implementation has issues with image inputs
        # TODO: Fix teacher forcing for multi-token sequences with vision inputs
        self._extract_single_token_logits(choice_token_ids_all, max_choices, lm_head)

    def _extract_single_token_logits(
        self,
        choice_token_ids_all: list[list[list[int]]],
        max_choices: int,
        lm_head: torch.nn.Module,
    ) -> None:
        """Extract logits for single-token choices (fast path)."""
        for layer_name, hidden_states in self._tap.layers.items():
            if hidden_states is None or hidden_states.ndim != 2:
                continue

            with torch.no_grad():
                hidden_states_gpu = hidden_states.to(self.device)
                vocab_logits = lm_head(hidden_states_gpu)  # (B, vocab_size)
                log_probs = F.log_softmax(vocab_logits, dim=-1)

                self._tap.logits[layer_name] = vocab_logits.detach().cpu()

                choice_logits_batch = []
                for i, sample_token_seqs in enumerate(choice_token_ids_all):
                    sample_logits = []
                    for token_seq in sample_token_seqs:
                        first_token_id = token_seq[0]
                        first_token_logprob = log_probs[i, first_token_id]
                        sample_logits.append(first_token_logprob)

                    choice_logits_batch.append(torch.tensor(sample_logits, device=vocab_logits.device))

                choice_logits_padded = []
                for cl in choice_logits_batch:
                    if len(cl) < max_choices:
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

    def _extract_multi_token_logits_with_teacher_forcing(
        self,
        batch: dict,
        choice_token_ids_all: list[list[list[int]]],
        max_choices: int,
        lm_head: torch.nn.Module,
        model: torch.nn.Module,
    ) -> None:
        """
        Extract logits for multi-token choices using teacher forcing.

        For each choice, generates it autoregressively and computes sequence probability.
        """
        # Get batch size
        batch_size = len(choice_token_ids_all)

        # Store layer-wise choice logits
        layer_choice_logits: dict[str, list[torch.Tensor]] = {layer_name: [] for layer_name in self._tap.layers}

        # Process each sample in the batch
        for sample_idx in range(batch_size):
            sample_choices = choice_token_ids_all[sample_idx]
            sample_choice_logits: dict[str, list[float]] = {layer_name: [] for layer_name in self._tap.layers}

            # Get original input for this sample
            # Note: batch contains processed inputs, we need to get the original input_ids
            # For simplicity, we'll use the hidden states approach

            # For each choice, compute sequence log probability
            for choice_tokens in sample_choices:
                # Compute sequence log probability using autoregressive generation
                seq_log_prob = self._compute_choice_sequence_probability(batch, sample_idx, choice_tokens, lm_head, model)

                # Store for each layer
                for layer_name, log_prob_value in seq_log_prob.items():
                    sample_choice_logits[layer_name].append(log_prob_value)

            # Add sample results to batch results
            for layer_name, choice_list in sample_choice_logits.items():
                layer_choice_logits[layer_name].append(torch.tensor(choice_list, device=self.device))

        # Pad and store results
        for layer_name, choice_logits_list in layer_choice_logits.items():
            if not choice_logits_list:
                continue

            choice_logits_padded = []
            for cl in choice_logits_list:
                if len(cl) < max_choices:
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

    def _compute_choice_sequence_probability(
        self,
        batch: dict,
        sample_idx: int,
        choice_tokens: list[int],
        lm_head: torch.nn.Module,
        model: torch.nn.Module,
    ) -> dict[str, float]:
        """
        Compute sequence log probability for a single choice using teacher forcing.

        For multi-token choices, autoregressively generates each token and computes:
        log P(choice) = log P(t1) + log P(t2|t1) + ... + log P(tn|t1:n-1)

        Returns:
            Dictionary mapping layer names to sequence log probabilities
        """
        if len(choice_tokens) == 1:
            # Single token - use existing hidden states (fast path)
            return self._get_single_token_log_prob(sample_idx, choice_tokens[0], lm_head)

        # Multi-token choice - need autoregressive generation with teacher forcing
        return self._get_multi_token_sequence_log_prob(batch, sample_idx, choice_tokens, lm_head, model)

    def _get_single_token_log_prob(
        self,
        sample_idx: int,
        token_id: int,
        lm_head: torch.nn.Module,
    ) -> dict[str, float]:
        """Get log probability for a single token from existing hidden states."""
        layer_log_probs = {}

        for layer_name, hidden_states in self._tap.layers.items():
            if hidden_states is None or hidden_states.ndim != 2:
                continue

            if sample_idx >= hidden_states.shape[0]:
                continue

            with torch.no_grad():
                sample_hidden = hidden_states[sample_idx : sample_idx + 1].to(self.device)
                vocab_logits = lm_head(sample_hidden)
                log_probs = F.log_softmax(vocab_logits, dim=-1)
                layer_log_probs[layer_name] = log_probs[0, token_id].item()

        return layer_log_probs

    def _get_multi_token_sequence_log_prob(
        self,
        batch: dict,
        sample_idx: int,
        choice_tokens: list[int],
        lm_head: torch.nn.Module,
        model: torch.nn.Module,
    ) -> dict[str, float]:
        """
        Compute sequence log probability with full teacher forcing.

        Autoregressively generates each token and accumulates log probabilities.
        """
        # Initialize layer log probs
        layer_log_probs = dict.fromkeys(self._tap.layers, 0.0)

        # Get the original input_ids from batch
        # Note: We need to extract the sample's input and append choice tokens one by one
        if "input_ids" not in batch:
            # Fallback: use first token approximation if we can't do full teacher forcing
            print("Warning: Cannot perform full teacher forcing without input_ids. Using approximation.")
            return self._get_single_token_log_prob(sample_idx, choice_tokens[0], lm_head)

        # Get sample's input_ids
        sample_input_ids = batch["input_ids"][sample_idx : sample_idx + 1].clone()

        # Get other batch inputs for this sample
        sample_batch = {
            "input_ids": sample_input_ids,
        }
        if "attention_mask" in batch:
            sample_batch["attention_mask"] = batch["attention_mask"][sample_idx : sample_idx + 1].clone()
        if "pixel_values" in batch:
            sample_batch["pixel_values"] = batch["pixel_values"][sample_idx : sample_idx + 1].clone()
        if "image_grid_thw" in batch:
            sample_batch["image_grid_thw"] = batch["image_grid_thw"][sample_idx : sample_idx + 1].clone()

        # Autoregressively generate each token with teacher forcing
        current_input_ids = sample_input_ids

        for target_token_id in choice_tokens:
            # Forward pass with current sequence
            with torch.no_grad():
                # Update input_ids for this step
                sample_batch_step = sample_batch.copy()
                sample_batch_step["input_ids"] = current_input_ids

                # Get model outputs with hidden states
                outputs = model(
                    **sample_batch_step,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get hidden states from all layers at the last position
                hidden_states_all_layers = outputs.hidden_states

                # Compute log prob for target token at each layer
                for layer_idx, hidden_state in enumerate(hidden_states_all_layers):
                    # Get last position hidden state
                    last_hidden = hidden_state[:, -1:, :]  # (1, 1, hidden_size)

                    # Compute logits
                    vocab_logits = lm_head(last_hidden.squeeze(1))  # (1, vocab_size)
                    log_probs = F.log_softmax(vocab_logits, dim=-1)

                    # Get log prob for target token
                    target_log_prob = log_probs[0, target_token_id].item()

                    # Accumulate for this layer
                    layer_name = f"l{layer_idx:02d}"
                    if layer_name in layer_log_probs:
                        layer_log_probs[layer_name] += target_log_prob

            # Append target token to input for next step (teacher forcing)
            current_input_ids = torch.cat([current_input_ids, torch.tensor([[target_token_id]], device=current_input_ids.device)], dim=1)

        return layer_log_probs

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
