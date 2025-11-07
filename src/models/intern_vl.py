"""InternVL feature extractor."""

import re
from typing import TYPE_CHECKING

from PIL import Image
import torch
from torchvision import transforms
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

        # Create image transform
        self.image_size = 448
        self.image_transform = self._build_transform(self.image_size)

        # Calculate num_image_token based on model config
        # InternVL uses patch-based tokenization
        patch_size = getattr(self.model.config.vision_config, "patch_size", 14)
        downsample_ratio = getattr(self.model.config, "downsample_ratio", 0.5)
        self.num_image_token = int((self.image_size // patch_size) ** 2 * (downsample_ratio**2))

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

    def _build_transform(self, input_size: int) -> transforms.Compose:
        """Build image transformation pipeline."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
                transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _find_closest_aspect_ratio(
        self, aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
    ) -> tuple[int, int]:
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self, image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False
    ) -> list[Image.Image]:
        """Dynamically preprocess image into multiple patches based on aspect ratio."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios_list = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(aspect_ratio, target_ratios_list, orig_width, orig_height, image_size)

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

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

                self._tap.v_enc = pooled.detach().float().to("cpu")

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

                self._tap.v_proj = pooled.detach().float().to("cpu")

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
                            cpu = pooled.detach().float().to("cpu")
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
                            self._tap.layers[tag] = pooled.detach().float().to("cpu")

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
        extract_logits: bool = False,
        options: list[list[str]] | None = None,
    ) -> TapOutput:
        """Extract features with or without images."""
        if texts is None:
            texts = [""] * (len(images) if images else 1)

        # InternVL3.5 requires processing one sample at a time due to dynamic patching
        # We'll aggregate results from individual forward passes
        batch_size = len(texts)

        # For batched processing, we need to handle each sample individually
        if batch_size > 1:
            # Process each sample individually and aggregate
            tap_outputs = []
            for i in range(batch_size):
                img_list = [images[i]] if (use_image and images is not None) else None
                text_list = [texts[i]]
                opt_list = [options[i]] if options is not None else None
                tap_out = self._forward_single(
                    images=img_list,
                    texts=text_list,
                    use_image=use_image,
                    decode=decode,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    generation_kwargs=generation_kwargs,
                    extract_logits=extract_logits,
                    options=opt_list,
                )
                tap_outputs.append(tap_out)

            # Aggregate results
            self._tap = self._aggregate_tap_outputs(tap_outputs)
            return self._tap

        # Single sample - direct processing
        return self._forward_single(
            images=images,
            texts=texts,
            use_image=use_image,
            decode=decode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            generation_kwargs=generation_kwargs,
            extract_logits=extract_logits,
            options=options,
        )

    def _forward_single(
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
        """Process a single sample."""
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

        # Convert images to pixel_values using dynamic_preprocess
        if use_image and images is not None:
            img = images[0]
            # Use dynamic preprocessing like official implementation
            processed_images = self._dynamic_preprocess(img, min_num=1, max_num=12, image_size=self.image_size, use_thumbnail=True)
            pixel_values = torch.stack([self.image_transform(im) for im in processed_images])
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = pixel_values.to(device=self.device, dtype=model_dtype)
            num_patches = pixel_values.shape[0]
        else:
            pixel_values = None
            num_patches = 0

        # Setup img_context_token_id for InternVL
        img_context_token = "<IMG_CONTEXT>"
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)
        self.model.img_context_token_id = img_context_token_id

        # Prepare inputs using InternVL's approach
        if use_image and pixel_values is not None:
            # Replace <image> with special tokens like official implementation
            img_start_token = "<img>"
            img_end_token = "</img>"
            image_tokens = img_start_token + img_context_token * self.num_image_token * num_patches + img_end_token
            prompts = [p.replace("<image>", image_tokens, 1) for p in prompts]

        # Tokenize text
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Add pixel values and image flags
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
            inputs["image_flags"] = torch.tensor([1], dtype=torch.long).to(self.device)
        else:
            # For text-only: set pixel_values to a dummy tensor
            # InternVL forward() expects pixel_values to be present
            dummy_pixel = torch.zeros(
                (1, 3, self.image_size, self.image_size), dtype=next(self.model.parameters()).dtype, device=self.device
            )
            inputs["pixel_values"] = dummy_pixel
            inputs["image_flags"] = torch.tensor([0], dtype=torch.long).to(self.device)

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

            # Prepare generation inputs (remove image_flags as it's not needed for generate)
            gen_inputs = {k: v for k, v in inputs.items() if k != "image_flags"}

            # For text-only mode, use language model directly to avoid assertion errors in InternVL's generate
            if use_image:
                gen = self.model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=_do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    return_dict_in_generate=True,
                    **gen_kwargs,
                )
            else:
                # Text-only: use language model directly
                # Remove pixel_values as language model doesn't need it
                text_gen_inputs = {k: v for k, v in gen_inputs.items() if k not in ["pixel_values"]}
                gen = self.model.language_model.generate(
                    **text_gen_inputs,
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
            self._extract_choice_logits(inputs, options)

        return self._tap

    def _extract_choice_logits(self, inputs: dict, options: list[list[str]]) -> None:  # noqa: ARG002
        """
        Extract logits for choice tokens from all layers.

        Args:
            inputs: Input dictionary (unused, kept for consistency)
            options: List of option lists for each sample (B, num_choices)
        """
        lm_head = self.model.language_model.get_output_embeddings()

        if lm_head is None:
            return

        # Tokenize all options to get first token IDs
        max_choices = max(len(opts) for opts in options)
        choice_token_ids = []

        for sample_options in options:
            sample_tokens = []
            for opt in sample_options:
                # Tokenize without special tokens and get first token
                token_ids = self.tokenizer.encode(opt, add_special_tokens=False)
                if len(token_ids) > 0:
                    sample_tokens.append(token_ids[0])
                else:
                    # Fallback: use unknown token
                    sample_tokens.append(self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0)
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

    def _aggregate_tap_outputs(self, tap_outputs: list[TapOutput]) -> TapOutput:
        """Aggregate multiple TapOutput objects into one."""
        aggregated = TapOutput()

        # Aggregate v_enc
        if all(t.v_enc is not None for t in tap_outputs):
            aggregated.v_enc = torch.cat([t.v_enc for t in tap_outputs if t.v_enc is not None], dim=0)

        # Aggregate v_proj
        if all(t.v_proj is not None for t in tap_outputs):
            aggregated.v_proj = torch.cat([t.v_proj for t in tap_outputs if t.v_proj is not None], dim=0)

        # Aggregate layers
        if tap_outputs:
            all_layer_keys: set[str] = set()
            for tap in tap_outputs:
                all_layer_keys.update(tap.layers.keys())

            for key in all_layer_keys:
                if all(key in t.layers for t in tap_outputs):
                    aggregated.layers[key] = torch.cat([t.layers[key] for t in tap_outputs], dim=0)

        # Aggregate gen_texts and gen_parsed
        if any(t.gen_texts for t in tap_outputs):
            aggregated.gen_texts = []
            for tap in tap_outputs:
                if tap.gen_texts:
                    aggregated.gen_texts.extend(tap.gen_texts)

        if any(t.gen_parsed for t in tap_outputs):
            aggregated.gen_parsed = []
            for tap in tap_outputs:
                if tap.gen_parsed:
                    aggregated.gen_parsed.extend(tap.gen_parsed)

        # Aggregate logits
        if tap_outputs:
            all_logit_keys: set[str] = set()
            for tap in tap_outputs:
                all_logit_keys.update(tap.logits.keys())

            for key in all_logit_keys:
                if all(key in t.logits for t in tap_outputs):
                    aggregated.logits[key] = torch.cat([t.logits[key] for t in tap_outputs], dim=0)

        # Aggregate choice_logits
        if tap_outputs:
            all_choice_logit_keys: set[str] = set()
            for tap in tap_outputs:
                all_choice_logit_keys.update(tap.choice_logits.keys())

            for key in all_choice_logit_keys:
                if all(key in t.choice_logits for t in tap_outputs):
                    aggregated.choice_logits[key] = torch.cat([t.choice_logits[key] for t in tap_outputs], dim=0)

        # Aggregate choice_token_ids
        if any(t.choice_token_ids for t in tap_outputs):
            aggregated.choice_token_ids = []
            for tap in tap_outputs:
                if tap.choice_token_ids:
                    aggregated.choice_token_ids.extend(tap.choice_token_ids)

        return aggregated

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
