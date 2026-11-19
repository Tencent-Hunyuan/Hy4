"""
Patches for ms-swift training (HY4).

Registers the hy_v4_vl model type and template for ms-swift, including:
    1. Model & Template registration: Register hy_v4_vl model_type
       with is_multimodal=True and VL architecture.
    2. Multimodal data processing (image/video encoding).
    3. Training optimizations (gradient checkpointing, grad norm, etc.).

Usage:
    swift sft --custom_register_path hy_v4_vl_swift_patches.py --model /path/to/hy4_ckpt ...
"""

import os
import gc
import json as _json
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Patch 1: VL Model & Template Registration
#
# Register HY4-VL as a multimodal model in ms-swift. The VL model uses
# HYV4VLForConditionalGeneration and needs image/video processing support.
# ============================================================================

from swift.model import register_model, ModelMeta, ModelGroup, Model
from swift.model.register import ModelLoader
from swift.model.model_arch import MultiModelKeys, register_model_arch
from swift.template import register_template, TemplateMeta

# Register HY4-VL model architecture mapping.
# This tells ms-swift which parameter prefixes correspond to vision_tower,
# language_model, etc. — required for freeze_vit to work correctly.
# HYV4VLForConditionalGeneration has:
#   self.model = HYV4VLModel (contains self.vision_tower and self.language_model)
#   self.lm_head
HYV4_VL_ARCH_NAME = 'hy_v4_vl'
register_model_arch(
    MultiModelKeys(
        HYV4_VL_ARCH_NAME,
        language_model=['model.language_model', 'lm_head'],
        vision_tower='model.vision_tower',
        aligner=[],
    ),
    exist_ok=True,
)


# Custom ModelLoader for HY4-VL: load HYV4VLForConditionalGeneration directly.
# This model is not registered in AutoModelForCausalLM's mapping (it's a VL model),
# and AutoModel would load HYV4VLModel (base model without lm_head/loss).
# We must use the concrete class to get the full model with lm_head and loss computation.
class HYV4VLModelLoader(ModelLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or HYV4VLForConditionalGeneration
        return super().get_model(model_dir, config, processor, model_kwargs)

# ---------------------------------------------------------------------------
# HYV4VLTemplate: multimodal template for HY4-VL
#
# Handles image processing, image token expansion, and MRoPE position_ids.
# Modeled after Qwen2VLTemplate but adapted for HY4-VL's architecture:
#   - image_token_id = 120120 (from config.json)
#   - cat_extra_token = False (no image_begin/end extra tokens)
#   - image token count = patch_h * (patch_w + 1)
#   - MRoPE position_ids computed by the processor
# ---------------------------------------------------------------------------
from swift.template.base import Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import Context, findall
from typing import List, Literal, Optional
from PIL import Image as PILImage

class HYV4VLTemplate(Template):
    """Multimodal template for HY4-VL models in ms-swift."""

    # HY4-VL image/video token IDs (from config.json)
    image_token_id = 120120
    video_token_id = 120124
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """Replace <image>/<video> placeholder with the actual image/video token string.

        For HY4-VL, the image placeholder token is the raw image_token repeated once.
        The actual expansion to N tokens happens in _encode() after we know the grid size.
        """
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            # Return a single image_token_id placeholder; _encode() will expand it
            return [[-100]]
        else:
            return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """Encode text + images into model inputs.

        1. Call parent _encode() to get input_ids and labels from the chat template.
        2. Process images through the HY4-VL image_processor to get pixel_values
           and image_grid_thw.
        3. Expand each single image_token placeholder to the correct number of
           image tokens based on the grid size.
        """
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # Process images
        if inputs.images:
            # Load and preprocess images
            images = []
            for img in inputs.images:
                if isinstance(img, str):
                    images.append(PILImage.open(img).convert('RGB'))
                else:
                    images.append(img)

            image_processor = processor.image_processor
            image_inputs = image_processor(images=images, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            merge_size = image_processor.merge_size

            # Find placeholder positions (-100 markers from replace_tag)
            idx_list = findall(input_ids, -100)

            def _get_new_tokens(i):
                """Calculate the number of image tokens for image i."""
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                # HY4-VL: patch_h * (patch_w + 1), no extra tokens (cat_extra_token=False)
                num_tokens = int(patch_h * (patch_w + 1))
                return [self.image_token_id] * num_tokens

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded.update(image_inputs)

        # Process videos (similar logic)
        if inputs.videos:
            video_processor = getattr(processor, 'video_processor', processor.image_processor)
            # Video processing parameters (fps, max_frames, size, etc.) are read
            # from the processor's config (video_preprocessor_config.json).
            # To adjust, modify the JSON config in the model directory.
            video_inputs = video_processor(videos=inputs.videos, return_tensors='pt')
            video_grid_thw = video_inputs['video_grid_thw']
            merge_size = processor.image_processor.merge_size

            idx_list = findall(input_ids, -200)

            def _get_new_video_tokens(i):
                grid_t, grid_h, grid_w = video_grid_thw[i]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                # Each frame: patch_h * (patch_w + 1) tokens (including newline per row)
                num_tokens = int(grid_t * patch_h * (patch_w + 1))
                return [self.video_token_id] * num_tokens

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_video_tokens)
            encoded.update(video_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        if loss_scale is not None:
            encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through inputs unchanged.

        HYV4VLForConditionalGeneration.forward() handles pixel_values and
        image embedding injection internally, so we don't need to do it here.
        Returning inputs as-is ensures pixel_values, image_grid_thw, etc.
        are preserved and passed to the model's forward method.
        """
        return inputs

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """Custom data collator that computes MRoPE position_ids after batching."""
        res = super()._data_collator(batch, padding_to=padding_to)

        # Compute MRoPE position_ids using the processor
        input_ids = res.get('input_ids')
        if input_ids is not None:
            image_grid_thw = res.get('image_grid_thw', None)
            video_grid_thw = res.get('video_grid_thw', None)

            if image_grid_thw is not None or video_grid_thw is not None:
                # Build mm_token_type_ids: 0=text, 1=image, 2=video
                # Required by HYV4VLModel.get_rope_index to distinguish modalities
                mm_token_type_ids = torch.zeros_like(input_ids)
                image_token_id = getattr(self, 'image_token_id', 120120)
                video_token_id = getattr(self, 'video_token_id', 120124)
                mm_token_type_ids[input_ids == image_token_id] = 1
                mm_token_type_ids[input_ids == video_token_id] = 2

                # Use the model's MRoPE position_ids computation
                try:
                    base_model = self.get_base_model(self._get_model())
                    if hasattr(base_model, 'get_rope_index'):
                        position_ids, _ = base_model.get_rope_index(
                            input_ids,
                            mm_token_type_ids=mm_token_type_ids,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            attention_mask=res.get('attention_mask'),
                        )
                        res['position_ids'] = position_ids
                except Exception as e:
                    logger.warning("[HYV4VLTemplate] Failed to compute MRoPE position_ids: %s", e)

                # Always pass mm_token_type_ids to the model — required by
                # HYV4VLModel.compute_3d_position_ids when multimodal data is present
                res['mm_token_type_ids'] = mm_token_type_ids

        return res


# Register hy_v4_vl template with HYV4VLTemplate for multimodal support
# Token format: <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>
# Image placeholder: <image> in user content → expanded to N image tokens
# Video placeholder: <video> in user content → expanded to N video tokens
register_template(
    TemplateMeta(
        template_type='hy_v4_vl',
        prefix=[],
        system_prefix=['<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{SYSTEM}}<｜hy_end:opensource｜>'],
        prompt=['<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{QUERY}}<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>'],
        chat_sep=['<｜hy_end:opensource｜>'],
        suffix=['<｜hy_end:opensource｜>'],
        default_system='<｜reasoning_mode:opensource｜>reasoning_effort:no_think',
        non_thinking_prefix='<think:opensource></think:opensource>',
        template_cls=HYV4VLTemplate,
    ),
    exist_ok=True,
)

# Also register the LLM-only template for backward compatibility
register_template(
    TemplateMeta(
        template_type='hy_v4',
        prefix=[],
        system_prefix=['<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{SYSTEM}}<｜hy_end:opensource｜>'],
        prompt=['<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{QUERY}}<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>'],
        chat_sep=['<｜hy_end:opensource｜>'],
        suffix=['<｜hy_end:opensource｜>'],
        default_system='<｜reasoning_mode:opensource｜>reasoning_effort:no_think',
        non_thinking_prefix='<think:opensource></think:opensource>',
    ),
    exist_ok=True,
)

# Register hy_v4_vl model (multimodal)
# Uses HYV4VLModelLoader to load via AutoModel instead of AutoModelForCausalLM.
register_model(
    ModelMeta(
        model_type='hy_v4_vl',
        model_groups=[
            ModelGroup([
                Model('Tencent-Hunyuan/Hy4-VL',
                      'Tencent-Hunyuan/Hy4-VL'),
            ]),
        ],
        loader=HYV4VLModelLoader,
        template='hy_v4_vl',
        model_arch=HYV4_VL_ARCH_NAME,
        architectures=['HYV4VLForConditionalGeneration'],
        is_multimodal=True,
    ),
    exist_ok=True,
)

# Also register the LLM-only model for backward compatibility
register_model(
    ModelMeta(
        model_type='hy_v4',
        model_groups=[
            ModelGroup([
                Model('Tencent-Hunyuan/Hy4',
                      'Tencent-Hunyuan/Hy4'),
            ]),
        ],
        template='hy_v4',
        architectures=['HYV4ForCausalLM'],
        is_multimodal=False,
    ),
    exist_ok=True,
)

logger.info(
    "[HYV4-VL Patch 1] Model types 'hy_v4_vl' (multimodal) and 'hy_v4' (text-only) "
    "registered in ms-swift."
)

# ============================================================================
# Patch 2: Skip grad norm computation for DeepSpeed ZeRO-3
# ============================================================================

def _apply_skip_grad_norm_patch():
    """Patch DeepSpeed ZeRO-3 optimizer to skip grad norm computation."""
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

        def _skip_get_norm_groups(self):
            return [torch.tensor(0.0)]

        DeepSpeedZeroOptimizer_Stage3._get_norm_groups = _skip_get_norm_groups
        logger.info(
            "[HYV4-VL Patch 2] Patched DeepSpeedZeroOptimizer_Stage3._get_norm_groups "
            "to skip grad norm computation."
        )
    except ImportError:
        logger.info("[HYV4-VL Patch 2] DeepSpeed not available, skipping grad norm patch.")

# ============================================================================
# Patch 3: Memory-efficient model loading for ZeRO-3 and FSDP1
# ============================================================================

def _apply_shard_loading_patch():
    """Ensure efficient model loading is used for both ZeRO-3 and FSDP1."""
    import sys
    from transformers import AutoConfig, PreTrainedModel

    _real_orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

    def _disable_router_logits_if_needed(model):
        if hasattr(model, 'config') and getattr(model.config, 'output_router_logits', False):
            model.config.output_router_logits = False
            print("[HYV4-VL Patch 3] Disabled output_router_logits.", flush=True)
        return model

    def _is_fsdp_requested():
        accel_fsdp = str(os.environ.get("ACCELERATE_USE_FSDP", "")).lower()
        if accel_fsdp in {"1", "true", "yes"}:
            return True
        return "--fsdp" in sys.argv

    def _build_meta_model_for_fsdp(cls, model_path, kwargs):
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        init_kwargs = {}
        torch_dtype = kwargs.get("torch_dtype", None)
        if torch_dtype is not None:
            init_kwargs["torch_dtype"] = torch_dtype
        if "attn_implementation" in kwargs:
            init_kwargs["attn_implementation"] = kwargs["attn_implementation"]
        if "experts_implementation" in kwargs:
            init_kwargs["experts_implementation"] = kwargs["experts_implementation"]

        with torch.device("meta"):
            model = cls._from_config(config, **init_kwargs)
        return _disable_router_logits_if_needed(model)

    def _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("low_cpu_mem_usage", True)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            logger.info(
                "[HYV4-VL Patch 3] FSDP mode: local_rank=%d != 0, creating model on meta device.",
                local_rank,
            )
            return _build_meta_model_for_fsdp(cls, pretrained_model_name_or_path, kwargs)

        logger.info(
            "[HYV4-VL Patch 3] FSDP mode: local_rank=0, loading real weights to CPU."
        )
        kwargs["device_map"] = {"": "cpu"}
        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        return _disable_router_logits_if_needed(model)

    def _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        model_path = pretrained_model_name_or_path
        print(f"[HYV4-VL Patch 3] _ensure_zero3_config_and_load called with path: {model_path}", flush=True)

        if not (isinstance(model_path, str) and os.path.isdir(model_path)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")
        if not (os.path.isfile(index_file) or os.path.isfile(single_file)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        if _is_fsdp_requested():
            return _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs)

        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            print("[HYV4-VL Patch 3] ZeRO-3 already enabled, using native from_pretrained.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        ds_config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", None)
        if ds_config_path is None:
            ds_config_path = os.environ.get("DEEPSPEED_CONFIG", None)

        if ds_config_path is None:
            for i, arg in enumerate(sys.argv):
                if arg == '--deepspeed' and i + 1 < len(sys.argv):
                    ds_config_path = sys.argv[i + 1]
                    break

        if ds_config_path is None or not os.path.isfile(ds_config_path):
            print("[HYV4-VL Patch 3] No DeepSpeed config found, using default from_pretrained.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        with open(ds_config_path, "r") as f:
            ds_config = _json.load(f)

        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
        if zero_stage != 3:
            print(f"[HYV4-VL Patch 3] Not ZeRO-3 (stage={zero_stage}), using default.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        print(f"[HYV4-VL Patch 3] Setting HfDeepSpeedConfig for ZeRO-3 native loading: {ds_config_path}", flush=True)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        _ds_config_obj = HfDeepSpeedConfig(ds_config_path)

        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        print("[HYV4-VL Patch 3] Native ZeRO-3 from_pretrained completed.", flush=True)
        return _disable_router_logits_if_needed(model)

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs)

    PreTrainedModel.from_pretrained = _patched_from_pretrained

    logger.info("[HYV4-VL Patch 3] Loading patch applied for ZeRO-3 and FSDP1.")

# ============================================================================
# Optional Patch: LoRA z3_leaf marking
# ============================================================================

def apply_lora_z3_leaf_patch(model):
    """Mark PEFT LoRA wrapper modules as ZeRO-3 leaf modules."""
    try:
        from deepspeed.utils import set_z3_leaf_module
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        logger.info("[HYV4-VL Optional] DeepSpeed or PEFT not available, skipping z3_leaf patch.")
        return

    z3_leaf_count = 0
    for module in model.modules():
        if isinstance(module, LoraLinear):
            set_z3_leaf_module(module, True)
            z3_leaf_count += 1

    logger.info("[HYV4-VL Optional] Marked %d LoraLinear modules with _z3_leaf=True.", z3_leaf_count)

# ============================================================================
# Patch 4: Fix logging_dir compatibility
# ============================================================================

def _apply_logging_dir_patch():
    """Fix SftArguments._add_version for transformers 5.x compatibility."""
    try:
        from swift.arguments.sft_args import SftArguments

        _orig_add_version = SftArguments._add_version

        def _patched_add_version(self):
            if not hasattr(self, 'logging_dir'):
                self.logging_dir = None
            if not hasattr(self, 'run_name'):
                self.run_name = None
            _orig_add_version(self)

        SftArguments._add_version = _patched_add_version
        logger.info("[HYV4-VL Patch 4] Patched SftArguments._add_version for transformers 5.x.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4-VL Patch 4] Could not apply logging_dir patch: %s", e)

# ============================================================================
# Patch 5: FSDP1 dtype + mixed precision guard
# ============================================================================

def _apply_fsdp_dtype_patch():
    """Monkey-patch Trainer to cast model to bf16 before FSDP wrap.

    Also sets ACCELERATE_MIXED_PRECISION=no at import time (before Accelerate
    initializes) to prevent FSDP from using fp16 as the compute dtype.
    Without this, Accelerate's FSDP bf16 mode sets MixedPrecision(param_dtype=fp16),
    which causes all-gathered parameters to be cast to fp16 during forward —
    fp16's limited dynamic range (max ~65504) leads to NaN overflow in large models.
    """
    # --- Level 0: Set env var BEFORE Accelerate reads it ---
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    os.environ["FSDP_MIXED_PRECISION"] = "no"
    print(
        "[HYV4-VL Patch 5] Set ACCELERATE_MIXED_PRECISION=no at import time "
        "(before Accelerate init) to prevent FSDP fp16 compute dtype.",
        flush=True
    )

    try:
        from transformers import Trainer

        _orig_prepare_for_training = Trainer._prepare_for_training

        def _patched_prepare_for_training(self, *args, **kwargs):
            # Detect FSDP mode using multiple signals for robustness
            is_fsdp = getattr(self, 'is_fsdp_enabled', False) or getattr(self.args, 'fsdp', False)

            if is_fsdp:
                # --- Step 1: Unify all parameter dtypes to bfloat16 ---
                model = self.model
                dtype_counts = {}
                for p in model.parameters():
                    dt = str(p.dtype)
                    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

                if len(dtype_counts) > 1:
                    print(
                        f"[HYV4-VL Patch 5] Mixed dtypes detected before FSDP wrap: {dtype_counts}. "
                        f"Casting all parameters to bfloat16.", flush=True
                    )
                    cast_count = 0
                    for p in model.parameters():
                        if p.dtype != torch.bfloat16 and p.dtype.is_floating_point:
                            p.data = p.data.to(torch.bfloat16)
                            cast_count += 1
                    print(f"[HYV4-VL Patch 5] Cast {cast_count} parameters to bfloat16.", flush=True)
                else:
                    print(
                        f"[HYV4-VL Patch 5] All params already uniform dtype: {dtype_counts}",
                        flush=True
                    )

                # --- Step 2: Force FSDP to use pure bf16 (no fp16 compute) ---
                try:
                    if hasattr(self, 'accelerator'):
                        # Level 1: Set accelerator state mixed_precision to "no"
                        if hasattr(self.accelerator, 'state'):
                            old_mp = getattr(self.accelerator.state, '_mixed_precision', None)
                            self.accelerator.state._mixed_precision = "no"
                            print(
                                f"[HYV4-VL Patch 5] Set accelerator.state._mixed_precision='no' "
                                f"(was: {old_mp}).", flush=True
                            )

                        # Level 2: Set FSDP plugin's mixed_precision_policy to pure bf16
                        fsdp_plugin = getattr(self.accelerator.state, 'fsdp_plugin', None)
                        if fsdp_plugin is not None:
                            try:
                                from torch.distributed.fsdp import MixedPrecision
                                bf16_policy = MixedPrecision(
                                    param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.bfloat16,
                                    buffer_dtype=torch.bfloat16,
                                )
                                fsdp_plugin.mixed_precision_policy = bf16_policy
                                print(
                                    f"[HYV4-VL Patch 5] Set fsdp_plugin.mixed_precision_policy "
                                    f"to pure bf16: {bf16_policy}",
                                    flush=True
                                )
                            except ImportError:
                                # Fallback: clear the policy if MixedPrecision not available
                                fsdp_plugin.mixed_precision_policy = None
                                print(
                                    "[HYV4-VL Patch 5] WARNING: Could not import MixedPrecision, "
                                    "set policy to None.", flush=True
                                )

                            if hasattr(fsdp_plugin, 'kwargs') and isinstance(fsdp_plugin.kwargs, dict):
                                fsdp_plugin.kwargs.pop('mixed_precision', None)

                        # Level 3: Disable native_amp to prevent autocast wrapping
                        if hasattr(self.accelerator, 'native_amp'):
                            old_amp = self.accelerator.native_amp
                            self.accelerator.native_amp = False
                            print(
                                f"[HYV4-VL Patch 5] Set accelerator.native_amp=False "
                                f"(was: {old_amp}).", flush=True
                            )

                    else:
                        print(
                            "[HYV4-VL Patch 5] WARNING: self.accelerator not found, "
                            "cannot disable mixed precision.", flush=True
                        )

                    # Level 4: Environment variables for any lazy initialization
                    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
                    os.environ["FSDP_MIXED_PRECISION"] = "no"

                except Exception as e:
                    print(f"[HYV4-VL Patch 5] WARNING: Failed to disable mixed precision: {e}", flush=True)

            return _orig_prepare_for_training(self, *args, **kwargs)

        Trainer._prepare_for_training = _patched_prepare_for_training
        print("[HYV4-VL Patch 5] Registered: unify model dtype to bf16 before FSDP wrap.", flush=True)
    except (ImportError, AttributeError) as e:
        print(f"[HYV4-VL Patch 5] Could not apply FSDP1 dtype patch: {e}", flush=True)

# ============================================================================
# Patch 6: Disable _compute_acc during training
# ============================================================================

def _apply_disable_compute_acc_patch():
    """Patch _compute_acc to be a no-op during training."""
    try:
        from swift.trainers.mixin import SwiftMixin

        def _noop_compute_acc(self, outputs, labels, cu_seqlens=None):
            return

        SwiftMixin._compute_acc = _noop_compute_acc
        print("[HYV4-VL Patch 6] Disabled _compute_acc to reduce memory usage.", flush=True)
    except (ImportError, AttributeError) as e:
        print(f"[HYV4-VL Patch 6] Could not apply _compute_acc patch: {e}", flush=True)

# ============================================================================
# Patch 7 (VL-specific): Freeze ViT encoder during SFT
#
# For VL SFT, the standard practice is to freeze the ViT encoder and only
# train the projector + LLM. This patch freezes ViT after model loading.
# ============================================================================

def _apply_freeze_vit_patch():
    """Patch 7: ViT freeze is now handled natively by ms-swift.

    We registered MultiModelKeys with vision_tower='model.vision_tower',
    so ms-swift's built-in freeze_vit=True (default) will automatically
    add 'model.vision_tower' to freeze_parameters. No manual patching needed.
    """
    logger.info("[HYV4-VL Patch 7] ViT freeze handled by ms-swift native freeze_vit + model_arch.")

# ============================================================================
# Auto-apply patches on import
# ============================================================================

def _apply_tokenizer_to_processing_class_patch():
    """Patch 8: Fix tokenizer → processing_class for transformers v5 compatibility.

    This patch wraps transformers.Seq2SeqTrainer.__init__ to intercept 'tokenizer'
    in kwargs and rename it to 'processing_class' before calling the original __init__.
    """
    try:
        from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer

        _original_hf_seq2seq_init = HfSeq2SeqTrainer.__init__

        def _patched_hf_seq2seq_init(self, *args, **kwargs):
            if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
                kwargs['processing_class'] = kwargs.pop('tokenizer')
            return _original_hf_seq2seq_init(self, *args, **kwargs)

        HfSeq2SeqTrainer.__init__ = _patched_hf_seq2seq_init
        logger.info("[HYV4-VL Patch 8] tokenizer→processing_class compatibility patch applied.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4-VL Patch 8] Could not apply patch: %s", e)


# ============================================================================
# Patch 9: Make CUDA pin_memory fault-tolerant for DeepSpeed ZeRO-3
# ============================================================================

def _apply_safe_pin_memory_patch():
    """Make CUDA pin_memory fault-tolerant for DeepSpeed ZeRO-3.

    DeepSpeed ZeRO-3 calls get_accelerator().pin_memory(tensor) during
    _create_param_groups_fp16_flat_cpu_memory to create pinned (page-locked)
    CPU buffers for parameter offloading.  This call is hard-coded and NOT
    controlled by the config's ``pin_memory: false`` setting.

    On some nodes, pin_memory() can sporadically fail with
    ``CUDA error: invalid argument`` due to transient driver/cgroup issues.
    This patch wraps the CUDA accelerator's pin_memory method so that if
    the call fails, it falls back to returning the original (unpinned)
    tensor instead of crashing the entire training job.
    """
    try:
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator

        _orig_pin_memory = CUDA_Accelerator.pin_memory

        def _safe_pin_memory(self, tensor):
            try:
                return _orig_pin_memory(self, tensor)
            except Exception as e:
                import torch
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[HYV4-VL Patch 9] WARNING: pin_memory() failed on rank {rank}: {e}. "
                      f"Falling back to unpinned CPU tensor.", flush=True)
                return tensor

        CUDA_Accelerator.pin_memory = _safe_pin_memory
        print("[HYV4-VL Patch 9] Patched CUDA_Accelerator.pin_memory with "
              "fault-tolerant fallback.", flush=True)
    except ImportError:
        pass


_apply_safe_pin_memory_patch()
_apply_skip_grad_norm_patch()
_apply_shard_loading_patch()
_apply_logging_dir_patch()
_apply_fsdp_dtype_patch()
_apply_disable_compute_acc_patch()
_apply_freeze_vit_patch()
_apply_tokenizer_to_processing_class_patch()

logger.info("[HYV4-VL] All ms-swift VL patches loaded successfully.")
