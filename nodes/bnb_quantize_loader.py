"""
BNB 4-bit Quantize Loader Node.

Loads a regular (non-quantized) model from safetensors and quantizes to 
BNB NF4/FP4 format layer-by-layer to avoid OOM.
"""

import gc
import logging
import re
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models_base
import comfy.latent_formats
import comfy.conds

from ..utils.safetensors_loader import MemoryEfficientSafeOpen

# Check for bitsandbytes
_BNB_AVAILABLE = False
_bnb_functional = None

try:
    import bitsandbytes.functional as bnb_F
    _BNB_AVAILABLE = True
    _bnb_functional = bnb_F
    logging.info("BNB4bitQuantizeLoader: bitsandbytes available")
except ImportError:
    logging.error(
        "BNB4bitQuantizeLoader: bitsandbytes NOT available. "
        "This node requires bitsandbytes: pip install bitsandbytes"
    )


def should_exclude(key: str, exclude_patterns: list) -> bool:
    """Check if a tensor key matches any exclusion pattern."""
    if not exclude_patterns:
        return False
    for pattern in exclude_patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        try:
            if re.search(pattern, key):
                return True
        except re.error:
            # Treat as substring match if not valid regex
            if pattern in key:
                return True
    return False


def is_quantizable_weight(key: str, shape: tuple) -> bool:
    """Check if a weight tensor should be quantized."""
    # Only quantize 2D weight tensors (Linear layers)
    if not key.endswith('.weight'):
        return False
    if len(shape) != 2:
        return False
    # Skip very small tensors
    if shape[0] < 64 or shape[1] < 64:
        return False
    return True


class BNB4bitFluxConfig(comfy.supported_models_base.BASE):
    """Minimal model config for BNB 4-bit Flux models."""
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Flux
    memory_usage_factor = 2.8
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def __init__(self, is_flux2=False):
        self.unet_config = {}
        self.latent_format = comfy.latent_formats.Flux2() if is_flux2 else comfy.latent_formats.Flux()
        self.unet_config["disable_unet_model_creation"] = True
        if is_flux2:
            self.memory_usage_factor = 2.8 * 4 * 2.36


class BNB4bitFluxModel(comfy.model_base.BaseModel):
    """Base model class for BNB 4-bit Flux loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out


class BNB4bitQuantizeLoader:
    """
    Load a regular model and quantize to BNB 4-bit (NF4/FP4) on-the-fly.
    
    Processes layers one-by-one to avoid OOM. Supports exclusion patterns
    for layers that should remain at high precision.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "quant_type": (["nf4", "fp4"],),
                "blocksize": ([64, 128, 256], {"default": 64}),
            },
            "optional": {
                "exclude_patterns": (
                    "STRING",
                    {
                        "default": "img_in,final_layer",
                        "multiline": False,
                        "tooltip": "Comma-separated patterns for layers to keep at high precision (e.g., 'img_in,norm,final_layer')",
                    },
                ),
                "model_type_override": (["auto", "flux2", "flux", "chroma", "chroma_radiance", "chroma_radiance_x0"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_quantize"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load a regular model and quantize to BNB 4-bit (NF4/FP4) on-the-fly. Memory-efficient layer-by-layer processing."

    def _detect_model_type(self, keys):
        """Detect model type from state dict keys."""
        def has_key_pattern(pattern):
            return any(pattern in k for k in keys)

        if has_key_pattern("double_stream_modulation_img.lin.weight"):
            return "flux2"

        has_distilled = has_key_pattern("distilled_guidance_layer.")
        has_nerf = has_key_pattern("nerf_blocks.")
        has_x0 = has_key_pattern("__x0__")

        if has_distilled:
            if has_nerf:
                if has_x0:
                    return "chroma_radiance_x0"
                else:
                    return "chroma_radiance"
            else:
                return "chroma"

        return "flux"

    def _count_blocks(self, keys, prefix):
        """Count number of blocks from state dict keys."""
        max_idx = -1
        for k in keys:
            if prefix in k:
                try:
                    idx = int(k.split(prefix)[1].split('.')[0])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    pass
        return max_idx + 1 if max_idx >= 0 else 0

    def load_and_quantize(
        self,
        unet_name,
        quant_type,
        blocksize,
        exclude_patterns="",
        model_type_override="auto",
    ):
        """Load and quantize a model to BNB 4-bit format."""
        if not _BNB_AVAILABLE:
            raise RuntimeError(
                "bitsandbytes is required for this node. "
                "Install with: pip install bitsandbytes"
            )

        import comfy.model_management as model_management
        import comfy.ldm.flux.model as flux_model

        try:
            from ..bnb4bit_ops import HybridBNB4bitOps
        except ImportError as e:
            logging.error(f"Failed to import HybridBNB4bitOps: {e}")
            raise

        # Parse exclusion patterns
        exclude_list = [p.strip() for p in exclude_patterns.split(',') if p.strip()]
        
        # Get model path
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        logging.info(f"BNB4bitQuantizeLoader: Loading {unet_name}")
        logging.info(f"  Quant type: {quant_type}, blocksize: {blocksize}")
        logging.info(f"  Exclude patterns: {exclude_list}")

        # Estimate model size and prepare memory by unloading cached models
        try:
            import os
            model_size_bytes = os.path.getsize(unet_path)
            model_size_gb = model_size_bytes / (1024**3)
            
            # Request memory from ComfyUI (unloads cached models)
            # We need ~1.5x the layer size for: load + CUDA copy + quantized output
            # But since we process layer-by-layer, we just need headroom for the largest layer
            # Estimate largest layer as ~0.5% of model size for safety
            headroom_gb = max(0.5, model_size_gb * 0.02)
            headroom_bytes = int(headroom_gb * 1024**3)
            
            logging.info(f"  Model size: {model_size_gb:.2f}GB, requesting {headroom_gb:.2f}GB VRAM headroom")
            model_management.free_memory(headroom_bytes, model_management.get_torch_device())
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"  Could not prepare memory: {e}")

        # Open model with memory-efficient loader
        quantized_sd = {}
        all_keys = []
        
        with MemoryEfficientSafeOpen(unet_path, mmap_mode=True) as f:
            all_keys = list(f.keys())
            total_keys = len(all_keys)
            quantized_count = 0
            excluded_count = 0
            
            for i, key in enumerate(all_keys):
                # Strip prefix if present
                clean_key = key
                if key.startswith("model.diffusion_model."):
                    clean_key = key[22:]
                
                # Load tensor
                tensor = f.get_tensor(key)
                header_meta = f.header.get(key, {})
                shape = header_meta.get("shape", list(tensor.shape))
                
                # Check if this should be quantized
                if is_quantizable_weight(clean_key, tuple(shape)):
                    # Check exclusion filter
                    if should_exclude(clean_key, exclude_list):
                        logging.debug(f"  [{i+1}/{total_keys}] EXCLUDE: {clean_key}")
                        # Store on CPU explicitly, then free original
                        quantized_sd[clean_key] = tensor.cpu().clone()
                        del tensor
                        excluded_count += 1
                        continue
                    
                    # Quantize using BNB
                    try:
                        # Move to CUDA for quantization
                        weight_cuda = tensor.to('cuda', dtype=torch.bfloat16)
                        # Free original tensor immediately to save RAM
                        del tensor
                        
                        # Quantize
                        packed, quant_state = _bnb_functional.quantize_4bit(
                            weight_cuda,
                            blocksize=blocksize,
                            quant_type=quant_type,
                            compress_statistics=False,
                        )
                        
                        # Free GPU tensor immediately after quantization
                        del weight_cuda
                        
                        # Store in BNB format (move to CPU)
                        quantized_sd[clean_key] = packed.cpu()
                        quantized_sd[f"{clean_key}.absmax"] = quant_state.absmax.cpu()
                        quantized_sd[f"{clean_key}.quant_map"] = quant_state.code.cpu()
                        
                        # Free quant_state GPU tensors
                        del packed, quant_state
                        
                        # Store metadata as JSON tensor
                        import json
                        
                        metadata = {
                            "dtype": "bfloat16",  # We converted to bf16 for quantization
                            "shape": list(shape),
                            "blocksize": blocksize,
                            "quant_type": quant_type,
                        }
                        json_bytes = json.dumps(metadata).encode('utf-8')
                        qs_tensor = torch.tensor(list(json_bytes), dtype=torch.uint8)
                        quantized_sd[f"{clean_key}.quant_state.bitsandbytes__{quant_type}"] = qs_tensor
                        
                        quantized_count += 1
                        
                        # Progress and periodic cleanup (every 20 layers)
                        if (i + 1) % 20 == 0:
                            logging.info(f"  [{i+1}/{total_keys}] Quantized {quantized_count} layers...")
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                    except Exception as e:
                        logging.warning(f"  Failed to quantize {clean_key}: {e}, keeping original")
                        # tensor might already be deleted, reload if needed
                        if 'tensor' not in dir():
                            tensor = f.get_tensor(key)
                        quantized_sd[clean_key] = tensor.cpu().clone()
                        del tensor
                        excluded_count += 1
                else:
                    # Non-quantizable tensor (bias, norm, etc.) - store on CPU
                    quantized_sd[clean_key] = tensor.cpu().clone()
                    del tensor

        logging.info(f"BNB4bitQuantizeLoader: Quantized {quantized_count} layers, excluded {excluded_count}")

        # Detect model type
        if model_type_override == "auto":
            model_type = self._detect_model_type(all_keys)
            logging.info(f"BNB4bitQuantizeLoader: Auto-detected model type: {model_type}")
        else:
            model_type = model_type_override
            logging.info(f"BNB4bitQuantizeLoader: Using override model type: {model_type}")

        is_flux2 = model_type == "flux2"
        is_chroma = model_type in ("chroma", "chroma_radiance", "chroma_radiance_x0")

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        unet_dtype = torch.bfloat16

        # Build FluxParams
        depth = self._count_blocks(all_keys, "double_blocks.")
        depth_single_blocks = self._count_blocks(all_keys, "single_blocks.")

        if model_type == "flux2":
            hidden_size = 6144
            context_in_dim = 15360
            vec_in_dim = 256
            params = flux_model.FluxParams(
                in_channels=128,
                out_channels=128,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=3.0,
                num_heads=48,
                depth=depth if depth > 0 else 8,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 48,
                axes_dim=[32, 32, 32, 32],
                theta=2000,
                patch_size=1,
                qkv_bias=False,
                guidance_embed=True,
                txt_ids_dims=[3],
                global_modulation=True,
                mlp_silu_act=True,
                ops_bias=False,
            )
        elif model_type == "chroma":
            hidden_size = 3072
            context_in_dim = 4096
            vec_in_dim = 768
            params = flux_model.FluxParams(
                in_channels=64,
                out_channels=64,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=2,
                qkv_bias=True,
                guidance_embed=False,
                txt_ids_dims=[],
            )
        elif model_type in ("chroma_radiance", "chroma_radiance_x0"):
            hidden_size = 3072
            context_in_dim = 4096
            vec_in_dim = 768
            params = flux_model.FluxParams(
                in_channels=3,
                out_channels=3,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=16,
                qkv_bias=True,
                guidance_embed=False,
                txt_ids_dims=[],
            )
        else:  # flux (default)
            hidden_size = 3072
            context_in_dim = 4096
            vec_in_dim = 768
            params = flux_model.FluxParams(
                in_channels=16,
                out_channels=16,
                vec_in_dim=vec_in_dim,
                context_in_dim=context_in_dim,
                hidden_size=hidden_size,
                mlp_ratio=4.0,
                num_heads=24,
                depth=depth if depth > 0 else 19,
                depth_single_blocks=depth_single_blocks if depth_single_blocks > 0 else 38,
                axes_dim=[16, 56, 56],
                theta=10000,
                patch_size=2,
                qkv_bias=True,
                guidance_embed=True,
                txt_ids_dims=[],
            )

        # Create model config and base model
        model_conf = BNB4bitFluxConfig(is_flux2=is_flux2)
        model_conf.set_inference_dtype(unet_dtype, unet_dtype)
        model = BNB4bitFluxModel(
            model_conf,
            model_type=comfy.model_base.ModelType.FLUX,
            device=load_device
        )

        logging.info(f"BNB4bitQuantizeLoader: Creating Flux model with HybridBNB4bitOps")

        # Create diffusion model with our custom ops
        model.diffusion_model = flux_model.Flux(
            device=offload_device,
            dtype=unet_dtype,
            operations=HybridBNB4bitOps,
            **{k: getattr(params, k) for k in params.__dataclass_fields__}
        )
        model.diffusion_model.eval()
        model.diffusion_model.dtype = unet_dtype

        # Load quantized weights
        m, u = model.diffusion_model.load_state_dict(quantized_sd, strict=False)
        if len(m) > 0:
            logging.warning(f"BNB4bitQuantizeLoader: missing keys: {len(m)}")
            logging.debug(f"Missing: {m[:10]}...")
        if len(u) > 0:
            logging.warning(f"BNB4bitQuantizeLoader: unexpected keys: {len(u)}")
            logging.debug(f"Unexpected: {u[:10]}...")

        logging.info(f"BNB4bitQuantizeLoader: Successfully loaded and quantized {unet_name}")

        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (patcher,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "BNB4bitQuantizeLoader": BNB4bitQuantizeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BNB4bitQuantizeLoader": "Load & Quantize to BNB 4-bit",
}
