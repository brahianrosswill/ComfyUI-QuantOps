"""
BNB 8-bit Text Encoder Loader Node.

Loads text encoders and converts to BNB INT8 (LLM.int8()) on-the-fly.
Uses bitsandbytes native INT8 tensor core kernels with outlier handling.
"""

import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.ops
import comfy.model_management

# Check for bitsandbytes
_BNB_AVAILABLE = False
_bnb = None

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
    _bnb = bnb
    logging.info("BNB8bitCLIPLoader: bitsandbytes available")
except ImportError:
    logging.warning(
        "BNB8bitCLIPLoader: bitsandbytes not available. "
        "Install with: pip install bitsandbytes"
    )


class BNB8bitLinear(torch.nn.Module):
    """
    Wrapper that loads FP16 weights and converts to BNB INT8 on first forward.
    
    Uses bitsandbytes.nn.Linear8bitLt for LLM.int8() quantization with
    outlier handling and native INT8 tensor core kernels.
    """
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Placeholder weight - will be replaced during load_state_dict
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
        
        # BNB linear module - created lazily
        self._bnb_linear = None
        self._quantized = False
    
    def _maybe_quantize(self, device):
        """Quantize to INT8 on first forward pass."""
        if self._quantized or not _BNB_AVAILABLE:
            return
        
        # Create BNB Linear8bitLt with the loaded weight
        self._bnb_linear = _bnb.nn.Linear8bitLt(
            self.in_features,
            self.out_features,
            bias=self.has_bias,
            has_fp16_weights=False,  # We'll quantize
            threshold=6.0,  # Default outlier threshold
        )
        
        # Copy weight data and quantize
        with torch.no_grad():
            # Move to CUDA for quantization
            weight_cuda = self.weight.data.to(device=device, dtype=torch.float16)
            self._bnb_linear.weight = _bnb.nn.Int8Params(
                weight_cuda,
                requires_grad=False,
                has_fp16_weights=False,
            )
            if self.bias is not None:
                self._bnb_linear.bias = torch.nn.Parameter(
                    self.bias.data.to(device=device, dtype=torch.float16),
                    requires_grad=False
                )
        
        self._quantized = True
        
        # Free original weight to save memory
        self.weight = None
        if hasattr(self, 'bias') and self.bias is not None:
            self.bias = None
    
    def forward(self, x):
        device = x.device
        
        if not _BNB_AVAILABLE:
            # Fallback to regular linear if BNB not available
            return torch.nn.functional.linear(x, self.weight.to(x.dtype), 
                                              self.bias.to(x.dtype) if self.bias is not None else None)
        
        # Quantize on first forward
        self._maybe_quantize(device)
        
        # Use BNB 8-bit forward
        return self._bnb_linear(x)


def make_bnb8bit_ops():
    """Create ops class that uses BNB 8-bit for Linear layers."""
    
    class BNB8bitOps(comfy.ops.manual_cast):
        class Linear(BNB8bitLinear):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(*args, device=device, dtype=dtype, **kwargs)
    
    return BNB8bitOps


# CLIPType options matching built-in CLIPLoader
CLIP_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux",
    "hunyuan_video",
    "flux2",
    "ovis",
]


class BNB8bitCLIPLoader:
    """
    Load text encoders and quantize to BNB INT8 (LLM.int8()) on-the-fly.
    
    Uses bitsandbytes native INT8 tensor core kernels with outlier handling.
    Outliers (values > threshold) are computed in FP16 for accuracy.
    
    Best for: T5XXL, CLIP-L, and other transformer text encoders.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (CLIP_TYPES,),
            },
            "optional": {
                "threshold": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Outlier threshold for LLM.int8(). Values above this are computed in FP16.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load text encoder and quantize to BNB INT8 (LLM.int8()). Uses native INT8 tensor cores with outlier handling."

    def load_clip(self, clip_name, type, threshold=6.0):
        """Load a CLIP/text encoder with BNB INT8 quantization."""
        if not _BNB_AVAILABLE:
            raise RuntimeError(
                "bitsandbytes is required for this node. "
                "Install with: pip install bitsandbytes"
            )
        
        # Get clip path
        clip_path = folder_paths.get_full_path("text_encoders", clip_name)
        
        # Convert type string to CLIPType enum
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )
        
        # Load state dict (FP16 weights)
        sd = comfy.utils.load_torch_file(clip_path, safe_load=True)
        
        logging.info(f"BNB8bitCLIPLoader: Loading {clip_name}")
        logging.info(f"  Type: {type}, Outlier threshold: {threshold}")
        
        # Set up model options with BNB 8-bit ops
        model_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device(),
            "custom_operations": make_bnb8bit_ops(),
        }
        
        # Load text encoder using ComfyUI's API
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        
        logging.info(f"BNB8bitCLIPLoader: Successfully loaded {clip_name}")
        
        return (clip,)


def extract_bnb8bit_state_dict(model):
    """
    Extract state dict from a model with BNB 8-bit quantized layers.
    
    Returns a state dict with:
    - layer.weight: INT8 quantized weights (CB matrix)
    - layer.weight.SCB: FP32 scale factors
    - layer.weight.quant_state.bitsandbytes__int8: metadata JSON
    - layer.bias: FP16 bias (if present)
    """
    import json
    
    state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, BNB8bitLinear) and module._quantized and module._bnb_linear is not None:
            bnb_module = module._bnb_linear
            weight = bnb_module.weight
            
            # Get the layer name prefix
            prefix = name + "." if name else ""
            
            # Extract INT8 weights (CB matrix)
            if hasattr(weight, 'CB') and weight.CB is not None:
                state_dict[f"{prefix}weight"] = weight.CB.cpu().to(torch.int8)
            elif hasattr(weight, 'data'):
                state_dict[f"{prefix}weight"] = weight.data.cpu().to(torch.int8)
            
            # Extract scale factors (SCB)
            if hasattr(weight, 'SCB') and weight.SCB is not None:
                state_dict[f"{prefix}weight.SCB"] = weight.SCB.cpu().to(torch.float32)
            
            # Store metadata
            metadata = {
                "format": "bnb_int8",
                "shape": list(bnb_module.weight.shape) if hasattr(bnb_module.weight, 'shape') else [module.out_features, module.in_features],
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_fp16_weights": False,
            }
            json_bytes = json.dumps(metadata).encode('utf-8')
            qs_tensor = torch.tensor(list(json_bytes), dtype=torch.uint8)
            state_dict[f"{prefix}weight.quant_state.bitsandbytes__int8"] = qs_tensor
            
            # Bias
            if bnb_module.bias is not None:
                state_dict[f"{prefix}bias"] = bnb_module.bias.cpu()
    
    return state_dict


class BNB8bitCLIPSaver:
    """
    Save a BNB INT8 quantized text encoder to safetensors.
    
    Saves the INT8 weights with scale factors for efficient loading.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename": ("STRING", {"default": "text_encoder_bnb8bit"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_clip"
    CATEGORY = "loaders/quantized"
    OUTPUT_NODE = True
    DESCRIPTION = "Save a BNB INT8 quantized text encoder to safetensors."

    def save_clip(self, clip, filename):
        """Save the quantized CLIP model."""
        import os
        from safetensors.torch import save_file
        
        # Get the underlying model
        # CLIP wrapper has .cond_stage_model or similar
        model = None
        if hasattr(clip, 'cond_stage_model'):
            model = clip.cond_stage_model
        elif hasattr(clip, 'patcher') and hasattr(clip.patcher, 'model'):
            model = clip.patcher.model
        elif hasattr(clip, 'model'):
            model = clip.model
        
        if model is None:
            raise ValueError("Could not find model inside CLIP object")
        
        # Extract quantized state dict
        state_dict = extract_bnb8bit_state_dict(model)
        
        if not state_dict:
            raise ValueError("No BNB 8-bit quantized layers found in model")
        
        # Save to text_encoders folder
        output_dir = folder_paths.get_folder_paths("text_encoders")[-1]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.safetensors")
        
        # Add metadata
        metadata = {
            "format": "bitsandbytes_int8",
            "format_version": "1.0",
        }
        
        save_file(state_dict, output_path, metadata=metadata)
        
        logging.info(f"BNB8bitCLIPSaver: Saved {len(state_dict)} tensors to {output_path}")
        
        return (output_path,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "BNB8bitCLIPLoader": BNB8bitCLIPLoader,
    "BNB8bitCLIPSaver": BNB8bitCLIPSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BNB8bitCLIPLoader": "Load Text Encoder (BNB INT8)",
    "BNB8bitCLIPSaver": "Save Text Encoder (BNB INT8)",
}
