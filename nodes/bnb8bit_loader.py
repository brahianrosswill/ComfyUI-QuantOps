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


def is_bnb8bit_prequantized(file_path: str) -> bool:
    """
    Check if a safetensors file contains pre-quantized BNB INT8 weights.
    
    Looks for .quant_state.bitsandbytes__int8 keys in the header.
    """
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            return any(k.endswith('.quant_state.bitsandbytes__int8') for k in keys)
    except Exception:
        return False


def tensor_to_dict(tensor_data):
    """Decode JSON metadata from uint8 tensor."""
    import json
    try:
        byte_data = bytes(tensor_data.tolist())
        json_str = byte_data.decode('utf-8')
        return json.loads(json_str)
    except Exception:
        return {}


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


class BNB8bitPrequantizedLinear(torch.nn.Module):
    """
    Linear layer that loads pre-quantized INT8 weights.
    
    Expects state dict to contain INT8 weights and SCB scale factors.
    """
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 prequantized_sd=None, layer_prefix=""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self._prequantized_sd = prequantized_sd or {}
        self._layer_prefix = layer_prefix
        
        # Placeholder weight - will be replaced during load_state_dict
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.int8),
            requires_grad=False
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
        
        self._bnb_linear = None
        self._loaded = False
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom loading to handle pre-quantized weights."""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        # Check if we have SCB scales in the prequantized state dict
        scb_key = f"{prefix}weight.SCB"
        full_key = f"{self._layer_prefix}.weight.SCB" if self._layer_prefix else f"{prefix}weight.SCB"
        
        if full_key in self._prequantized_sd:
            self._scb = self._prequantized_sd[full_key]
        elif scb_key in self._prequantized_sd:
            self._scb = self._prequantized_sd[scb_key]
        else:
            self._scb = None
    
    def _maybe_create_bnb_linear(self, device):
        """Create BNB linear from loaded INT8 weights."""
        if self._loaded or not _BNB_AVAILABLE:
            return
        
        # Create BNB Linear8bitLt
        self._bnb_linear = _bnb.nn.Linear8bitLt(
            self.in_features,
            self.out_features,
            bias=self.has_bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        
        # Set the pre-quantized weight
        with torch.no_grad():
            weight_int8 = self.weight.data.to(device=device)
            self._bnb_linear.weight = _bnb.nn.Int8Params(
                weight_int8,
                requires_grad=False,
                has_fp16_weights=False,
            )
            # Set the scales if we have them
            if hasattr(self, '_scb') and self._scb is not None:
                self._bnb_linear.weight.SCB = self._scb.to(device=device)
            
            if self.bias is not None:
                self._bnb_linear.bias = torch.nn.Parameter(
                    self.bias.data.to(device=device, dtype=torch.float16),
                    requires_grad=False
                )
        
        self._loaded = True
        self.weight = None
        self.bias = None
    
    def forward(self, x):
        device = x.device
        
        if not _BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes required for pre-quantized INT8 inference")
        
        self._maybe_create_bnb_linear(device)
        return self._bnb_linear(x)


def make_bnb8bit_ops():
    """Create ops class that uses BNB 8-bit for Linear layers."""
    
    class BNB8bitOps(comfy.ops.manual_cast):
        class Linear(BNB8bitLinear):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(*args, device=device, dtype=dtype, **kwargs)
    
    return BNB8bitOps


def make_bnb8bit_prequantized_ops(prequantized_sd):
    """Create ops class that loads pre-quantized INT8 weights."""
    
    class BNB8bitPrequantizedOps(comfy.ops.manual_cast):
        class Linear(BNB8bitPrequantizedLinear):
            _shared_sd = prequantized_sd
            
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(*args, device=device, dtype=dtype, 
                                prequantized_sd=self._shared_sd, **kwargs)
    
    return BNB8bitPrequantizedOps


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
    DESCRIPTION = "Load text encoder and quantize to BNB INT8 (LLM.int8()). Auto-detects pre-quantized models. Uses native INT8 tensor cores with outlier handling."

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
        
        # Check if already pre-quantized
        is_prequantized = is_bnb8bit_prequantized(clip_path)
        
        if is_prequantized:
            logging.info(f"BNB8bitCLIPLoader: Loading pre-quantized {clip_name}")
            return self._load_prequantized(clip_path, clip_type)
        else:
            logging.info(f"BNB8bitCLIPLoader: Loading and quantizing {clip_name}")
            logging.info(f"  Type: {type}, Outlier threshold: {threshold}")
            return self._load_and_quantize(clip_path, clip_type, threshold)
    
    def _load_and_quantize(self, clip_path, clip_type, threshold):
        """Load FP16 model and quantize to INT8 on first forward."""
        # Load state dict (FP16 weights)
        sd = comfy.utils.load_torch_file(clip_path, safe_load=True)
        
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
        
        logging.info(f"BNB8bitCLIPLoader: Successfully loaded (will quantize on first use)")
        
        return (clip,)
    
    def _load_prequantized(self, clip_path, clip_type):
        """Load pre-quantized INT8 model directly."""
        from safetensors import safe_open
        
        # Load state dict
        sd = {}
        with safe_open(clip_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        
        # Create ops that can handle pre-quantized weights
        model_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device(),
            "custom_operations": make_bnb8bit_prequantized_ops(sd),
        }
        
        # Build a clean state dict for ComfyUI (without quant metadata keys)
        clean_sd = {}
        for key, value in sd.items():
            # Skip quant_state and SCB keys - they're handled by the ops
            if '.quant_state.' in key or key.endswith('.SCB'):
                continue
            clean_sd[key] = value
        
        # Load text encoder using ComfyUI's API
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[clean_sd],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        
        logging.info(f"BNB8bitCLIPLoader: Successfully loaded pre-quantized model")
        
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
        # Check for both quantize-on-load and pre-quantized linear types
        is_bnb8bit_linear = isinstance(module, BNB8bitLinear) and module._quantized and module._bnb_linear is not None
        is_prequant_linear = isinstance(module, BNB8bitPrequantizedLinear) and module._loaded and module._bnb_linear is not None
        
        if is_bnb8bit_linear or is_prequant_linear:
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
        
        # Try to find the underlying models in CLIP wrapper
        # CLIP in ComfyUI has .cond_stage_model (for SD) or multiple clip_l/clip_g/t5xxl models
        models_to_check = []
        
        # Direct model attributes
        if hasattr(clip, 'cond_stage_model'):
            models_to_check.append(clip.cond_stage_model)
        if hasattr(clip, 'patcher') and hasattr(clip.patcher, 'model'):
            models_to_check.append(clip.patcher.model)
        if hasattr(clip, 'model'):
            models_to_check.append(clip.model)
        
        # For Flux/SD3 style CLIP with multiple encoders
        for attr in ['clip_l', 'clip_g', 't5xxl', 'clip_h', 't5_model']:
            if hasattr(clip, attr):
                models_to_check.append(getattr(clip, attr))
            # Also check nested cond_stage_model
            if hasattr(clip, 'cond_stage_model') and hasattr(clip.cond_stage_model, attr):
                models_to_check.append(getattr(clip.cond_stage_model, attr))
        
        # Extract from all found models
        state_dict = {}
        for model in models_to_check:
            if model is not None:
                extracted = extract_bnb8bit_state_dict(model)
                state_dict.update(extracted)
        
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
