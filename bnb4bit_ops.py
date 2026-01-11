"""
Native BNB 4-bit Operations for NF4/FP4 quantized models.

This module provides custom ops that handle bitsandbytes-compatible 4-bit quantized
models using native bitsandbytes kernels when available, with a pure-PyTorch fallback.

State dict format (per quantized weight):
    {prefix}weight: Packed 4-bit indices, shape [numel/2, 1], dtype uint8
    {prefix}weight.absmax: Per-block scales, shape [num_blocks], dtype float32
    {prefix}weight.quant_map: Codebook, shape [16], dtype float32
    {prefix}weight.quant_state.bitsandbytes__nf4: JSON metadata as uint8

The JSON metadata contains:
    - dtype: Original weight dtype (e.g., "bfloat16")
    - shape: Original weight shape (e.g., [3072, 3072])
    - blocksize: Elements per quantization block (e.g., 64)
    - quant_type: "nf4" or "fp4"
"""

import json
import logging
import torch
import torch.nn.functional as F
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight


# =============================================================================
# BNB Availability Check
# =============================================================================

_BNB_AVAILABLE = False
_bnb_functional = None
_bnb_matmul_4bit = None
_QuantState = None

try:
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.functional import QuantState
    # matmul_4bit location changed in bitsandbytes 0.48+
    try:
        from bitsandbytes.autograd._functions import matmul_4bit as bnb_matmul_4bit_fn
    except ImportError:
        from bitsandbytes import matmul_4bit as bnb_matmul_4bit_fn
    
    _BNB_AVAILABLE = True
    _bnb_functional = bnb_F
    _bnb_matmul_4bit = bnb_matmul_4bit_fn
    _QuantState = QuantState
    logging.info("ComfyUI-QuantOps: bitsandbytes available, using native 4-bit kernels")
except Exception as e:
    logging.warning(
        f"ComfyUI-QuantOps: bitsandbytes import failed: {e}. "
        "Using pure-PyTorch fallback. Install bitsandbytes for better performance: pip install bitsandbytes"
    )


def is_bnb_available() -> bool:
    """Check if bitsandbytes is available for native 4-bit operations."""
    return _BNB_AVAILABLE


# =============================================================================
# Fallback: Pure-PyTorch Codebooks (when BNB not available)
# =============================================================================

# NF4 (Normal Float 4-bit) quantization table
NF4_QUANT_MAP = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)

# FP4 (Floating Point 4-bit) quantization table
FP4_QUANT_MAP = torch.tensor([
    0.0,
    0.00520833,
    0.16666667,
    0.25,
    0.33333333,
    0.5,
    0.66666667,
    1.0,
    -0.0,
    -0.00520833,
    -0.16666667,
    -0.25,
    -0.33333333,
    -0.5,
    -0.66666667,
    -1.0,
], dtype=torch.float32)


# =============================================================================
# Utility Functions
# =============================================================================

def tensor_to_dict(tensor_data: torch.Tensor) -> dict:
    """Convert a uint8 tensor containing JSON bytes back to a dictionary."""
    byte_data = bytes(tensor_data.cpu().tolist())
    json_str = byte_data.decode("utf-8")
    return json.loads(json_str)


def get_quant_map(quant_type: str, device: torch.device) -> torch.Tensor:
    """Get the quantization codebook for NF4 or FP4."""
    if quant_type == "nf4":
        return NF4_QUANT_MAP.to(device)
    elif quant_type == "fp4":
        return FP4_QUANT_MAP.to(device)
    else:
        logging.warning(f"Unknown quant_type '{quant_type}', defaulting to NF4")
        return NF4_QUANT_MAP.to(device)


def preprocess_bnb_state_dict(state_dict: dict) -> dict:
    """
    Preprocess a BNB 4-bit quantized state dict for model detection.

    ComfyUI's model detection examines weight tensor shapes to determine model
    architecture. BNB 4-bit packed weights have shape [N*K/2, 1] instead of
    original [N, K], causing detection to fail.

    This function omits packed weight keys from the detection state dict,
    allowing hardcoded defaults to be used.
    """
    new_sd = {}
    packed_weight_keys = set()

    # First pass: identify all BNB 4-bit packed weight keys
    for key in state_dict.keys():
        if '.quant_state.bitsandbytes__nf4' in key or '.quant_state.bitsandbytes__fp4' in key:
            weight_key = key.rsplit('.quant_state.', 1)[0]
            packed_weight_keys.add(weight_key)

    # Second pass: copy all keys EXCEPT the packed weight keys themselves
    for key, value in state_dict.items():
        if key in packed_weight_keys:
            logging.debug(f"Omitting packed weight {key} for detection (defaults will be used)")
            continue
        new_sd[key] = value

    logging.info(f"BNB preprocess: omitted {len(packed_weight_keys)} packed weight keys for detection")
    return new_sd


def get_original_shape(state_dict: dict, weight_key: str) -> tuple:
    """
    Get original shape of a BNB 4-bit quantized weight from its quant_state.
    """
    for suffix in ['.quant_state.bitsandbytes__nf4', '.quant_state.bitsandbytes__fp4']:
        qs_key = weight_key + suffix
        if qs_key in state_dict:
            try:
                qs = tensor_to_dict(state_dict[qs_key])
                shape = qs.get('shape', None)
                if shape:
                    return tuple(shape)
            except Exception as e:
                logging.warning(f"Failed to parse quant_state for {weight_key}: {e}")
    return None


def _build_qs_dict_for_layer(state_dict: dict, prefix: str, quant_type: str) -> dict:
    """
    Build a QuantState-compatible dict from state dict keys for a layer.
    
    Args:
        state_dict: Full model state dict
        prefix: Layer prefix (e.g., "model.layers.0.attn.qkv.")
        quant_type: "nf4" or "fp4"
    
    Returns:
        Dict suitable for QuantState.from_dict()
    """
    qs_dict = {}
    
    # Get the packed quant_state tensor (contains JSON metadata)
    qs_key = f"{prefix}weight.quant_state.bitsandbytes__{quant_type}"
    if qs_key in state_dict:
        qs_dict[f"quant_state.bitsandbytes__{quant_type}"] = state_dict[qs_key]
    
    # Get absmax
    absmax_key = f"{prefix}weight.absmax"
    if absmax_key in state_dict:
        qs_dict["absmax"] = state_dict[absmax_key]
    
    # Get quant_map
    qmap_key = f"{prefix}weight.quant_map"
    if qmap_key in state_dict:
        qs_dict["quant_map"] = state_dict[qmap_key]
    
    return qs_dict


# =============================================================================
# Fallback: Pure-PyTorch Dequantization
# =============================================================================

def dequantize_bnb_4bit_pytorch(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    quant_map: torch.Tensor,
    blocksize: int,
    original_shape: tuple,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize BNB 4-bit packed weights using pure PyTorch (fallback).
    """
    device = packed.device
    quant_map = quant_map.to(device)
    absmax = absmax.to(device)

    packed_flat = packed.flatten()

    # Unpack nibbles
    low_indices = (packed_flat & 0x0F).to(torch.long)
    high_indices = (packed_flat >> 4).to(torch.long)
    indices = torch.stack([low_indices, high_indices], dim=-1).flatten()

    # Look up values in quant_map
    values = quant_map[indices]

    # Scale by absmax
    n_blocks = absmax.numel()
    n_elements = values.numel()
    values_per_block = n_elements // n_blocks

    if values_per_block * n_blocks <= n_elements:
        values_blocked = values[:n_blocks * values_per_block].view(n_blocks, values_per_block)
        dequantized = values_blocked * absmax.view(-1, 1).to(values.dtype)
    else:
        dequantized = values * absmax.repeat_interleave(values_per_block)[:n_elements].to(values.dtype)

    original_numel = 1
    for s in original_shape:
        original_numel *= s

    dequantized_flat = dequantized.flatten()[:original_numel]
    return dequantized_flat.view(original_shape).to(target_dtype)


# =============================================================================
# Main Ops Class
# =============================================================================

class HybridBNB4bitOps(manual_cast):
    """
    Hybrid BNB 4-bit operations class for NF4/FP4 quantized models.

    Uses native bitsandbytes kernels when available, falls back to pure-PyTorch.
    
    LoRA patching is NOT supported via dynamic ops. Use a dedicated LoRA merge
    node that does layer-by-layer dequant -> merge -> requant.
    """

    class Linear(manual_cast.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            kwargs['device'] = 'cpu'
            super().__init__(in_features, out_features, *args, **kwargs)
            
            # BNB 4-bit state
            self.is_bnb_4bit = False
            self.quant_state = None  # Native BNB QuantState object
            self.packed_weight = None
            
            # Fallback state (when BNB not available)
            self.absmax = None
            self.quant_map = None
            self.blocksize = 64
            self.original_shape = None
            self.original_dtype = torch.bfloat16
            self.quant_type = "nf4"

        def reset_parameters(self):
            return None

        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            """
            Custom state dict loading that handles BNB 4-bit format.
            
            Reconstructs native BNB QuantState when bitsandbytes is available.
            """
            weight_key = prefix + 'weight'

            # Check for BNB 4-bit format
            quant_state_key_nf4 = prefix + 'weight.quant_state.bitsandbytes__nf4'
            quant_state_key_fp4 = prefix + 'weight.quant_state.bitsandbytes__fp4'

            quant_state_tensor = state_dict.pop(quant_state_key_nf4, None)
            if quant_state_tensor is not None:
                self.quant_type = "nf4"
            else:
                quant_state_tensor = state_dict.pop(quant_state_key_fp4, None)
                if quant_state_tensor is not None:
                    self.quant_type = "fp4"

            if quant_state_tensor is not None:
                self.is_bnb_4bit = True

                # Load packed weight
                self.packed_weight = state_dict.pop(weight_key, None)
                if self.packed_weight is not None:
                    self.packed_weight = self.packed_weight.to(torch.uint8)

                # Load absmax
                absmax_key = prefix + 'weight.absmax'
                absmax_tensor = state_dict.pop(absmax_key, None)
                if absmax_tensor is not None:
                    absmax_tensor = absmax_tensor.to(torch.float32)

                # Load quant_map
                quant_map_key = prefix + 'weight.quant_map'
                quant_map_tensor = state_dict.pop(quant_map_key, None)
                if quant_map_tensor is not None:
                    quant_map_tensor = quant_map_tensor.to(torch.float32)

                # Try to build native BNB QuantState
                if _BNB_AVAILABLE and _QuantState is not None:
                    try:
                        # Build qs_dict for QuantState.from_dict()
                        qs_dict = {
                            f"quant_state.bitsandbytes__{self.quant_type}": quant_state_tensor,
                            "absmax": absmax_tensor,
                            "quant_map": quant_map_tensor,
                        }
                        self.quant_state = _QuantState.from_dict(qs_dict, device=torch.device('cpu'))
                        logging.debug(f"BNB 4-bit layer {weight_key}: using native QuantState")
                    except Exception as e:
                        logging.warning(f"Failed to create native QuantState for {weight_key}: {e}")
                        self.quant_state = None

                # Fallback: parse metadata manually
                if self.quant_state is None:
                    try:
                        quant_state_meta = tensor_to_dict(quant_state_tensor)
                        self.blocksize = quant_state_meta.get('blocksize', 64)
                        self.original_shape = tuple(quant_state_meta.get('shape', []))
                        dtype_str = quant_state_meta.get('dtype', 'bfloat16')
                        self.original_dtype = getattr(torch, dtype_str, torch.bfloat16)
                        logging.debug(f"BNB 4-bit layer {weight_key}: using PyTorch fallback")
                    except Exception as e:
                        logging.warning(f"Failed to parse quant_state for {weight_key}: {e}")
                        self.blocksize = 64
                        self.original_shape = None

                    self.absmax = absmax_tensor
                    self.quant_map = quant_map_tensor if quant_map_tensor is not None else get_quant_map(self.quant_type, torch.device('cpu'))

                # Set dummy weight to satisfy module structure
                self.weight = torch.nn.Parameter(
                    torch.empty(1, dtype=torch.float32),
                    requires_grad=False
                )

            else:
                # Not a BNB 4-bit layer, use standard loading
                self.is_bnb_4bit = False
                weight_tensor = state_dict.pop(weight_key, None)
                if weight_tensor is not None:
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + 'bias'
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def _dequantize_weight_pytorch(self, input_dtype: torch.dtype) -> torch.Tensor:
            """Fallback: dequantize using pure PyTorch."""
            if self.packed_weight is None or self.absmax is None:
                raise RuntimeError("BNB 4-bit layer missing packed_weight or absmax")

            # Infer shape if not stored
            if self.original_shape is None or len(self.original_shape) == 0:
                n_blocks = self.absmax.numel()
                n_elements = self.packed_weight.numel() * 2
                out_features = n_blocks * self.blocksize // (n_elements // n_blocks)
                in_features = n_elements // out_features if out_features > 0 else n_elements
                self.original_shape = (out_features, in_features)
                logging.warning(f"Inferred shape {self.original_shape} for BNB 4-bit layer")

            return dequantize_bnb_4bit_pytorch(
                self.packed_weight,
                self.absmax,
                self.quant_map,
                self.blocksize,
                self.original_shape,
                input_dtype,
            )

        def forward_comfy_cast_weights(self, input):
            """Forward pass with BNB 4-bit support and CPU offloading."""
            if self.is_bnb_4bit:
                device = input.device
                
                # Track original device for offloading back
                original_device = self.packed_weight.device
                should_offload = original_device != device
                
                # Move packed weight to device
                if should_offload:
                    self.packed_weight = self.packed_weight.to(device)

                # Native BNB path
                if self.quant_state is not None and _BNB_AVAILABLE:
                    self.quant_state.to(device)
                    
                    # Use native matmul_4bit
                    output = _bnb_matmul_4bit(
                        input,
                        self.packed_weight.t(),
                        quant_state=self.quant_state,
                        bias=self.bias.to(device=device, dtype=input.dtype) if self.bias is not None else None,
                    )
                    
                    # Offload back to original device (CPU) after forward
                    if should_offload:
                        self.packed_weight = self.packed_weight.to(original_device)
                        self.quant_state.to(original_device)
                    
                    return output
                
                # Fallback: PyTorch dequantization
                if self.absmax.device != device:
                    self.absmax = self.absmax.to(device)
                if self.quant_map.device != device:
                    self.quant_map = self.quant_map.to(device)

                weight = self._dequantize_weight_pytorch(input.dtype)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=device, dtype=input.dtype)

                output = F.linear(input, weight, bias)
                
                # Offload back
                if should_offload:
                    self.packed_weight = self.packed_weight.to(original_device)
                    self.absmax = self.absmax.to(original_device)
                    self.quant_map = self.quant_map.to(original_device)
                
                return output

            # Standard manual_cast path for non-BNB layers
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = F.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def forward(self, *args, **kwargs):
            if self.is_bnb_4bit or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """
            Convert weight for LoRA patching.
            
            Returns None for BNB 4-bit layers to signal that dynamic LoRA patching
            is not supported. Use a dedicated LoRA merge node instead.
            """
            if self.is_bnb_4bit:
                # Signal that LoRA patching is not supported
                logging.warning(
                    "LoRA patching not supported for BNB 4-bit layers. "
                    "Use a dedicated LoRA merge node for layer-by-layer patching."
                )
                return None
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            if return_weight:
                return weight
            
            if weight is None:
                # LoRA patching was skipped for BNB layer
                return

            if inplace_update and not self.is_bnb_4bit:
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            # After patching, no longer in 4-bit mode
            self.is_bnb_4bit = False
            self.quant_state = None
            self.packed_weight = None
            self.absmax = None

    # Normalization layers - use standard manual_cast versions
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

    # Convolution layers - use standard manual_cast versions
    class Conv1d(manual_cast.Conv1d):
        pass

    class Conv2d(manual_cast.Conv2d):
        pass

    class Conv3d(manual_cast.Conv3d):
        pass

    class ConvTranspose1d(manual_cast.ConvTranspose1d):
        pass

    class ConvTranspose2d(manual_cast.ConvTranspose2d):
        pass

    class Embedding(manual_cast.Embedding):
        pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
