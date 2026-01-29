"""
SDNQ (Stochastic Differentiable Neural Quantization) Layout.

Supports arbitrary bit-widths, custom dtypes, and optional SVD correction.
Matches the implementation in convert_to_quant-reference.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Union, List

# Import from ComfyUI core
from comfy.quant_ops import QuantizedTensor, register_layout_op

# Import base layout types from comfy_kitchen
try:
    from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams
except ImportError:
    # Fallback for older versions
    from comfy.quant_ops import QuantizedLayout
    BaseLayoutParams = object

# --- SDNQ Specific Constants ---

SDNQ_DTYPE_DICT = {
    ### Integers
    "int32": {"min": -2147483648, "max": 2147483647, "num_bits": 32, "sign": 1, "exponent": 0, "mantissa": 31, "target_dtype": torch.int32, "torch_dtype": torch.int32, "storage_dtype": torch.int32, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int16": {"min": -32768, "max": 32767, "num_bits": 16, "sign": 1, "exponent": 0, "mantissa": 15, "target_dtype": torch.int16, "torch_dtype": torch.int16, "storage_dtype": torch.int16, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int8": {"min": -128, "max": 127, "num_bits": 8, "sign": 1, "exponent": 0, "mantissa": 7, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True, "is_packed": False},
    ### Custom Integers
    "int7": {"min": -64, "max": 63, "num_bits": 7, "sign": 1, "exponent": 0, "mantissa": 6, "target_dtype": "int7", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "sign": 1, "exponent": 0, "mantissa": 5, "target_dtype": "int6", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int5": {"min": -16, "max": 15, "num_bits": 5, "sign": 1, "exponent": 0, "mantissa": 4, "target_dtype": "int5", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "sign": 1, "exponent": 0, "mantissa": 3, "target_dtype": "int4", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int3": {"min": -4, "max": 3, "num_bits": 3, "sign": 1, "exponent": 0, "mantissa": 2, "target_dtype": "int3", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int2": {"min": -2, "max": 1, "num_bits": 2, "sign": 1, "exponent": 0, "mantissa": 1, "target_dtype": "int2", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    ### Unsigned Integers
    "uint32": {"min": 0, "max": 4294967295, "num_bits": 32, "sign": 0, "exponent": 0, "mantissa": 32, "target_dtype": torch.uint32, "torch_dtype": torch.uint32, "storage_dtype": torch.uint32, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint16": {"min": 0, "max": 65535, "num_bits": 16, "sign": 0, "exponent": 0, "mantissa": 16, "target_dtype": torch.uint16, "torch_dtype": torch.uint16, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "sign": 0, "exponent": 0, "mantissa": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": False},
    ### Custom Unsigned Integers
    "uint7": {"min": 0, "max": 127, "num_bits": 7, "sign": 0, "exponent": 0, "mantissa": 7, "target_dtype": "uint7", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "sign": 0, "exponent": 0, "mantissa": 6, "target_dtype": "uint6", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint5": {"min": 0, "max": 31, "num_bits": 5, "sign": 0, "exponent": 0, "mantissa": 5, "target_dtype": "uint5", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "sign": 0, "exponent": 0, "mantissa": 4, "target_dtype": "uint4", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint3": {"min": 0, "max": 7, "num_bits": 3, "sign": 0, "exponent": 0, "mantissa": 3, "target_dtype": "uint3", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "sign": 0, "exponent": 0, "mantissa": 2, "target_dtype": "uint2", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "sign": 0, "exponent": 0, "mantissa": 1, "target_dtype": torch.bool, "torch_dtype": torch.bool, "storage_dtype": torch.bool, "is_unsigned": True, "is_integer": True, "is_packed": True},
    ### Floats
    "float32": {"min": -3.40282e+38, "max": 3.40282e+38, "num_bits": 32, "sign": 1, "exponent": 8, "mantissa": 23, "target_dtype": torch.float32, "torch_dtype": torch.float32, "storage_dtype": torch.float32, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "bfloat16": {"min": -3.38953e+38, "max": 3.38953e+38, "num_bits": 16, "sign": 1, "exponent": 8, "mantissa": 7, "target_dtype": torch.bfloat16, "torch_dtype": torch.bfloat16, "storage_dtype": torch.bfloat16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float16": {"min": -65504.0, "max": 65504.0, "num_bits": 16, "sign": 1, "exponent": 5, "mantissa": 10, "target_dtype": torch.float16, "torch_dtype": torch.float16, "storage_dtype": torch.float16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e4m3fn": {"min": -448.0, "max": 448.0, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e5m2": {"min": -57344.0, "max": 57344.0, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False, "is_packed": False},
}

SDNQ_DTYPE_DICT["fp32"] = SDNQ_DTYPE_DICT["float32"]
SDNQ_DTYPE_DICT["bf16"] = SDNQ_DTYPE_DICT["bfloat16"]
SDNQ_DTYPE_DICT["fp16"] = SDNQ_DTYPE_DICT["float16"]
SDNQ_DTYPE_DICT["fp8"] = SDNQ_DTYPE_DICT["float8_e4m3fn"]
SDNQ_DTYPE_DICT["bool"] = SDNQ_DTYPE_DICT["uint1"]
SDNQ_DTYPE_DICT["int1"] = SDNQ_DTYPE_DICT["uint1"]

# --- Unpacking Logic ---

def unpack_uint4(tensor: torch.Tensor) -> torch.Tensor:
    res = torch.empty((tensor.shape[0], 2), device=tensor.device, dtype=torch.uint8)
    res[:, 0] = torch.bitwise_and(tensor, 0x0F)
    res[:, 1] = torch.bitwise_right_shift(tensor, 4)
    return res.view(-1)

def unpack_uint2(tensor: torch.Tensor) -> torch.Tensor:
    res = torch.empty((tensor.shape[0], 4), device=tensor.device, dtype=torch.uint8)
    res[:, 0] = torch.bitwise_and(tensor, 0x03)
    res[:, 1] = torch.bitwise_and(torch.bitwise_right_shift(tensor, 2), 0x03)
    res[:, 2] = torch.bitwise_and(torch.bitwise_right_shift(tensor, 4), 0x03)
    res[:, 3] = torch.bitwise_and(torch.bitwise_right_shift(tensor, 6), 0x03)
    return res.view(-1)

def unpack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    res = torch.empty((tensor.shape[0], 8), device=tensor.device, dtype=torch.uint8)
    for i in range(8):
        res[:, i] = torch.bitwise_and(torch.bitwise_right_shift(tensor, i), 0x01)
    return res.view(-1)

unpacked_int_function_dict = {
    "uint4": unpack_uint4, "int4": unpack_uint4,
    "uint2": unpack_uint2, "int2": unpack_uint2,
    "uint1": unpack_uint1, "bool": unpack_uint1,
}

def unpack_weight(qdata: torch.Tensor, weights_dtype: str, scale: torch.Tensor, zero_point: Optional[torch.Tensor], group_size: int, original_shape: torch.Size) -> torch.Tensor:
    dtype_info = SDNQ_DTYPE_DICT.get(weights_dtype)
    if dtype_info is None:
        raise ValueError(f"Unknown SDNQ dtype: {weights_dtype}")
    
    # 1. Unpack bits if packed
    if dtype_info["is_packed"]:
        if weights_dtype in unpacked_int_function_dict:
            weight = unpacked_int_function_dict[weights_dtype](qdata).to(torch.float32)
        else:
            # Fallback for complex packing or formats not explicitly implemented here
            # (e.g. float widths < 8). For now, we only implement the integer ones.
            weight = qdata.to(torch.float32)
    else:
        weight = qdata.to(torch.float32)

    # 2. Reshape to handle groups
    if group_size > 0:
        # Reconstruct the grouped shape used during quantization
        # We need to flatten to the correct number of elements first
        weight = weight.flatten()
        
        out_features = original_shape[0]
        in_features = original_shape[1] if len(original_shape) > 1 else original_shape[0]
        num_groups = in_features // group_size
        
        if len(original_shape) == 2: # Linear
            weight = weight.reshape(out_features, num_groups, group_size)
        elif len(original_shape) == 4: # Conv2d
            # SDNQ Conv2d reduction is usually on axis 1
            weight = weight.reshape(out_features, num_groups, group_size, original_shape[2], original_shape[3])
        else:
            # Generic fallback for other dimensions
            weight = weight.reshape(*original_shape[:-1], num_groups, group_size)

    # 3. Apply scale and zero point
    if dtype_info["is_unsigned"] and zero_point is not None:
        weight = weight.mul(scale).add(zero_point)
    else:
        if dtype_info["is_integer"] and not dtype_info["is_unsigned"]:
            # Symmetric integer
            weight = weight.add(dtype_info["min"]).mul(scale)
        else:
            weight = weight.mul(scale)

    # 4. Final reshape to original shape
    return weight.reshape(original_shape)

# --- SDNQ Layout ---

class SDNQLayout(QuantizedLayout):
    """
    SDNQ quantization layout.
    Supports arbitrary bit-widths, optional SVD correction, and custom dtypes.
    """

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """SDNQ layout parameters."""
        weights_dtype: str = "int8"
        group_size: int = -1
        zero_point: Optional[torch.Tensor] = None
        svd_up: Optional[torch.Tensor] = None
        svd_down: Optional[torch.Tensor] = None
        transposed: bool = False
        unpack_shape: Optional[Tuple[int, ...]] = None

        def _tensor_fields(self) -> List[str]:
            fields = ["scale"]
            if self.zero_point is not None:
                fields.append("zero_point")
            if self.svd_up is not None:
                fields.append("svd_up")
            if self.svd_down is not None:
                fields.append("svd_down")
            return fields

    @classmethod
    def quantize(cls, tensor, **kwargs) -> Tuple[torch.Tensor, "SDNQLayout.Params"]:
        """
        Quantization is not implemented in the layout itself for SDNQ.
        It should be done via the conversion pipeline.
        """
        raise NotImplementedError("SDNQ quantization should be performed via the conversion pipeline.")

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: "SDNQLayout.Params") -> torch.Tensor:
        """
        Dequantize an SDNQ-formatted tensor.
        W = dequant(Q) + (svd_up @ svd_down if available)
        """
        weights_dtype = params.weights_dtype
        scale = params.scale
        zero_point = params.zero_point
        svd_up = params.svd_up
        svd_down = params.svd_down
        group_size = params.group_size
        orig_dtype = params.orig_dtype
        is_transposed = params.transposed
        
        # Use unpack_shape if provided (standard for deferred transpose)
        # otherwise use orig_shape (backwards compatible)
        # IF transposed is True, and unpack_shape is None, orig_shape is already transposed!
        if is_transposed and params.unpack_shape is None:
             # Legacy/manual transpose: try to recover original shape
             if len(params.orig_shape) == 2:
                 unpack_shape = (params.orig_shape[1], params.orig_shape[0])
             else:
                 unpack_shape = params.orig_shape
        else:
             unpack_shape = params.unpack_shape if params.unpack_shape is not None else params.orig_shape

        # 1. Unpack and basic dequant
        weight = unpack_weight(qdata, weights_dtype, scale, zero_point, group_size, unpack_shape)
        weight = weight.to(orig_dtype)

        # 2. Add SVD correction if present
        if svd_up is not None and svd_down is not None:
            # Robust SVD reconstruction
            svd_up_o, svd_down_o = _get_sdnq_svd_tensors(params, device=weight.device, dtype=orig_dtype)
            
            if svd_up_o is not None and svd_down_o is not None and svd_up_o.shape[1] == svd_down_o.shape[0]:
                correction = torch.mm(svd_up_o, svd_down_o)
                
                # Reshape correction to match weight if it was a conv layer
                if weight.ndim > 2:
                    correction = correction.reshape(weight.shape)
                
                weight = weight.add(correction)

        # 3. Handle deferred transpose if requested
        if is_transposed:
            weight = weight.t()

        # 4. Handle possible shape mismatch after all processing
        if weight.shape != params.orig_shape:
             if weight.numel() == torch.Size(params.orig_shape).numel():
                 weight = weight.reshape(params.orig_shape)
             else:
                 logging.warning(f"SDNQLayout: Final shape mismatch: {weight.shape} vs {params.orig_shape}")

        return weight

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> Tuple[torch.Tensor, ...]:
        """Extract raw tensors for computation."""
        p = qtensor._params
        return (
            qtensor._qdata,
            p.scale,
            p.zero_point,
            p.svd_up,
            p.svd_down
        )

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: "SDNQLayout.Params") -> Dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization."""
        res = {
            "": qdata,
            "_scale": params.scale,
        }
        if params.zero_point is not None:
            res["_zero_point"] = params.zero_point
        if params.svd_up is not None:
            res["_svd_up"] = params.svd_up
        if params.svd_down is not None:
            res["_svd_down"] = params.svd_down
        return res

# --- Operation Handlers ---

def _get_sdnq_svd_tensors(params, device=None, dtype=None):
    """Robustly extract and orient SVD tensors."""
    svd_up = params.svd_up
    svd_down = params.svd_down
    if svd_up is None or svd_down is None:
        return None, None
    
    if device is not None:
        svd_up = svd_up.to(device)
        svd_down = svd_down.to(device)
    if dtype is not None:
        svd_up = svd_up.to(dtype)
        svd_down = svd_down.to(dtype)

    unpack_shape = params.unpack_shape if params.unpack_shape is not None else params.orig_shape
    m_orig = unpack_shape[0]
    flat_n_orig = 1
    if len(unpack_shape) > 1:
        for d in unpack_shape[1:]:
            flat_n_orig *= d
    else:
        flat_n_orig = unpack_shape[0]

    # Orient correctly
    if svd_up.shape[0] != m_orig and svd_up.shape[1] == m_orig:
        svd_up = svd_up.t()
    if svd_down.shape[1] != flat_n_orig and svd_down.shape[0] == flat_n_orig:
        svd_down = svd_down.t()
        
    return svd_up, svd_down

@register_layout_op(torch.ops.aten.linear.default, SDNQLayout)
def sdnq_linear(func, args, kwargs):
    """
    SDNQ linear operation.
    Optimized to compute SVD correction separately and cache dequantized base weights.
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if isinstance(weight, QuantizedTensor):
        params = weight._params
        
        # Split execution if SVD is present
        if params.svd_up is not None and params.svd_down is not None:
            # 1. Base part (dequantize only base, with caching)
            device = input_tensor.device
            dtype = input_tensor.dtype
            
            weight_base = getattr(weight, "_sdnq_cache_base", None)
            if weight_base is None or weight_base.device != device or weight_base.dtype != dtype:
                base_params = SDNQLayout.Params(
                    scale=params.scale,
                    orig_dtype=params.orig_dtype,
                    orig_shape=params.orig_shape,
                    weights_dtype=params.weights_dtype,
                    group_size=params.group_size,
                    zero_point=params.zero_point,
                    svd_up=None,
                    svd_down=None,
                    transposed=params.transposed,
                    unpack_shape=params.unpack_shape,
                )
                # Ensure dequantization happens on the target device
                qdata = weight._qdata.to(device)
                weight_base = SDNQLayout.dequantize(qdata, base_params).to(dtype)
                try:
                    object.__setattr__(weight, "_sdnq_cache_base", weight_base)
                except AttributeError:
                    pass
            
            res = torch.nn.functional.linear(input_tensor, weight_base, bias)
            
            # 2. SVD Part (also cached if possible)
            svd_up_o = getattr(weight, "_sdnq_cache_svd_up", None)
            svd_down_o = getattr(weight, "_sdnq_cache_svd_down", None)
            
            if svd_up_o is None or svd_up_o.device != device or svd_up_o.dtype != dtype:
                svd_up_o, svd_down_o = _get_sdnq_svd_tensors(params, device=device, dtype=dtype)
                try:
                    object.__setattr__(weight, "_sdnq_cache_svd_up", svd_up_o)
                    object.__setattr__(weight, "_sdnq_cache_svd_down", svd_down_o)
                except AttributeError:
                    pass

            if svd_up_o is not None and svd_down_o is not None:
                # Optimized low-rank update: (X @ V) @ U^T or (X @ U) @ V
                x_flat = input_tensor.flatten(0, -2)
                if params.transposed:
                    # Logical W = Q + U @ V -> X @ (Q + U @ V) = X @ Q + (X @ U) @ V
                    # svd_up: (Out_stored, Rank), svd_down: (Rank, In_stored)
                    res_svd = torch.mm(torch.mm(x_flat, svd_up_o), svd_down_o)
                else:
                    # Logical W = Q + U @ V^T -> X @ (Q^T + V @ U^T) = X @ Q^T + (X @ V) @ U^T
                    # svd_up: (Out, Rank), svd_down: (Rank, In)
                    res_svd = torch.mm(torch.mm(x_flat, svd_down_o.t()), svd_up_o.t())
                res = res + res_svd.reshape(res.shape)
            return res

        # Cache dequantized weight even if no SVD
        weight_dequant = getattr(weight, "_sdnq_cache_full", None)
        if weight_dequant is None or weight_dequant.device != input_tensor.device or weight_dequant.dtype != input_tensor.dtype:
            weight_dequant = weight.dequantize().to(input_tensor.device, input_tensor.dtype)
            try:
                object.__setattr__(weight, "_sdnq_cache_full", weight_dequant)
            except AttributeError:
                pass
        weight = weight_dequant
        
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)

@register_layout_op(torch.ops.aten.mm.default, SDNQLayout)
def sdnq_mm(func, args, kwargs):
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        params = weight._params
        if params.svd_up is not None and params.svd_down is not None:
            base_params = SDNQLayout.Params(
                scale=params.scale,
                orig_dtype=params.orig_dtype,
                orig_shape=params.orig_shape,
                weights_dtype=params.weights_dtype,
                group_size=params.group_size,
                zero_point=params.zero_point,
                svd_up=None,
                svd_down=None,
                transposed=params.transposed,
                unpack_shape=params.unpack_shape,
            )
            weight_base = SDNQLayout.dequantize(weight._qdata, base_params).to(input_tensor.device, input_tensor.dtype)
            res = torch.mm(input_tensor, weight_base)
            
            svd_up, svd_down = _get_sdnq_svd_tensors(params, device=input_tensor.device, dtype=input_tensor.dtype)
            if svd_up is not None and svd_down is not None:
                x_flat = input_tensor.flatten(0, -2)
                if params.transposed:
                    res_svd = torch.mm(torch.mm(x_flat, svd_down.t()), svd_up.t())
                else:
                    res_svd = torch.mm(torch.mm(x_flat, svd_up), svd_down)
                res = res + res_svd.reshape(res.shape)
            return res
        weight = weight.dequantize()
        
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return torch.mm(input_tensor, weight)

@register_layout_op(torch.ops.aten.addmm.default, SDNQLayout)
def sdnq_addmm(func, args, kwargs):
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(weight, QuantizedTensor):
        params = weight._params
        if params.svd_up is not None and params.svd_down is not None:
            base_params = SDNQLayout.Params(
                scale=params.scale,
                orig_dtype=params.orig_dtype,
                orig_shape=params.orig_shape,
                weights_dtype=params.weights_dtype,
                group_size=params.group_size,
                zero_point=params.zero_point,
                svd_up=None,
                svd_down=None,
                transposed=params.transposed,
                unpack_shape=params.unpack_shape,
            )
            weight_base = SDNQLayout.dequantize(weight._qdata, base_params).to(input_tensor.device, input_tensor.dtype)
            res = torch.addmm(bias, input_tensor, weight_base, **kwargs)
            
            svd_up, svd_down = _get_sdnq_svd_tensors(params, device=input_tensor.device, dtype=input_tensor.dtype)
            if svd_up is not None and svd_down is not None:
                x_flat = input_tensor.flatten(0, -2)
                if params.transposed:
                    res_svd = torch.mm(torch.mm(x_flat, svd_down.t()), svd_up.t())
                else:
                    res_svd = torch.mm(torch.mm(x_flat, svd_up), svd_down)
                res = res + res_svd.reshape(res.shape)
            return res
        weight = weight.dequantize()

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return torch.addmm(bias, input_tensor, weight, **kwargs)

@register_layout_op(torch.ops.aten.convolution.default, SDNQLayout)
def sdnq_convolution(func, args, kwargs):
    """SDNQ convolution operation with caching."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2]

    if isinstance(weight, QuantizedTensor):
        params = weight._params
        device = input_tensor.device
        dtype = input_tensor.dtype

        if params.svd_up is not None and params.svd_down is not None:
            # 1. Base part (cached)
            weight_base = getattr(weight, "_sdnq_cache_base", None)
            if weight_base is None or weight_base.device != device or weight_base.dtype != dtype:
                base_params = SDNQLayout.Params(
                    scale=params.scale,
                    orig_dtype=params.orig_dtype,
                    orig_shape=params.orig_shape,
                    weights_dtype=params.weights_dtype,
                    group_size=params.group_size,
                    zero_point=params.zero_point,
                    svd_up=None,
                    svd_down=None,
                    transposed=params.transposed,
                    unpack_shape=params.unpack_shape,
                )
                qdata = weight._qdata.to(device)
                weight_base = SDNQLayout.dequantize(qdata, base_params).to(dtype)
                try:
                    object.__setattr__(weight, "_sdnq_cache_base", weight_base)
                except AttributeError:
                    pass
            
            res = torch.ops.aten.convolution.default(input_tensor, weight_base, bias, *args[3:], **kwargs)
            
            # 2. SVD Part (cached)
            svd_up_o = getattr(weight, "_sdnq_cache_svd_up", None)
            svd_down_o = getattr(weight, "_sdnq_cache_svd_down", None)
            if svd_up_o is None or svd_up_o.device != device or svd_up_o.dtype != dtype:
                svd_up_o, svd_down_o = _get_sdnq_svd_tensors(params, device=device, dtype=dtype)
                try:
                    object.__setattr__(weight, "_sdnq_cache_svd_up", svd_up_o)
                    object.__setattr__(weight, "_sdnq_cache_svd_down", svd_down_o)
                except AttributeError:
                    pass

            if svd_up_o is not None and svd_down_o is not None:
                orig_shape = params.unpack_shape if params.unpack_shape else params.orig_shape
                if len(orig_shape) == 4:
                    # Optimized low-rank convolution update
                    M, C, H, W = orig_shape
                    R = svd_up_o.shape[1]
                    V_reshaped = svd_down_o.reshape(R, C // args[8], H, W)
                    U_reshaped = svd_up_o.reshape(M, R, 1, 1)
                    temp = torch.nn.functional.conv2d(input_tensor, V_reshaped, None, args[3], args[4], args[5], args[8])
                    res_svd = torch.nn.functional.conv2d(temp, U_reshaped, None)
                    res = res + res_svd
                else:
                    weight_svd = torch.mm(svd_up_o, svd_down_o).reshape(orig_shape)
                    res_svd = torch.ops.aten.convolution.default(input_tensor, weight_svd, None, *args[3:], **kwargs)
                    res = res + res_svd
            return res

        # Cache dequantized weight even if no SVD
        weight_dequant = getattr(weight, "_sdnq_cache_full", None)
        if weight_dequant is None or weight_dequant.device != device or weight_dequant.dtype != dtype:
            weight_dequant = weight.dequantize().to(device, dtype)
            try:
                object.__setattr__(weight, "_sdnq_cache_full", weight_dequant)
            except AttributeError:
                pass
        weight = weight_dequant

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    return torch.ops.aten.convolution.default(input_tensor, weight, bias, *args[3:], **kwargs)

@register_layout_op(torch.ops.aten.t.default, SDNQLayout)
def _handle_sdnq_transpose(func, args, kwargs):
    """Handle transpose as a logical no-op for SDNQ."""
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old_shape = input_tensor._params.orig_shape
    new_shape = (old_shape[1], old_shape[0])
    old_params = input_tensor._params

    # Ensure unpack_shape is set to the original non-transposed shape
    unpack_shape = old_params.unpack_shape
    if unpack_shape is None:
        if old_params.transposed:
             unpack_shape = (old_shape[1], old_shape[0])
        else:
             unpack_shape = old_shape

    # Reconstruct params with flipped transposed flag
    new_params = SDNQLayout.Params(
        scale=old_params.scale,
        orig_dtype=old_params.orig_dtype,
        orig_shape=new_shape,
        weights_dtype=old_params.weights_dtype,
        group_size=old_params.group_size,
        zero_point=old_params.zero_point,
        svd_up=old_params.svd_up,
        svd_down=old_params.svd_down,
        transposed=not old_params.transposed,
        unpack_shape=unpack_shape,
    )
    return QuantizedTensor(input_tensor._qdata, "SDNQLayout", new_params)

@register_layout_op(torch.ops.aten.view.default, SDNQLayout)
def sdnq_view(qt, args, kwargs):
    input_tensor = qt
    if isinstance(input_tensor, QuantizedTensor):
        # SDNQ is fallback-heavy, dequantizing is safest for shape changes
        # especially since packing makes logical shape != storage shape.
        return torch.ops.aten.view.default(input_tensor.dequantize(), *args, **kwargs)
    return torch.ops.aten.view.default(input_tensor, *args, **kwargs)
