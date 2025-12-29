"""
FP8 Variant Layouts (Row-wise and Block-wise scaling)

These layouts extend ComfyUI's base TensorCoreFP8Layout with finer-grained scaling:
- RowWiseFP8Layout: One scale per row (M scales for MxN weight)
- BlockWiseFP8Layout: One scale per 2D block (MxN blocks)

When PyTorch scaled_grouped_mm is available (2.10+), uses native FP8 matmul.
Otherwise falls back to Triton kernels, then dequantization.
"""

import torch
import logging
from typing import Tuple, Dict

# Import from ComfyUI core
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op

# Try to import FP8 Triton kernels
try:
    from ..kernels.fp8_kernels import (
        _check_triton_available,
        fp8_act_quant,
        fp8_gemm_blockwise,
        fp8_addmm_blockwise,
        fp8_gemm_rowwise,
    )

    _HAS_FP8_KERNELS = _check_triton_available()
except ImportError:
    _HAS_FP8_KERNELS = False
    logging.debug("FP8 Triton kernels not available, using dequantize fallback")


# ==============================================================================
# PyTorch scaled_grouped_mm support (requires PyTorch 2.10+ and Hopper/Ada GPU)
# ==============================================================================
_HAS_GROUPED_MM = False
_ScalingType = None

try:
    from torch.nn.functional import scaled_grouped_mm, ScalingType as _ScalingType
    # Verify the enum has the values we need
    if hasattr(_ScalingType, 'RowWise') and hasattr(_ScalingType, 'BlockWise1x128'):
        _HAS_GROUPED_MM = True
        logging.debug("scaled_grouped_mm available (PyTorch 2.10+)")
except ImportError:
    logging.debug("scaled_grouped_mm not available, will use Triton/dequant fallback")


class RowWiseFP8Layout(QuantizedLayout):
    """
    Row-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-row scaling factors, shape (out_features,) - stored as dequant scale
    - orig_dtype: Original dtype before quantization
    """

    @classmethod
    def quantize(
        cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize a 2D tensor with row-wise scaling.
        """
        orig_dtype = tensor.dtype

        if tensor.ndim != 2:
            raise ValueError(
                f"RowWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape
        fp8_max = torch.finfo(dtype).max

        if scale is None:
            # Compute per-row absolute maximum
            row_max = tensor.abs().amax(dim=1, keepdim=True)  # (M, 1)
            quant_scale = fp8_max / row_max.clamp_min(1e-12)  # (M, 1)
        else:
            # scale is provided as dequant scale, convert to quant scale
            quant_scale = (
                (1.0 / scale).unsqueeze(1) if scale.ndim == 1 else (1.0 / scale)
            )

        # Apply scale per-row
        tensor_scaled = tensor * quant_scale

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata = tensor_scaled.to(dtype)

        # Store dequant scale (reciprocal of quant scale)
        dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)

        layout_params = {
            "scale": dequant_scale.to(torch.float32),
            "orig_dtype": orig_dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs) -> torch.Tensor:
        """Dequantize FP8 tensor with row-wise scaling."""
        # Convert to target dtype (matching core ComfyUI pattern)
        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = scale.to(
            dtype=orig_dtype, device=plain_tensor.device
        ).unsqueeze(1)  # (M, 1)
        plain_tensor.mul_(scale_broadcast)
        return plain_tensor

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        """Extract raw tensors for computation."""
        return qtensor._qdata, qtensor._layout_params["scale"]


class BlockWiseFP8Layout(QuantizedLayout):
    """
    True 2D block-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-block scaling factors, shape (M//block_size, N//block_size)
    - block_size: Size of quantization blocks
    - orig_dtype: Original dtype before quantization
    """

    @classmethod
    def quantize(
        cls, tensor, scale=None, block_size=64, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize a 2D tensor with 2D block-wise scaling.
        """
        orig_dtype = tensor.dtype

        if tensor.ndim != 2:
            raise ValueError(
                f"BlockWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape

        if M % block_size != 0 or N % block_size != 0:
            raise ValueError(
                f"BlockWiseFP8Layout requires dimensions divisible by block_size={block_size}. "
                f"Got shape ({M}, {N})"
            )

        fp8_max = torch.finfo(dtype).max

        # Reshape to 2D blocks
        tensor_blocked = tensor.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        tensor_blocked = tensor_blocked.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)

        if scale is None:
            # Compute per-block absolute maximum
            block_max = tensor_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
            quant_scale = fp8_max / block_max.clamp_min(1e-12)
        else:
            quant_scale = 1.0 / scale

        # Apply scale per-block
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(-1)
        tensor_scaled = tensor_blocked * scale_broadcast

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata_blocked = tensor_scaled.to(dtype)

        # Reshape back
        qdata = qdata_blocked.permute(0, 2, 1, 3).reshape(M, N)
        dequant_scale = 1.0 / quant_scale

        layout_params = {
            "scale": dequant_scale.to(torch.float32),
            "block_size": block_size,
            "orig_dtype": orig_dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, block_size, orig_dtype, **kwargs) -> torch.Tensor:
        """Dequantize FP8 tensor with 2D block-wise scaling."""
        M, N = qdata.shape

        # Reshape to blocks
        qdata_blocked = qdata.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)

        # Convert to target dtype (matching core ComfyUI pattern)
        dequantized = torch.ops.aten._to_copy.default(qdata_blocked, dtype=orig_dtype)

        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = (
            scale.to(dtype=orig_dtype, device=dequantized.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        dequantized.mul_(scale_broadcast)

        # Reshape back
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._layout_params["scale"],
            qtensor._layout_params["block_size"],
        )


# ==============================================================================
# Helper Functions
# ==============================================================================

def _check_grouped_mm_device_support(device: torch.device) -> bool:
    """
    Check if device supports scaled_grouped_mm.
    
    PyTorch currently only supports compute capability 9.0 and 10.0.
    Ada (8.9), Hopper (9.0), but NOT Blackwell SM 12.0 yet.
    """
    if not _HAS_GROUPED_MM:
        return False
    
    if not device.type == 'cuda':
        return False
    
    try:
        major, minor = torch.cuda.get_device_capability(device)
        # PyTorch only supports SM 9.0 and 10.0 for scaled_grouped_mm
        # SM 8.9 (Ada), SM 9.0 (Hopper), SM 10.0 - but NOT SM 12.0 (Blackwell) yet
        return (major == 9 and minor == 0) or (major == 10 and minor == 0) or (major == 8 and minor == 9)
    except Exception:
        return False


def _fast_fp8_quantize(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast activation quantization using fixed scale=1.0 with clamp.
    
    This is significantly faster than dynamic scale computation because it
    avoids the per-forward amax() reduction. Trades some precision for speed.
    Used by Hybrid Loader for optimal inference throughput.
    
    Args:
        tensor: Input tensor of any shape
        dtype: Target FP8 dtype
        
    Returns:
        (qdata, scale) where scale is always 1.0
    """
    fp8_max = torch.finfo(dtype).max
    # Clamp to FP8 representable range and cast directly
    qdata = torch.clamp(tensor, min=-fp8_max, max=fp8_max).to(dtype)
    scale = torch.ones((), device=tensor.device, dtype=torch.float32)
    return qdata, scale


def _dynamic_fp8_quantize_rowwise(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
    use_fast_path: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize activations with row-wise scaling for scaled_grouped_mm.
    
    Args:
        tensor: Input tensor of shape (..., K)
        dtype: Target FP8 dtype
        use_fast_path: If True, use fast clamp-based quantization (recommended)
        
    Returns:
        (qdata, scale) where scale has shape (M,) for row-wise or () for fast path
    """
    # Fast path: fixed scale=1.0, just clamp and cast
    if use_fast_path:
        return _fast_fp8_quantize(tensor, dtype)
    
    # Precise path: compute per-row scales
    fp8_max = torch.finfo(dtype).max
    orig_shape = tensor.shape
    
    # Flatten to 2D for row-wise processing
    tensor_2d = tensor.reshape(-1, orig_shape[-1])
    
    # Compute per-row max
    row_max = tensor_2d.abs().amax(dim=1, keepdim=True)  # (M, 1)
    scale = row_max / fp8_max
    scale = scale.clamp_min(1e-12)
    
    # Quantize
    qdata = (tensor_2d / scale).clamp(-fp8_max, fp8_max).to(dtype)
    
    # Reshape scale to match grouped_mm expectations (M,)
    return qdata.reshape(orig_shape), scale.squeeze(-1).to(torch.float32)


def _dynamic_fp8_quantize_blockwise(
    tensor: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
    use_fast_path: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize activations with 1xK block-wise scaling for scaled_grouped_mm.
    
    For activations, we use 1xblock_size blocks (one scale per block_size elements in K dim).
    This matches ScalingType.BlockWise1x128.
    
    Args:
        tensor: Input tensor of shape (..., K)
        block_size: Block size for scaling (default 128)
        dtype: Target FP8 dtype
        use_fast_path: If True, use fast clamp-based quantization (recommended)
        
    Returns:
        (qdata, scale) where scale has shape (..., K//block_size) or () for fast path
    """
    # Fast path: fixed scale=1.0, just clamp and cast (much faster)
    if use_fast_path:
        return _fast_fp8_quantize(tensor, dtype)
    
    # Precise path: compute per-block scales
    fp8_max = torch.finfo(dtype).max
    orig_shape = tensor.shape
    K = orig_shape[-1]
    
    # Pad if needed
    if K % block_size != 0:
        pad_size = block_size - (K % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
        K = tensor.shape[-1]
    
    # Flatten to 2D for processing
    tensor_2d = tensor.reshape(-1, K)
    M = tensor_2d.shape[0]
    
    # Reshape to blocks: (M, K//block_size, block_size)
    tensor_blocks = tensor_2d.reshape(M, K // block_size, block_size)
    
    # Compute per-block max
    block_max = tensor_blocks.abs().amax(dim=2, keepdim=True)  # (M, K//bs, 1)
    scale = block_max / fp8_max
    scale = scale.clamp_min(1e-12)
    
    # Quantize per block
    qdata_blocks = (tensor_blocks / scale).clamp(-fp8_max, fp8_max).to(dtype)
    
    # Reshape back
    qdata = qdata_blocks.reshape(M, K)
    scale = scale.squeeze(-1).to(torch.float32)  # (M, K//block_size)
    
    # Trim padding if applied
    if K != orig_shape[-1]:
        qdata = qdata[..., :orig_shape[-1]]
    
    return qdata.reshape(*orig_shape[:-1], qdata.shape[-1]), scale


# ==============================================================================
# Operation Handlers (scaled_grouped_mm → Triton → dequant fallback chain)
# ==============================================================================


@register_layout_op(torch.ops.aten.linear.default, "RowWiseFP8Layout")
def rowwise_fp8_linear(func, args, kwargs):
    """
    Row-wise FP8 linear operation.
    
    Fallback chain: scaled_grouped_mm → Triton kernels → dequantization
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(input_tensor, weight, bias)

    w_qdata, w_scale = RowWiseFP8Layout.get_plain_tensors(weight)
    orig_dtype = weight._layout_params.get("orig_dtype", torch.bfloat16)
    
    # Path 1: Try scaled_grouped_mm (PyTorch 2.10+, Hopper/Ada)
    if _check_grouped_mm_device_support(w_qdata.device):
        try:
            # Dynamic activation quantization with row-wise scaling
            input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
            a_qdata, a_scale = _dynamic_fp8_quantize_rowwise(
                input_2d.to(device=w_qdata.device), 
                dtype=w_qdata.dtype
            )
            
            logging.debug(
                f"FP8 rowwise: Using scaled_grouped_mm, "
                f"input={a_qdata.shape}, weight={w_qdata.shape}"
            )
            
            # Weight needs transpose for linear: input @ weight.T
            result = scaled_grouped_mm(
                a_qdata,
                w_qdata.t().contiguous(),
                a_scale, _ScalingType.RowWise,
                w_scale, _ScalingType.RowWise,
                bias=bias,
                output_dtype=orig_dtype,
            )
            
            # Reshape back if needed
            if len(input_tensor.shape) > 2:
                result = result.reshape(*input_tensor.shape[:-1], -1)
            
            return result
            
        except Exception as e:
            logging.debug(f"scaled_grouped_mm failed, trying Triton: {e}")

    # Path 2: Try Triton FP8 kernels
    if _HAS_FP8_KERNELS and w_qdata.is_cuda:
        act_block_size = 128
        if input_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            try:
                a_qdata, a_scale = fp8_act_quant(
                    input_tensor.to(device=w_qdata.device),
                    block_size=act_block_size,
                    dtype=w_qdata.dtype,
                )

                logging.debug(
                    f"FP8 rowwise: Using Triton kernel, "
                    f"input={a_qdata.shape}, weight={w_qdata.shape}"
                )

                result = fp8_gemm_rowwise(
                    a_qdata, a_scale, w_qdata, w_scale,
                    input_block_size=act_block_size,
                )

                if bias is not None:
                    result = result + bias.to(device=result.device, dtype=result.dtype)

                return result.to(orig_dtype)
            except Exception as e:
                logging.debug(f"Triton kernel failed: {e}")

    # Path 3: Dequantize fallback
    logging.debug("FP8 rowwise: Using dequant fallback")
    weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "RowWiseFP8Layout")
def rowwise_fp8_mm(func, args, kwargs):
    """Row-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "RowWiseFP8Layout")
def rowwise_fp8_addmm(func, args, kwargs):
    """Row-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "RowWiseFP8Layout")
@register_layout_op(torch.ops.aten.t.default, "RowWiseFP8Layout")
def rowwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for row-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale = RowWiseFP8Layout.get_plain_tensors(input_tensor)
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(
            func(*ar, **kwargs), "RowWiseFP8Layout", input_tensor._layout_params
        )
    return func(*args, **kwargs)


# Debug: track which layers have been logged
_blockwise_logged_layers = set()

@register_layout_op(torch.ops.aten.linear.default, "BlockWiseFP8Layout")
def blockwise_fp8_linear(func, args, kwargs):
    """
    Block-wise FP8 linear operation.
    
    Fallback chain: scaled_grouped_mm → Triton kernels → dequantization
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(input_tensor, weight, bias)

    # Debug: log once per layer
    layer_id = id(weight)
    
    w_qdata, w_scale, w_block_size = BlockWiseFP8Layout.get_plain_tensors(weight)
    orig_dtype = weight._layout_params.get("orig_dtype", torch.bfloat16)
    
    if layer_id not in _blockwise_logged_layers:
        device_ok = _check_grouped_mm_device_support(w_qdata.device)
        print(f"[QuantOps] blockwise: shape={weight.shape}, block_size={w_block_size}, device_ok={device_ok}, HAS_GROUPED_MM={_HAS_GROUPED_MM}")
        _blockwise_logged_layers.add(layer_id)
    
    # Path 1: Try scaled_grouped_mm (PyTorch 2.10+, Hopper/Ada/Blackwell)
    if _check_grouped_mm_device_support(w_qdata.device):
        # Map block_size to ScalingType
        scaling_type_map = {
            16: _ScalingType.BlockWise1x16 if _ScalingType else None,
            32: _ScalingType.BlockWise1x32 if _ScalingType else None,
            128: _ScalingType.BlockWise1x128 if _ScalingType else None,
        }
        w_scaling_type = scaling_type_map.get(w_block_size, _ScalingType.BlockWise1x128 if _ScalingType else None)
        
        if w_scaling_type is not None:
            try:
                # Dynamic activation quantization with blockwise scaling
                input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
                a_qdata, a_scale = _dynamic_fp8_quantize_blockwise(
                    input_2d.to(device=w_qdata.device),
                    block_size=w_block_size,
                    dtype=w_qdata.dtype,
                )
                
                if layer_id not in _blockwise_logged_layers:
                    print(f"[QuantOps] Using scaled_grouped_mm, block_size={w_block_size}")
                    _blockwise_logged_layers.add(layer_id)
                
                # Weight needs transpose for linear: input @ weight.T
                result = scaled_grouped_mm(
                    a_qdata,
                    w_qdata.t().contiguous(),
                    a_scale, _ScalingType.BlockWise1x128,  # Activations use 1xK blocks
                    w_scale, w_scaling_type,
                    bias=bias,
                    output_dtype=orig_dtype,
                )
                
                # Reshape back if needed
                if len(input_tensor.shape) > 2:
                    result = result.reshape(*input_tensor.shape[:-1], -1)
                
                return result
                
            except Exception as e:
                print(f"[QuantOps] scaled_grouped_mm FAILED: {e}")

    # Path 2: Try Triton FP8 kernels
    if _HAS_FP8_KERNELS and w_qdata.is_cuda:
        # If input is already quantized FP8, use it directly
        if isinstance(input_tensor, QuantizedTensor):
            try:
                a_qdata, a_scale, a_block_size = BlockWiseFP8Layout.get_plain_tensors(input_tensor)
                
                if layer_id not in _blockwise_logged_layers:
                    print(f"[QuantOps] Using Triton FP8 kernel (pre-quantized)")
                
                if bias is not None:
                    result = fp8_addmm_blockwise(
                        a_qdata, a_scale, w_qdata, w_scale,
                        bias=bias.to(device=a_qdata.device),
                        input_block_size=w_block_size,
                    )
                else:
                    result = fp8_gemm_blockwise(
                        a_qdata, a_scale, w_qdata, w_scale,
                        input_block_size=w_block_size,
                    )
                return result.to(orig_dtype)
            except Exception as e:
                if layer_id not in _blockwise_logged_layers:
                    print(f"[QuantOps] Triton (pre-quant) FAILED: {e}")
        
        # Dynamically quantize input
        elif input_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            try:
                a_qdata, a_scale = fp8_act_quant(
                    input_tensor.to(device=w_qdata.device),
                    block_size=w_block_size,
                    dtype=w_qdata.dtype,
                )

                if layer_id not in _blockwise_logged_layers:
                    print(f"[QuantOps] Using Triton FP8 kernel (dynamic quant)")

                if bias is not None:
                    result = fp8_addmm_blockwise(
                        a_qdata, a_scale, w_qdata, w_scale,
                        bias=bias.to(device=a_qdata.device),
                        input_block_size=w_block_size,
                    )
                else:
                    result = fp8_gemm_blockwise(
                        a_qdata, a_scale, w_qdata, w_scale,
                        input_block_size=w_block_size,
                    )
                return result.to(orig_dtype)
            except Exception as e:
                if layer_id not in _blockwise_logged_layers:
                    print(f"[QuantOps] Triton (dynamic quant) FAILED: {e}")

    # Path 3: Dequantize fallback
    if layer_id not in _blockwise_logged_layers:
        print(f"[QuantOps] Using dequant fallback (SLOW)")
    
    weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "BlockWiseFP8Layout")
def blockwise_fp8_mm(func, args, kwargs):
    """Block-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "BlockWiseFP8Layout")
def blockwise_fp8_addmm(func, args, kwargs):
    """Block-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "BlockWiseFP8Layout")
@register_layout_op(torch.ops.aten.t.default, "BlockWiseFP8Layout")
def blockwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for block-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale, block_size = BlockWiseFP8Layout.get_plain_tensors(
            input_tensor
        )
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(
            func(*ar, **kwargs), "BlockWiseFP8Layout", input_tensor._layout_params
        )
    return func(*args, **kwargs)
