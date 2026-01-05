"""
NVFP4 (FP4 E2M1) Block Quantization Layout.

Implements NVIDIA's 4-bit floating point format with two-level scaling:
1. Per-tensor scale (float32 scalar)
2. Per-block scale (FP8 E4M3, 16-element blocks in cuBLAS tiled layout)

This layout supports:
- Loading NVFP4-quantized models from convert_to_quant
- Dequantization via comfy-kitchen (if available) or pure PyTorch fallback
- Hardware-accelerated matmul on Blackwell GPUs (SM >= 10.0)
"""

import json
import logging
from typing import Dict, Tuple, Any

import torch

from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op

from ..utils.float_utils import (
    F4_E2M1_MAX,
    F4_E2M1_EBITS,
    F4_E2M1_MBITS,
    F8_E4M3_MAX,
    roundup,
    pack_uint4,
    unpack_uint4,
    to_blocked,
    from_blocked,
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
    _float8_round,
)

logger = logging.getLogger(__name__)

# Block size for NVFP4 quantization (fixed by format)
NVFP4_BLOCK_SIZE = 16


def _try_comfy_kitchen_dequantize(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Try to use comfy-kitchen for dequantization.

    Args:
        qdata: Packed FP4 data (uint8)
        scale: Per-tensor scale (float32)
        block_scale: Block scales in cuBLAS tiled layout (float8_e4m3fn)
        output_dtype: Target output dtype

    Returns:
        Dequantized tensor

    Raises:
        ImportError: If comfy-kitchen is not available
    """
    import comfy_kitchen as ck

    return ck.dequantize_nvfp4(qdata, scale, block_scale, output_dtype)


def _pure_pytorch_dequantize(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_scale: torch.Tensor,
    output_dtype: torch.dtype,
    orig_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Pure PyTorch dequantization fallback.

    Args:
        qdata: Packed FP4 data (uint8), shape (M, K//2)
        scale: Per-tensor scale (float32 scalar)
        block_scale: Block scales in cuBLAS tiled layout (float8_e4m3fn)
        output_dtype: Target output dtype
        orig_shape: Original tensor shape before padding

    Returns:
        Dequantized tensor of shape orig_shape
    """
    # Unpack FP4 data: (M, K//2) -> (M, K)
    data_unpacked = unpack_uint4(qdata)

    # Convert unpacked FP4 to float32
    data_f32 = _floatx_unpacked_to_f32(data_unpacked, F4_E2M1_EBITS, F4_E2M1_MBITS)

    padded_shape = data_f32.shape
    M, K = padded_shape

    # Reshape to blocks: (M, K) -> (M, K//16, 16)
    data_reshaped = data_f32.reshape(M, -1, NVFP4_BLOCK_SIZE)

    # Unswizzle block_scales from cuBLAS tiled layout
    num_blocks_per_row = K // NVFP4_BLOCK_SIZE
    block_scales_unswizzled = from_blocked(
        block_scale.reshape(-1), num_rows=M, num_cols=num_blocks_per_row
    )

    # Compute total scale: per_tensor_scale * block_scale
    # block_scale is in FP8, need to cast to float32 for multiplication
    total_scale = scale.to(torch.float32) * block_scales_unswizzled.to(torch.float32)

    # Apply scaling: (M, K//16, 16) * (M, K//16, 1)
    data_dequantized = data_reshaped * total_scale.unsqueeze(-1)

    # Reshape back and slice to original shape
    result = data_dequantized.reshape(padded_shape)

    # Remove padding if necessary
    if orig_shape != padded_shape:
        result = result[: orig_shape[0], : orig_shape[1]]

    return result.to(output_dtype)


class NVFP4Layout(QuantizedLayout):
    """
    NVFP4 E2M1 block quantization with per-tensor and block scaling.

    Auto-pads to 16x16 alignment for cuBLAS compatibility.

    Note:
        Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
        Shape operations (view, reshape, transpose) are limited due to
        packed format and block scales.
    """

    BLOCK_SIZE = NVFP4_BLOCK_SIZE
    MIN_SM_VERSION = (10, 0)

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize a 2D tensor to NVFP4 format.

        Args:
            tensor: Input 2D tensor
            scale: Per-tensor scale (computed if None)

        Returns:
            Tuple of (qdata, layout_params)
        """
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        device = tensor.device

        # Pad to 16x16 alignment
        rows, cols = orig_shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)

        if padded_rows != rows or padded_cols != cols:
            tensor_padded = torch.zeros(
                (padded_rows, padded_cols), device=device, dtype=tensor.dtype
            )
            tensor_padded[:rows, :cols] = tensor
            tensor = tensor_padded

        # Compute per-tensor scale if not provided
        if scale is None:
            amax = torch.amax(tensor.abs())
            scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)

        scale = scale.to(device=device, dtype=torch.float32)

        # Reshape to blocks
        M, K = tensor.shape
        tensor_blocks = tensor.reshape(M, -1, cls.BLOCK_SIZE)

        # Compute per-block scales
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)
        block_scale = block_max / F4_E2M1_MAX
        block_scale_fp32 = block_scale.to(torch.float32)

        # Scale block scales by per-tensor scale
        scaled_block_scales = block_scale_fp32 / scale
        scaled_block_scales = torch.clamp(
            scaled_block_scales, min=1e-12, max=F8_E4M3_MAX
        )
        scaled_block_scales_fp8 = _float8_round(scaled_block_scales)

        # Compute total scale for data
        total_scale = scale * scaled_block_scales_fp8

        # Scale and quantize data
        data_scaled = tensor_blocks / total_scale.unsqueeze(-1)
        data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
        data_scaled = data_scaled.reshape(M, K)

        # Convert to FP4 E2M1 format
        data_lp = _f32_to_floatx_unpacked(
            data_scaled.float(), F4_E2M1_EBITS, F4_E2M1_MBITS
        )

        # Pack two FP4 values per uint8
        qdata = pack_uint4(data_lp)

        # Convert block scales to cuBLAS tiled layout
        blocked_scales = to_blocked(
            scaled_block_scales.to(torch.float8_e4m3fn), flatten=False
        )

        layout_params = {
            "scale": scale,
            "block_scale": blocked_scales,
            "orig_dtype": orig_dtype,
            "orig_shape": orig_shape,
        }

        return qdata, layout_params

    @staticmethod
    def dequantize(
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_scale: torch.Tensor,
        orig_dtype: torch.dtype,
        orig_shape: Tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor back to float.

        Args:
            qdata: Packed FP4 data (uint8)
            scale: Per-tensor scale (float32)
            block_scale: Block scales in cuBLAS tiled layout (float8_e4m3fn)
            orig_dtype: Target output dtype
            orig_shape: Original tensor shape before padding

        Returns:
            Dequantized tensor of shape orig_shape
        """
        # Try comfy-kitchen first for potential hardware acceleration
        try:
            result = _try_comfy_kitchen_dequantize(
                qdata, scale, block_scale, orig_dtype
            )
            # Slice to original shape if padded
            if result.shape != orig_shape:
                result = result[: orig_shape[0], : orig_shape[1]]
            return result
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"comfy-kitchen dequantize failed, using fallback: {e}")

        # Pure PyTorch fallback
        return _pure_pytorch_dequantize(
            qdata, scale, block_scale, orig_dtype, orig_shape
        )

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Extract raw tensors for computation.

        Returns:
            (qdata, scale, block_scale)
        """
        return (
            qtensor._qdata,
            qtensor._layout_params["scale"],
            qtensor._layout_params["block_scale"],
        )

    @classmethod
    def get_padded_shape(cls, orig_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get padded shape for 16x16 alignment."""
        if len(orig_shape) != 2:
            raise ValueError(f"NVFP4 requires 2D shape, got {len(orig_shape)}D")
        rows, cols = orig_shape
        return (roundup(rows, 16), roundup(cols, 16))

    @classmethod
    def get_storage_shape(cls, orig_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get storage shape (accounts for packing)."""
        padded = cls.get_padded_shape(orig_shape)
        return (padded[0], padded[1] // 2)


# =============================================================================
# Operation Handlers
# =============================================================================


@register_layout_op(torch.ops.aten.linear.default, "NVFP4Layout")
def nvfp4_linear(func, args, kwargs):
    """
    NVFP4 linear operation: output = input @ weight.T + bias

    Currently uses dequantization. Hardware matmul requires Blackwell GPU
    and comfy-kitchen with scaled_mm_nvfp4 support.
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Dequantize weight if quantized
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    # Dequantize input if quantized
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "NVFP4Layout")
def nvfp4_mm(func, args, kwargs):
    """NVFP4 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "NVFP4Layout")
def nvfp4_addmm(func, args, kwargs):
    """NVFP4 addmm operation (dequant-fallback)."""
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


@register_layout_op(torch.ops.aten.t.default, "NVFP4Layout")
def nvfp4_transpose(func, args, kwargs):
    """
    Handle transpose for NVFP4 tensors.

    Note: NVFP4 cannot efficiently transpose the packed format,
    so we dequantize first.
    """
    input_tensor = args[0]

    if isinstance(input_tensor, QuantizedTensor):
        # Dequantize and transpose
        dequantized = input_tensor.dequantize()
        return func(dequantized)

    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "NVFP4Layout")
def nvfp4_view(func, args, kwargs):
    """
    Handle view for NVFP4 tensors.

    Note: NVFP4 cannot efficiently reshape the packed format,
    so we dequantize first.
    """
    input_tensor = args[0]

    if isinstance(input_tensor, QuantizedTensor):
        # Dequantize and view
        dequantized = input_tensor.dequantize()
        return func(dequantized, *args[1:], **kwargs)

    return func(*args, **kwargs)
