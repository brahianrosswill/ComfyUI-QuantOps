"""
Float utilities for sub-byte quantization formats.

Provides FP4 E2M1 encoding/decoding, uint4 packing, and cuBLAS tiled layout
transformations for NVFP4 quantization support.

Derived from:
- comfy-kitchen (Comfy Org, Apache-2.0)
- PyTorch AO (Meta Platforms, BSD-3-Clause)
"""

import torch
from typing import Tuple


def _n_ones(n: int) -> int:
    """Generate a bitmask with n ones."""
    return (1 << n) - 1


# =============================================================================
# Float Format Constants
# =============================================================================

EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

# FP4 E2M1 constants
F4_E2M1_MAX = 6.0
F4_E2M1_EPS = 0.5
F4_E2M1_EBITS = 2
F4_E2M1_MBITS = 1

# FP8 E4M3 constants
F8_E4M3_MAX = 448.0
F8_E4M3_EPS = 0.125

# NVFP4 block size (fixed by format)
NVFP4_BLOCK_SIZE = 16


# =============================================================================
# Utility Functions
# =============================================================================

def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return (a + b - 1) // b


def down_size(size: Tuple[int, ...]) -> Tuple[int, ...]:
    """Halve the last dimension (for packing two values per byte)."""
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size: Tuple[int, ...]) -> Tuple[int, ...]:
    """Double the last dimension (for unpacking)."""
    return (*size[:-1], size[-1] * 2)


# =============================================================================
# FP4 E2M1 Encoding/Decoding
# =============================================================================

def _float8_round(x: torch.Tensor) -> torch.Tensor:
    """Round tensor through FP8 and back to float32."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def _f32_to_floatx_unpacked(
    x: torch.Tensor, ebits: int, mbits: int
) -> torch.Tensor:
    """
    Convert FP32 numbers to sub-byte floating point numbers.

    Args:
        x: Input tensor of dtype torch.float
        ebits: Number of exponent bits
        mbits: Number of mantissa bits

    Returns:
        torch.Tensor of dtype torch.uint8, where the bit encoding is stored
        in the least significant bits. e.g. fp4: bits 0-3 empty, bits 4-7 in
        fp4_e2m1 encoding.

    Note:
        No special values (NaN, inf) are supported. Values outside the
        representable range after rounding are clamped to the maximum
        magnitude (sign is preserved).
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # Calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # All E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # Reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # Save the sign
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # Set everything to positive, will add sign back at the end
    x = x ^ sign
    x = x.view(torch.float)

    # Rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(
        torch.logical_not(saturate_mask), x < min_normal
    )
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    # Branch 2: conversion to denormal as well as rounding up to normal
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Branch 3: stay in normal range, adjust the exponent and round
    normal_x = x.view(torch.int32)
    # Resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # Update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # Rounding bias part 2
    normal_x += mant_odd
    # Take the bits
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    # Combine the branches
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # Add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Mask out bits to get just the sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _floatx_unpacked_to_f32(
    x: torch.Tensor, ebits: int, mbits: int
) -> torch.Tensor:
    """
    Convert sub-byte floating point numbers to FP32.

    Args:
        x: Input tensor of dtype uint8, where the bit encoding is stored
           in the least significant bits.
        ebits: Number of exponent bits
        mbits: Number of mantissa bits

    Returns:
        torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # Save the sign
    sign_lp = x & sign_mask

    # Set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    # 1. Calculate zero mask
    zero_mask = x_pos == 0

    # 2. Calculate the denormal path mask
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    # 3. Calculate the normal path
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # Shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    # 4. Add the zero and denormal casts
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # Fast path for FP4_E2M1
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32
    else:
        # Iterate over all possible values of mantissa
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                left_shift = mbits - i
                mantissa_f32_val = (mantissa_cmp - (1 << i)) << (
                    left_shift + MBITS_F32 - mbits
                )
                exp_biased_f32_val = (denormal_exp_biased - left_shift) << MBITS_F32
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32_val + mantissa_f32_val
                )
        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # Add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


# =============================================================================
# Uint4 Packing/Unpacking
# =============================================================================

def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """
    Pack two uint4 values into one uint8.

    Args:
        uint8_data: Tensor with values in [0, 15] stored as uint8

    Returns:
        Packed tensor with half the last dimension size.
        Even indices go to upper nibble, odd indices to lower nibble.
    """
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed = (uint8_data[::2] << 4) | uint8_data[1::2]
    return packed.view(down_size(shape))


def unpack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 to two uint4 values.

    Args:
        uint8_data: Packed tensor

    Returns:
        Unpacked tensor with double the last dimension size.
        Upper nibble becomes even indices, lower nibble becomes odd indices.
    """
    assert uint8_data.is_contiguous()
    shape = uint8_data.shape

    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        up_size(shape)
    )
    return unpacked


# =============================================================================
# cuBLAS Tiled Layout Conversion
# =============================================================================

def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange a matrix to cuBLAS 2D block scaling factors layout.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        flatten: If True, return flattened tensor; else return 2D

    Returns:
        Rearranged tensor in cuBLAS tiled layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    if flatten:
        return rearranged.flatten()

    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(
    blocked_matrix: torch.Tensor, num_rows: int, num_cols: int
) -> torch.Tensor:
    """
    Reverse cuBLAS tiled layout back to standard (H, W) layout.

    Args:
        blocked_matrix: Swizzled tensor from cuBLAS layout
        num_rows: Desired output rows (unpadded)
        num_cols: Desired output cols (unpadded)

    Returns:
        Unswizzled tensor of shape (num_rows, num_cols)
    """
    n_row_blocks = ceil_div(num_rows, 128)
    n_col_blocks = ceil_div(num_cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


# =============================================================================
# High-Level FP4 Conversion Functions
# =============================================================================

def fp4_to_f32(packed_data: torch.Tensor) -> torch.Tensor:
    """
    Convert packed FP4 E2M1 data to float32.

    Args:
        packed_data: Packed uint8 tensor (2 FP4 values per byte)

    Returns:
        Float32 tensor with double the last dimension
    """
    unpacked = unpack_uint4(packed_data)
    return _floatx_unpacked_to_f32(unpacked, F4_E2M1_EBITS, F4_E2M1_MBITS)


def f32_to_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 tensor to packed FP4 E2M1 format.

    Args:
        tensor: Float32 tensor (last dim must be even)

    Returns:
        Packed uint8 tensor with half the last dimension
    """
    fp4_unpacked = _f32_to_floatx_unpacked(
        tensor.float(), F4_E2M1_EBITS, F4_E2M1_MBITS
    )
    return pack_uint4(fp4_unpacked)
