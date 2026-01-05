"""
Hybrid NVFP4 Operations for ComfyUI inference.

This module provides custom ops that handle NVFP4 (FP4 E2M1) quantized models,
supporting:
- Legacy and new state dict key formats
- Dequantization with comfy-kitchen or pure PyTorch fallback
- LoRA patching (dequantize -> patch -> inference)
"""

import json
import logging
from typing import Optional, Tuple

import torch
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, LAYOUTS

from .utils.float_utils import (
    F4_E2M1_EBITS,
    F4_E2M1_MBITS,
    NVFP4_BLOCK_SIZE,
    unpack_uint4,
    from_blocked,
    _floatx_unpacked_to_f32,
)
from .utils.hardware_check import check_nvfp4_hardware_support, check_comfy_kitchen_available

logger = logging.getLogger(__name__)


class HybridNVFP4Ops:
    """
    Hybrid NVFP4 operations class for ComfyUI inference.

    Handles NVFP4-quantized models with:
    - State dict loading with proper scale extraction
    - Dequantization fallback for non-Blackwell GPUs
    - LoRA patching support via convert_weight/set_weight
    """

    class Linear(manual_cast.Linear):
        """
        Linear layer with NVFP4 quantization support.

        Attributes:
            scale_weight: Per-tensor scale (float32 scalar)
            block_scale: Per-block scales in cuBLAS tiled layout (float8_e4m3fn)
            orig_shape: Original weight shape before padding
            is_quantized: Whether this layer has NVFP4 weights
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight: Optional[torch.Tensor] = None
            self.block_scale: Optional[torch.Tensor] = None
            self.orig_shape: Optional[Tuple[int, int]] = None
            self.is_quantized: bool = False
            self.quant_format: Optional[str] = None

        def reset_parameters(self):
            """Skip default initialization for quantized weights."""
            return None

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            """
            Custom state dict loading that detects NVFP4 format.

            NVFP4 state dict keys:
            - {prefix}weight: uint8 packed FP4 data (M, K//2)
            - {prefix}weight_scale: float32 per-tensor scale
            - {prefix}block_scale: float8_e4m3fn block scales (cuBLAS tiled)
            - {prefix}comfy_quant (optional): JSON metadata
            """
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            block_scale_key = f"{prefix}block_scale"
            metadata_key = f"{prefix}comfy_quant"

            # Also check legacy key format
            legacy_scale_key = f"{prefix}scale_weight"

            if weight_key not in state_dict:
                return super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

            weight = state_dict[weight_key]

            # Detect NVFP4 format: uint8 weight + block_scale present
            if weight.dtype == torch.uint8 and block_scale_key in state_dict:
                self.is_quantized = True
                self.quant_format = "nvfp4"

                # Load scales
                if scale_key in state_dict:
                    self.scale_weight = state_dict[scale_key]
                elif legacy_scale_key in state_dict:
                    self.scale_weight = state_dict[legacy_scale_key]
                else:
                    logger.warning(
                        f"NVFP4 weight at {weight_key} missing scale, using 1.0"
                    )
                    self.scale_weight = torch.tensor(1.0, dtype=torch.float32)

                self.block_scale = state_dict[block_scale_key]

                # Parse metadata if available
                if metadata_key in state_dict:
                    try:
                        metadata_tensor = state_dict[metadata_key]
                        metadata_bytes = metadata_tensor.numpy().tobytes()
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        self.orig_shape = tuple(metadata.get("orig_shape", weight.shape))
                    except Exception as e:
                        logger.debug(f"Could not parse NVFP4 metadata: {e}")
                        # Infer original shape from packed data
                        self.orig_shape = (weight.shape[0], weight.shape[1] * 2)
                else:
                    # Infer original shape from packed data
                    self.orig_shape = (weight.shape[0], weight.shape[1] * 2)

                logger.debug(
                    f"Loaded NVFP4 layer: {prefix}, "
                    f"weight={weight.shape}, orig_shape={self.orig_shape}"
                )

                # Store packed weight directly
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

                # Mark scale/block_scale keys as expected
                for key in [scale_key, legacy_scale_key, block_scale_key, metadata_key]:
                    if key in unexpected_keys:
                        unexpected_keys.remove(key)
                    if key in state_dict and key in missing_keys:
                        missing_keys.remove(key)

                return

            # Not NVFP4, use parent implementation
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        def _dequantize_weight(
            self,
            weight: torch.Tensor,
            scale: torch.Tensor,
            block_scale: torch.Tensor,
            input_dtype: torch.dtype,
        ) -> torch.Tensor:
            """
            Dequantize NVFP4 weight to float.

            Tries comfy-kitchen first, falls back to pure PyTorch.

            Args:
                weight: Packed uint8 FP4 data
                scale: Per-tensor scale
                block_scale: Block scales in cuBLAS tiled layout
                input_dtype: Target output dtype

            Returns:
                Dequantized weight tensor
            """
            # Try comfy-kitchen
            try:
                import comfy_kitchen as ck

                result = ck.dequantize_nvfp4(weight, scale, block_scale, input_dtype)
                if self.orig_shape and result.shape != self.orig_shape:
                    result = result[: self.orig_shape[0], : self.orig_shape[1]]
                return result
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"comfy-kitchen dequantize failed: {e}")

            # Pure PyTorch fallback
            return self._pure_pytorch_dequantize(
                weight, scale, block_scale, input_dtype
            )

        def _pure_pytorch_dequantize(
            self,
            weight: torch.Tensor,
            scale: torch.Tensor,
            block_scale: torch.Tensor,
            input_dtype: torch.dtype,
        ) -> torch.Tensor:
            """Pure PyTorch NVFP4 dequantization."""
            # Unpack FP4 data: (M, K//2) -> (M, K)
            data_unpacked = unpack_uint4(weight)

            # Convert unpacked FP4 to float32
            data_f32 = _floatx_unpacked_to_f32(
                data_unpacked, F4_E2M1_EBITS, F4_E2M1_MBITS
            )

            M, K = data_f32.shape

            # Reshape to blocks: (M, K) -> (M, K//16, 16)
            data_reshaped = data_f32.reshape(M, -1, NVFP4_BLOCK_SIZE)

            # Unswizzle block_scales from cuBLAS tiled layout
            num_blocks_per_row = K // NVFP4_BLOCK_SIZE
            block_scales_unswizzled = from_blocked(
                block_scale.reshape(-1).to(data_f32.device),
                num_rows=M,
                num_cols=num_blocks_per_row,
            )

            # Compute total scale: per_tensor_scale * block_scale
            total_scale = scale.to(torch.float32) * block_scales_unswizzled.to(
                torch.float32
            )

            # Apply scaling: (M, K//16, 16) * (M, K//16, 1)
            data_dequantized = data_reshaped * total_scale.unsqueeze(-1)

            # Reshape back
            result = data_dequantized.reshape(M, K)

            # Remove padding if necessary
            if self.orig_shape and result.shape != self.orig_shape:
                result = result[: self.orig_shape[0], : self.orig_shape[1]]

            return result.to(input_dtype)

        def forward_comfy_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """Forward pass with proper NVFP4 handling."""
            if self.is_quantized and self.quant_format == "nvfp4":
                # Dequantize weight
                weight = self._dequantize_weight(
                    self.weight,
                    self.scale_weight,
                    self.block_scale,
                    input.dtype,
                )
                weight = weight.to(device=input.device)

                # Handle bias
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)

                return torch.nn.functional.linear(input, weight, bias)

            # Non-quantized path: use parent implementation
            weight, bias, offload_stream = cast_bias_weight(
                self, input, offloadable=True
            )
            result = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return result

        def forward(self, *args, **kwargs):
            """Standard forward that routes to cast_weights version."""
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """
            Convert weight for LoRA patching - dequantize NVFP4.

            Called by ComfyUI's model patcher before applying LoRA.
            """
            if self.is_quantized and self.quant_format == "nvfp4":
                return self._dequantize_weight(
                    weight,
                    self.scale_weight,
                    self.block_scale,
                    torch.float32,
                )
            return weight

        def set_weight(
            self,
            weight,
            inplace_update=False,
            seed=None,
            return_weight=False,
            **kwargs,
        ):
            """
            Set weight after LoRA patching.

            After patching, the weight is no longer quantized, so we
            clear the quantization state.
            """
            self.is_quantized = False
            self.quant_format = None
            self.scale_weight = None
            self.block_scale = None
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

            if return_weight:
                return weight

    # =========================================================================
    # Normalization Layers (passthrough to parent)
    # =========================================================================

    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

    # =========================================================================
    # Convolution Layers (passthrough to parent for now)
    # =========================================================================

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
        """Factory method for creating Conv layers."""
        if dims == 1:
            return cls.Conv1d(*args, **kwargs)
        elif dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        raise ValueError(f"Unsupported conv dims: {dims}")
