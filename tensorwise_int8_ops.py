"""
Tensorwise INT8 Operations

Custom ComfyUI operations for tensorwise INT8 quantization.
Uses torch._int_mm with dynamic per-row activation quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch import Tensor

from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.model_patcher import LowVramPatch

# Import quantization utilities
from .quant_layouts.tensorwise_int8_layout import (
    quantize_int8,
    quantize_int8_tensorwise,
    quantize_int8_axiswise,
    dequantize,
)


class TensorWiseInt8Ops(manual_cast):
    """
    Custom ComfyUI operations for tensorwise INT8 quantization.

    Uses torch._int_mm for native INT8 matmul with dynamic per-row
    activation quantization. Provides ~2x speedup on RTX 30-series.

    Supports fused LoRA without full weight dequantization.

    Usage:
        model_options = {"custom_operations": TensorWiseInt8Ops}
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    """

    class Linear(manual_cast.Linear):
        """Linear layer with direct INT8 weight loading and fast _int_mm forward."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_scale = None
            self.input_scale = None  # Optional: for static activation quant
            self._is_quantized = False
            self.compute_dtype = torch.bfloat16

        def reset_parameters(self):
            """Skip weight initialization - we load from state dict."""
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
            Load INT8 weights and scales directly from state dict.
            No dequant/requant needed.
            """
            weight_key = prefix + "weight"
            scale_key = prefix + "weight_scale"
            input_scale_key = prefix + "input_scale"
            bias_key = prefix + "bias"

            # Pop scale tensors
            weight_scale = state_dict.pop(scale_key, None)
            input_scale = state_dict.pop(input_scale_key, None)

            # Pop comfy_quant metadata if present
            state_dict.pop(prefix + "comfy_quant", None)

            # Get weight tensor
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                    # Direct INT8 load
                    self._is_quantized = True
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)

                    # Store scale as scalar or tensor
                    if isinstance(weight_scale, torch.Tensor):
                        if weight_scale.numel() == 1:
                            self.weight_scale = weight_scale.float().item()
                        else:
                            self.weight_scale = weight_scale.float()
                    else:
                        self.weight_scale = float(weight_scale)

                    # Store input scale if present (for static quantization)
                    if input_scale is not None:
                        if isinstance(input_scale, torch.Tensor):
                            self.input_scale = input_scale.float()
                        else:
                            self.input_scale = torch.tensor(input_scale, dtype=torch.float32)

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    # High-precision weight - keep unquantized
                    self._is_quantized = False
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    # Unknown dtype - store as-is
                    self._is_quantized = False
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def _dequantize_weight(self, weight, scale, input_dtype):
            """Dequantize tensorwise INT8 weight to float."""
            if weight.dtype == torch.int8 and scale is not None:
                return dequantize(weight, scale).to(input_dtype)
            return weight.to(input_dtype)

        def forward_comfy_cast_weights(self, input):
            """Forward pass with proper INT8 handling."""
            if not self._is_quantized:
                # Non-quantized path - use standard ComfyUI cast
                weight, bias, offload_stream = cast_bias_weight(
                    self, input, offloadable=True
                )
                out = F.linear(input, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return out

            # Quantized path - use fast int8 matmul
            compute_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

            # Flatten to 2D for matmul
            x_shape = input.shape
            x_2d = input.reshape(-1, x_shape[-1])

            # Move weight to input device
            weight = self.weight.data.to(device=input.device)

            # Always use INT8 matmul (no dequantize fallback to prevent OOM)
            if self.input_scale is not None:
                # Static quantization path
                y = _int8_forward_static(
                    x_2d, weight, self.weight_scale,
                    self.input_scale, self.bias, compute_dtype
                )
            else:
                # Dynamic activation quantization (default)
                y = _int8_forward_dynamic(
                    x_2d, weight, self.weight_scale,
                    self.bias, compute_dtype
                )

            # Reshape back
            return y.reshape(*x_shape[:-1], y.shape[-1])

        def forward_fused_lora(self, input):
            """
            Memory-efficient LoRA forward pass.

            Computes: output = base_int8_matmul(x, W) + lora_contribution(x)

            Instead of dequantizing the full weight, we:
            1. Run native INT8 matmul for base (uses dynamic activation quant)
            2. Compute LoRA contribution separately: x @ B.T @ A.T * scale
            3. Sum the results

            This avoids materializing the full bf16 weight tensor.
            """
            input_dtype = input.dtype
            compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

            # Log that we're using fused LoRA path (only first few times)
            if not hasattr(TensorWiseInt8Ops.Linear, '_fused_lora_log_count'):
                TensorWiseInt8Ops.Linear._fused_lora_log_count = 0
            if TensorWiseInt8Ops.Linear._fused_lora_log_count < 3:
                weight = self.weight.data
                logging.info(f"TensorWiseINT8: Using fused LoRA path - input={input.shape}, weight={weight.shape}")
                TensorWiseInt8Ops.Linear._fused_lora_log_count += 1

            # Flatten to 2D for matmul
            x_shape = input.shape
            x_2d = input.reshape(-1, x_shape[-1])

            # Move weight to input device
            weight = self.weight.data.to(device=input.device)

            # 1. Base INT8 output (no LoRA applied) - always use INT8 matmul
            if self.input_scale is not None:
                base_out = _int8_forward_static(
                    x_2d, weight, self.weight_scale,
                    self.input_scale, None, compute_dtype
                )
            else:
                base_out = _int8_forward_dynamic(
                    x_2d, weight, self.weight_scale,
                    None, compute_dtype
                )

            # 2. Compute LoRA contributions separately (LoRA weights are small, no OOM risk)
            lora_out = None
            for patch_fn in self.weight_function:
                if isinstance(patch_fn, LowVramPatch):
                    # Extract patches for this layer
                    patches = patch_fn.patches.get(patch_fn.key, [])
                    for patch_data in patches:
                        # patch_data: (strength_patch, adapter, strength_model, offset, function)
                        strength_patch = patch_data[0]
                        adapter = patch_data[1]
                        strength_model = patch_data[2]

                        # Check if adapter has weights (LoRA-style adapters)
                        if hasattr(adapter, 'weights') and adapter.weights is not None:
                            weights = adapter.weights
                            # weights[0] = mat1 (lora_up), weights[1] = mat2 (lora_down), weights[2] = alpha
                            mat1 = weights[0]  # [out_dim, rank]
                            mat2 = weights[1]  # [rank, in_dim]
                            alpha = weights[2] if weights[2] is not None else 1.0
                            rank = mat2.shape[0]
                            scale = strength_patch * strength_model * (alpha / rank)

                            # Move to device
                            mat1 = mat1.to(device=input.device, dtype=input_dtype)
                            mat2 = mat2.to(device=input.device, dtype=input_dtype)

                            # Compute: x @ mat2.T @ mat1.T * scale
                            # input: [B, seq, in_dim], mat2: [rank, in_dim], mat1: [out_dim, rank]
                            temp = F.linear(x_2d, mat2)  # [M, rank]
                            lora_contrib = F.linear(temp, mat1) * scale  # [M, out_dim]

                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                        else:
                            # Non-LoRA adapter without decomposed weights - cannot apply without dequant
                            raise RuntimeError(
                                f"TensorWiseINT8: Unsupported adapter type (no decomposed weights). "
                                f"Cannot apply to INT8 quantized model without dequantization. "
                                f"Adapter: {type(adapter).__name__}"
                            )
                else:
                    # Non-LowVramPatch function - unsupported
                    raise RuntimeError(
                        f"TensorWiseINT8: Unsupported patch function type: {type(patch_fn).__name__}. "
                        f"Only LowVramPatch with decomposed LoRA weights is supported for INT8 models."
                    )

            # 3. Combine base + LoRA
            out = base_out
            if lora_out is not None:
                out = out + lora_out

            # Add bias
            if self.bias is not None:
                bias = self.bias.to(device=input.device, dtype=input_dtype)
                out = out + bias

            # Reshape back
            return out.reshape(*x_shape[:-1], out.shape[-1])

        def forward(self, *args, **kwargs):
            """Main forward - dispatches to appropriate path."""
            weight = self.weight
            if isinstance(weight, nn.Parameter):
                weight = weight.data

            # Check if we have LoRA patches AND quantized weight
            has_lora = len(self.weight_function) > 0
            is_quant = self._is_quantized and weight.dtype == torch.int8

            if has_lora and is_quant:
                # Use fused LoRA path to avoid full weight dequantization
                return self.forward_fused_lora(*args, **kwargs)
            elif self.comfy_cast_weights or has_lora or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                # INT8 needs our special forward path
                if weight.dtype == torch.int8:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize INT8."""
            if weight.dtype == torch.int8 and self.weight_scale is not None:
                return self._dequantize_weight(weight, self.weight_scale, torch.float32)
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            # For now, keep as dequantized (re-quantization is complex for INT8)
            if return_weight:
                return weight

            if inplace_update:
                self.weight.data.copy_(weight)
            else:
                self.weight = nn.Parameter(weight, requires_grad=False)

            # Mark as no longer quantized after patching
            self._is_quantized = False
            self.weight_scale = None

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


# ==============================================================================
# INT8 Forward Functions
# ==============================================================================


@torch.no_grad()
def _int8_forward_dynamic(
    x: Tensor,
    weight: Tensor,
    weight_scale: float,
    bias: Tensor | None,
    compute_dtype: torch.dtype
) -> Tensor:
    """
    Forward with dynamic per-row activation quantization.

    Args:
        x: Input tensor [M, K]
        weight: INT8 weight [N, K]
        weight_scale: Per-tensor weight scale (scalar)
        bias: Optional bias [N]
        compute_dtype: Output dtype

    Returns:
        Output tensor [M, N]
    """
    # Dynamic per-row activation quantization
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)

    # Try Triton kernel first
    try:
        from .kernels.tensorwise_kernels import mm_8bit
        res = mm_8bit(x_8, weight.T)
    except ImportError:
        # Fallback to torch._int_mm
        res = torch._int_mm(x_8, weight.T)

    # Rescale and convert dtype
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)

    # Add bias
    if bias is not None:
        res_scaled = res_scaled + bias.to(device=res_scaled.device, dtype=compute_dtype)

    return res_scaled


@torch.no_grad()
def _int8_forward_static(
    x: Tensor,
    weight: Tensor,
    weight_scale: float,
    input_scale: Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype
) -> Tensor:
    """
    Forward with static (calibrated) activation quantization.

    Args:
        x: Input tensor [M, K]
        weight: INT8 weight [N, K]
        weight_scale: Per-tensor weight scale (scalar)
        input_scale: Calibrated input scale (scalar)
        bias: Optional bias [N]
        compute_dtype: Output dtype

    Returns:
        Output tensor [M, N]
    """
    # Static activation quantization using calibrated scale
    x_8 = quantize_int8(x, input_scale)

    # Try Triton kernel first
    try:
        from .kernels.tensorwise_kernels import mm_8bit
        res = mm_8bit(x_8, weight.T)
    except ImportError:
        # Fallback to torch._int_mm
        res = torch._int_mm(x_8, weight.T)

    # Rescale: combined scale is weight_scale * input_scale
    res_scaled = res.float().mul_(weight_scale * input_scale).to(compute_dtype)

    # Add bias
    if bias is not None:
        res_scaled = res_scaled + bias.to(device=res_scaled.device, dtype=compute_dtype)

    return res_scaled
