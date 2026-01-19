"""
Unified Quantization Operations for ComfyUI.

This module provides custom ops that correctly read group_size from per-layer
metadata for all quantized models (INT8, FP8, MXFP8, NVFP4).

The issue: Core ComfyUI's MixedPrecisionOps reads block_size from QUANT_ALGOS
fallback instead of per-layer metadata, causing wrong block boundaries.
"""

import json
import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class


class QuantizedOps(manual_cast):
    """
    Unified quantization operations class that correctly handles block_size from metadata.

    Supports all quantization formats via comfy-kitchen QuantizedTensor dispatch:
    - INT8 blockwise (BlockWiseINT8Layout)
    - FP8 tensorwise (TensorCoreFP8Layout)
    - FP8 blockwise/rowwise (BlockWiseFP8Layout, RowWiseFP8Layout)
    - MXFP8 (TensorCoreMXFP8Layout)
    - NVFP4 (TensorCoreNVFP4Layout)
    """


    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = None
            self.block_size = None
            self.is_quantized = False
            self.layout_type = None
            self.quant_format = None

        def reset_parameters(self):
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
            Custom state dict loading that correctly reads group_size from per-layer metadata.
            """
            weight_key = prefix + "weight"

            # Get weight_scale (block scale for NVFP4, tensor scale for FP8)
            scale = state_dict.pop(prefix + "weight_scale", None)
            
            # Get weight_scale_2 (per-tensor scale for NVFP4)
            scale_2 = state_dict.pop(prefix + "weight_scale_2", None)

            # Remove input_scale if present (not used for weight dequantization)
            state_dict.pop(prefix + "input_scale", None)

            # Parse comfy_quant metadata for layout type and block_size
            comfy_quant_tensor = state_dict.pop(prefix + "comfy_quant", None)
            layer_conf = None

            if comfy_quant_tensor is not None:
                try:
                    # Decode the comfy_quant tensor to dict
                    layer_conf = json.loads(comfy_quant_tensor.numpy().tobytes())
                    self.quant_format = layer_conf.get("format", None)
                    # KEY FIX: Read group_size from per-layer metadata!
                    self.block_size = layer_conf.get("group_size", None)
                    logging.debug(
                        f"HybridFP8Ops: Parsed comfy_quant for {prefix}: format={self.quant_format}, group_size={self.block_size}"
                    )
                except Exception as e:
                    logging.debug(
                        f"HybridFP8Ops: Failed to parse comfy_quant metadata: {e}"
                    )

            # Load weight tensor
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                # Check if this is NVFP4 (uint8 packed format with scale_2)
                is_nvfp4 = (
                    self.quant_format == "nvfp4" or 
                    (weight_tensor.dtype == torch.uint8 and scale_2 is not None)
                )
                
                if is_nvfp4:
                    self.is_quantized = True
                    self.layout_type = "TensorCoreNVFP4Layout"
                    self.block_size = 16  # NVFP4 uses 16x16 blocks
                    
                    from comfy.quant_ops import TensorCoreNVFP4Layout
                    
                    # Get orig_dtype from metadata or default to bfloat16
                    orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                    DTYPE_MAP = {
                        "torch.bfloat16": torch.bfloat16,
                        "torch.float16": torch.float16,
                        "torch.float32": torch.float32,
                    }
                    orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                    
                    # Get orig_shape from metadata or compute from packed storage
                    if layer_conf and "orig_shape" in layer_conf:
                        orig_shape = tuple(layer_conf["orig_shape"])
                    else:
                        # NVFP4 packs 2 values per byte: logical cols = storage cols * 2
                        orig_shape = (weight_tensor.shape[0], weight_tensor.shape[1] * 2)
                    
                    # scale = block_scale (FP8 E4M3), scale_2 = per-tensor scale (float32)
                    layout_params = TensorCoreNVFP4Layout.Params(
                        scale=scale_2.to(torch.float32) if scale_2 is not None else torch.tensor(1.0),
                        orig_dtype=orig_dtype,
                        orig_shape=orig_shape,
                        block_scale=scale,  # FP8 E4M3 block scales
                    )
                    
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                    logging.debug(
                        f"HybridFP8Ops: Loaded NVFP4 layer {prefix}, block_scale shape={scale.shape if scale is not None else None}"
                    )
                
                # Check if this is an FP8 tensor
                elif weight_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    self.is_quantized = True
                    self.scale_weight = scale

                    # Determine layout type from format
                    if self.quant_format is not None:
                        qconfig = QUANT_ALGOS.get(self.quant_format, {})
                        self.layout_type = qconfig.get(
                            "comfy_tensor_layout", "TensorCoreFP8Layout"
                        )

                        # Fallback block_size from QUANT_ALGOS only if not in metadata
                        if self.block_size is None:
                            self.block_size = qconfig.get("group_size", None)
                    else:
                        # Infer layout from scale shape
                        if scale is not None:
                            if scale.ndim == 0 or (
                                scale.ndim == 1 and scale.numel() == 1
                            ):
                                self.layout_type = "TensorCoreFP8Layout"
                            elif (
                                scale.ndim == 1
                                and scale.numel() == weight_tensor.shape[0]
                            ):
                                self.layout_type = "RowWiseFP8Layout"
                            elif scale.ndim == 2:
                                self.layout_type = "BlockWiseFP8Layout"
                                # Infer block_size from scale shape
                                if self.block_size is None:
                                    M, N = weight_tensor.shape
                                    scale_M, scale_N = scale.shape
                                    if M % scale_M == 0 and N % scale_N == 0:
                                        self.block_size = M // scale_M
                            else:
                                self.layout_type = "TensorCoreFP8Layout"
                        else:
                            self.layout_type = "TensorCoreFP8Layout"

                    # Check if the layout is registered
                    try:
                        get_layout_class(self.layout_type)
                    except KeyError:
                        logging.warning(
                            f"HybridFP8Ops: Layout '{self.layout_type}' not registered, using TensorCoreFP8Layout"
                        )
                        self.layout_type = "TensorCoreFP8Layout"



                    # Create layout_params based on layout_type
                    if self.layout_type == "TensorCoreMXFP8Layout":
                        from comfy_kitchen.tensor import TensorCoreMXFP8Layout
                        # Get orig_dtype from comfy_quant metadata if available
                        orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                        DTYPE_MAP = {
                            "torch.bfloat16": torch.bfloat16,
                            "torch.float16": torch.float16,
                            "torch.float32": torch.float32,
                        }
                        orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                        
                        # Get orig_shape from metadata or use current shape
                        orig_shape = tuple(layer_conf.get("orig_shape", list(weight_tensor.shape))) if layer_conf else tuple(weight_tensor.shape)
                        
                        # Convert E8M0 scales from uint8 to float8_e8m0fnu (safetensors stores as uint8)
                        if scale is not None and scale.dtype == torch.uint8:
                            scale = scale.view(torch.float8_e8m0fnu)
                        
                        layout_params = TensorCoreMXFP8Layout.Params(
                            scale=scale,
                            orig_dtype=orig_dtype,
                            orig_shape=orig_shape,
                        )
                        logging.debug(
                            f"HybridFP8Ops: Loading MXFP8 layer {prefix}, scale shape={scale.shape if scale is not None else None}"
                        )
                    elif self.layout_type == "BlockWiseFP8Layout":
                        from .quant_layouts.fp8_variants import BlockWiseFP8Layout
                        block_size = self.block_size if self.block_size is not None else 64
                        if self.block_size is None:
                            logging.warning(
                                f"HybridFP8Ops: No block_size found for {prefix}, using fallback 64"
                            )
                        layout_params = BlockWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                            block_size=block_size,
                        )
                    elif self.layout_type == "RowWiseFP8Layout":
                        from .quant_layouts.fp8_variants import RowWiseFP8Layout
                        layout_params = RowWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                        )
                    else:
                        # TensorCoreFP8Layout or other - use comfy's layout
                        from comfy.quant_ops import TensorCoreFP8Layout
                        layout_params = TensorCoreFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                        )

                    # Create QuantizedTensor
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                    logging.debug(
                        f"QuantizedOps: Loaded FP8 layer {prefix} with layout={self.layout_type}, block_size={self.block_size}"
                    )
                
                # Check if this is INT8
                elif weight_tensor.dtype == torch.int8 or self.quant_format == "int8_blockwise":
                    self.is_quantized = True
                    self.layout_type = "BlockWiseINT8Layout"
                    self.block_size = self.block_size if self.block_size is not None else 128
                    
                    from comfy_kitchen.tensor import BlockWiseINT8Layout
                    
                    # Get orig_dtype from metadata or default to bfloat16
                    orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                    DTYPE_MAP = {
                        "torch.bfloat16": torch.bfloat16,
                        "torch.float16": torch.float16,
                        "torch.float32": torch.float32,
                    }
                    orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                    
                    # Get orig_shape from metadata or use current shape
                    orig_shape = tuple(layer_conf.get("orig_shape", list(weight_tensor.shape))) if layer_conf else tuple(weight_tensor.shape)
                    
                    layout_params = BlockWiseINT8Layout.Params(
                        scale=scale.to(torch.float32) if scale is not None else None,
                        orig_dtype=orig_dtype,
                        orig_shape=orig_shape,
                        block_size=self.block_size,
                        is_weight=True,
                    )
                    
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                    logging.debug(
                        f"QuantizedOps: Loaded INT8 layer {prefix}, block_size={self.block_size}"
                    )
                
                else:
                    # Non-quantized weight - high-precision layer
                    self.is_quantized = False
                    self.scale_weight = None
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + "bias"
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def forward_comfy_cast_weights(self, input):
            """Forward pass with proper FP8 handling."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data

            input_dtype = input.dtype

            # Handle QuantizedTensor (triggers dispatch to layout handlers)
            if isinstance(weight, QuantizedTensor):
                # Move to input device if needed
                if weight.device != input.device:
                    weight = weight.to(device=input.device)

                # Update orig_dtype for dequantization
                if hasattr(weight, "_params"):
                    object.__setattr__(weight._params, "orig_dtype", input_dtype)

                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)

                # For MXFP8: quantize input to QuantizedTensor so handler uses scaled_mm
                if self.layout_type == "TensorCoreMXFP8Layout":
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        input = QuantizedTensor.from_float(input, "TensorCoreMXFP8Layout")
                        output = torch.nn.functional.linear(input, weight, bias)
                        if tensor_3d:
                            output = output.reshape(input_shape[0], input_shape[1], -1)
                        return output
                    else:
                        # Fallback for non-2D: dequantize weight
                        return torch.nn.functional.linear(
                            input.reshape(input_shape), weight.dequantize(), bias
                        )

                # For NVFP4: quantize input to QuantizedTensor so handler uses scaled_mm_nvfp4
                if self.layout_type == "TensorCoreNVFP4Layout":
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        input = QuantizedTensor.from_float(input, "TensorCoreNVFP4Layout")
                        output = torch.nn.functional.linear(input, weight, bias)
                        if tensor_3d:
                            output = output.reshape(input_shape[0], input_shape[1], -1)
                        return output
                    else:
                        # Fallback for non-2D: dequantize weight
                        return torch.nn.functional.linear(
                            input.reshape(input_shape), weight.dequantize(), bias
                        )

                # For INT8: quantize input to QuantizedTensor so handler uses scaled_mm_int8
                if self.layout_type == "BlockWiseINT8Layout":
                    input_shape = input.shape
                    tensor_3d = input.ndim == 3
                    
                    if tensor_3d:
                        input = input.reshape(-1, input_shape[2])
                    
                    if input.ndim == 2:
                        input = QuantizedTensor.from_float(input, "BlockWiseINT8Layout")
                        output = torch.nn.functional.linear(input, weight, bias)
                        if tensor_3d:
                            output = output.reshape(input_shape[0], input_shape[1], -1)
                        return output
                    else:
                        # Fallback for non-2D: dequantize weight
                        return torch.nn.functional.linear(
                            input.reshape(input_shape), weight.dequantize(), bias
                        )

                # This triggers QuantizedTensor dispatch -> layout-specific handler
                return torch.nn.functional.linear(input, weight, bias)

            # Fallback: dequantize FP8 weight manually if needed
            if self.is_quantized and weight.dtype in [
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]:
                weight = weight.to(device=input.device)

                if self.scale_weight is not None:
                    scale = self.scale_weight.to(device=input.device)
                    weight_dequant = self._dequantize_weight(weight, scale, input_dtype)
                else:
                    weight_dequant = weight.to(input_dtype)

                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)
                return torch.nn.functional.linear(input, weight_dequant, bias)

            # Standard manual_cast path for non-quantized weights
            weight, bias, offload_stream = cast_bias_weight(
                self, input, offloadable=True
            )
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def _dequantize_weight(self, weight, scale, input_dtype):
            """Dequantize FP8 weight to float.

            Handles:
            - TensorCoreFP8Layout: scalar scale
            - RowWiseFP8Layout: scale shape (M,)
            - BlockWiseFP8Layout: scale shape (M//block_size, N//block_size)
            """
            M, N = weight.shape

            # Scalar scale (tensor-wise)
            if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                return weight.to(input_dtype) * scale.item()

            # Row-wise scale
            if scale.ndim == 1 and scale.shape[0] == M:
                scale_broadcast = scale.unsqueeze(1).to(
                    device=weight.device, dtype=input_dtype
                )
                return weight.to(input_dtype) * scale_broadcast

            # Block-wise scale
            if scale.ndim == 2 and self.block_size is not None:
                block_size = self.block_size
                if M % block_size == 0 and N % block_size == 0:
                    qdata_blocked = weight.reshape(
                        M // block_size, block_size, N // block_size, block_size
                    )
                    qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
                    scale_broadcast = (
                        scale.unsqueeze(-1)
                        .unsqueeze(-1)
                        .to(device=weight.device, dtype=input_dtype)
                    )
                    dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                    return dequant.permute(0, 2, 1, 3).reshape(M, N)

            # Fallback: try broadcasting
            logging.warning(
                f"FP8 scale shape {scale.shape} for weight {weight.shape}, using broadcast"
            )
            return weight.to(input_dtype) * scale.to(
                device=weight.device, dtype=input_dtype
            )

        def forward(self, *args, **kwargs):
            if (
                self.comfy_cast_weights
                or len(self.weight_function) > 0
                or len(self.bias_function) > 0
            ):
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                weight = self.weight
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data

                # Quantized weights need our special forward path
                if weight.dtype in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.int8,
                ] or isinstance(weight, QuantizedTensor):
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize FP8."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()

            if (
                weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                and self.scale_weight is not None
            ):
                return self._dequantize_weight(weight, self.scale_weight, torch.float32)

            return weight

        def set_weight(
            self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs
        ):
            """Set weight after LoRA patching - requantize if layout is available."""
            if getattr(self, 'layout_type', None) is not None:
                # Requantize using the layout's quantize method
                weight = QuantizedTensor.from_float(
                    weight, 
                    self.layout_type, 
                    scale="recalculate", 
                    stochastic_rounding=seed if seed else 0,
                    inplace_ops=True
                )
                # Match the weight dtype for proper dispatch
                if hasattr(self.weight, 'dtype'):
                    weight = weight.to(self.weight.dtype)
            else:
                # Non-quantized path
                weight = weight.to(self.weight.dtype)

            if return_weight:
                return weight

            assert inplace_update is False  # Inplace update not supported with requant
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

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


# Backward compatibility alias
HybridFP8Ops = QuantizedOps

