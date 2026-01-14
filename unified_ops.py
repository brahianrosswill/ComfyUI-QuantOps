"""
Unified Hybrid Operations for multi-format quantization.

This module provides a single ops class that handles ALL quantization formats:
- NVFP4 (uint8 packed with FP8 block scales)
- MXFP8 (float8_e4m3fn with E8M0 block scales)
- FP8 blockwise/rowwise/tensor-scaled
- INT8 blockwise/lodewise
- High-precision (bf16/fp16/fp32) passthrough

Per-layer format is determined by:
1. .comfy_quant metadata tensor (per-layer)
2. _quantization_metadata safetensors header (global fallback)
"""

import json
import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class

# Try to import INT8 layout
try:
    from .quant_layouts.int8_layout import BlockWiseINT8Layout
    _HAS_INT8_LAYOUT = True
except ImportError:
    _HAS_INT8_LAYOUT = False
    logging.debug("INT8 layout not available")


class UnifiedHybridOps(manual_cast):
    """
    Unified operations class that handles all quantization formats per-layer.
    
    Each layer is dispatched based on weight dtype + .comfy_quant metadata:
    - uint8 + nvfp4 -> TensorCoreNVFP4Layout
    - float8_e4m3fn + mxfp8 -> TensorCoreMXFP8Layout
    - float8_e4m3fn + blockwise/rowwise -> BlockWiseFP8Layout/RowWiseFP8Layout
    - float8_e4m3fn (tensor-scaled) -> TensorCoreFP8Layout
    - int8 -> BlockWiseINT8Layout
    - bf16/fp16/fp32 -> Passthrough (standard linear)
    """
    
    # Class-level storage for global quantization metadata from safetensors header
    _global_quant_metadata = None
    
    @classmethod
    def set_global_metadata(cls, metadata):
        """Set global quantization metadata from safetensors header."""
        cls._global_quant_metadata = metadata
    
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
            Custom state dict loading with per-layer format dispatch.
            
            Reads format from:
            1. Per-layer .comfy_quant tensor
            2. Global _quantization_metadata header (fallback)
            """
            weight_key = prefix + "weight"

            # Get weight scales
            scale = state_dict.pop(prefix + "weight_scale", None)
            scale_2 = state_dict.pop(prefix + "weight_scale_2", None)
            
            # Remove input_scale if present
            state_dict.pop(prefix + "input_scale", None)

            # Parse per-layer comfy_quant metadata
            comfy_quant_tensor = state_dict.pop(prefix + "comfy_quant", None)
            layer_conf = None

            if comfy_quant_tensor is not None:
                try:
                    layer_conf = json.loads(comfy_quant_tensor.numpy().tobytes())
                    self.quant_format = layer_conf.get("format", None)
                    self.block_size = layer_conf.get("group_size", None)
                    logging.debug(
                        f"UnifiedHybridOps: Parsed comfy_quant for {prefix}: "
                        f"format={self.quant_format}, group_size={self.block_size}"
                    )
                except Exception as e:
                    logging.debug(f"UnifiedHybridOps: Failed to parse comfy_quant: {e}")
            
            # Fallback to global metadata if no per-layer config
            if layer_conf is None and UnifiedHybridOps._global_quant_metadata is not None:
                try:
                    # Strip prefix to get layer name for lookup
                    layer_name = prefix.rstrip(".")
                    global_conf = UnifiedHybridOps._global_quant_metadata
                    if isinstance(global_conf, dict):
                        # Check for layer-specific config in global metadata
                        if layer_name in global_conf:
                            layer_conf = global_conf[layer_name]
                        elif "default" in global_conf:
                            layer_conf = global_conf["default"]
                        
                        if layer_conf:
                            self.quant_format = layer_conf.get("format", None)
                            self.block_size = layer_conf.get("group_size", None)
                            logging.debug(
                                f"UnifiedHybridOps: Using global metadata for {prefix}: "
                                f"format={self.quant_format}"
                            )
                except Exception as e:
                    logging.debug(f"UnifiedHybridOps: Failed to parse global metadata: {e}")

            # Load weight tensor
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                # === NVFP4 (uint8 packed) ===
                if self.quant_format == "nvfp4" or (
                    weight_tensor.dtype == torch.uint8 and scale_2 is not None
                ):
                    self._load_nvfp4(weight_tensor, scale, scale_2, layer_conf, prefix)
                
                # === INT8 ===
                elif weight_tensor.dtype == torch.int8:
                    self._load_int8(weight_tensor, scale, layer_conf, prefix)
                
                # === FP8 variants ===
                elif weight_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    self._load_fp8(weight_tensor, scale, layer_conf, prefix)
                
                # === High-precision passthrough ===
                else:
                    self.is_quantized = False
                    self.scale_weight = None
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    logging.debug(
                        f"UnifiedHybridOps: High-precision layer {prefix}, dtype={weight_tensor.dtype}"
                    )
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + "bias"
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def _load_nvfp4(self, weight_tensor, scale, scale_2, layer_conf, prefix):
            """Load NVFP4 quantized layer."""
            self.is_quantized = True
            self.layout_type = "TensorCoreNVFP4Layout"
            self.block_size = 16  # NVFP4 uses 16x16 blocks
            
            from comfy.quant_ops import TensorCoreNVFP4Layout
            
            # Get orig_dtype from metadata
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
                orig_shape = (weight_tensor.shape[0], weight_tensor.shape[1] * 2)
            
            layout_params = TensorCoreNVFP4Layout.Params(
                scale=scale_2.to(torch.float32) if scale_2 is not None else torch.tensor(1.0),
                orig_dtype=orig_dtype,
                orig_shape=orig_shape,
                block_scale=scale,
            )
            
            self.weight = torch.nn.Parameter(
                QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                requires_grad=False,
            )
            logging.debug(
                f"UnifiedHybridOps: Loaded NVFP4 layer {prefix}, "
                f"block_scale shape={scale.shape if scale is not None else None}"
            )

        def _load_int8(self, weight_tensor, scale, layer_conf, prefix):
            """Load INT8 quantized layer."""
            self.is_quantized = True
            self.scale_weight = scale
            self.layout_type = "BlockWiseINT8Layout"
            self.block_size = layer_conf.get("group_size", 128) if layer_conf else 128
            
            if scale is not None and _HAS_INT8_LAYOUT:
                layout_params = BlockWiseINT8Layout.Params(
                    scale=scale.to(torch.float32),
                    orig_dtype=torch.bfloat16,
                    orig_shape=tuple(weight_tensor.shape),
                    block_size=self.block_size,
                    is_weight=True,
                )
                self.weight = torch.nn.Parameter(
                    QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                    requires_grad=False,
                )
                logging.debug(
                    f"UnifiedHybridOps: Loaded INT8 layer {prefix} with BlockWiseINT8Layout, "
                    f"block_size={self.block_size}"
                )
            else:
                # Fallback: store raw INT8 tensor for manual dequant
                self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                logging.debug(
                    f"UnifiedHybridOps: Loaded INT8 layer {prefix} without layout (fallback)"
                )

        def _load_fp8(self, weight_tensor, scale, layer_conf, prefix):
            """Load FP8 quantized layer (MXFP8, blockwise, rowwise, tensor-scaled)."""
            self.is_quantized = True
            self.scale_weight = scale

            # Determine layout type from format
            if self.quant_format == "mxfp8":
                self.layout_type = "TensorCoreMXFP8Layout"
            elif self.quant_format is not None:
                qconfig = QUANT_ALGOS.get(self.quant_format, {})
                self.layout_type = qconfig.get("comfy_tensor_layout", "TensorCoreFP8Layout")
                if self.block_size is None:
                    self.block_size = qconfig.get("group_size", None)
            else:
                # Infer from scale shape
                if scale is not None:
                    if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                        self.layout_type = "TensorCoreFP8Layout"
                    elif scale.ndim == 1 and scale.numel() == weight_tensor.shape[0]:
                        self.layout_type = "RowWiseFP8Layout"
                    elif scale.ndim == 2:
                        self.layout_type = "BlockWiseFP8Layout"
                        if self.block_size is None:
                            M, N = weight_tensor.shape
                            scale_M, scale_N = scale.shape
                            if M % scale_M == 0 and N % scale_N == 0:
                                self.block_size = M // scale_M
                    else:
                        self.layout_type = "TensorCoreFP8Layout"
                else:
                    self.layout_type = "TensorCoreFP8Layout"

            # Check if layout is registered
            try:
                get_layout_class(self.layout_type)
            except KeyError:
                logging.warning(
                    f"UnifiedHybridOps: Layout '{self.layout_type}' not registered, "
                    f"using TensorCoreFP8Layout"
                )
                self.layout_type = "TensorCoreFP8Layout"

            # Create layout params
            layout_params = self._create_fp8_params(weight_tensor, scale, layer_conf)
            
            self.weight = torch.nn.Parameter(
                QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                requires_grad=False,
            )
            logging.debug(
                f"UnifiedHybridOps: Loaded FP8 layer {prefix} with {self.layout_type}, "
                f"block_size={self.block_size}"
            )

        def _create_fp8_params(self, weight_tensor, scale, layer_conf):
            """Create layout params based on FP8 layout type."""
            if self.layout_type == "TensorCoreMXFP8Layout":
                from comfy_kitchen.tensor import TensorCoreMXFP8Layout
                
                orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                DTYPE_MAP = {
                    "torch.bfloat16": torch.bfloat16,
                    "torch.float16": torch.float16,
                    "torch.float32": torch.float32,
                }
                orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.bfloat16)
                orig_shape = tuple(layer_conf.get("orig_shape", list(weight_tensor.shape))) if layer_conf else tuple(weight_tensor.shape)
                
                # Convert E8M0 scales from uint8
                if scale is not None and scale.dtype == torch.uint8:
                    scale = scale.view(torch.float8_e8m0fnu)
                
                return TensorCoreMXFP8Layout.Params(
                    scale=scale,
                    orig_dtype=orig_dtype,
                    orig_shape=orig_shape,
                )
            
            elif self.layout_type == "BlockWiseFP8Layout":
                from .quant_layouts.fp8_variants import BlockWiseFP8Layout
                block_size = self.block_size if self.block_size is not None else 64
                return BlockWiseFP8Layout.Params(
                    scale=scale.to(torch.float32) if scale is not None else None,
                    orig_dtype=torch.bfloat16,
                    orig_shape=tuple(weight_tensor.shape),
                    block_size=block_size,
                )
            
            elif self.layout_type == "RowWiseFP8Layout":
                from .quant_layouts.fp8_variants import RowWiseFP8Layout
                return RowWiseFP8Layout.Params(
                    scale=scale.to(torch.float32) if scale is not None else None,
                    orig_dtype=torch.bfloat16,
                    orig_shape=tuple(weight_tensor.shape),
                )
            
            else:
                from comfy.quant_ops import TensorCoreFP8Layout
                return TensorCoreFP8Layout.Params(
                    scale=scale.to(torch.float32) if scale is not None else None,
                    orig_dtype=torch.bfloat16,
                    orig_shape=tuple(weight_tensor.shape),
                )

        def forward_comfy_cast_weights(self, input):
            """Forward pass with unified format dispatch."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data

            input_dtype = input.dtype

            # Handle QuantizedTensor (triggers dispatch to layout handlers)
            if isinstance(weight, QuantizedTensor):
                if weight.device != input.device:
                    weight = weight.to(device=input.device)

                if hasattr(weight, "_params"):
                    object.__setattr__(weight._params, "orig_dtype", input_dtype)

                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)

                # MXFP8: quantize input for scaled_mm
                if self.layout_type == "TensorCoreMXFP8Layout":
                    return self._forward_mxfp8(input, weight, bias)

                # NVFP4: quantize input for scaled_mm
                if self.layout_type == "TensorCoreNVFP4Layout":
                    return self._forward_nvfp4(input, weight, bias)

                # Let QuantizedTensor dispatch handle other layouts
                return torch.nn.functional.linear(input, weight, bias)

            # Fallback: dequantize FP8/INT8 weight manually
            if self.is_quantized:
                if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    return self._forward_fp8_fallback(input, weight)
                elif weight.dtype == torch.int8:
                    return self._forward_int8_fallback(input, weight)

            # Standard manual_cast path for non-quantized weights
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def _forward_mxfp8(self, input, weight, bias):
            """MXFP8 forward with input quantization."""
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
                return torch.nn.functional.linear(
                    input.reshape(input_shape), weight.dequantize(), bias
                )

        def _forward_nvfp4(self, input, weight, bias):
            """NVFP4 forward with input quantization."""
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
                return torch.nn.functional.linear(
                    input.reshape(input_shape), weight.dequantize(), bias
                )

        def _forward_fp8_fallback(self, input, weight):
            """FP8 fallback with manual dequantization."""
            input_dtype = input.dtype
            weight = weight.to(device=input.device)
            
            if self.scale_weight is not None:
                scale = self.scale_weight.to(device=input.device)
                weight_dequant = self._dequantize_fp8(weight, scale, input_dtype)
            else:
                weight_dequant = weight.to(input_dtype)
            
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=input.device, dtype=input_dtype)
            return torch.nn.functional.linear(input, weight_dequant, bias)

        def _forward_int8_fallback(self, input, weight):
            """INT8 fallback with manual dequantization."""
            input_dtype = input.dtype
            weight = weight.to(device=input.device)
            
            if self.scale_weight is not None:
                scale = self.scale_weight.to(device=input.device)
                weight_dequant = self._dequantize_int8(weight, scale, input_dtype)
            else:
                weight_dequant = weight.to(input_dtype)
            
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=input.device, dtype=input_dtype)
            return torch.nn.functional.linear(input, weight_dequant, bias)

        def _dequantize_fp8(self, weight, scale, input_dtype):
            """Dequantize FP8 weight."""
            M, N = weight.shape

            if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                return weight.to(input_dtype) * scale.item()

            if scale.ndim == 1 and scale.shape[0] == M:
                scale_broadcast = scale.unsqueeze(1).to(device=weight.device, dtype=input_dtype)
                return weight.to(input_dtype) * scale_broadcast

            if scale.ndim == 2 and self.block_size is not None:
                block_size = self.block_size
                if M % block_size == 0 and N % block_size == 0:
                    qdata_blocked = weight.reshape(
                        M // block_size, block_size, N // block_size, block_size
                    )
                    qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
                    scale_broadcast = (
                        scale.unsqueeze(-1).unsqueeze(-1)
                        .to(device=weight.device, dtype=input_dtype)
                    )
                    dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                    return dequant.permute(0, 2, 1, 3).reshape(M, N)

            logging.warning(f"FP8 scale shape {scale.shape} for weight {weight.shape}, using broadcast")
            return weight.to(input_dtype) * scale.to(device=weight.device, dtype=input_dtype)

        def _dequantize_int8(self, weight, scale, input_dtype):
            """Dequantize INT8 weight."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            N, K = weight.shape
            block_size = self.block_size if self.block_size else 128
            k_blocks = K // block_size if K % block_size == 0 else (K + block_size - 1) // block_size
            
            # Lodewise: scale shape (N, K//block_size)
            if scale.ndim == 2 and scale.shape[0] == N:
                if K % block_size == 0 and scale.shape == (N, k_blocks):
                    qdata_blocked = weight.reshape(N, k_blocks, block_size)
                    scale_broadcast = scale.unsqueeze(-1).to(device=weight.device, dtype=input_dtype)
                    dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                    return dequant.reshape(N, K)
            
            # Blockwise: scale shape (N//block_size, K//block_size)
            if scale.ndim == 2 and N % block_size == 0 and K % block_size == 0:
                expected_shape = (N // block_size, K // block_size)
                if scale.shape == expected_shape:
                    qdata_blocked = weight.reshape(
                        N // block_size, block_size, K // block_size, block_size
                    )
                    qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
                    scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1).to(
                        device=weight.device, dtype=input_dtype
                    )
                    dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                    return dequant.permute(0, 2, 1, 3).reshape(N, K)
            
            # 1D scale fallback
            if scale.ndim == 1:
                if scale.shape[0] == N:
                    scale_broadcast = scale.unsqueeze(-1).to(device=weight.device, dtype=input_dtype)
                    return weight.to(input_dtype) * scale_broadcast
                elif scale.shape[0] == 1:
                    return weight.to(input_dtype) * scale.item()
            
            logging.warning(f"INT8 scale shape {scale.shape} for weight {weight.shape}, using broadcast")
            return weight.to(input_dtype) * scale.to(device=weight.device, dtype=input_dtype)

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
                if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.int8]:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                if isinstance(weight, QuantizedTensor):
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()

            if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and self.scale_weight is not None:
                return self._dequantize_fp8(weight, self.scale_weight, torch.float32)
            
            if weight.dtype == torch.int8 and self.scale_weight is not None:
                return self._dequantize_int8(weight, self.scale_weight, torch.float32)

            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching - requantize if layout available."""
            if getattr(self, 'layout_type', None) is not None:
                weight = QuantizedTensor.from_float(
                    weight,
                    self.layout_type,
                    scale="recalculate",
                    stochastic_rounding=seed if seed else 0,
                    inplace_ops=True
                )
                if hasattr(self.weight, 'dtype'):
                    weight = weight.to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)

            if return_weight:
                return weight

            assert inplace_update is False
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
