"""
Hybrid SDNQ Operations.

Handles loading and inference for SDNQ (Stochastic Differentiable Neural Quantization)
quantized models.
"""

import torch
import torch.nn.functional as F
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor
from comfy.model_patcher import LowVramPatch


class SDNQLayerMixin:
    """Mixin to add SDNQ loading and forward logic to layers."""
    
    def _load_sdnq_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Custom state dict loading for SDNQ format."""
        weight_key = prefix + "weight"

        # 1. Check for comfy_quant metadata
        comfy_quant_tensor = state_dict.pop(prefix + "comfy_quant", None)
        if comfy_quant_tensor is None:
            # Fallback to standard loading for non-quantized layers
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
            return

        # 2. Parse metadata
        try:
            from .comfy_quant_helpers import tensor_to_dict
            meta = tensor_to_dict(comfy_quant_tensor)
        except Exception as e:
            logging.error(f"HybridSDNQOps: Failed to parse comfy_quant metadata for {prefix}: {e}")
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
            return

        # 3. Verify format
        if meta.get("format") != "sdnq":
            # Put it back so other ops can try
            state_dict[prefix + "comfy_quant"] = comfy_quant_tensor
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
            return

        # 4. Extract SDNQ-specific tensors
        qdata = state_dict.pop(weight_key, None)
        scale = state_dict.pop(prefix + "weight_scale", None)
        zero_point = state_dict.pop(prefix + "weight_zp", None)
        svd_up = state_dict.pop(prefix + "svd_up", None)
        svd_down = state_dict.pop(prefix + "svd_down", None)

        # Ignore input_scale if present
        state_dict.pop(prefix + "input_scale", None)

        if qdata is not None and scale is not None:
            self.is_quantized = True
            
            from .quant_layouts.sdnq_layout import SDNQLayout
            
            # Determine original dtype
            orig_dtype_str = meta.get("orig_dtype", "float16")
            DTYPE_MAP = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            orig_dtype = DTYPE_MAP.get(orig_dtype_str, torch.float16)
            
            # Reconstruct Params
            unpack_shape = tuple(meta.get("original_shape"))
            expected_shape = self.weight.shape
            orig_shape = unpack_shape
            transposed = False
            
            if unpack_shape != expected_shape:
                if len(unpack_shape) == 2 and len(expected_shape) == 2:
                    if unpack_shape[::-1] == expected_shape:
                        logging.debug(f"HybridSDNQOps: Transpose detected for {weight_key}: metadata={unpack_shape}, module={expected_shape}")
                        transposed = True
                        orig_shape = expected_shape
                    else:
                        logging.warning(f"HybridSDNQOps: Shape mismatch for {weight_key}: metadata={unpack_shape}, module={expected_shape}")
                else:
                    logging.warning(f"HybridSDNQOps: Shape mismatch for {weight_key}: metadata={unpack_shape}, module={expected_shape}")

            layout_params = SDNQLayout.Params(
                scale=scale.to(torch.float32),
                orig_dtype=orig_dtype,
                orig_shape=orig_shape,
                weights_dtype=meta.get("weights_dtype", "int8"),
                group_size=meta.get("group_size", -1),
                zero_point=zero_point,
                svd_up=svd_up,
                svd_down=svd_down,
                transposed=transposed,
                unpack_shape=unpack_shape,
            )
            
            # Wrap in QuantizedTensor
            # Note: comfy_kitchen QuantizedTensor requires layout name as string
            self.weight = torch.nn.Parameter(
                QuantizedTensor(qdata, "SDNQLayout", layout_params),
                requires_grad=False
            )
            logging.debug(f"HybridSDNQOps: Loaded SDNQ layer {weight_key} (dtype={layout_params.weights_dtype}, svd={svd_up is not None})")
        else:
            if qdata is None: missing_keys.append(weight_key)
            if scale is None: missing_keys.append(prefix + "weight_scale")

        # Handle bias
        bias_key = prefix + "bias"
        bias_tensor = state_dict.pop(bias_key, None)
        if bias_tensor is not None:
            self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
        else:
            self.bias = None

    def _sdnq_forward(self, input, *args, **kwargs):
        """Generic SDNQ forward that uses dispatcher."""
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        
        if isinstance(weight, QuantizedTensor):
            # Ensure device match
            if weight.device != input.device:
                weight = weight.to(device=input.device)
            
            # Update orig_dtype for output consistency
            if hasattr(weight, "_params"):
                object.__setattr__(weight._params, "orig_dtype", input.dtype)
            
            # Support passing explicit bias or using self.bias
            bias = kwargs.get("bias", self.bias)
            if bias is not None:
                bias = bias.to(device=input.device, dtype=input.dtype)
            
            # Linear case
            if hasattr(self, 'in_features'):
                return torch.nn.functional.linear(input, weight, bias)
            
            # Convolution case
            if hasattr(self, 'kernel_size'):
                return torch.nn.functional.convolution(
                    input, weight, bias, self.stride, self.padding, self.dilation, False, self.padding, self.groups
                )
        
        # Fallback to standard path
        return None

class HybridSDNQOps(manual_cast):
    """
    Hybrid SDNQ operations class.
    
    Handles:
    - Loading from SDNQ state dicts (with optional SVD and custom bit-widths)
    - Lazy dequantization during forward pass (via QuantizedTensor dispatch)
    """

    class Linear(SDNQLayerMixin, manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_quantized = False

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args, **kwargs):
            return self._load_sdnq_from_state_dict(*args, **kwargs)

        def forward_comfy_cast_weights(self, input):
            out = self._sdnq_forward(input)
            if out is not None:
                return out

            # Standard path for non-quantized weights
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def forward_fused_lora(self, input):
            """
            Memory-efficient LoRA forward pass for SDNQ.
            
            Computes: output = base_sdnq_matmul(x, W) + lora_contribution(x)
            
            This avoids dequantizing the full SDNQ weight (which involves SVD reconstruction
            and full tensor materialization) when LoRAs are present.
            """
            input_dtype = input.dtype
            
            # Log usage of fused LoRA path
            if not hasattr(self.__class__, "_fused_lora_log_count"):
                self.__class__._fused_lora_log_count = 0
            if self.__class__._fused_lora_log_count < 1:
                logging.info(f"SDNQ: Using memory-efficient fused LoRA path for {input.shape}")
                self.__class__._fused_lora_log_count += 1

            # 1. Base SDNQ output (no LoRA applied, no bias applied yet)
            # We explicitly pass bias=None to get just the matmul result
            base_out = self._sdnq_forward(input, bias=None)
            
            if base_out is None:
                # Fallback if not quantized
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
                base_out = torch.nn.functional.linear(input, weight, None)
                uncast_bias_weight(self, weight, bias, offload_stream)

            # 2. Compute LoRA contributions separately
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
                            temp = F.linear(input, mat2)  # [B, seq, rank]
                            lora_contrib = F.linear(temp, mat1) * scale  # [B, seq, out_dim]
                            
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                        else:
                            # Fallback for non-LoRA adapters: apply the patch function to dequantized weight
                            # This is memory-heavy but ensures correctness
                            logging.warning(f"SDNQ Fused LoRA: Falling back to dequant for non-LoRA adapter")
                            weight_fp = self.convert_weight(self.weight.data).to(input.device, dtype=input_dtype)
                            patched_weight = patch_fn(weight_fp)
                            # Compute the delta contribution
                            lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                else:
                    # Non-LowVramPatch function - fall back to calling it
                    logging.warning(f"SDNQ Fused LoRA: Unknown patch function type, falling back")
                    weight_fp = self.convert_weight(self.weight.data).to(input.device, dtype=input_dtype)
                    patched_weight = patch_fn(weight_fp)
                    lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                    if lora_out is None:
                        lora_out = lora_contrib
                    else:
                        lora_out = lora_out + lora_contrib
            
            # 3. Combine base + LoRA + bias
            out = base_out
            if lora_out is not None:
                out = out + lora_out
            
            # Add bias
            if self.bias is not None:
                bias = self.bias.to(device=input.device, dtype=input_dtype)
                out = out + bias
            
            return out

        def forward(self, *args, **kwargs):
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            # Check if we have LoRA patches AND quantized weight
            has_lora = len(self.weight_function) > 0
            # Also check for raw uint8 (packed SDNQ storage)
            is_quant = isinstance(weight, QuantizedTensor) or (getattr(self, "is_quantized", False) and weight.dtype == torch.uint8)
            
            if has_lora and is_quant:
                # Use fused LoRA path to avoid full weight dequantization (SVD reconstruction)
                return self.forward_fused_lora(*args, **kwargs)
            
            if self.is_quantized or self.comfy_cast_weights or has_lora or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            if return_weight:
                return weight
            if inplace_update and not isinstance(self.weight.data, QuantizedTensor):
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.is_quantized = False

    class Conv1d(SDNQLayerMixin, manual_cast.Conv1d):
        def reset_parameters(self): return None
        def _load_from_state_dict(self, *args, **kwargs):
            return self._load_sdnq_from_state_dict(*args, **kwargs)
        def forward_comfy_cast_weights(self, input):
            out = self._sdnq_forward(input)
            if out is not None: return out
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out
        def forward(self, *args, **kwargs):
            if getattr(self, "is_quantized", False) or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv2d(SDNQLayerMixin, manual_cast.Conv2d):
        def reset_parameters(self): return None
        def _load_from_state_dict(self, *args, **kwargs):
            return self._load_sdnq_from_state_dict(*args, **kwargs)
        def forward_comfy_cast_weights(self, input):
            out = self._sdnq_forward(input)
            if out is not None: return out
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out
        def forward(self, *args, **kwargs):
            if getattr(self, "is_quantized", False) or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv3d(SDNQLayerMixin, manual_cast.Conv3d):
        def reset_parameters(self): return None
        def _load_from_state_dict(self, *args, **kwargs):
            return self._load_sdnq_from_state_dict(*args, **kwargs)
        def forward_comfy_cast_weights(self, input, autopad=None):
            out = self._sdnq_forward(input)
            if out is not None: return out
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = self._conv_forward(input, weight, bias, autopad=autopad)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out
        def forward(self, *args, **kwargs):
            if getattr(self, "is_quantized", False) or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0 or "autopad" in kwargs:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    # Use standard manual_cast versions for other layers
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

    class ConvTranspose1d(manual_cast.ConvTranspose1d):
        pass

    class ConvTranspose2d(manual_cast.ConvTranspose2d):
        pass

    class Embedding(manual_cast.Embedding):
        pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 1:
            return cls.Conv1d(*args, **kwargs)
        elif dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
