"""
Hybrid SDNQ Operations.

Handles loading and inference for SDNQ (Stochastic Differentiable Neural Quantization)
quantized models.
"""

import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor


class HybridSDNQOps(manual_cast):
    """
    Hybrid SDNQ operations class.
    
    Handles:
    - Loading from SDNQ state dicts (with optional SVD and custom bit-widths)
    - Lazy dequantization during forward pass (via QuantizedTensor dispatch)
    """

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_quantized = False

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
            Custom state dict loading for SDNQ format.
            """
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

        def forward_comfy_cast_weights(self, input):
            """Forward pass with SDNQ handling."""
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
                
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)
                
                # This triggers SDNQLayout.dequantize -> matmul
                return torch.nn.functional.linear(input, weight, bias)

            # Standard path for non-quantized weights
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def forward(self, *args, **kwargs):
            if self.is_quantized or self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize SDNQ."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            if return_weight:
                return weight

            # Since we don't have a robust on-the-fly SDNQ quantizer yet,
            # we keep the patched weight in full precision.
            if inplace_update and not isinstance(self.weight.data, QuantizedTensor):
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            self.is_quantized = False

    # Use standard manual_cast versions for other layers
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

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
        if dims == 1:
            return cls.Conv1d(*args, **kwargs)
        elif dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")
