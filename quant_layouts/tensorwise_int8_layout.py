import torch
import logging

try:
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from comfy_kitchen.tensor.int8 import tensorwise_int8_linear as orig_linear
    from comfy.quant_ops import register_layout_op, QuantizedTensor
    from ..kernels.int8_kernels import int8_linear_per_channel

    @register_layout_op(torch.ops.aten.linear.default, "TensorWiseINT8Layout")
    def tensorwise_int8_linear_patched(func, args, kwargs):
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None

        if isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
            plain_weight, scale_b, _, _ = TensorWiseINT8Layout.get_plain_tensors(weight)
            
            # Check if scale is per-row/per-channel (shape [N] or [N, 1])
            # If so, use our custom fallback instead of comfy_kitchen's which might not support it
            if scale_b.numel() > 1:
                try:
                    plain_input = input_tensor.dequantize()
                    return int8_linear_per_channel(plain_input, plain_weight, scale_b, bias)
                except Exception as e:
                    logging.warning(f"Row-wise INT8 linear failed, falling back: {e}")
                    return torch.nn.functional.linear(input_tensor.dequantize(), weight.dequantize(), bias)
            
        elif not isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor):
            plain_weight, scale_b, _, _ = TensorWiseINT8Layout.get_plain_tensors(weight)
            
            if scale_b.numel() > 1:
                try:
                    return int8_linear_per_channel(input_tensor, plain_weight, scale_b, bias)
                except Exception as e:
                    logging.warning(f"Row-wise INT8 linear dynamic act quant failed, falling back: {e}")
                    return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
                    
        # Fallback to original comfy_kitchen implementation
        return orig_linear(func, args, kwargs)

    logging.info("ComfyUI-QuantOps: Patched TensorWiseINT8Layout for per-channel scale support")

except ImportError:
    pass
