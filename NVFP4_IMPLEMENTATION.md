# NVFP4 Inference Implementation Guide

This document provides comprehensive instructions for implementing NVFP4 (FP4 E2M1) inference support in ComfyUI-QuantOps.

## Overview

NVFP4 is NVIDIA's 4-bit floating point format (E2M1) with two-level scaling:
1. **Per-tensor scale** (float32 scalar)
2. **Per-block scale** (FP8 E4M3, 16-element blocks in cuBLAS tiled layout)

### Hardware Requirements

| GPU Type | Compute Capability | Status |
|----------|-------------------|--------|
| Blackwell Datacenter (B100, B200) | SM ≥ 10.0 | ✅ Supported |
| Blackwell Consumer (RTX 50xx) | SM ≥ 12.0 | ✅ Supported |
| Ada Lovelace (RTX 40xx) | SM 8.9 | ❌ Not supported |
| Ampere (RTX 30xx) | SM 8.0-8.6 | ❌ Not supported |

> [!IMPORTANT]
> On Windows, CUDA 13.0+ is required due to a kernel compilation bug in CUDA 12.9.

---

## Reference Implementations

### 1. comfy-kitchen (Comfy Org)

The primary reference for NVFP4 kernels.

**Repository**: https://github.com/Comfy-Org/comfy-kitchen

**Installation**:
```bash
pip install comfy-kitchen  # Pre-built wheel (requires CUDA 13.0+ runtime)
```

**Key Files**:
- [`comfy_kitchen/tensor/nvfp4.py`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/nvfp4.py) - TensorCoreNVFP4Layout class
- [`comfy_kitchen/backends/eager/quantization.py`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/eager/quantization.py) - Pure PyTorch fallback
- [`comfy_kitchen/float_utils.py`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/float_utils.py) - FP4 encode/decode, cuBLAS layout

**API Usage**:
```python
import comfy_kitchen as ck

# Quantize
qdata, block_scales = ck.quantize_nvfp4(tensor, per_tensor_scale, pad_16x=True)

# Dequantize
dequant = ck.dequantize_nvfp4(qdata, per_tensor_scale, block_scales, output_dtype)

# Hardware matmul (Blackwell only)
result = ck.scaled_mm_nvfp4(a, b, scale_a, scale_b, block_scale_a, block_scale_b)
```

### 2. convert_to_quant

Quantization tool that converts models to NVFP4 format.

**Repository**: https://github.com/silveroxides/convert_to_quant (feature/nvfp4-support branch)

**Key Files**:
- [`convert_to_quant/utils/float_utils.py`](https://github.com/silveroxides/convert_to_quant/blob/feature/nvfp4-support/convert_to_quant/utils/float_utils.py) - Standalone FP4 utilities
- [`convert_to_quant/converters/nvfp4_converter.py`](https://github.com/silveroxides/convert_to_quant/blob/feature/nvfp4-support/convert_to_quant/converters/nvfp4_converter.py) - NVFP4Converter class

---

## Tensor Layout Specification

### Quantized Data Layout
```
Shape: (M, K // 2)
Dtype: uint8 (2 FP4 values packed per byte)

Packing: pack_uint4(fp4_values)
  - fp4_values[::2] → upper nibble
  - fp4_values[1::2] → lower nibble
```

### Block Scales Layout (cuBLAS Tiled)

Block scales use NVIDIA's cuBLAS 2D block layout for efficient GPU access.

```
Block size: 16 elements
Scale dtype: float8_e4m3fn
Tile pattern: 128×4 blocks swizzled

Conversion functions:
- to_blocked(scales, flatten=False) → cuBLAS layout
- from_blocked(blocked, num_rows, num_cols) → standard layout
```

Reference: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

### State Dict Keys

For a quantized layer `model.layer.weight`:
```python
{
    "model.layer.weight": tensor(uint8, shape=(M, K//2)),        # Packed FP4 data
    "model.layer.weight_scale": tensor(float32, shape=()),       # Per-tensor scale
    "model.layer.block_scale": tensor(float8_e4m3fn, swizzled),  # Block scales
    "model.layer.comfy_quant": tensor(uint8, json_metadata),     # Metadata
}
```

### Metadata Format (.comfy_quant)

```json
{
    "format": "nvfp4",
    "group_size": 16,
    "orig_dtype": "torch.bfloat16",
    "orig_shape": [4096, 4096]
}
```

---

## Implementation Steps

### Step 1: Create NVFP4 Layout Class

Create `quant_layouts/nvfp4_layout.py`:

```python
import torch
from comfy.quant_ops import QuantizedLayout

class NVFP4Layout(QuantizedLayout):
    """NVFP4 E2M1 block quantization layout."""
    
    BLOCK_SIZE = 16
    
    @staticmethod
    def from_tensors(qdata, scale_weight, block_scale, orig_dtype, orig_shape):
        """Reconstruct QuantizedTensor from state dict components."""
        # TODO: Implement based on TensorCoreNVFP4Layout pattern
        pass
    
    @staticmethod
    def dequantize(qdata, scale_weight, block_scale, output_dtype):
        """Dequantize NVFP4 to float."""
        try:
            import comfy_kitchen as ck
            return ck.dequantize_nvfp4(qdata, scale_weight, block_scale, output_dtype)
        except ImportError:
            # Pure PyTorch fallback
            from .nvfp4_fallback import dequantize_nvfp4_eager
            return dequantize_nvfp4_eager(qdata, scale_weight, block_scale, output_dtype)
```

### Step 2: Create Operations Class

Create `nvfp4_ops.py` following the pattern of `fp8_ops.py`:

```python
import torch
from comfy.ops import manual_cast

class HybridNVFP4Ops:
    """NVFP4 operations class for ComfyUI inference."""
    
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = None
            self.block_scale = None
            self.is_quantized = False
        
        def _load_from_state_dict(self, state_dict, prefix, ...):
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            block_scale_key = f"{prefix}block_scale"
            metadata_key = f"{prefix}comfy_quant"
            
            if weight_key in state_dict:
                weight = state_dict[weight_key]
                if weight.dtype == torch.uint8:
                    # NVFP4 packed format detected
                    self.is_quantized = True
                    self.scale_weight = state_dict.get(scale_key)
                    self.block_scale = state_dict.get(block_scale_key)
            # ...
        
        def _dequantize_weight(self, weight, scale, block_scale, input_dtype):
            """Dequantize NVFP4 weight to float."""
            try:
                import comfy_kitchen as ck
                return ck.dequantize_nvfp4(weight, scale, block_scale, input_dtype)
            except ImportError:
                # Fallback to pure PyTorch
                pass
        
        def forward_comfy_cast_weights(self, input):
            if self.is_quantized:
                weight = self._dequantize_weight(
                    self.weight, self.scale_weight, self.block_scale, input.dtype
                )
            else:
                weight = self.weight
            return torch.nn.functional.linear(input, weight, self.bias)
```

### Step 3: Register Layout in ComfyUI

Add to `__init__.py`:

```python
from comfy.quant_ops import LAYOUTS, QUANT_ALGOS

# Register NVFP4 layout
LAYOUTS["NVFP4Layout"] = NVFP4Layout
QUANT_ALGOS["nvfp4"] = {
    "layout_class": NVFP4Layout,
    "group_size": 16,
}
```

### Step 4: Hardware Matmul (Optional)

For Blackwell GPUs, bypass dequantization with hardware FP4 matmul:

```python
def forward_nvfp4_matmul(self, input):
    """Hardware-accelerated NVFP4 matmul (Blackwell only)."""
    if not self._check_blackwell_capability():
        return self._dequant_fallback(input)
    
    # Quantize input to NVFP4
    input_q, input_scale, input_block_scale = quantize_nvfp4(input)
    
    # Hardware matmul
    result = torch._scaled_mm(
        input_q.view(torch.float4_e2m1fn_x2),
        self.weight.view(torch.float4_e2m1fn_x2),
        input_block_scale.view(-1),
        self.block_scale.view(-1),
        out_dtype=input.dtype,
    )
    
    return result * (input_scale * self.scale_weight)
```

---

## Testing Checklist

- [ ] Model loads without errors
- [ ] Dequantization produces expected outputs
- [ ] Memory footprint matches expected (50% of FP8)
- [ ] Inference speed acceptable (with/without hardware matmul)
- [ ] LoRA patching works (dequantize → patch → requantize if needed)

---

## Resources

| Resource | URL |
|----------|-----|
| comfy-kitchen source | https://github.com/Comfy-Org/comfy-kitchen |
| convert_to_quant (NVFP4 branch) | https://github.com/silveroxides/convert_to_quant/tree/feature/nvfp4-support |
| cuBLAS block layout docs | https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout |
| PyTorch FP8 docs | https://pytorch.org/docs/stable/generated/torch._scaled_mm.html |
| FP4 E2M1 format | https://arxiv.org/abs/2310.08659 (NVIDIA FP4 paper) |
