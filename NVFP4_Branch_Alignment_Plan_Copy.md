# NVFP4 Branch Alignment - Complete System Integration

## Architecture Overview

ComfyUI has **two parallel quantization paths**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Path 1: ComfyUI Core (mixed_precision_ops)                              │
│                                                                          │
│ model_options={}  →  pick_operations()  →  mixed_precision_ops          │
│                            │                       │                     │
│                            ▼                       ▼                     │
│                    Linear._load_from_state_dict reads comfy_quant       │
│                            │                                             │
│                            ▼                                             │
│                    get_layout_class() → TensorCoreNVFP4Layout           │
│                            │                                             │
│                            ▼                                             │
│                    QuantizedTensor wraps weight                          │
│                            │                                             │
│                            ▼                                             │
│                    torch dispatch → _LAYOUT_DISPATCH_TABLE → ops        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Path 2: QuantOps Custom (HybridXxxOps via custom_operations)            │
│                                                                          │
│ model_options={"custom_operations": HybridINT8Ops}                       │
│                            │                                             │
│                            ▼                                             │
│    HybridINT8Ops.Linear replaces torch.nn.Linear                        │
│                            │                                             │
│                            ▼                                             │
│    Linear._load_from_state_dict handles state (legacy key compat)       │
│                            │                                             │
│                            ▼                                             │
│    Linear.forward_comfy_cast_weights → dequantize or native matmul      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Both paths coexist**. QuantOps adds value by:
1. Legacy key format support (`scale_weight` → `weight_scale`)
2. Backend selection (pytorch/triton)
3. Custom kernel registration with comfy-kitchen
4. PyTorch fallback for non-Blackwell

---

## What Needs to Happen

### 1. Register Layouts with comfy-kitchen

File: `__init__.py`

comfy-kitchen uses `register_layout_class()` to populate `LAYOUTS` dict. Layouts must be registered **and** have operation handlers registered via `register_layout_op()`.

```python
from comfy.quant_ops import register_layout_class, QUANT_ALGOS
from .quant_layouts.nvfp4_layout import QuantOpsNVFP4Layout
from .quant_layouts.mxfp8_layout import QuantOpsMXFP8Layout

register_layout_class("QuantOpsNVFP4Layout", QuantOpsNVFP4Layout)
register_layout_class("QuantOpsMXFP8Layout", QuantOpsMXFP8Layout)
```

### 2. Create Layout Classes with Operation Handlers

File: `quant_layouts/nvfp4_layout.py`

Layout must:
- Inherit from `QuantizedLayout`
- Define `Params` dataclass
- Implement `quantize()`, `dequantize()`, `get_plain_tensors()`, `state_dict_tensors()`
- Register ops via `@register_layout_op(torch.ops.aten.linear.default, LayoutClass)`

```python
from comfy_kitchen.tensor.base import (
    QuantizedLayout, BaseLayoutParams, register_layout_op, dequantize_args
)

class QuantOpsNVFP4Layout(QuantizedLayout):
    MIN_SM_VERSION = (10, 0)  # Blackwell for fast matmul
    
    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        block_scale: torch.Tensor
        transposed: bool = False
        
        def _tensor_fields(self):
            return ["scale", "block_scale"]
    
    @classmethod
    def dequantize(cls, qdata, params):
        # Tiered: ck CUDA → ck eager → PyTorch
        ...

@register_layout_op(torch.ops.aten.linear.default, QuantOpsNVFP4Layout)
def _handle_nvfp4_linear(qt, args, kwargs):
    # Check hardware capability
    if QuantOpsNVFP4Layout.supports_fast_matmul():
        # Use ck.scaled_mm_nvfp4
        ...
    else:
        # Dequant fallback
        return torch.nn.functional.linear(*dequantize_args(args))
```

### 3. Create Ops Class for Custom Node Path

File: `nvfp4_ops.py` (feature branch has this, needs update)

For the custom_operations path (used by QuantizedModelLoader/QuantizedUNETLoader), we need a `HybridNVFP4Ops` class that:
- Handles legacy/new key formats
- Manages backend selection
- Works alongside or instead of the layout path

```python
class HybridNVFP4Ops(manual_cast):
    class Linear(manual_cast.Linear):
        def _load_from_state_dict(...):
            # Handle nvfp4 specific keys
            # Create QuantizedTensor with QuantOpsNVFP4Layout
            
        def forward_comfy_cast_weights(self, input):
            # If weight is QuantizedTensor: dispatch happens automatically
            # Otherwise: use manual dequant path
```

### 4. Update Nodes

File: `nodes/loader_nodes.py`

Add NVFP4/MXFP8 to format options and route to appropriate ops:

```python
"quant_format": ([
    "auto", "int8", 
    "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise",
    "nvfp4", "mxfp8"
],)
```

```python
elif quant_format == "nvfp4":
    try:
        from ..nvfp4_ops import HybridNVFP4Ops
        model_options = {"custom_operations": HybridNVFP4Ops}
    except ImportError:
        # Fall back to ComfyUI core path (no custom_operations)
        pass
```

### 5. PyTorch Fallback Utils

File: `utils/float_utils.py`

Keep and update utilities for non-Blackwell fallback:

```python
# FP4 E2M1 constants
F4_E2M1_EBITS = 2
F4_E2M1_MBITS = 1
NVFP4_BLOCK_SIZE = 16

# MXFP8 constants  
E8M0_BIAS = 127
MXFP8_BLOCK_SIZE = 32

def unpack_uint4(packed): ...
def _floatx_unpacked_to_f32(x, ebits, mbits): ...
def from_blocked(blocked_matrix, num_rows, num_cols): ...
def e8m0_to_f32(x): ...
```

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `__init__.py` | Modify | Add NVFP4/MXFP8 layout registration |
| `quant_layouts/nvfp4_layout.py` | Create | Layout + operation handlers |
| `quant_layouts/mxfp8_layout.py` | Create | Layout + operation handlers |
| `nvfp4_ops.py` | Update | Custom ops class with QuantizedTensor integration |
| `nodes/loader_nodes.py` | Modify | Add format options, ops routing |
| `utils/float_utils.py` | Update | Add `e8m0_to_f32()`, ensure constants |
| `utils/hardware_check.py` | Delete | Use `QuantizedLayout.supports_fast_matmul()` |

---

## Key Differences from Feature Branch

| Aspect | Feature Branch | Required |
|--------|---------------|----------|
| Layout location | Custom `NVFP4Layout` | Use comfy-kitchen patterns |
| Op registration | Manual dispatch | `@register_layout_op` decorator |
| Key names | `block_scale` | `weight_scale`/`weight_scale_2` match ComfyUI |
| MXFP8 | Missing | Add full support |
| Hardware check | Separate module | Use `MIN_SM_VERSION` + `supports_fast_matmul()` |

---

## Verification

```bash
# Syntax
python -m py_compile __init__.py
python -m py_compile quant_layouts/nvfp4_layout.py
python -m py_compile quant_layouts/mxfp8_layout.py
python -m py_compile nvfp4_ops.py
python -m py_compile nodes/loader_nodes.py

# Integration
# Start ComfyUI, check logs for:
# - "ComfyUI-QuantOps: Registered layouts: [... QuantOpsNVFP4Layout, QuantOpsMXFP8Layout]"
# - No import errors
```
