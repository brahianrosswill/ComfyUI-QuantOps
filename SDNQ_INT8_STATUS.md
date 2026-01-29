# SDNQ Int8 Matmul - Current State & Issues

## Status: INCOMPLETE - Scale Application Issue

### Recent Fixes (2026-01-29)

1. ✅ **Transpose Handling** (Lines 315-322)
   - Fixed: `params.transposed = True` means weight is already [K, N]
   - Now correctly reshapes based on transpose flag
   - No unnecessary transpose when already transposed

2. ✅ **SVD Order for Transposed Weights** (Lines 327-333)
   - Fixed: Was doing `x @ svd_up @ svd_down` (backwards)
   - Now: `x @ svd_down @ svd_up` for transposed case
   - Shapes: x[M,16384] @ svd_down[16384,32] @ svd_up[32,4096] = [M,4096]

3. ✅ **Scale View Fix** (Line 358)
   - Changed from `flatten()` to `reshape(-1)` to avoid copying params.scale
   - Prevents memory leak from modifying original param tensor

### Current Blocking Issue

**Still creating large intermediate tensors at lines 361-362**

```python
output_fp = output_int32.to(torch.float32) * input_scale  # Creates [M, N] tensor
output_fp = output_fp * weight_scale_1d.unsqueeze(0)      # Another [M, N] tensor
```

For M=8240, N=4096: Each operation creates **128MB** intermediate tensors.

### Correct Solution (From sdnext)

Use `torch.addcmul()` for fused multiply-add (line 60 of dequantizer.py):

```python
# sdnext does: result = torch.addcmul(bias, weight, scale)
# This is: result = bias + (weight * scale) in ONE operation

# For int8 output:
# Step 1: Scale by input_scale with bias fusion
if bias is not None:
    # output = bias + (int32_result * input_scale)
    output_fp = torch.addcmul(bias.unsqueeze(0), output_int32.to(dtype), input_scale)
else:
    output_fp = output_int32.to(dtype) * input_scale

# Step 2: Scale by weight_scale in-place
output_fp.mul_(weight_scale_1d.unsqueeze(0))
```

**OR** even better, merge into single op if bias can be pre-scaled.

### Files Modified
- `quant_layouts/sdnq_layout.py`: Lines 296-375 (int8 matmul), lines 386-394 (dispatch)

### Fixed Issues (Don't Regress)
1. ✅ SVD handling: Lines 325-340, correct order for transposed weights
2. ✅ Weight shape: Lines 315-322, handle transpose flag correctly
3. ✅ Scale view: Line 358 use `reshape(-1)` not `flatten()`
4. ✅ Transpose detection: Lines 359-362, skip .t() when already transposed

### Next Steps

Replace lines 361-362 with `torch.addcmul()` pattern from sdnext to:
1. Reduce intermediate tensor allocations
2. Fuse bias addition with first scale multiply
3. Apply second scale in-place

### Test Case
- Model: 8GB SDNQ int8 Flux
- Expected: ~8GB VRAM
- Current: Unknown (latest fix applied, needs testing)

---
**Status**: Transpose and SVD fixes applied. Need to use torch.addcmul for efficient scale/bias application.

