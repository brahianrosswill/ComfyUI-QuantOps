# SDNQ Int4 Matmul Implementation Plan

## Investigation Findings

### Existing Infrastructure

**ComfyUI-QuantOps**:
- ❌ No int4 matmul kernels
- ✅ SDNQ unpacking exists: `unpack_uint4()` at line 67-71
- ✅ Int8 matmul pattern to follow: `_sdnq_int8_matmul()` lines 296-363

**PyTorch**: ✅ `torch.ops.aten._weight_int4pack_mm` available (PyTorch 2.10)

**sdnext**: ❌ No int4 matmul - dequantizes to float

## Critical Issues Found in Initial Plan

### 🔴 Issue 1: Missing SVD Handling
**Risk**: Same bug as int8 implementation - shape mismatch errors  
**Fix Required**: Must copy proven SVD pattern from lines 317-334 (int8) or lines 427-437 (dequant)

### 🔴 Issue 2: Wrong Dequantization in Fallback
**Risk**: Line 99 calls `SDNQLayout.dequantize()` which has the SAME int8 bug at line 134  
**Fix Required**: Cannot use dequantize for int4 fallback - must inline correct dequant logic

### 🔴 Issue 3: Vague "handle scaling and bias"
**Risk**: Same scale broadcasting bugs as int8 (fixed at line 333-336)  
**Fix Required**: Explicit scale merging: `input_scale * weight_scale.squeeze(-1).unsqueeze(0)`

### 🔴 Issue 4: No Shape Validation
**Risk**: PyTorch int4pack_mm may expect different packing format than SDNQ  
**Fix Required**: Must validate/repack if formats don't match

## Detailed Implementation Plan

### Phase 1: Add `_sdnq_int4_matmul` Function

**Location**: `quant_layouts/sdnq_layout.py`, insert after `_sdnq_int8_matmul` (line 364)

```python
def _sdnq_int4_matmul(input_tensor, weight_qt, bias, params):
    """
    SDNQ int4 matmul adapted from int8 version.
    CRITICAL: Must handle SVD and scales exactly like int8.
    """
    # Small batch fallback (same as int8)
    if torch.numel(input_tensor) / input_tensor.shape[-1] < 32:
        # CRITICAL: Cannot use SDNQLayout.dequantize() - it has the int8 bug at line 134!
        # Must inline CORRECT dequantization:
        qdata_unpacked = unpack_uint4(weight_qt._qdata).to(torch.float32)
        # Correct dequant for signed int4: just multiply by scale (NO adding dtype_info["min"])
        weight_fp = qdata_unpacked.mul(params.scale)
        weight_fp = weight_fp.reshape(params.orig_shape)
        return torch.nn.functional.linear(input_tensor, weight_fp, bias)
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    weight_int4_packed = weight_qt._qdata.to(device)  # uint8 with 2×int4 per byte
    weight_scale = params.scale.to(device, dtype=torch.float32)
    output_shape = (*input_tensor.shape[:-1], weight_qt.shape[0])
    
    # CRITICAL: SVD handling - EXACT copy of lines 317-334 from int8
    if params.svd_up is not None and params.svd_down is not None:
        svd_up, svd_down = _get_sdnq_svd_tensors(params, device, dtype)
        x_flat = input_tensor.flatten(0, -2)
        if params.transposed:
            svd_correction = torch.mm(torch.mm(x_flat, svd_up), svd_down)
        else:
            svd_correction = torch.mm(torch.mm(x_flat, svd_down.t()), svd_up.t())
        
        if bias is not None:
            bias = bias.to(device, dtype) + svd_correction.to(dtype)
        else:
            bias = svd_correction.to(dtype)
    
    # Dynamically quantize activation (same as int8)
    input_flat = input_tensor.flatten(0, -2).to(weight_scale.dtype)
    input_scale = torch.amax(input_flat.abs(), dim=-1, keepdims=True) / 127.0
    input_int8 = (input_flat / input_scale).round().clamp(-127, 127).to(torch.int8)
    
    # CRITICAL: Scale merging - EXACT copy from line 333-336
    merged_scale = input_scale * weight_scale.squeeze(-1).unsqueeze(0)
    if merged_scale.dtype == torch.float16:
        merged_scale = merged_scale.to(torch.float32)
    
    # Try PyTorch int4pack_mm
    try:
        # CRITICAL: Verify packing format matches PyTorch expectations
        # PyTorch may expect different layout than SDNQ's row-major 2×int4/byte
        # TODO: Add format conversion if needed
        
        # Unpack int4 for now (less efficient but correct)
        weight_unpacked = unpack_uint4(weight_int4_packed)  # Now flat
        # Reconstruct to [N, K] and convert to int8 range
        N, K = params.orig_shape
        weight_int8 = weight_unpacked.reshape(N, K).to(torch.int8).t()  # [K, N] for matmul
        
        # Use standard int8 path since PyTorch int4pack format unclear
        output_int32 = torch._int_mm(input_int8, weight_int8)
        
    except Exception as e:
        # Fallback: unpack and use int8 path
        logging.warning(f"Int4 native path failed: {e}, using int8 fallback")
        weight_unpacked = unpack_uint4(weight_int4_packed)
        N, K = params.orig_shape
        weight_int8 = weight_unpacked.reshape(N, K).to(torch.int8).t()
        output_int32 = torch._int_mm(input_int8, weight_int8)
    
    # CRITICAL: Dequantize exactly like int8 (lines 357-363)
    output_fp = output_int32.to(merged_scale.dtype) * merged_scale
    if bias is not None:
        output_fp = output_fp + bias.to(device, merged_scale.dtype)
    
    output = output_fp.reshape(output_shape)
    return output.to(dtype)
```

### Phase 2: Update Dispatch

**Location**: `sdnq_linear()` at line 379

```python
# ADD after int8 check:
elif params.weights_dtype in ("int4", "uint4"):
    return _sdnq_int4_matmul(input_tensor, weight, bias, params)
```

### Phase 3: Testing

**Create**: `test_sdnq_int4.py` (copy `test_sdnq_int8.py` and adapt)

```python
# Key test: int4 range is [-8, 7] not [-128, 127]
weight_int4 = (weight_fp / weight_scale).round().clamp(-8, 7)
```

## Risk Mitigation Checklist

- [x] SVD handling: Use proven pattern from int8 (lines 317-334)
- [x] Scale merging: Use exact formula from int8 (line 333-336)
- [x] Dequantization: Avoid buggy SDNQLayout.dequantize(), inline correct logic
- [x] Shape validation: Explicit unpacking and reshaping with orig_shape
- [x] Small batch fallback: Use correct inline dequant
- [x] Error handling: Try-except with fallback path

## Known Limitations

1. **Performance**: Unpack int4→int8 loses int4 density benefit
   - Fix: Implement proper int4 Triton kernel (future work)
   - Current: Correctness > performance

2. **PyTorch int4pack_mm format**: Unclear if compatible with SDNQ packing
   - Current plan: Unpack and use int8 path (safer)
   - Future: Investigate PyTorch format and optimize

## Verification Steps

1. Unit test with 64 tokens (>32 threshold)
2. Compare vs float dequant reference
3. Check SVD handling with transposed weights
4. Test with real SDNQ int4 model
5. Verify no shape mismatches


## Investigation Findings

### Existing Infrastructure

**ComfyUI-QuantOps**:
- ❌ No int4 kernels or layouts
- ✅ Has int8 kernels in `int8_kernels.py` (can serve as template)
- ✅ SDNQ layout already handles int4 **storage** (packed format)

**sdnext Reference**:
- ✅ Supports int4 weights (`int4`, `uint4` packed as 2× values per uint8)
- ✅ Has pack/unpack functions in `packed_int.py`
- ❌ No int4 matmul - **dequantizes to float** just like current ComfyUI

**PyTorch**:
- ✅ PyTorch 2.6.0 detected
- 🔍 Checking: `torch.ops.aten._weight_int4pack_mm` availability

### Key Insight

**Neither sdnext nor ComfyUI have int4 matmul kernels**. Both dequantize int4 → float for matmul. We would be implementing this from scratch.

## Implementation Options

### Option 1: PyTorch Native (Recommended Start)

Use `torch.ops.aten._weight_int4pack_mm` if available (PyTorch 2.3+).

**Advantages**:
- Optimized by PyTorch/NVIDIA
- Minimal code (just interface layer)
- Tensor Core acceleration on Ampere+

**Disadvantages**:
- API may have specific format requirements
- Less control over scaling/packing layout

### Option 2: Custom Triton Kernel

Implement Triton kernel that unpacks int4 and performs matmul.

**Advantages**:
- Full control over format
- Works on AMD GPUs
- Can optimize for SDNQ's specific layout

**Disadvantages**:
- More development effort
- Need to handle 2× int4 unpacking per byte
- Harder to match native performance

### Option 3: Hybrid Approach (Recommended)

1. Try PyTorch native first
2. Fall back to Triton if unavailable
3. Fall back to dequant if both fail

## Proposed Implementation

### Phase 1: Investigate PyTorch int4pack_mm

Check compatibility with SDNQ int4 format:

```python
# Test if torch._weight_int4pack_mm exists and works with SDNQ layout
# SDNQ packs: 2× int4 values per uint8, row-major
# Need to verify if PyTorch expects same format
```

### Phase 2: Add SDNQ int4 Matmul Handler

**[NEW] `kernels/int4_kernels.py`** (if PyTorch native doesn't work):

```python
@triton.jit
def int4_gemm_kernel(...):
    """
    Int4 GEMM with unpacking.
    Unpacks 2× int4 from each uint8 byte.
    Accumulates in int32, scales to float32.
    """
    # Load packed uint8
    # Unpack to 2× int4: low_nibble = byte & 0x0F, high_nibble = byte >> 4
    # Sign-extend if signed int4
    # Perform int8/int16 matmul
    # Scale output
```

**[MODIFY] `quant_layouts/sdnq_layout.py`**:

```python
def _sdnq_int4_matmul(input_tensor, weight_qt, bias, params):
    """SDNQ int4 matmul using PyTorch native or Triton fallback."""
    
    # Small batch fallback
    if torch.numel(input_tensor) / input_tensor.shape[-1] < 32:
        weight_dequant = SDNQLayout.dequantize(weight_qt._qdata, params)
        return torch.nn.functional.linear(input_tensor, weight_dequant, bias)
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    weight_int4_packed = weight_qt._qdata.to(device)  # uint8, 2× int4 per byte
    weight_scale = params.scale.to(device, dtype=torch.float32)
    
    # Try PyTorch native first
    if hasattr(torch.ops.aten, '_weight_int4pack_mm'):
        try:
            # Quantize activation
            input_flat = input_tensor.flatten(0, -2)
            input_scale = torch.amax(input_flat.abs(), dim=-1, keepdim=True) / 127.0
            input_int8 = (input_flat / input_scale).round().clamp(-127, 127).to(torch.int8)
            
            # Int4 matmul (if format matches)
            output = torch.ops.aten._weight_int4pack_mm(
                input_int8, weight_int4_packed, weight_scale
            )
            # ... handle scaling and bias
            return output.to(dtype)
        except:
            pass  # Fall through to Triton
    
    # Triton fallback
    from ..kernels.int4_kernels import int4_gemm
    # ... implement Triton path
```

**Update dispatch in `sdnq_linear`**:

```python
if params.weights_dtype == "int8":
    return _sdnq_int8_matmul(...)
elif params.weights_dtype in ("int4", "uint4"):
    return _sdnq_int4_matmul(...)
```

## Challenges

1. **Unpacking overhead**: 2× int4 per byte requires bit manipulation
2. **Accumulation precision**: Need int32 accumulation to avoid overflow
3. **Scale handling**: Per-block scales need correct broadcasting
4. **Format compatibility**: PyTorch int4pack format may differ from SDNQ

## Verification

1. **Unit test**: Create test similar to `test_sdnq_int8.py`
2. **Compare**: int4 output vs dequantized float reference
3. **Performance**: Benchmark int4 vs int8 vs float
4. **Real model**: Test with SDNQ int4 quantized model

## Recommended Next Steps

1. ✅ Check PyTorch int4pack_mm availability (in progress)
2. Create proof-of-concept with PyTorch native API
3. If PyTorch works: integrate into SDNQ
4. If not: implement Triton kernel based on int8 pattern
5. Add comprehensive tests
