# BNB 8-bit Save Feature - Work In Progress

**Status**: On Ice  
**Date**: 2026-01-11  
**Branch**: `bnb-attempt`

## Objective

Enable `BNB8bitCLIPLoader` to save quantized BNB INT8 text encoders after on-the-fly quantization, so they can be directly loaded later without re-quantizing.

## Current State

The save functionality is mostly implemented but may still have issues. Latest commit fixes the key matching logic.

### What Works
1. **Loading FP16 and quantizing on-the-fly**: Uses `bitsandbytes.nn.Linear8bitLt` via `BNB8bitLinear` wrapper
2. **Loading pre-quantized models**: Detects `.quant_state.bitsandbytes__int8` keys and uses `BNB8bitPrequantizedLinear`
3. **Finding BNB8bitLinear modules**: Now successfully finds 252 modules in Qwen 3 4B model

### Key Issue Fixed (Latest Commit)

**The comparison direction was wrong.**

State dict keys and module names have different prefixes:
- **State dict key**: `model.layers.0.self_attn.q_proj.weight`
- **Module name**: `qwen3_4b.transformer.model.layers.0.self_attn.q_proj`

The module name ENDS WITH the key (minus `.weight`), not the other way around.

Fixed in `_load_and_quantize()`:
```python
# WRONG: key_without_weight.endswith(mod_name)
# RIGHT: mod_name.endswith(key_without_weight)
```

## Architecture

### Save Flow (in `_load_and_quantize`)

1. Load original state dict from file
2. Create CLIP model with `make_bnb8bit_ops()` custom operations
3. If `output_path` provided:
   - Find model via `clip.cond_stage_model` or `clip.patcher.model`
   - Collect all `BNB8bitLinear` modules with `named_modules()`
   - Iterate through original state dict
   - For 2D weight tensors, find matching module by checking `mod_name.endswith(key_without_weight)`
   - Force quantize the module with `_maybe_quantize(device)`
   - Extract INT8 weights (`weight.CB`), scales (`weight.SCB`), and metadata
   - Save with `safetensors.torch.save_file()`

### Key Files

- `nodes/bnb8bit_loader.py`: Main implementation
  - `BNB8bitLinear`: Wrapper for on-the-fly quantization
  - `BNB8bitPrequantizedLinear`: Wrapper for loading pre-quantized
  - `BNB8bitCLIPLoader`: The ComfyUI node
  - `_load_and_quantize()`: Handles quantization and saving

### Save Format

Saved safetensors includes:
- Original non-linear layers (embeddings, norms, biases) as-is
- Linear weights as INT8 tensors
- `.SCB` scale tensors for each linear layer
- `.quant_state.bitsandbytes__int8` metadata tensors (JSON encoded)

Metadata format:
```json
{
  "format": "bnb_int8",
  "shape": [out_features, in_features],
  "in_features": int,
  "out_features": int
}
```

File-level metadata:
```json
{
  "format": "bitsandbytes_int8",
  "format_version": "1.0"
}
```

## Next Steps to Complete

1. **Test the latest fix**: Run with `save_quantized=True` and verify quantization count > 0
2. **Verify saved file**: Check that the output has INT8 weights + scales + metadata
3. **Test loading saved file**: Load the `*_bnb8bit.safetensors` and verify it works
4. **Handle edge cases**: 
   - Different model architectures (CLIP-L, CLIP-G, T5XXL)
   - Different state dict key formats
5. **Clean up**: Remove debug logging, add proper error messages

## Lessons Learned

1. **Don't iterate blindly** - Understand the data structures (state dict keys vs module names)
2. **Add clear error messages** - Silent fallbacks mask real problems
3. **Test incrementally** - Verify each step works before moving on
4. **ComfyUI CLIP wrapper is complex** - Direct model access in `_load_and_quantize` is better than traversing after return

## Commands

```bash
# Switch to branch
git checkout bnb-attempt

# Pull latest
git pull origin bnb-attempt

# Test in ComfyUI
# Use BNB8bitCLIPLoader node with save_quantized=True
```
