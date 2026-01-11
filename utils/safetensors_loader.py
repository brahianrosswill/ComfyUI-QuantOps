"""
Safetensors loader utilities for QuantOps.

Provides memory-efficient loading of safetensors files with guaranteed
float32 scale conversion for comfy_kitchen compatibility.

Copied and adapted from ComfyUI getkeys.py.
"""

import mmap
import json
import torch
import struct
import re
from typing import Dict, Any, Optional, Tuple


def tensor_to_dict(tensor_data: torch.Tensor) -> dict:
    """Convert uint8 tensor to dictionary."""
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode("utf-8")
    return json.loads(json_str)


class MemoryEfficientSafeOpen:
    """Memory-efficient safetensors file reader."""

    def __init__(self, filename: str, device: str = "cpu", mmap_mode: bool = False):
        self.filename = filename
        self.device = device
        self.mmap_mode = mmap_mode
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")
        self.mmap_obj = None

        if self.mmap_mode:
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap_obj:
            self.mmap_obj.close()
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start != offset_end:
            self.file.seek(self.header_size + 8 + offset_start)
            # Use bytearray to create a writable, freeable buffer
            # (not memoryview which creates a non-freeable view into mmap)
            tensor_bytes = bytearray(offset_end - offset_start)
            self.file.readinto(tensor_bytes)
        else:
            tensor_bytes = None

        return self._deserialize_tensor(tensor_bytes, metadata)

    def get_tensor_as_dict(self, key: str) -> dict:
        """Get a uint8 tensor and convert it to a dictionary."""
        tensor = self.get_tensor(key)
        metadata = self.header[key]

        if metadata["dtype"] != "U8":
            raise ValueError(f"Tensor '{key}' has dtype {metadata['dtype']}, expected U8 (uint8)")

        return tensor_to_dict(tensor)

    def _read_header(self) -> Tuple[dict, int]:
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata) -> torch.Tensor:
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str: str, shape) -> torch.Tensor:
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


def load_fp8_state_dict(
    filepath: str,
    device: str = "cpu",
    force_scale_float32: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """
    Load a safetensors file and return state_dict with float32 scales.

    Args:
        filepath: Path to safetensors file
        device: Device to load tensors to
        force_scale_float32: If True, convert all scale tensors to float32

    Returns:
        Tuple of (state_dict, metadata)
        - state_dict: Dict of tensors with scales guaranteed float32
        - metadata: File metadata if present
    """
    state_dict = {}
    metadata = None

    with MemoryEfficientSafeOpen(filepath, device=device) as f:
        # Get metadata if present
        if "__metadata__" in f.header:
            metadata = f.header["__metadata__"]

        for key in f.keys():
            tensor = f.get_tensor(key)

            # Check if this is a scale tensor that needs float32 conversion
            if force_scale_float32 and _is_scale_tensor(key):
                if tensor.dtype in (torch.float16, torch.bfloat16):
                    tensor = tensor.to(torch.float32)

            state_dict[key] = tensor

    return state_dict, metadata


def _is_scale_tensor(key: str) -> bool:
    """Check if a tensor key is a scale parameter."""
    scale_patterns = [
        "weight_scale",
        "scale_weight",
        "input_scale",
        "scale_input",
        "weight_scale_2",  # NVFP4 tensor scale
    ]
    return any(pattern in key for pattern in scale_patterns)


def get_layer_metadata(
    filepath: str,
    layer_prefix: str,
) -> Optional[dict]:
    """
    Get comfy_quant metadata for a specific layer.

    Args:
        filepath: Path to safetensors file
        layer_prefix: Layer prefix (e.g., "model.layers.0.attn.qkv.")

    Returns:
        Dict with layer metadata or None if not found
    """
    comfy_quant_key = f"{layer_prefix}comfy_quant"

    with MemoryEfficientSafeOpen(filepath, device="cpu") as f:
        if comfy_quant_key in f.keys():
            try:
                return f.get_tensor_as_dict(comfy_quant_key)
            except (ValueError, json.JSONDecodeError):
                pass
    return None
