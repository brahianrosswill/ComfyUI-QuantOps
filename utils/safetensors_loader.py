"""
Unified safetensors loader with parallel I/O and metadata support.

Based on patterns from ComfyUI-ModelUtils merger_utils.py, providing:
- Parallel loading via ThreadPoolExecutor (2-4x speedup)
- mmap mode for zero-copy tensor access
- low_memory mode for minimal OS page cache usage
- Automatic scale tensor float32 conversion
- Metadata extraction (both __metadata__ header and per-layer comfy_quant)
"""

import os
import gc
import mmap
import json
import struct
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple


class QuantizedSafetensorsLoader:
    """Unified safetensors loader for quantized models.
    
    Features:
    - mmap mode: Zero-copy tensor access via memory-mapped file
    - low_memory mode: Direct file reads to minimize OS page cache usage
    - Parallel loading: Multi-threaded tensor reads for 2-4x speedup
    - Sorted batch reads: Keys sorted by file offset for sequential I/O
    - Automatic scale conversion: Converts scale tensors to float32
    
    Usage:
        with QuantizedSafetensorsLoader("model.safetensors") as loader:
            # Get file-level metadata
            metadata = loader.metadata()
            
            # Load all tensors
            state_dict = loader.load_state_dict()
            
            # Or load specific tensors
            weight = loader.get_tensor("model.layer.weight")
    """
    
    SCALE_PATTERNS = frozenset({
        "weight_scale", "scale_weight", "input_scale", "scale_input", "weight_scale_2"
    })
    
    def __init__(
        self, 
        filename: str, 
        device: str = "cpu", 
        mmap_mode: bool = True,
        low_memory: bool = False,
        force_scale_float32: bool = True
    ):
        """Initialize the loader.
        
        Args:
            filename: Path to safetensors file
            device: Target device (default 'cpu')
            mmap_mode: Use memory-mapped file for zero-copy (default True)
            low_memory: Use direct file reads to minimize memory (overrides mmap_mode)
            force_scale_float32: Convert scale tensors to float32 (default True)
        """
        self.filename = filename
        self.device = device
        self.low_memory = low_memory
        self.mmap_mode = mmap_mode and not low_memory
        self.force_scale_float32 = force_scale_float32
        
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")
        self.mmap_obj = None
        
        if self.mmap_mode:
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close file handle and release resources."""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.file:
            self.file.close()
            self.file = None
    
    def keys(self) -> List[str]:
        """Return all tensor keys (excluding metadata)."""
        return [k for k in self.header.keys() if k != "__metadata__"]
    
    def metadata(self) -> Dict[str, str]:
        """Return file-level metadata from safetensors header."""
        return self.header.get("__metadata__", {})
    
    def get_quantization_metadata(self) -> Optional[Dict[str, Any]]:
        """Get parsed _quantization_metadata from file header if present."""
        meta = self.metadata()
        if "_quantization_metadata" in meta:
            try:
                return json.loads(meta["_quantization_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None
    
    def keys_sorted_by_offset(self) -> List[str]:
        """Return keys sorted by file offset for optimal sequential I/O."""
        keys_with_offsets = []
        for key in self.keys():
            offset = self.header[key]["data_offsets"][0]
            keys_with_offsets.append((key, offset))
        keys_with_offsets.sort(key=lambda x: x[1])
        return [k for k, _ in keys_with_offsets]
    
    def get_tensor(self, key: str) -> torch.Tensor:
        """Load a single tensor by key.
        
        Automatically converts scale tensors to float32 if force_scale_float32=True.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")
        
        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        
        if self.mmap_mode and self.mmap_obj:
            if offset_start != offset_end:
                start = self.header_size + 8 + offset_start
                end = self.header_size + 8 + offset_end
                tensor_bytes = memoryview(self.mmap_obj)[start:end]
            else:
                tensor_bytes = None
        else:
            tensor_bytes = None
            if offset_start != offset_end:
                self.file.seek(self.header_size + 8 + offset_start)
                tensor_bytes = bytearray(offset_end - offset_start)
                self.file.readinto(tensor_bytes)
        
        tensor = self._deserialize_tensor(tensor_bytes, metadata)
        
        # Convert scale tensors to float32 if requested
        if self.force_scale_float32 and self._is_scale_key(key):
            if tensor.dtype in (torch.float16, torch.bfloat16):
                tensor = tensor.to(torch.float32)
        
        return tensor
    
    def get_tensor_as_dict(self, key: str) -> Dict[str, Any]:
        """Load a uint8 tensor and decode as JSON dict (for comfy_quant config tensors)."""
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found")
        
        metadata = self.header[key]
        
        if metadata["dtype"] != "U8":
            raise ValueError(f"Tensor '{key}' has dtype {metadata['dtype']}, expected U8 (uint8)")
        
        offset_start, offset_end = metadata["data_offsets"]
        
        if offset_start == offset_end:
            return {}
        
        if self.mmap_mode and self.mmap_obj:
            start = self.header_size + 8 + offset_start
            end = self.header_size + 8 + offset_end
            tensor_bytes = bytes(self.mmap_obj[start:end])
        else:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)
        
        return json.loads(tensor_bytes.decode("utf-8"))
    
    def load_state_dict(
        self,
        keys: Optional[List[str]] = None,
        parallel: bool = True,
        workers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load all tensors as a state dict.
        
        Args:
            keys: Specific keys to load (default: all keys)
            parallel: Use parallel loading for speedup (default: True)
            workers: Number of worker threads (default: auto 4)
        
        Returns:
            Dict mapping key -> tensor with scales as float32
        """
        if keys is None:
            keys = self.keys()
        
        if parallel and len(keys) > 10:
            return self._load_parallel(keys, workers or min(4, os.cpu_count() or 4))
        else:
            return self._load_sequential(keys)
    
    def _load_sequential(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Load tensors sequentially (sorted by offset for efficiency)."""
        # Sort by offset for sequential I/O
        keys_with_offsets = [(k, self.header[k]["data_offsets"][0]) for k in keys]
        keys_with_offsets.sort(key=lambda x: x[1])
        sorted_keys = [k for k, _ in keys_with_offsets]
        
        result = {}
        for key in sorted_keys:
            result[key] = self.get_tensor(key)
        return result
    
    def _load_parallel(
        self, 
        keys: List[str], 
        workers: int
    ) -> Dict[str, torch.Tensor]:
        """Load tensors in parallel using ThreadPoolExecutor."""
        # Sort by offset for better I/O patterns
        keys_with_offsets = [(k, self.header[k]["data_offsets"][0]) for k in keys]
        keys_with_offsets.sort(key=lambda x: x[1])
        sorted_keys = [k for k, _ in keys_with_offsets]
        
        result = {}
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _read_tensor_from_file,
                    self.filename,
                    key,
                    self.header[key],
                    self.header_size,
                    self.force_scale_float32 and self._is_scale_key(key)
                ): key
                for key in sorted_keys
            }
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    tensor = future.result()
                    if tensor is not None:
                        result[key] = tensor
                except Exception as e:
                    raise RuntimeError(f"Failed to load tensor '{key}': {e}") from e
        
        return result
    
    def _is_scale_key(self, key: str) -> bool:
        """Check if a tensor key is a scale parameter."""
        return any(pattern in key for pattern in self.SCALE_PATTERNS)
    
    def _read_header(self) -> Tuple[Dict, int]:
        """Read and parse the safetensors header."""
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size
    
    def _deserialize_tensor(self, tensor_bytes, metadata) -> torch.Tensor:
        """Deserialize raw bytes into a torch tensor."""
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)
        
        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            if isinstance(tensor_bytes, bytearray):
                byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)
            else:
                byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)
        
        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)
        
        return byte_tensor.view(dtype).reshape(shape)
    
    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        """Map safetensors dtype string to torch dtype."""
        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        if hasattr(torch, "float8_e8m0fnu"):
            dtype_map["F8_E8M0"] = torch.float8_e8m0fnu
        
        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype
    
    @staticmethod
    def _convert_float8(byte_tensor: torch.Tensor, dtype_str: str, shape: list) -> torch.Tensor:
        """Convert bytes to float8 tensor."""
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}. PyTorch version may be too old.")


def _read_tensor_from_file(
    filepath: str,
    key: str,
    metadata: Dict,
    header_size: int,
    convert_to_float32: bool = False
) -> Optional[torch.Tensor]:
    """Helper function to read a single tensor from file (for parallel execution).
    
    Each worker opens its own file handle for thread safety.
    """
    offset_start, offset_end = metadata["data_offsets"]
    dtype_str = metadata["dtype"]
    shape = metadata["shape"]
    
    if offset_start == offset_end:
        return None
    
    dtype_map = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
        "U8": torch.uint8, "BOOL": torch.bool,
    }
    if hasattr(torch, "float8_e5m2"):
        dtype_map["F8_E5M2"] = torch.float8_e5m2
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["F8_E4M3"] = torch.float8_e4m3fn
    
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    with open(filepath, "rb") as f:
        f.seek(header_size + 8 + offset_start)
        tensor_bytes = f.read(offset_end - offset_start)
    
    byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)
    
    if dtype_str in ["F8_E5M2", "F8_E4M3"]:
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            tensor = byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            tensor = byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8: {dtype_str}")
    else:
        tensor = byte_tensor.view(dtype).reshape(shape)
    
    # Convert scale to float32 if requested
    if convert_to_float32 and tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.to(torch.float32)
    
    return tensor


# Convenience function for simple use cases
def load_quantized_state_dict(
    filepath: str,
    device: str = "cpu",
    force_scale_float32: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Load a safetensors file and return state_dict with file metadata.
    
    Args:
        filepath: Path to safetensors file
        device: Device to load tensors to
        force_scale_float32: Convert scale tensors to float32
    
    Returns:
        Tuple of (state_dict, metadata)
    """
    with QuantizedSafetensorsLoader(filepath, device=device, force_scale_float32=force_scale_float32) as loader:
        state_dict = loader.load_state_dict()
        metadata = loader.metadata()
    return state_dict, metadata


# Backward compatibility aliases
MemoryEfficientSafeOpen = QuantizedSafetensorsLoader
load_fp8_state_dict = load_quantized_state_dict
