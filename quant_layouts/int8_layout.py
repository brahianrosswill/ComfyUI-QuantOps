"""
Block-wise INT8 Quantization Layout - Thin Wrapper

This module re-exports BlockWiseINT8Layout from comfy_kitchen.tensor for backward
compatibility. All quantization and dispatch is handled by comfy-kitchen.

Note: The set_backend() method is a no-op as comfy-kitchen handles backend
selection globally via ck.enable_backend() / ck.disable_backend().
"""

import logging

# Re-export from comfy-kitchen
from comfy_kitchen.tensor import BlockWiseINT8Layout as _CKBlockWiseINT8Layout

# Re-export base classes for type hints
from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams


class BlockWiseINT8Layout(_CKBlockWiseINT8Layout):
    """Block-wise INT8 quantization layout.
    
    This class wraps comfy_kitchen's BlockWiseINT8Layout for compatibility with
    existing code that uses set_backend() or get_backend().
    
    Backend selection is now handled globally by comfy-kitchen:
        import comfy_kitchen as ck
        ck.enable_backend("triton")  # Enable triton
        ck.disable_backend("triton")  # Disable triton (uses eager backend)
    """
    
    # Class-level backend state (for compatibility, but effectively a no-op)
    _backend = "triton"
    
    @classmethod
    def set_backend(cls, backend: str):
        """Set backend preference (compatibility method - effectively a no-op).
        
        Note: comfy-kitchen now handles backend selection globally.
        This method exists only for backward compatibility.
        
        Args:
            backend: "triton" or "pytorch" (ignored, logs warning)
        """
        if backend not in ("triton", "pytorch"):
            raise ValueError(f"Unsupported backend: {backend}. Use 'triton' or 'pytorch'.")
        
        cls._backend = backend
        logging.debug(
            f"BlockWiseINT8Layout.set_backend('{backend}') called - "
            f"note: backend is now managed by comfy-kitchen globally"
        )
    
    @classmethod
    def get_backend(cls) -> str:
        """Get current backend preference (compatibility method)."""
        return cls._backend


# For backward compatibility with code that imports directly
__all__ = ["BlockWiseINT8Layout", "QuantizedLayout", "BaseLayoutParams"]
