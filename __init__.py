"""
ComfyUI-QuantOps: Extended Quantization Layouts for ComfyUI

This custom node extends ComfyUI's quantization system with additional layouts:
- INT8 blockwise (with optional Triton acceleration)
- Row-wise and Block-wise FP8 variants

All layouts are lazy-loaded to avoid import errors when optional dependencies
(like Triton) are not installed.
"""

import logging


# Register layouts into ComfyUI's registry
# This happens at import time, before any model loading
def _register_layouts():
    """Register our custom layouts into ComfyUI's LAYOUTS and QUANT_ALGOS dicts."""
    try:
        from comfy.quant_ops import LAYOUTS, QUANT_ALGOS
        import torch

        # Import our layouts (this also registers their operation handlers)
        from .quant_layouts.int8_layout import BlockWiseINT8Layout
        from .quant_layouts.fp8_variants import RowWiseFP8Layout, BlockWiseFP8Layout

        # Try to import NVFP4 layout (may fail if dependencies missing)
        try:
            from .quant_layouts.nvfp4_layout import NVFP4Layout
            _HAS_NVFP4 = True
        except ImportError as e:
            logging.debug(f"NVFP4 layout not available: {e}")
            _HAS_NVFP4 = False

        # Register layouts (use setdefault to not override if already present)
        LAYOUTS.setdefault("BlockWiseINT8Layout", BlockWiseINT8Layout)
        LAYOUTS.setdefault("RowWiseFP8Layout", RowWiseFP8Layout)
        LAYOUTS.setdefault("BlockWiseFP8Layout", BlockWiseFP8Layout)
        if _HAS_NVFP4:
            LAYOUTS.setdefault("NVFP4Layout", NVFP4Layout)

        # Register QUANT_ALGOS
        QUANT_ALGOS.setdefault(
            "int8_blockwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "BlockWiseINT8Layout",
                "group_size": 128,
                "asymmetric_layout": True,
            },
        )
        QUANT_ALGOS.setdefault(
            "float8_e4m3fn_rowwise",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "RowWiseFP8Layout",
            },
        )
        QUANT_ALGOS.setdefault(
            "float8_e4m3fn_blockwise",
            {
                "storage_t": torch.float8_e4m3fn,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "BlockWiseFP8Layout",
                "group_size": 64,
            },
        )
        if _HAS_NVFP4:
            QUANT_ALGOS.setdefault(
                "nvfp4",
                {
                    "storage_t": torch.uint8,
                    "parameters": {"weight_scale", "block_scale"},
                    "comfy_tensor_layout": "NVFP4Layout",
                    "group_size": 16,
                },
            )

        logging.info(f"ComfyUI-QuantOps: Registered layouts: {list(LAYOUTS.keys())}")

    except Exception as e:
        logging.error(f"ComfyUI-QuantOps: Failed to register layouts: {e}")


# Register layouts on import
_register_layouts()

# Import nodes for ComfyUI discovery
from .nodes.loader_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
