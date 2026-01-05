"""
Hardware capability detection for NVFP4 support.

Provides functions to check if the current system supports NVFP4 hardware-
accelerated matmul, including GPU compute capability and CUDA version checks.
"""

import logging
import platform
from typing import Tuple

import torch


# Minimum compute capability for NVFP4 hardware matmul
# - SM 10.0+: Blackwell datacenter (B100, B200)
# - SM 12.0+: Blackwell consumer (RTX 50xx)
MIN_SM_DATACENTER = (10, 0)
MIN_SM_CONSUMER = (12, 0)

# CUDA version requirement on Windows (kernel compilation bug in 12.9)
MIN_CUDA_VERSION_WINDOWS = (13, 0)


def get_compute_capability(device: int = 0) -> Tuple[int, int]:
    """
    Get the compute capability of a CUDA device.

    Args:
        device: CUDA device index

    Returns:
        Tuple of (major, minor) compute capability
    """
    if not torch.cuda.is_available():
        return (0, 0)

    try:
        props = torch.cuda.get_device_properties(device)
        return (props.major, props.minor)
    except Exception:
        return (0, 0)


def get_cuda_version() -> Tuple[int, int]:
    """
    Get the current CUDA runtime version.

    Returns:
        Tuple of (major, minor) CUDA version
    """
    if not torch.cuda.is_available():
        return (0, 0)

    try:
        # torch.version.cuda returns string like "12.4" or "13.0"
        cuda_version_str = torch.version.cuda
        if cuda_version_str is None:
            return (0, 0)

        parts = cuda_version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


def is_blackwell_gpu(device: int = 0) -> bool:
    """
    Check if the GPU is a Blackwell-class device (SM >= 10.0).

    Args:
        device: CUDA device index

    Returns:
        True if the GPU supports Blackwell features
    """
    cc = get_compute_capability(device)
    return cc >= MIN_SM_DATACENTER


def check_comfy_kitchen_available() -> Tuple[bool, str]:
    """
    Check if comfy-kitchen is available for hardware-accelerated NVFP4.

    Returns:
        (is_available, message)
    """
    try:
        import comfy_kitchen as ck

        # Check for NVFP4-specific functions
        if hasattr(ck, "quantize_nvfp4") and hasattr(ck, "dequantize_nvfp4"):
            return True, "comfy-kitchen available with NVFP4 support"
        else:
            return False, "comfy-kitchen found but missing NVFP4 functions"

    except ImportError:
        return False, "comfy-kitchen not installed"


def check_nvfp4_hardware_support(device: int = 0) -> Tuple[bool, str]:
    """
    Check if NVFP4 hardware-accelerated matmul is supported.

    This checks for:
    1. CUDA availability
    2. Blackwell compute capability (SM >= 10.0)
    3. CUDA version >= 13.0 on Windows (kernel compilation bug in 12.9)

    Args:
        device: CUDA device index to check

    Returns:
        (is_supported, reason_message)
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    # Check compute capability
    cc = get_compute_capability(device)
    if cc == (0, 0):
        return False, "Could not determine GPU compute capability"

    if cc < MIN_SM_DATACENTER:
        return (
            False,
            f"GPU compute capability {cc[0]}.{cc[1]} < 10.0 "
            f"(requires Blackwell SM 10.0+ for hardware NVFP4)"
        )

    # Check CUDA version on Windows
    if is_windows():
        cuda_ver = get_cuda_version()
        if cuda_ver < MIN_CUDA_VERSION_WINDOWS:
            return (
                False,
                f"CUDA {cuda_ver[0]}.{cuda_ver[1]} < 13.0 "
                f"(Windows requires CUDA 13.0+ for NVFP4 kernel compilation)"
            )

    # All checks passed
    return True, f"NVFP4 hardware support available (SM {cc[0]}.{cc[1]})"


def check_nvfp4_dequant_support() -> Tuple[bool, str]:
    """
    Check if NVFP4 dequantization fallback is supported.

    This always returns True as pure PyTorch dequantization works on any GPU.
    However, it logs a warning if comfy-kitchen is not available.

    Returns:
        (is_supported, reason_message)
    """
    ck_available, ck_msg = check_comfy_kitchen_available()

    if ck_available:
        return True, f"NVFP4 dequantization via comfy-kitchen"

    # Pure PyTorch fallback is always available
    return True, f"NVFP4 dequantization via pure PyTorch fallback ({ck_msg})"


def log_nvfp4_support_status(device: int = 0) -> None:
    """
    Log the current NVFP4 support status.

    Args:
        device: CUDA device index to check
    """
    logger = logging.getLogger(__name__)

    # Check hardware matmul support
    hw_supported, hw_msg = check_nvfp4_hardware_support(device)
    if hw_supported:
        logger.info(f"NVFP4 hardware matmul: {hw_msg}")
    else:
        logger.debug(f"NVFP4 hardware matmul not available: {hw_msg}")

    # Check dequantization support
    dq_supported, dq_msg = check_nvfp4_dequant_support()
    logger.debug(f"NVFP4 dequantization: {dq_msg}")

    # Check comfy-kitchen
    ck_available, ck_msg = check_comfy_kitchen_available()
    if ck_available:
        logger.debug(f"comfy-kitchen: {ck_msg}")
    else:
        logger.debug(f"comfy-kitchen: {ck_msg}")
