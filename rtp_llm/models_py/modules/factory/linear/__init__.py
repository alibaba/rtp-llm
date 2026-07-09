"""Linear factory module

Uses strategy pattern for creating Linear layers.
"""

import logging

from rtp_llm.device.device_type import DeviceType, get_device_type

from .factory import LinearFactory
from .linear_base import LinearBase
from ..platform_ext_loader import load_platform_extension

__all__ = ["LinearFactory", "LinearBase"]

# ============================================================================
# Device-specific Linear implementation registration
# ============================================================================

device_type = get_device_type()
try:
    if device_type == DeviceType.ROCm:
        # Import to trigger ROCm Linear strategy registration
        import rtp_llm.models_py.modules.factory.linear.impl.rocm  # noqa: F401
    elif device_type == DeviceType.Cuda:
        import rtp_llm.models_py.modules.factory.linear.impl.cuda  # noqa: F401
except Exception as e:
    logging.warning(f"Failed to import Linear implementation: {e}")

extension = load_platform_extension()
if extension and hasattr(extension, "register_linear"):
    extension.register_linear(device_type=device_type, linear_factory=LinearFactory)
