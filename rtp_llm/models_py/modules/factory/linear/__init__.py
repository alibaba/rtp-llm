"""Linear factory module

Uses strategy pattern for creating Linear layers.
"""

from rtp_llm.ops.compute_ops import DeviceType, get_device

from .factory import LinearFactory
from .linear_base import LinearBase

__all__ = ["LinearFactory", "LinearBase"]

# ============================================================================
# Device-specific Linear implementation registration
# ============================================================================

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    # Import to trigger ROCm Linear strategy registration
    import rtp_llm.models_py.modules.factory.linear.impl.rocm  # noqa: F401
else:
    # Import to trigger CUDA Linear strategy registration
    import rtp_llm.models_py.modules.factory.linear.impl.cuda  # noqa: F401
