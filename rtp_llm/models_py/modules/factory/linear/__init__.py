"""Linear factory module

Uses strategy pattern for creating Linear layers.
Device-specific registration is driven by device.register_linear_impl().
"""

import logging

from rtp_llm.device import get_current_device

from .factory import LinearFactory
from .linear_base import LinearBase

__all__ = ["LinearFactory", "LinearBase"]

# ============================================================================
# Device-specific Linear implementation registration
# ============================================================================

try:
    get_current_device().register_linear_impl()
except Exception as e:
    logging.warning(f"Failed to register Linear implementation: {e}")
