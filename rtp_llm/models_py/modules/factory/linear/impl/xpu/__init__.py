"""XPU Linear implementations and registration.

Uses PyTorch F.linear for all computations on Intel XPU.
"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered XPU Linear strategies")

from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from .f16_linear import XpuF16Linear

LinearFactory.register(XpuF16Linear)
