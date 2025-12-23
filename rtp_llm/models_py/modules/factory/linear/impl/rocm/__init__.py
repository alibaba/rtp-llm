"""ROCm Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered ROCm Linear strategies")


from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

from .f16_linear import RocmF16LinearWithSwizzle, RocmF16LinearNoSwizzle
from .fp8_deepgemm_linear import RocmFp8DeepGEMMLinear
from .fp8_ptpc_linear import RocmFp8PTPCLinear

# Register ROCm strategies
LinearFactory.register(RocmF16LinearWithSwizzle)
LinearFactory.register(RocmF16LinearNoSwizzle)
LinearFactory.register(RocmFp8PTPCLinear)
LinearFactory.register(RocmFp8DeepGEMMLinear)
