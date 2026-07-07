"""ROCm Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered ROCm Linear strategies")


from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

from .f16_linear import RocmF16LinearNoSwizzle, RocmF16LinearWithSwizzle
from .fp8_deepgemm_linear import RocmFp8DeepGEMMLinear
from .fp8_ptpc_linear import RocmFp8PTPCLinearNoSwizzle, RocmFp8PTPCLinearWithSwizzle

# Register ROCm strategies
LinearFactory.register(RocmF16LinearWithSwizzle)
LinearFactory.register(RocmF16LinearNoSwizzle)
LinearFactory.register(RocmFp8PTPCLinearWithSwizzle)
LinearFactory.register(RocmFp8PTPCLinearNoSwizzle)
LinearFactory.register(RocmFp8DeepGEMMLinear)
