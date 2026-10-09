"""Ascend Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered Ascend Linear strategies")


from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

from .f16_linear import AscendF16Linear

LinearFactory.register(AscendF16Linear)
