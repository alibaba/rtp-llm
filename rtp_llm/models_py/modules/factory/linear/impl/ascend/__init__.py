"""Ascend Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered Ascend Linear strategies")


from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

from .f16_linear import AscendF16Linear
from .w8a8_mxfp8_linear import AscendW8A8MXFP8Linear

LinearFactory.register(AscendF16Linear)
LinearFactory.register(AscendW8A8MXFP8Linear)
