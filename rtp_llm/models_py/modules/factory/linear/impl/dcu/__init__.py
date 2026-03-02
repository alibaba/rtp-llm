"""DCU Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered DCU Linear strategies")


from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory


from .linear import DcuLinear


LinearFactory.register(DcuLinear)
