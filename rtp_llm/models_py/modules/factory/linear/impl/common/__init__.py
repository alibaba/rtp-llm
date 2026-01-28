"""Common Linear implementations and registration"""

from rtp_llm.models_py.modules.factory.linear import LinearFactory

from .f16_linear import F16Linear

# Register common strategies
LinearFactory.register(F16Linear)
