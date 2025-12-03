"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import is_cuda

# Register CUDA strategies
from .f16_linear import CudaF16Linear

LinearFactory.register(CudaF16Linear)

if is_cuda():
    from .fp8_deepgemm_linear import CudaFp8DeepGEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8DeepGEMMLinear)
