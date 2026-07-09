"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda


# Register CUDA strategies only on CUDA devices
if is_cuda():
    from .f16_linear import CudaF16Linear
    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    LinearFactory.register(CudaF16Linear)

    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
