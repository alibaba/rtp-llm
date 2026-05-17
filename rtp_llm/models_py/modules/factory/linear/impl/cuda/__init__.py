"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda

# Register CUDA strategies
from .f16_linear import CudaF16Linear

LinearFactory.register(CudaF16Linear)

if is_cuda():
    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    major, minor = get_sm()
    if major >= 10:
        try:
            from .fp4_linear import CudaFp4GEMMLinear

            LinearFactory.register(CudaFp4GEMMLinear)
        except ImportError as e:
            logger.warning(f"FP4 Linear not available: {e}")

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
