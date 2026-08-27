"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda, is_sm120

# Register CUDA strategies
from .f16_linear import CudaF16Linear

LinearFactory.register(CudaF16Linear)

if is_cuda():
    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)

    if is_sm120():
        try:
            from .fp8_vllm_blockwise_sm120_linear import CudaFp8VllmBlockwiseLinear

            LinearFactory.register(CudaFp8VllmBlockwiseLinear)
        except ImportError as e:
            logger.warning("SM120 FP8 blockwise backend unavailable: %s", e)

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
