"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda, is_sm12x

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
        except ImportError as error:
            logger.warning("Optional FP4 Linear backend is unavailable: %s", error)
        else:
            LinearFactory.register(CudaFp4GEMMLinear)

    if is_sm12x() and (major, minor) == (12, 0):
        try:
            from .fp8_vllm_blockwise_sm120_linear import (
                CudaFp8VllmBlockwiseLinear,
                sm120_blockwise_backend_available,
            )

            if sm120_blockwise_backend_available():
                LinearFactory.register(CudaFp8VllmBlockwiseLinear)
            else:
                logger.warning(
                    "SM120 CUTLASS FP8 binding unavailable; rebuild with --config=cuda12_9"
                )
        except ImportError as error:
            logger.warning(
                "Optional SM120 CUTLASS FP8 backend is unavailable: %s", error
            )

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
