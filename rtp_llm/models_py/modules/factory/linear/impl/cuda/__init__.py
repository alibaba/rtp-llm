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

    # Keep optional FP4 imports isolated from the mandatory FP8 registrations.
    # FlashInfer FP4 is a Blackwell-only API and may be absent from otherwise
    # valid CUDA wheels used on SM8x/SM9x.
    major, _ = get_sm()
    if major >= 10:
        if is_sm120():
            # Consumer-Blackwell wheels may omit the FlashInfer FP4 API; keep
            # registration best-effort so the rest of the factory still loads.
            try:
                from .fp4_linear import CudaFp4GEMMLinear, fp4_backend_available

                if fp4_backend_available():
                    LinearFactory.register(CudaFp4GEMMLinear)
                else:
                    logger.warning("SM120 FP4 backend APIs are unavailable")
            except (ImportError, AttributeError, OSError, RuntimeError) as e:
                logger.warning("SM120 FP4 backend unavailable: %s", e)
        else:
            try:
                from .fp4_linear import CudaFp4GEMMLinear
            except ImportError as error:
                logger.warning("Optional FP4 Linear backend is unavailable: %s", error)
            else:
                LinearFactory.register(CudaFp4GEMMLinear)

    if is_sm120():
        try:
            from .fp8_vllm_blockwise_sm120_linear import (
                CudaFp8VllmBlockwiseLinear,
                sm120_blockwise_fp8_available,
            )

            if sm120_blockwise_fp8_available():
                LinearFactory.register(CudaFp8VllmBlockwiseLinear)
            else:
                logger.warning("SM120 FP8 blockwise backend is unavailable")
        except ImportError as e:
            logger.warning("SM120 FP8 blockwise backend unavailable: %s", e)

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
