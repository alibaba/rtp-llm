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
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)

    if is_sm12x():
        try:
            from .fp8_vllm_blockwise_sm120_linear import (
                CudaFp8VllmBlockwiseLinear,
                _has_cutlass_scaled_mm_blockwise_sm120_fp8,
            )

            if _has_cutlass_scaled_mm_blockwise_sm120_fp8():
                LinearFactory.register(CudaFp8VllmBlockwiseLinear)
            else:
                logger.warning(
                    "CudaFp8VllmBlockwiseLinear unavailable on sm12x: "
                    "cutlass_scaled_mm_blockwise_sm120_fp8 was not compiled. "
                    "Rebuild with --config=cuda12_9 to enable this backend."
                )
        except ImportError as e:
            logger.warning(
                "CudaFp8VllmBlockwiseLinear unavailable on sm12x: %s; "
                "rebuild with --config=cuda12_9 to enable this backend.",
                e,
            )

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
