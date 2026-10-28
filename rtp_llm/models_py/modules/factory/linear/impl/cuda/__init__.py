"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda


# Register CUDA strategies only on CUDA devices
if is_cuda():
    from .f16_linear import CudaF16Linear

    # Keep the unquantized fallback available even if an optional FP8 backend
    # cannot be imported on the current CUDA architecture.
    LinearFactory.register(CudaF16Linear)

    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    # Register independent FP8 backends before the optional FP4 import. Missing
    # rtp_kernel.nvfp4 must not remove working FP8 strategies from the factory.
    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)

    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)
