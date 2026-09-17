"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda

# Register CUDA strategies only on NVIDIA. Importing this package on ROCm
# (e.g. generic_moe fused-quant helpers) must not register CudaF16Linear.
if is_cuda():
    from .f16_linear import CudaF16Linear

    LinearFactory.register(CudaF16Linear)
    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)

        try:
            from .mxfp8_linear import CudaMxfp8Linear

            LinearFactory.register(CudaMxfp8Linear)
        except ImportError as e:
            logger.warning(f"MXFP8 Linear not available: {e}")

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
