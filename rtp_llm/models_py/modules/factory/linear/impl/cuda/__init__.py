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

    # Register NVFP4 Linear if available
    try:
        from .fp4_linear import CudaFp4GEMMLinear, has_flashinfer_fp4

        if has_flashinfer_fp4():
            LinearFactory.register(CudaFp4GEMMLinear)
            logger.info("Registered CudaFp4GEMMLinear")
    except ImportError:
        logger.error("CudaFp4GEMMLinear not available (flashinfer FP4 support not found)")
