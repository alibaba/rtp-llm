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
    from .fp8_deepgemm_linear import CudaFp8DeepGEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear

        LinearFactory.register(CudaFp4GEMMLinear)

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8DeepGEMMLinear)

    # SM90 swapAB-aware FP8 dense GEMM via flashinfer (auto opt-out via
    # RTP_LLM_USE_FLASHINFER_FP8_GEMM=0).
    if major == 9:
        try:
            from .fp8_flashinfer_linear import CudaFp8FlashinferLinear

            LinearFactory.register(CudaFp8FlashinferLinear)
        except Exception as e:
            logger.warning("CudaFp8FlashinferLinear not registered: %s", e)
