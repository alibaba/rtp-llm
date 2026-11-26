"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import get_sm, is_cuda


def _register_optional(import_fn, name: str) -> None:
    try:
        strategy = import_fn()
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        logger.warning("%s is unavailable: %s", name, error)
        return
    if strategy is not None:
        LinearFactory.register(strategy)


def _load_fp4_linear():
    from .fp4_linear import CudaFp4GEMMLinear, fp4_backend_available

    if not fp4_backend_available():
        logger.warning("FP4 backend APIs are unavailable")
        return None
    return CudaFp4GEMMLinear


# Register CUDA strategies
from .f16_linear import CudaF16Linear

LinearFactory.register(CudaF16Linear)

if is_cuda():
    from .fp8_gemm_linear import CudaFp8GEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear

    major, _ = get_sm()
    if major >= 10:
        _register_optional(_load_fp4_linear, "Optional FP4 Linear backend")

    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8GEMMLinear)
