"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)

try:
    import sys
    import os
    import nvidia_cutlass_dsl
    pkg_dir = nvidia_cutlass_dsl.__path__[0]
    python_packages_dir = os.path.join(pkg_dir, 'python_packages')

    if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
        sys.path.insert(0, python_packages_dir)
        logger.debug(f"Added to sys.path: {python_packages_dir}")
except ImportError:
    pass

logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import is_cuda, get_sm

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
