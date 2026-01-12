"""CUDA MOE strategies"""

from .fp8_per_block import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorNoDPStrategy,
)

from .fp4 import (CudaFp4EpLowLatencyStrategy,
                  CudaFp4EpNormalStrategy,
                  CudaFp4NoDPStrategy)

from .no_quant import CudaNoQuantEpLowLatencyStrategy

__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorNoDPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    "CudaFp4EpLowLatencyStrategy",
    "CudaFp4EpNormalStrategy",
    "CudaFp4NoDPStrategy"
]
