"""CUDA MOE strategies"""

from .fp8_per_block import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorSingleGpuStrategy,
)
from .no_quant import CudaNoQuantEpLowLatencyStrategy

__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorSingleGpuStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
]
