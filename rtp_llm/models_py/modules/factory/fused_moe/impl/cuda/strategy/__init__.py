"""CUDA MOE strategies"""

from .af_disaggregate import CudaAfDisaggregateStrategy
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
from .no_quant import CudaNoQuantCppStrategy, CudaNoQuantEpLowLatencyStrategy

__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    "CudaNoQuantCppStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorNoDPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    # AF disaggregate
    "CudaAfDisaggregateStrategy",
]
