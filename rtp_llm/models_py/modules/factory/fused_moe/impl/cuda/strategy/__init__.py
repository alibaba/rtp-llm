"""CUDA MOE strategies"""

from .fp8_per_block import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockPureTPMaskedStrategy,
    CudaFp8PerBlockPureTPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorPureTPStrategy,
)
from .w4a8_int4_per_channel import (
    CudaW4a8Int4PerChannelEpLowLatencyStrategy,
    CudaW4a8Int4PerChannelEpNormalStrategy,
    CudaW4a8Int4PerChannelPureTPStrategy,
)
from .no_quant import (
    CudaNoQuantCppStrategy,
    CudaNoQuantDpNormalStrategy,
    CudaNoQuantEpLowLatencyStrategy,
)
from .fp4 import (CudaFp4EpLowLatencyStrategy,
                  CudaFp4EpNormalStrategy,
                  CudaFp4PureTPStrategy)


__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    "CudaNoQuantCppStrategy",
    "CudaNoQuantDpNormalStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockPureTPMaskedStrategy",
    "CudaFp8PerBlockPureTPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorPureTPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    # W4A8 INT4 PerChannel
    "CudaW4a8Int4PerChannelEpLowLatencyStrategy",
    "CudaW4a8Int4PerChannelEpNormalStrategy",
    "CudaW4a8Int4PerChannelPureTPStrategy",
    "CudaFp4EpLowLatencyStrategy",
    "CudaFp4EpNormalStrategy",
    "CudaFp4PureTPStrategy"
]
