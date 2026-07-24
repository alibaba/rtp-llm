"""CUDA MOE strategies"""

from .fp4 import (
    CudaFp4B12xNoDPStrategy,
    CudaFp4EpLowLatencyStrategy,
    CudaFp4EpNormalStrategy,
    CudaFp4NoDPStrategy,
)
from .fp8_per_block import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPMaskedStrategy,
    CudaFp8PerBlockNoDPStrategy,
    CudaFp8PerBlockPureCPStrategy,
    CudaFp8PerBlockPureDPStrategy,
    CudaSm120Fp8GroupedGemmNoDPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorNoDPStrategy,
)
from .no_quant import (
    CudaNoQuantCppStrategy,
    CudaNoQuantDpNormalStrategy,
    CudaNoQuantEpLowLatencyStrategy,
)
from .w4a8_int4_per_channel import (
    CudaW4a8Int4PerChannelEpLowLatencyStrategy,
    CudaW4a8Int4PerChannelEpNormalStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
)

__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    "CudaNoQuantCppStrategy",
    "CudaNoQuantDpNormalStrategy",
    # FP8 PerBlock
    "CudaSm120Fp8GroupedGemmNoDPStrategy",
    "CudaFp8PerBlockNoDPMaskedStrategy",
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockPureCPStrategy",
    "CudaFp8PerBlockPureDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorNoDPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    # W4A8 INT4 PerChannel
    "CudaW4a8Int4PerChannelEpLowLatencyStrategy",
    "CudaW4a8Int4PerChannelEpNormalStrategy",
    "CudaW4a8Int4PerChannelNoDPStrategy",
    "CudaFp4B12xNoDPStrategy",
    "CudaFp4EpLowLatencyStrategy",
    "CudaFp4EpNormalStrategy",
    "CudaFp4NoDPStrategy",
]
