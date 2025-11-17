from rtp_llm.models_py.modules.common.moe.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)
from rtp_llm.models_py.modules.cuda.moe.strategy import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPStrategy,
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorSingleGpuStrategy,
    CudaNoQuantEpLowLatencyStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe import (
    FusedMoeFactory,
    StrategyRegistry,
)

registry = StrategyRegistry()
registry.register(CudaFp8PerTensorEpLowLatencyStrategy())
registry.register(CudaFp8PerTensorEpNormalStrategy())
registry.register(CudaFp8PerBlockEpLowLatencyStrategy())
registry.register(CudaFp8PerBlockEpNormalStrategy())
registry.register(CudaFp8PerBlockNoDPStrategy())
registry.register(CudaFp8PerTensorSingleGpuStrategy())
registry.register(CudaNoQuantEpLowLatencyStrategy())
registry.register(BatchedTritonStrategy())
FusedMoeFactory.set_registry(registry)
