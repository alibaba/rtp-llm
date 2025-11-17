from rtp_llm.models_py.modules.common.moe.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe import (
    FusedMoeFactory,
    StrategyRegistry,
)
from rtp_llm.models_py.modules.rocm.moe.strategy import (
    RocmEpLowLatencyStrategy,
    RocmEpNormalStrategy,
)

registry = StrategyRegistry()
registry.register(RocmEpLowLatencyStrategy())
registry.register(RocmEpNormalStrategy())
registry.register(BatchedTritonStrategy())
FusedMoeFactory.set_registry(registry)
