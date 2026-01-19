"""FusedMoe factory module

Uses strategy pattern and builder pattern for refactored MOE factory.

Main components:
- FusedMoeFactory: Main factory class
- MoeStrategy: Strategy base class
- RouterBuilder/ExecutorBuilder: Builder classes
- StrategyRegistry: Strategy registry

Note: DeepEpInitializer is located in rtp_llm.models_py.distributed.deepep_initializer

Usage example:
    from rtp_llm.models_py.modules.factory import FusedMoeFactory

    moe = FusedMoeFactory.create_fused_moe(config, weights)
"""

from rtp_llm.ops.compute_ops import DeviceType, get_device

from .defs.fused_moe import FusedMoe
from .factory import FusedMoeFactory
from .strategy_registry import StrategyRegistry

__all__ = ["FusedMoeFactory", "StrategyRegistry", "FusedMoe"]

# ============================================================================
# Device-specific MoE strategy registration
# ============================================================================

device_type = get_device().get_device_type()

# Import common strategies
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)

if device_type == DeviceType.ROCm:
    # ========== ROCm Registry ==========

    # MoE strategies
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy import (
        RocmEpLowLatencyStrategy,
        RocmEpNormalStrategy,
    )

    registry = StrategyRegistry()
    registry.register(RocmEpLowLatencyStrategy())
    registry.register(RocmEpNormalStrategy())
    registry.register(BatchedTritonStrategy())
    FusedMoeFactory.set_registry(registry)

else:
    # ========== CUDA Registry ==========

    # MoE strategies
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
        CudaFp8PerBlockEpLowLatencyStrategy,
        CudaFp8PerBlockEpNormalStrategy,
        CudaFp8PerBlockNoDPStrategy,
        CudaFp8PerTensorEpLowLatencyStrategy,
        CudaFp8PerTensorEpNormalStrategy,
        CudaFp8PerTensorNoDPStrategy,
        CudaNoQuantCppStrategy,
        CudaNoQuantDpNormalStrategy,
        CudaNoQuantEpLowLatencyStrategy,
    )

    registry = StrategyRegistry()
    registry.register(CudaFp8PerTensorEpLowLatencyStrategy())
    registry.register(CudaFp8PerTensorEpNormalStrategy())
    registry.register(CudaFp8PerBlockEpLowLatencyStrategy())
    registry.register(CudaFp8PerBlockEpNormalStrategy())
    registry.register(CudaFp8PerBlockNoDPStrategy())
    registry.register(CudaFp8PerTensorNoDPStrategy())
    registry.register(CudaNoQuantEpLowLatencyStrategy())
    registry.register(CudaNoQuantDpNormalStrategy())
    registry.register(CudaNoQuantCppStrategy())
    registry.register(BatchedTritonStrategy())
    FusedMoeFactory.set_registry(registry)
