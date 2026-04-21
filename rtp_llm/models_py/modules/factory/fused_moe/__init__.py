"""FusedMoe factory module

Uses strategy pattern and builder pattern for refactored MOE factory.
Device-specific strategy registration is driven by device.get_moe_strategy_candidates().

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

import logging

from rtp_llm.device import get_current_device

from .defs.fused_moe import FusedMoe
from .factory import FusedMoeFactory
from .strategy_registry import StrategyRegistry

__all__ = ["FusedMoeFactory", "StrategyRegistry", "FusedMoe"]

# ============================================================================
# Device-specific MoE strategy registration
# ============================================================================

registry = StrategyRegistry()

for strategy_cls in get_current_device().get_moe_strategy_candidates():
    try:
        registry.register(strategy_cls())
    except Exception as e:
        logging.debug(f"Skipping MoE strategy {strategy_cls.__name__}: {e}")

FusedMoeFactory.set_registry(registry)
