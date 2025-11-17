"""FusedMoe factory module

Uses strategy pattern and builder pattern for refactored MOE factory.

Main components:
- FusedMoeFactory: Main factory class
- MoeStrategy: Strategy base class
- RouterBuilder/ExecutorBuilder: Builder classes
- StrategyRegistry: Strategy registry

Note: DeepEpInitializer is located in rtp_llm.models_py.distributed.deepep_initializer

Usage example:
    from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory

    moe = FusedMoeFactory.create_fused_moe(config, weights)
"""

from .factory import FusedMoeFactory
from .strategies.strategy_registry import StrategyRegistry

__all__ = ["FusedMoeFactory", "StrategyRegistry"]
