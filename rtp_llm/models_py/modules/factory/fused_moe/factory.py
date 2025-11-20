"""Refactored FusedMoe factory class

Uses strategy pattern to simplify FusedMoe creation logic.
"""

from typing import Dict, Optional

import torch

from rtp_llm.models_py.modules.common.moe.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.runtime_config import RuntimeConfig

from .strategies.strategy_registry import StrategyRegistry


class FusedMoeFactory:
    """FusedMoe factory class

    Responsible for creating appropriate FusedMoe instances based on configuration.
    """

    _registry: Optional[StrategyRegistry] = None

    def __init__(self):
        assert FusedMoeFactory._registry is not None, "_registry should not be None"
        self.registry = FusedMoeFactory._registry

    @classmethod
    def set_registry(cls, r: StrategyRegistry):
        cls._registry = r

    def create_fused_moe(
        self, config: RuntimeConfig, weights: Dict[str, torch.Tensor]
    ) -> FusedMoe:
        """Create FusedMoe instance

        Automatically selects the appropriate strategy based on configuration,
        and creates corresponding Router and Executor.

        Args:
            config: Runtime config, including model config and runtime flags
            weights: Weight dictionary

        Returns:
            FusedMoe instance

        Raises:
            ValueError: If no suitable strategy is found or configuration is not supported
        """
        strategy = self.registry.get_strategy(config)

        router = strategy.create_router(config.model_config)
        executor = strategy.create_executor(config.model_config, weights)

        return FusedMoe(router, executor, expert_num=config.model_config.expert_num)
