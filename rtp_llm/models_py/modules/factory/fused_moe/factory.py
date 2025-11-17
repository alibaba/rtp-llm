"""Refactored FusedMoe factory class

Uses strategy pattern to simplify FusedMoe creation logic.
"""

from typing import Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.moe.fused_moe import FusedMoe

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
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> FusedMoe:
        """Create FusedMoe instance

        Automatically selects the appropriate strategy based on configuration,
        and creates corresponding Router and Executor.

        Args:
            config: Model initialization parameters
            weights: Weight dictionary

        Returns:
            FusedMoe instance

        Raises:
            ValueError: If no suitable strategy is found or configuration is not supported
        """
        strategy = self.registry.get_strategy(config)

        router = strategy.create_router(config)
        executor = strategy.create_executor(config, weights)

        return FusedMoe(router, executor, expert_num=config.expert_num)
