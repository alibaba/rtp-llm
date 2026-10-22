"""Refactored FusedMoe factory class

Uses strategy pattern to simplify FusedMoe creation logic.
"""

from typing import Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.utils.model_weight import W

from .defs.fused_moe import FusedMoe
from .strategy_registry import StrategyRegistry


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

    @torch.inference_mode()
    def create_fused_moe(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> FusedMoe:
        """Create FusedMoe instance

        Automatically selects the appropriate strategy based on configuration,
        and creates corresponding Router and Executor.

        Args:
            config: MOE configuration adapter
            weights: Weight dictionary

        Returns:
            FusedMoe instance

        Raises:
            ValueError: If no suitable strategy is found or configuration is not supported
        """
        logical_to_physical = weights.get(W.log2phy)
        logical_count = weights.get(W.logic_expert_cnt)
        if (logical_to_physical is None) != (logical_count is None):
            raise ValueError("EPLB requires both log2phy and logic_expert_cnt")
        strategy = self.registry.get_strategy(config)

        router = strategy.create_router(config)
        executor = strategy.create_executor(config, weights)

        eplb_layer = None
        if logical_to_physical is not None:
            from rtp_llm.eplb.runtime import EplbLayer

            eplb_layer = EplbLayer(
                config.layer_id,
                logical_to_physical,
                logical_count,
                executor.get_eplb_weights(),
                ep_rank=config.ep_rank,
                # tp_size describes the MoE input view: CP and DP inputs are
                # independent, while TP ranks must agree on each token's copy.
                replica_offset=config.ep_rank if config.tp_size == 1 else 0,
            )

        strategy_name = (
            getattr(strategy, "strategy_name", None) or strategy.__class__.__name__
        )
        return FusedMoe(
            router,
            executor,
            expert_num=config.physical_expert_num,
            strategy_name=strategy_name,
            eplb_layer=eplb_layer,
        )
