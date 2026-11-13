"""Refactored FusedMoe factory class

Uses strategy pattern to simplify FusedMoe creation logic.
"""

from typing import Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

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
        self,
        config: MoEConfigAdapter,
        weights: Dict[str, torch.Tensor],
        *,
        batch_invariant: bool = False,
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
        if batch_invariant:
            from .defs.quant_config import FusedMoEQuantConfig
            from .impl.cuda.executors.triton_fused_executor import (
                TritonFusedMoeExecutor,
            )
            from .impl.cuda.routers.pure_tp_router import PureTpRouterNoQuant

            if config.moe_quant_method not in (None, "FP8_PER_BLOCK"):
                raise ValueError(
                    "RTP_BATCH_INVARIANT does not support this expert quantization"
                )
            quant = (
                FusedMoEQuantConfig(
                    quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
                )
                if config.moe_quant_method == "FP8_PER_BLOCK"
                else FusedMoEQuantConfig()
            )
            return FusedMoe(
                PureTpRouterNoQuant(config, quant, batch_invariant=True),
                TritonFusedMoeExecutor(config, quant, weights, batch_invariant=True),
                expert_num=config.expert_num,
                strategy_name="batch_invariant",
            )

        strategy = self.registry.get_strategy(config)

        router = strategy.create_router(config)
        executor = strategy.create_executor(config, weights)

        strategy_name = (
            getattr(strategy, "strategy_name", None) or strategy.__class__.__name__
        )
        return FusedMoe(
            router,
            executor,
            expert_num=config.expert_num,
            strategy_name=strategy_name,
        )
