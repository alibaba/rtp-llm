"""Strategy registry

Manages registration and selection of all MOE strategies.
"""

import logging
from typing import List

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

from .defs.strategy_base import MoeStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Strategy registry

    Responsible for managing all registered strategies and selecting the most
    appropriate strategy based on configuration.
    """

    def __init__(self):
        """Initialize registry"""
        self._strategies: List[MoeStrategy] = []

    def register(self, strategy: MoeStrategy) -> None:
        """Register a strategy

        Args:
            strategy: Strategy instance to register
        """
        self._strategies.append(strategy)

    def list_strategies(self) -> List[MoeStrategy]:
        """List all registered strategies sorted by priority (descending)

        Returns:
            List of strategies sorted by priority (highest first)
        """
        return sorted(self._strategies, key=lambda s: s.priority, reverse=True)

    def clear(self) -> None:
        """Clear all registered strategies"""
        self._strategies.clear()

    def get_strategy(self, config: MoEConfigAdapter) -> MoeStrategy:
        """Get appropriate strategy based on configuration

        First finds all strategies that can handle the configuration,
        then selects the one with the highest priority.

        Args:
            config: MOE configuration adapter

        Returns:
            Most appropriate strategy instance (highest priority among candidates)

        Raises:
            ValueError: If no suitable strategy is found
        """
        # Find all candidate strategies that can handle this config
        logger.debug(
            f"[StrategyRegistry] Evaluating {len(self._strategies)} strategies..."
        )
        candidates = [
            strategy for strategy in self._strategies if strategy.can_handle(config)
        ]
        logger.debug(f"[StrategyRegistry] Found {len(candidates)} candidate(s)")

        if not candidates:
            quant_method = (
                config.model_config.quant_config.get_method()
                if config.model_config.quant_config is not None
                else None
            )
            logger.error(
                f"No suitable MOE strategy found. Config details: "
                f"effective_quant_config={config.quant_config}, "
                f"ep_size={config.ep_size}, "
                f"world_size={config.world_size}, "
                f"tp_size={config.tp_size}, "
                f"use_deepep_low_latency={config.moe_config.use_deepep_low_latency if config.moe_config else False}"
            )
            if quant_method == "W8A8_INT8_PER_CHANNEL_COMPRESSED":
                raise ValueError(
                    "W8A8_INT8_PER_CHANNEL_COMPRESSED weights were loaded, but "
                    "no registered MOE compute backend can consume them; install "
                    "or register a backend with W8A8 INT8 per-channel execution "
                    "support"
                )
            raise ValueError(
                f"No suitable MOE strategy found for configuration. "
                f"Please check quant_config, ep_size, and parallelism settings."
            )

        # get_attributes() is not a plain accessor -- it does lazy imports and
        # some backends log from it -- so resolve it once and reuse it for the
        # candidate log and selection.
        scored = [(strategy, strategy.get_attributes()) for strategy in candidates]

        # Sort by priority (descending, higher priority first)
        scored.sort(key=lambda pair: pair[1].calculate_priority(), reverse=True)

        # Log all candidate strategies
        logger.info(f"Found {len(scored)} candidate strategy(ies) for MOE:")
        for strategy, attrs in scored:
            logger.info(
                f"  - {strategy.__class__.__name__}: "
                f"{attrs} (priority={attrs.calculate_priority()})"
            )

        # Select the strategy with highest priority (first in sorted list)
        selected, selected_attrs = scored[0]

        logger.info(
            f"Selected strategy: {selected.__class__.__name__} "
            f"with priority {selected_attrs.calculate_priority()}"
        )

        return selected
