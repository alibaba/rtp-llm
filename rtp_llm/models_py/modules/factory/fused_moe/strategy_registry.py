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

        # A pre-quantized checkpoint may require MOE experts to consume a
        # specific activation format. Granularity is part of the contract: an
        # INT8 per-tensor strategy must not satisfy a scheme that needs INT8 per
        # token.
        model_quant_config = config.model_config.quant_config
        required_act_spec = (
            model_quant_config.get_moe_activation_quant_spec()
            if model_quant_config is not None
            else None
        )
        spec_hint = ""
        if required_act_spec is not None:
            required_dtype, required_per_act_token = required_act_spec
            spec_hint = (
                f" Quantization method {model_quant_config.get_method()} needs a "
                f"strategy producing {required_dtype} activations "
                f"({'per token' if required_per_act_token else 'not per token'}); "
                "MOE layers cannot serve this checkpoint unless one is registered."
            )

        if not candidates:
            logger.error(
                f"No suitable MOE strategy found. Config details: "
                f"quant_config={config.model_config.quant_config}, "
                f"ep_size={config.ep_size}, "
                f"world_size={config.world_size}, "
                f"tp_size={config.tp_size}, "
                f"use_deepep_low_latency={config.moe_config.use_deepep_low_latency if config.moe_config else False}"
            )
            raise ValueError(
                f"No suitable MOE strategy found for configuration. "
                f"Please check quant_config, ep_size, and parallelism settings."
                f"{spec_hint}"
            )

        # get_attributes() is not a plain accessor -- it does lazy imports and
        # some backends log from it -- so resolve it once and reuse it for the
        # activation-format filter, the candidate log and the selection.
        scored = [(strategy, strategy.get_attributes()) for strategy in candidates]

        if required_act_spec is not None:
            matching = [
                (strategy, attrs)
                for strategy, attrs in scored
                if (
                    attrs.quant_config.quant_dtype,
                    attrs.quant_config.per_act_token_quant,
                )
                == required_act_spec
            ]
            if not matching:
                provided = sorted(
                    str(
                        (
                            attrs.quant_config.quant_dtype,
                            attrs.quant_config.per_act_token_quant,
                        )
                    )
                    for _, attrs in scored
                )
                raise ValueError(
                    f"None of the {len(scored)} candidate strategy(ies) matches the "
                    f"required activation format (they provide {provided})."
                    f"{spec_hint}"
                )
            # Narrow the selection too, so the strategy that gets picked is the
            # one the check passed on rather than a higher-priority fallback.
            scored = matching

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
