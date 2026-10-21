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


def _supports_quant_method(strategy: MoeStrategy, quant_method: str | None) -> bool:
    supported = getattr(strategy, "supported_moe_quant_method", None)
    if supported is not None:
        if isinstance(supported, (list, tuple, set, frozenset)):
            return quant_method in supported
        return supported == quant_method
    # In-tree MoeStrategy implementations without an FP8/FP4 declaration use
    # older quantization contracts and must not consume FP8/FP4 weights. A
    # legacy out-of-tree strategy object has no such declaration or base class;
    # defer to its can_handle() method to preserve that extension contract.
    return quant_method != "FP8_FP4" or not isinstance(strategy, MoeStrategy)


def _requested_strategy_context(config: MoEConfigAdapter) -> str:
    requested = getattr(config, "moe_strategy", "auto") or "auto"
    model_config = getattr(config, "model_config", None)
    model_scope = getattr(model_config, "model_type", None)
    if not model_scope:
        model_scope = type(
            model_config if model_config is not None else config
        ).__name__
    return (
        f"Requested MOE_STRATEGY={requested!r} for model scope "
        f"{model_scope!r} in the generic fused-MoE factory."
    )


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
        return sorted(
            self._strategies,
            key=lambda s: getattr(s, "priority", 0),
            reverse=True,
        )

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
        quant_method = getattr(config, "moe_quant_method", None)
        strategies = [
            strategy
            for strategy in self._strategies
            if _supports_quant_method(strategy, quant_method)
        ]
        requested = getattr(config, "moe_strategy", "auto") or "auto"
        if requested != "auto":
            strategies = [
                strategy
                for strategy in strategies
                if getattr(strategy, "strategy_name", None) == requested
            ]
        candidates = [
            strategy for strategy in strategies if strategy.can_handle(config)
        ]
        logger.debug(f"[StrategyRegistry] Found {len(candidates)} candidate(s)")

        if not candidates:
            quant_method = getattr(config, "moe_quant_method", None)
            model_config = getattr(config, "model_config", None)
            quant_config = getattr(model_config, "quant_config", None)
            if quant_method is None:
                quant_method = (
                    quant_config.get_method() if quant_config is not None else None
                )
            # Strategies dropped by can_handle() for an unimportable router or
            # executor record why. Surface it: otherwise this reads as a config
            # problem when the real cause is a missing dependency.
            skipped = [
                f"{s.__class__.__name__}: {s.skip_reason}"
                for s in strategies
                if getattr(s, "skip_reason", None)
            ]
            logger.error(
                "No suitable MOE strategy found. Config details: "
                "effective_quant_config=%r, ep_size=%r, world_size=%r, tp_size=%r, "
                "use_deepep_low_latency=%r, skipped_for_missing_deps=%r",
                getattr(config, "quant_config", quant_config),
                getattr(config, "ep_size", None),
                getattr(config, "world_size", None),
                getattr(config, "tp_size", None),
                getattr(
                    getattr(config, "moe_config", None),
                    "use_deepep_low_latency",
                    False,
                ),
                skipped,
            )
            if quant_method == "W8A8_INT8_PER_CHANNEL_COMPRESSED":
                raise ValueError(
                    "W8A8_INT8_PER_CHANNEL_COMPRESSED weights were loaded, but "
                    "no registered MOE compute backend can consume them; install "
                    "or register a backend with W8A8 INT8 per-channel execution "
                    f"support. {_requested_strategy_context(config)}"
                )
            if quant_method == "FP8_FP4" and getattr(
                config, "has_redundant_experts", False
            ):
                raise ValueError(
                    "FP8/FP4 MOE strategies do not support EPLB redundant "
                    "experts; disable EPLB or register an EPLB-aware backend "
                    f"(logical_experts={getattr(config, 'expert_num', None)}, "
                    "physical_experts="
                    f"{getattr(config, 'physical_expert_num', None)}). "
                    f"{_requested_strategy_context(config)}"
                )
            if skipped:
                raise ValueError(
                    "No suitable MOE strategy found: every candidate was skipped "
                    "because its router/executor could not be imported, so this is "
                    "a missing-dependency problem rather than a configuration one. "
                    f"Skipped: {skipped}. {_requested_strategy_context(config)}"
                )
            raise ValueError(
                "No suitable MOE strategy found for configuration. "
                "Please check quant_config, ep_size, and parallelism settings. "
                f"{_requested_strategy_context(config)}"
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
