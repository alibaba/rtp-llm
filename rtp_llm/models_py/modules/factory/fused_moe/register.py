"""Register the PPU fused-MoE strategies with the open-source factory.

The open-source fused_moe package builds its StrategyRegistry in a
``get_device_type()`` branch and then drains the ``fused_moe`` backend slot, so
record a hook here instead of adding a PPU arm to that branch.
"""

import logging

from rtp_llm.utils.backend_registry import register_backend_hook

_registered = False

# CLI names of the strategies registered below. The public --moe_strategy
# parser cannot advertise these, so extend its choices through the
# moe_strategy_choices slot; without that the flag rejects them outright.
_CLI_STRATEGIES = (
    "no_quant_dp_normal_deepgemm",
    "no_quant_ep_low_latency_deepgemm",
    "w8a8_int8_dp_normal_deepgemm",
    "w8a8_int8_ep_low_latency_deepgemm",
)


def _extend_strategy_choices(parser, **_ignored) -> None:
    for action in parser._actions:
        if "--moe_strategy" not in action.option_strings:
            continue
        if action.choices is None:
            action.choices = list(_CLI_STRATEGIES)
            return
        # The public parser uses a list, but tolerate a tuple so an upstream
        # change of container type cannot break startup.
        merged = list(action.choices)
        for name in _CLI_STRATEGIES:
            if name not in merged:
                merged.append(name)
        action.choices = (
            merged if isinstance(action.choices, list) else tuple(merged)
        )
        return
    logging.warning(
        "[ppu moe register] --moe_strategy not found; PPU strategies stay unselectable"
    )


def _register_strategies(registry, **_ignored) -> None:
    from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.strategy.no_quant import (
        PpuNoQuantDpNormalDeepGemmStrategy,
        PpuNoQuantEpLowLatencyDeepGemmStrategy,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.strategy.w8a8_int8 import (
        PpuW8A8Int8DpNormalDeepGemmStrategy,
        PpuW8A8Int8EpLowLatencyDeepGemmStrategy,
    )

    registry.register(PpuNoQuantDpNormalDeepGemmStrategy())
    registry.register(PpuNoQuantEpLowLatencyDeepGemmStrategy())
    registry.register(PpuW8A8Int8DpNormalDeepGemmStrategy())
    registry.register(PpuW8A8Int8EpLowLatencyDeepGemmStrategy())
    logging.info("[ppu moe register] Registered BF16 + W8A8 INT8 DeepGEMM strategies")


def install() -> None:
    """Record the registration hook. Must run before the factory is imported."""
    global _registered
    if _registered:
        return
    _registered = True
    register_backend_hook("fused_moe", _register_strategies)
    register_backend_hook("moe_strategy_choices", _extend_strategy_choices)
    logging.info("[ppu moe register] Fused-MoE backend hooks recorded")
