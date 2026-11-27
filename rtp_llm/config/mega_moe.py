"""MegaMoE recipes shared by weight preparation and execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MegaMoeExecutionPolicy:
    """Model numerical semantics, independent of the selected kernel family."""

    w1_layout: str = "up_gate"
    activation_clamp: float | None = None
    fast_math: bool = False


FP4_MEGA_MOE_STRATEGIES = frozenset({"mega_moe", "mega_moe_se", "mega_moe_fused"})
FP8_MEGA_MOE_STRATEGIES = frozenset({"mega_moe_fp8", "mega_moe_fp8_se"})
MEGA_MOE_STRATEGIES = FP4_MEGA_MOE_STRATEGIES | FP8_MEGA_MOE_STRATEGIES
FUSED_SHARED_STRATEGIES = frozenset(
    {"mega_moe_se", "mega_moe_fp8_se", "mega_moe_fused"}
)

_fp4_extension_strategies: dict[str, bool] = {}


def register_prepared_fp4_strategy(name: str, *, fused_shared: bool) -> None:
    """Register a prepared FP4 recipe and its shared-expert capability."""
    if fused_shared:
        _fp4_extension_strategies[name] = True
    else:
        _fp4_extension_strategies.setdefault(name, False)


def is_mega_moe_strategy(name: str | None) -> bool:
    return name in MEGA_MOE_STRATEGIES or name in _fp4_extension_strategies


def is_fp4_mega_moe_strategy(name: str) -> bool:
    return name in FP4_MEGA_MOE_STRATEGIES or name in _fp4_extension_strategies


def has_fused_shared_experts(name: str) -> bool:
    return name in FUSED_SHARED_STRATEGIES or _fp4_extension_strategies.get(name, False)
