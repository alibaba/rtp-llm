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
