"""DeepGEMM FP8 MegaMoE implementation selection shared with its C++ API."""

import os


def mega_moe_fp8_impl(*, shared_expert_gates: bool = False) -> str:
    """Validate without overriding DeepGEMM's per-launch environment dispatch."""
    impl = os.environ.get("DG_MEGA_MOE_FP8_IMPL", "") or "optimized"
    if impl not in ("optimized", "legacy"):
        raise ValueError(
            f"Invalid DG_MEGA_MOE_FP8_IMPL: {impl!r}; expected optimized or legacy "
            "(unset/empty defaults to optimized)"
        )
    if impl == "legacy" and shared_expert_gates:
        raise ValueError(
            "DG_MEGA_MOE_FP8_IMPL=legacy does not support shared_expert_gates; "
            "use optimized for mega_moe_fp8_se, or use mega_moe_fp8 with "
            "separate shared experts"
        )
    return impl
