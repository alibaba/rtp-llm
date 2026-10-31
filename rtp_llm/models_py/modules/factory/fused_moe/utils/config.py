"""Environment-backed configuration shared by generic MoE components."""

import os


def strict_fused_moe_enabled() -> bool:
    return os.environ.get("MOE_STRICT_FUSED", "1") != "0"


def mega_shared_expert_block32_enabled() -> bool:
    """Decode-only opt-in for a group-32 shared expert fused into MegaMoE.

    The fused activation/combine rounds differently from the standalone BF16
    shared expert, so this stays off unless explicitly requested.
    """
    return os.environ.get("MOE_MEGA_SHARED_EXPERT", "0") == "1"
