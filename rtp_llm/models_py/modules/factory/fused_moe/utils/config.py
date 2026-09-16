"""Environment-backed configuration shared by generic MoE components."""

import os


def strict_fused_moe_enabled() -> bool:
    return os.environ.get("MOE_STRICT_FUSED", "1") != "0"


def shared_expert_mode() -> str:
    return os.environ.get("MOE_SHARED_EXPERT_MODE", "sequential").strip().lower()


def mega_moe_input_packer_mode() -> str:
    return os.environ.get("MEGA_MOE_INPUT_PACKER", "fused").strip().lower()
