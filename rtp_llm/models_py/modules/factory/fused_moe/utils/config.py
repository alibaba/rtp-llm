"""Environment-backed configuration shared by generic MoE components."""

import os


def strict_fused_moe_enabled() -> bool:
    return os.environ.get("MOE_STRICT_FUSED", "1") != "0"


def shared_expert_mode() -> str:
    return os.environ.get("MOE_SHARED_EXPERT_MODE", "sequential").strip().lower()


def mega_moe_input_packer_mode() -> str:
    return os.environ.get("MEGA_MOE_INPUT_PACKER", "fused").strip().lower()


def resolve_mega_moe_max_tokens_per_rank(default_capacity: int) -> int:
    """Resolve the communication-buffer capacity in input tokens per EP rank.

    The startup override is independent of scheduler limits. DeepGEMM applies
    its own alignment; execution still checks the allocated buffer's capacity.
    """
    name = "MEGA_MOE_MAX_TOKENS_PER_RANK"
    value = os.environ.get(name, "").strip()
    if not value:
        return max(int(default_capacity), 1)
    try:
        capacity = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if capacity <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return capacity
