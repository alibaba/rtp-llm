"""Shared-expert weight residency options for model loaders and execution."""

import os
from typing import Optional


def shared_expert_weight_shard_enabled(
    *, legacy_env_name: Optional[str] = None
) -> bool:
    """Resolve the common option; callers enforce model and role support."""
    env_name = "SHARED_EXPERT_WEIGHT_SHARD"
    if env_name not in os.environ and legacy_env_name is not None:
        env_name = legacy_env_name
    raw = os.environ.get(env_name, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{env_name} must be 0 or 1, got {raw!r}")
    return raw == "1"
