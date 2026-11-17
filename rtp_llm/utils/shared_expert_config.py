"""Shared-expert weight residency options for model loaders and execution."""


def shared_expert_weight_shard_enabled(model_config) -> bool:
    """Read the typed model option; callers enforce model and role support."""
    enabled = getattr(model_config, "shared_expert_weight_shard", False)
    if type(enabled) is not bool:
        raise ValueError("shared_expert_weight_shard must be a boolean")
    return enabled
