"""Data-only cache experiment report configuration shared by loader and CLI."""


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("cache comparison policy must be a mapping")
    extra = set(policy) - {"comparison", "alignment_event"}
    if extra:
        raise ValueError(f"unknown cache comparison fields: {sorted(extra)}")
    if policy.get("comparison") != "cache_scale_in":
        raise ValueError("comparison must be cache_scale_in")
    event = policy.get("alignment_event")
    if event is not None and (not isinstance(event, str) or not event.strip()):
        raise ValueError("alignment_event must be a nonempty event name or null")
    return dict(policy)
