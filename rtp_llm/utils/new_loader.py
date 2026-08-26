from typing import Optional, Protocol


class NewLoaderConfigSource(Protocol):
    use_new_loader: Optional[bool]


def is_new_loader_enabled(
    model_config: NewLoaderConfigSource, *, default_enabled: bool = False
) -> bool:
    """Resolve an explicit loader override against the model-specific default."""
    if not isinstance(default_enabled, bool):
        raise TypeError("default_enabled must be a bool")
    configured = model_config.use_new_loader
    if configured is None:
        return default_enabled
    if not isinstance(configured, bool):
        raise TypeError("model_config.use_new_loader must be a bool or None")
    return configured
