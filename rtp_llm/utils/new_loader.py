from typing import Protocol


class NewLoaderConfigSource(Protocol):
    use_new_loader: bool


def is_new_loader_enabled(model_config: NewLoaderConfigSource) -> bool:
    """Return the single process/model decision used by all loader entry points."""
    configured = model_config.use_new_loader
    if not isinstance(configured, bool):
        raise TypeError("model_config.use_new_loader must be a bool")
    return configured
