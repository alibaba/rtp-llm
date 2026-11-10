from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import Pipeline

__all__ = ["Pipeline"]


def __getattr__(name: str) -> Any:
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
