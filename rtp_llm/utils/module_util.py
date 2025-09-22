import functools
import importlib.util
from types import ModuleType
from typing import Any, Callable


@functools.cache
def has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


def resolve_symbol(module: ModuleType, new: str, old: str) -> Callable[..., Any] | None:
    """Return the *new* symbol if it exists, otherwise the *old* one."""
    if hasattr(module, new):
        return getattr(module, new)
    if hasattr(module, old):
        print(
            "Found legacy symbol `%s`. Please upgrade the package "
            "so that `%s` is available. Support for the legacy symbol "
            "will be removed in a future release.",
            old,
            new,
        )
        return getattr(module, old)
    return None
