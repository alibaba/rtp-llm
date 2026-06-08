import importlib
from typing import Callable


def resolve_func(dotted_path: str) -> Callable:
    module_path, _, func_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid dotted path '{dotted_path}': must contain at least one dot"
        )
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{func_name}'"
        )
    if not callable(func):
        raise TypeError(
            f"'{dotted_path}' resolved to {type(func).__name__}, not a callable"
        )
    return func
