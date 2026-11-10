import os
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

PROFILE_STAGES = os.environ.get("RTP_LLM_CP_PROFILE_STAGES") == "1"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def profile_stage(
    name: str, *, layer_arg: int | None = None, layer_attr: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        if not PROFILE_STAGES:
            return function
        from torch.profiler import record_function

        @wraps(function)
        def recorded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            label = name
            if layer_arg is not None:
                layer = args[layer_arg] if len(args) > layer_arg else kwargs["layer"]
                label = f"{name}(layer={layer})"
            elif layer_attr is not None:
                label = f"{name}(layer={getattr(args[0], layer_attr)})"
            with record_function(label):
                return function(*args, **kwargs)

        return recorded

    return decorate
