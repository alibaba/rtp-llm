from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_benchmark_workspace_scope: ContextVar[bool] = ContextVar(
    "benchmark_workspace_scope", default=False
)


def in_benchmark_workspace_scope() -> bool:
    return _benchmark_workspace_scope.get()


@contextmanager
def benchmark_workspace_scope() -> Iterator[None]:
    token = _benchmark_workspace_scope.set(True)
    try:
        yield
    finally:
        _benchmark_workspace_scope.reset(token)
