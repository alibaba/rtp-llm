"""Small, grep-friendly timing helpers for the sleep/wake control path.

The sleep path crosses Python, C++ and the multi-rank gRPC controller.  Keep
the wire format here deliberately boring: one log line, stable key names, and
monotonic durations.  Callers can add rank/controller metadata through
``fields`` without having to duplicate formatting logic.
"""

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Callable, Iterator, Mapping, Optional, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class _TimingScope:
    operation: str
    started: float


_ACTIVE_SCOPE: ContextVar[_TimingScope | None] = ContextVar(
    "sleep_timing_scope", default=None
)


def _rank_fields(scope: str) -> dict[str, object]:
    """Best-effort rank identity for Python-side records.

    Importing ``collective_torch`` is intentionally deferred: this helper is
    also used by the frontend controller, where distributed Python state does
    not exist yet.  Environment fallbacks keep the records useful during
    bootstrap and in unit tests.
    """

    if scope != "python":
        return {}
    values: dict[str, object] = {}
    try:
        from rtp_llm.models_py.distributed import collective_torch

        config = getattr(collective_torch, "_parallelism_config", None)
        if config is not None:
            for name in ("world_rank", "local_rank", "dp_rank", "tp_rank", "ep_rank"):
                value = getattr(config, name, None)
                if value is not None:
                    values[name] = value
    except Exception:
        pass
    env_names = {
        "world_rank": "RANK",
        "local_rank": "LOCAL_RANK",
        "dp_rank": "DP_RANK",
        "tp_rank": "TP_RANK",
        "ep_rank": "EP_RANK",
    }
    for name, env_name in env_names.items():
        if name not in values and os.environ.get(env_name) is not None:
            values[name] = os.environ[env_name]
    return values


def log_sleep_timing(
    operation: str,
    phase: str,
    elapsed_ms: float,
    *,
    total_ms: Optional[float] = None,
    status: str = "ok",
    scope: str = "python",
    fields: Optional[Mapping[str, object]] = None,
) -> None:
    """Emit one standardized sleep/wake timing record.

    ``fields`` is sorted to make records deterministic and is restricted to
    single-line values so a memory/error detail cannot corrupt grep output.
    """

    values = {
        "op": operation,
        "scope": scope,
        "phase": phase,
        "status": status,
        "elapsed_ms": f"{elapsed_ms:.3f}",
    }
    if total_ms is not None:
        values["total_ms"] = f"{total_ms:.3f}"
    merged_fields = _rank_fields(scope)
    if fields:
        merged_fields.update(fields)
    if merged_fields:
        for key, value in sorted(merged_fields.items()):
            text = str(value).replace("\n", " ").replace("\r", " ").replace(" ", "_")
            values[str(key)] = text
    logging.info(
        "[SleepTiming] %s",
        " ".join(f"{key}={value}" for key, value in values.items()),
    )


@contextmanager
def timed_sleep_phase(
    operation: str,
    phase: str,
    *,
    scope: str = "python",
    fields: Optional[Mapping[str, object]] = None,
) -> Iterator[None]:
    """Time a phase and emit an ``ok``/``error`` record on every exit path.

    Nested phases of the same operation share one monotonic start time through
    ``ContextVar``.  That keeps ``total_ms`` an actual operation elapsed time
    while retaining the phase-local ``elapsed_ms`` value.
    """

    active_scope = _ACTIVE_SCOPE.get()
    scope_token = None
    if active_scope is None or active_scope.operation != operation:
        active_scope = _TimingScope(operation=operation, started=perf_counter())
        scope_token = _ACTIVE_SCOPE.set(active_scope)
    phase_started = perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        now = perf_counter()
        log_sleep_timing(
            operation,
            phase,
            (now - phase_started) * 1000.0,
            total_ms=(now - active_scope.started) * 1000.0,
            status=status,
            scope=scope,
            fields=fields,
        )
        if scope_token is not None:
            _ACTIVE_SCOPE.reset(scope_token)


def timed_sleep_method(
    operation: str, phase: str
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a synchronous hook method with the same timing record."""

    def decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with timed_sleep_phase(operation, phase):
                return func(*args, **kwargs)

        return wrapped

    return decorate
