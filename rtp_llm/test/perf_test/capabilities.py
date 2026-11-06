"""Central capability matrix for unified perf-test requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional, Tuple

from rtp_llm.test.perf_test.errors import CapabilityError

MODES: FrozenSet[str] = frozenset({"prefill", "decode"})
ACTIONS: FrozenSet[str] = frozenset({"run", "resume", "profile", "report"})
BACKENDS: FrozenSet[str] = frozenset({"local"})

_SUPPORTED: FrozenSet[Tuple[str, str, str]] = frozenset(
    {
        ("prefill", "local", "run"),
        ("prefill", "local", "resume"),
        ("prefill", "local", "profile"),
        ("prefill", "local", "report"),
        ("decode", "local", "run"),
    }
)


@dataclass(frozen=True)
class Capability:
    mode: str
    backend: str
    action: str
    offline_only: bool = False

    @property
    def key(self) -> Tuple[str, str, str]:
        return self.mode, self.backend, self.action


CapabilityValidator = Callable[[str, str, str], Capability]


def _validate_value(path: str, value: str, allowed: FrozenSet[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise CapabilityError(f"{path}: expected one of: {choices}; got {value!r}")


def validate_capability(
    mode: str,
    action: str,
    backend: str,
    *,
    backend_validator: Optional[CapabilityValidator] = None,
) -> Capability:
    """Validate a request before any filesystem or execution side effect."""
    _validate_value("mode", mode, MODES)
    _validate_value("action", action, ACTIONS)
    if backend_validator is not None:
        return backend_validator(mode, action, backend)
    _validate_value("backend", backend, BACKENDS)
    key = (mode, backend, action)
    if key not in _SUPPORTED:
        raise CapabilityError(
            f"unsupported capability: mode={mode}, action={action}, backend={backend}"
        )
    return Capability(
        mode=mode,
        backend=backend,
        action=action,
        offline_only=False,
    )
