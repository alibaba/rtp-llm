"""Runtime switch surface for DSV4 modules (strangler pattern).

One entry point for NEW DSV4 runtime switches.  It exists to stop three
recurring deployment hazards:

1. **Silent no-op switches.**  A reader that exists in the source tree but
   not on the import path the wheel actually loads is indistinguishable
   from a working one (the ``package-root/internal_source`` empty-switch
   finding in ROADMAP 3.1).  Every switch resolved here prints exactly one
   ``[DSV4_CONFIG] name=value`` audit line when first consumed, so the
   engine log proves which switches the running process really saw.
2. **Typo'd values discovered mid-run.**  Validation is fail-loud: the
   consuming module resolves its switches at import time, so an invalid
   value aborts at process start with a readable message instead of
   poisoning a decode arm hours later.
3. **Switch sprawl.**  The ~85 existing ad-hoc ``os.environ`` readers are
   deliberately NOT migrated (strangler pattern): a reader moves onto this
   surface only when it is touched for other reasons.

Design mirrors ``chunk_env.py``: stdlib only (``os`` + ``logging``), one
env read per switch per process, no re-parsing on the hot path, no
third-party dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "get_switch",
    "register_switch",
    "parse_bool",
    "parse_nonneg_int",
    "parse_percentile_list",
]

# name -> (default, validator). Registration validates at import time even
# if nobody consumes the switch on this path yet.
_META: Dict[str, Tuple[Any, Optional[Callable[[str, Any], Any]]]] = {}
# name -> resolved value.  The env is read exactly once per process; later
# get_switch calls return the cached value (see chunk_env.py philosophy).
_VALUES: Dict[str, Any] = {}

_TRUTHY = ("1", "true", "on", "yes")
_FALSY = ("0", "false", "off", "no")


def _resolve(name: str, default: Any, validator: Any) -> Any:
    if name in _VALUES:
        return _VALUES[name]
    raw = os.environ.get(name)
    source = "default"
    if raw is None:
        value = default
    else:
        source = "env"
        try:
            value = validator(raw, default) if validator is not None else raw
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"invalid {name}={raw!r}: {e}.  Set a valid value or unset "
                f"it to fall back to {default!r}."
            ) from e
    _VALUES[name] = value
    _META.setdefault(name, (default, validator))
    # One audit line per switch, ever: _resolve only runs before caching.
    logging.info("[DSV4_CONFIG] %s=%r (%s)", name, value, source)
    return value


def register_switch(name: str, default: Any, validator: Any = None) -> Any:
    """Declare a switch and validate it NOW (fail-loud at import time).

    Returns the resolved value so callers can bind it to a module constant
    in one line, exactly like ``get_switch``.
    """
    return _resolve(name, default, validator)


def get_switch(name: str, default: Any = None, validator: Any = None) -> Any:
    """Resolve a switch: one env read, fail-loud validation, one audit line.

    ``default=None`` on an unregistered name is treated as a typo: an
    unknown switch would silently read ``None`` downstream, which is the
    failure mode this module exists to prevent.

    On a repeat consumption of an already-resolved switch, a differently
    supplied ``default``/``validator`` is ignored: the first resolution wins
    (registration-time semantics), so a switch has exactly one canonical
    value per process.
    """
    if name in _VALUES:
        return _VALUES[name]
    if name in _META:
        default, validator = _META[name]
        return _resolve(name, default, validator)
    if default is None:
        raise KeyError(
            f"unknown runtime switch {name!r}: register it with "
            f"register_switch(name, default, validator) at import time, or "
            f"pass an explicit default to get_switch()"
        )
    return _resolve(name, default, validator)


def parse_bool(raw: str, default: Any) -> bool:
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ValueError(f"expected one of {sorted(_TRUTHY + _FALSY)}")


def parse_nonneg_int(raw: str, default: Any) -> int:
    v = int(raw.strip())
    if v < 0:
        raise ValueError("expected a non-negative integer")
    return v


def parse_percentile_list(raw: str, default: Any) -> Tuple[int, ...]:
    """Parse ``"p50,p99"`` (or ``"50,99"``) into a tuple of ints.

    Empty/whitespace-only means "off" and returns ``()`` — the current
    behaviour of every caller, so unset == no percentiles.
    """
    text = raw.strip()
    if not text:
        return ()
    out = []
    for part in text.split(","):
        token = part.strip().lower()
        if token.startswith("p"):
            token = token[1:]
        if not token.isdigit():
            raise ValueError(f"bad percentile {part!r} (expected p0..p100)")
        pct = int(token)
        if not 0 <= pct <= 100:
            raise ValueError(f"percentile {part!r} outside 0..100")
        out.append(pct)
    return tuple(out)


def _reset_for_tests() -> None:
    """Clear the one-shot caches.  Tests only; production never calls this."""
    _META.clear()
    _VALUES.clear()
