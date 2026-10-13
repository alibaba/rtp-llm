"""Process-wide warmup switches.

``WARM_UP`` is the global master switch.  ``MODEL_WARM_UP`` controls model
initialization and other model-side warmups, and is effective only when the
global switch is enabled.

CUDA graph capture has its own internal preparation forwards and deliberately
does not use this module.
"""

from __future__ import annotations

import os
from typing import Any

WARM_UP_ENV = "WARM_UP"
MODEL_WARM_UP_ENV = "MODEL_WARM_UP"

_FALSE_VALUES = frozenset(("0", "false", "off", "no", "n", "f"))


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSE_VALUES


def _env_bool(name: str, default: bool = True) -> bool:
    return _as_bool(os.environ.get(name), default)


def configure_warmup(warm_up: bool, model_warm_up: bool) -> None:
    """Publish parsed switches for model construction and child processes."""
    os.environ[WARM_UP_ENV] = "1" if _as_bool(warm_up) else "0"
    os.environ[MODEL_WARM_UP_ENV] = "1" if _as_bool(model_warm_up) else "0"


def global_warm_up_enabled() -> bool:
    """Return the global warmup master switch (default: enabled)."""
    return _env_bool(WARM_UP_ENV, True)


def model_warm_up_enabled() -> bool:
    """Return the effective model-side warmup switch."""
    return global_warm_up_enabled() and _env_bool(MODEL_WARM_UP_ENV, True)
