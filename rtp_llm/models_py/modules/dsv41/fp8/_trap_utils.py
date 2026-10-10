"""Device-trap and backend-selection flags for DSV4 FP8 cache access."""

from __future__ import annotations

import os

DSV4_TRAP_INVALID_KV_ACCESS_ENV = "DSV4_TRAP_INVALID_KV_ACCESS"
DSV4_VALIDATE_INVALID_KV_ACCESS_ENV = "DSV4_VALIDATE_INVALID_KV_ACCESS"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def trap_invalid_kv_access_enabled() -> bool:
    return _env_flag(DSV4_TRAP_INVALID_KV_ACCESS_ENV, True)


def invalid_kv_access_validation_enabled() -> bool:
    return _env_flag(DSV4_VALIDATE_INVALID_KV_ACCESS_ENV, False)
