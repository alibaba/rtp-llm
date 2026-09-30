"""Shared, CUDA-free interpretation of sleep startup and reclaim settings."""

import os
from typing import Optional

LEVEL_ENV = "SLEEP_MODE_LEVEL"
ENABLE_ENV = "ENABLE_SLEEP_MODE"
RUNTIME_CACHES_ENV = "RTP_LLM_SLEEP_FREE_RUNTIME_CACHES"
LEGACY_RUNTIME_CACHES_ENV = "RTP_LLM_SLEEP_FREE_MEGA_SYMM"
COLLECTIVE_MEMORY_ENV = "SLEEP_RELEASE_COLLECTIVE_MEMORY"


def resolve_sleep_level(level: Optional[int], enabled: Optional[bool]) -> int:
    """Level is canonical; an enable-only legacy configuration selects level 1."""
    if level is None:
        return 1 if enabled else 0
    if type(level) is not int or level not in (0, 1, 2):
        raise ValueError("SLEEP_MODE_LEVEL must be 0 (disabled), 1, or 2")
    if enabled is not None and enabled != (level > 0):
        raise ValueError(
            "ENABLE_SLEEP_MODE conflicts with SLEEP_MODE_LEVEL; "
            "remove ENABLE_SLEEP_MODE and select 0, 1, or 2 with SLEEP_MODE_LEVEL"
        )
    return level


def sleep_level_from_env() -> int:
    """Also support isolated weight-loader callers before server argument setup."""
    level = os.environ.get(LEVEL_ENV)
    enabled = os.environ.get(ENABLE_ENV)
    return resolve_sleep_level(
        int(level) if level is not None else None,
        enabled == "1" if enabled is not None else None,
    )


def resource_release_enabled(
    name: str, *, default: Optional[bool] = None, legacy_alias: Optional[str] = None
) -> bool:
    """Unset follows sleep; explicit zero wins even over an enabled old alias.

    This is the requested policy, not a capability/safety decision. Resource
    owners must still apply their CUDA-graph and collective-runtime interlocks.
    Startup parsing validates/normalizes bool strings before process spawn.
    """
    value = os.environ.get(name)
    if value is None and legacy_alias is not None:
        value = os.environ.get(legacy_alias)
    if value is not None:
        return value == "1"
    return sleep_level_from_env() > 0 if default is None else default
