"""Shared environment parsing for DSV4 long-context chunk knobs."""

from __future__ import annotations

import logging
import os

DSV4_CHUNK_TOKENS_ENV = "DSV4_CHUNK_TOKENS"
DEFAULT_DSV4_CHUNK_TOKENS = 16384


def chunked_moe_enabled() -> bool:
    if dsv4_global_chunk_tokens_configured():
        return moe_chunk_tokens_from_env() > 0
    return os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"


def moe_chunk_tokens_from_env(default: int = DEFAULT_DSV4_CHUNK_TOKENS) -> int:
    min_value = 0 if dsv4_global_chunk_tokens_configured() else 1
    return dsv4_chunk_tokens_from_env(
        "DSV4_MOE_CHUNK_TOKENS", default, min_value=min_value
    )


def dsv4_global_chunk_tokens_configured() -> bool:
    return DSV4_CHUNK_TOKENS_ENV in os.environ


def dsv4_chunk_tokens_from_env(
    specific_env: str,
    default: int = DEFAULT_DSV4_CHUNK_TOKENS,
    *,
    min_value: int = 0,
) -> int:
    """Read a DSV4 chunk size with ``DSV4_CHUNK_TOKENS`` as override.

    When configured, ``DSV4_CHUNK_TOKENS`` wins over the caller's historical
    env.  Otherwise the historical per-path env is used so existing scripts
    keep their behavior.
    """
    env_name = DSV4_CHUNK_TOKENS_ENV
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        env_name = specific_env
        raw_value = os.environ.get(specific_env, str(default))

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logging.warning(
            "[DSV4] invalid %s=%r; using default=%d",
            env_name,
            raw_value,
            default,
        )
        value = default
    return max(value, int(min_value))
