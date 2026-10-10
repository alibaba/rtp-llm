"""Shared environment parsing for DSV4 long-context chunk knobs."""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional

DSV4_CHUNK_TOKENS_ENV = "DSV4_CHUNK_TOKENS"
DEFAULT_DSV4_CHUNK_TOKENS = 16384


def chunked_moe_enabled(options=None) -> bool:
    if dsv4_global_chunk_tokens_configured(options):
        return moe_chunk_tokens_from_env(options=options) > 0
    source = os.environ if options is None else options
    return source.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"


def moe_chunk_tokens_from_env(
    default: int = DEFAULT_DSV4_CHUNK_TOKENS, *, options=None
) -> int:
    min_value = 0 if dsv4_global_chunk_tokens_configured(options) else 1
    return dsv4_chunk_tokens_from_env(
        "DSV4_MOE_CHUNK_TOKENS", default, min_value=min_value, options=options
    )


def dsv4_global_chunk_tokens_configured(options=None) -> bool:
    return DSV4_CHUNK_TOKENS_ENV in (os.environ if options is None else options)


def dsv4_chunk_tokens_from_env(
    specific_env: str,
    default: int = DEFAULT_DSV4_CHUNK_TOKENS,
    *,
    min_value: int = 0,
    options: Optional[Mapping[str, str]] = None,
) -> int:
    """Read a DSV4 chunk size with ``DSV4_CHUNK_TOKENS`` as override.

    When configured, ``DSV4_CHUNK_TOKENS`` wins over the caller's historical
    env.  Otherwise the historical per-path env is used so existing scripts
    keep their behavior.
    """
    source = os.environ if options is None else options
    env_name = DSV4_CHUNK_TOKENS_ENV
    raw_value = source.get(env_name)
    if raw_value is None:
        env_name = specific_env
        raw_value = source.get(specific_env, str(default))

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
