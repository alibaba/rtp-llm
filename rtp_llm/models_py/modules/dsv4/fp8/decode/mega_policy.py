"""Deployment policy shared by the legacy and direct-parameter V4 constructors."""

import logging
import os
from typing import Optional

_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_ENV_VALUES = frozenset(("0", "false", "no", "off", ""))


def _optional_env_bool(name: str) -> Optional[bool]:
    if name not in os.environ:
        return None
    value = os.environ[name].strip().lower()
    if value in _TRUE_ENV_VALUES:
        return True
    if value in _FALSE_ENV_VALUES:
        return False
    raise RuntimeError(
        f"{name} must be one of {sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, "
        f"got {os.environ[name]!r}"
    )


def mega_decode_enabled(args, device) -> bool:
    # SWA-only drafts have no Mega components even when deployment forces Mega.
    request = _optional_env_bool("DSV4_MEGA")
    if request is False or not any(args.compress_ratios[: args.n_layers]):
        return False
    from .mega_support import mega_decode_unavailable_reason

    reason = mega_decode_unavailable_reason(args, device)
    if reason is not None:
        if request is True:
            raise RuntimeError(f"DSV4_MEGA=1 is unsupported: {reason}")
        logging.info(
            "DSV4 Mega decode is unavailable; using the ordinary path: %s", reason
        )
        return False
    logging.info(
        "DSV4 Mega attention decode enabled on %s (TP1, dim=%d); set DSV4_MEGA=0 to disable",
        device,
        args.dim,
    )
    return True
