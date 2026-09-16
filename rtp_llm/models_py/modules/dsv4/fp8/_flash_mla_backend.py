"""Select the sparse-prefill FlashMLA implementation on PPU."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_BACKEND_ENV = "DSV4_PPU_FLASH_MLA_BACKEND"


def _is_m890p() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_name(torch.cuda.current_device()) == "ZW-M890P"
    except RuntimeError:
        return False


@lru_cache(maxsize=1)
def get_flash_mla_sparse_fwd() -> Callable:
    """Return the selected sparse-prefill callable.

    M890P defaults to ``sinian``; other devices default to ``runtime``.
    ``auto`` explicitly prefers the SinianOps PPU wheel on M890P and falls back to the
    runtime FlashMLA package when the wheel is absent. ``sinian`` makes a
    missing or incompatible wheel fatal; ``runtime`` is the rollback arm.
    """

    requested = os.environ.get(_BACKEND_ENV)
    if requested is None:
        requested = "sinian" if _is_m890p() else "runtime"
    requested = requested.strip().lower()
    if requested not in ("auto", "sinian", "runtime"):
        raise ValueError(
            f"invalid {_BACKEND_ENV}={requested!r}; expected auto, sinian, or runtime"
        )

    if requested == "sinian" or (requested == "auto" and _is_m890p()):
        try:
            from sinian_ops.flash_mla import flash_mla_sparse_fwd

            logger.info("DSV4_FLASH_MLA_BACKEND backend=sinian_ops")
            return flash_mla_sparse_fwd
        except (ImportError, OSError):
            if requested == "sinian":
                raise
            logger.info(
                "SinianOps FlashMLA is unavailable; using runtime FlashMLA",
                exc_info=True,
            )

    from flash_mla import flash_mla_sparse_fwd

    logger.info("DSV4_FLASH_MLA_BACKEND backend=runtime")
    return flash_mla_sparse_fwd


__all__ = ["get_flash_mla_sparse_fwd"]
