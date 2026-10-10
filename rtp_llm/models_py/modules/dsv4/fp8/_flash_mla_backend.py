"""Resolve the sparse-prefill FlashMLA implementation from the runtime stack."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable


@lru_cache(maxsize=1)
def get_flash_mla_sparse_fwd() -> Callable:
    from flash_mla import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd


__all__ = ["get_flash_mla_sparse_fwd"]
