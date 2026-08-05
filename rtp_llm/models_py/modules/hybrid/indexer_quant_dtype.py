"""Shared INDEXER_QUANT_DTYPE resolution for model config and Indexer runtime."""

import os
from typing import Tuple

import torch

INDEXER_QUANT_DTYPE_ENV = "INDEXER_QUANT_DTYPE"
_SUPPORTED_INDEXER_QUANT_DTYPES: Tuple[str, ...] = ("fp8", "fp4")
_BLACKWELL_MIN_CC = (10, 0)


def resolve_indexer_quant_dtype_from_env(default: str = "fp8") -> str:
    """Read ``INDEXER_QUANT_DTYPE`` once at model-config time."""
    raw = os.environ.get(INDEXER_QUANT_DTYPE_ENV, default).strip().lower()
    if raw not in _SUPPORTED_INDEXER_QUANT_DTYPES:
        raise ValueError(
            f"{INDEXER_QUANT_DTYPE_ENV}={raw!r} not supported; "
            f"expected one of {_SUPPORTED_INDEXER_QUANT_DTYPES}"
        )
    return raw


def validate_indexer_quant_dtype(quant_dtype: str) -> str:
    """Validate ``attn_config.indexer_quant_dtype`` at Indexer init."""
    raw = (quant_dtype or "fp8").strip().lower()
    if raw not in _SUPPORTED_INDEXER_QUANT_DTYPES:
        raise ValueError(
            f"indexer_quant_dtype={raw!r} not supported; "
            f"expected one of {_SUPPORTED_INDEXER_QUANT_DTYPES}"
        )
    if raw == "fp4":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "indexer_quant_dtype=fp4 requires CUDA; no CUDA device available"
            )
        cc = torch.cuda.get_device_capability()
        if cc < _BLACKWELL_MIN_CC:
            raise RuntimeError(
                f"indexer_quant_dtype=fp4 requires Blackwell (SM>=10.0); "
                f"current device capability {cc}"
            )
    return raw
