"""Tensor dump utility for diagnosing fuse kernel precision.

When env var DUMP_FUSE_TENSORS is set to a directory path, each fuse
kernel call site runs BOTH the fused and unfused paths (on cloned inputs
for the unfused path) and saves the output tensors.  Only the first
forward pass is dumped (controlled by _seq counter).

Usage:
    DUMP_FUSE_TENSORS=precision_test/tensor_dumps \
        ENABLE_GRAPHFX_FUSION=1 bash py_start_glm5.sh
    # send ONE request, then kill server
    python precision_test/compare_tensors.py precision_test/tensor_dumps
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_DUMP_DIR: Optional[str] = os.environ.get("DUMP_FUSE_TENSORS")
_seq: int = 0
_MAX: int = 50


def fuse_dump_active() -> bool:
    return _DUMP_DIR is not None and _seq < _MAX


def dump_fuse_tensors(
    kernel_name: str,
    layer_idx: int,
    fused: dict[str, torch.Tensor],
    unfused: dict[str, torch.Tensor],
) -> None:
    global _seq
    if _DUMP_DIR is None or _seq >= _MAX:
        return
    idx = _seq
    _seq += 1
    d = os.path.join(_DUMP_DIR, f"{idx:03d}_L{layer_idx}_{kernel_name}")
    os.makedirs(d, exist_ok=True)
    for k, v in fused.items():
        if v is not None:
            torch.save(v.detach().cpu().clone(), os.path.join(d, f"fused_{k}.pt"))
    for k, v in unfused.items():
        if v is not None:
            torch.save(v.detach().cpu().clone(), os.path.join(d, f"unfused_{k}.pt"))
