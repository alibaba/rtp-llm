"""Env-gated tensor recorder for DSV4 forward bisection.

MOEDBG=0 (default): zero overhead, no allocations.
MOEDBG=1: per-call buffer, dumped at end of forward via `dump`.

Usage:
    from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
    _rt.begin()                        # call at start of forward
    _rt.record("embed_out", h)         # callable from anywhere in the call tree
    ...
    _rt.dump(step=N, extra={...})      # call at end of forward

Output: $MOEDBG_DIR/$MOEDBG_CASE/rank{R}_step{N}.pt with structure:
    {"tensors": {name: cpu_tensor (full, only if numel <= _FULL_THRESHOLD)},
     "hashes":  {name: md5},
     "stats":   {name: {shape, dtype, mean, std, abs_max, n_nan, n_inf, numel}},
     "extra":   {cp_size, cp_rank, global_positions, ...}}
"""

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

_MOEDBG = int(os.environ.get("MOEDBG", "0"))
_DBG_BASE = os.environ.get("MOEDBG_DIR", "/tmp/moedbg_runs")
_DBG_CASE = os.environ.get("MOEDBG_CASE", "default")
_FULL_THRESHOLD = 1_000_000
# Skip recording for forward passes whose seqlen exceeds this (warmup
# fires a single max_seq_len pass; recording it at MOEDBG=2 OOMs / hangs
# health checks before the real query arrives).  0 disables the gate.
_DBG_MAX_SEQ = int(os.environ.get("MOEDBG_MAX_SEQ", "0"))

ENABLED = _MOEDBG > 0
LEVEL = _MOEDBG  # 1 = top-level only, 2 = top + per-layer detail

_local = threading.local()


def _get_buf() -> Optional[List[Tuple[str, torch.Tensor]]]:
    return getattr(_local, "buf", None)


def begin(seqlen: Optional[int] = None) -> None:
    """Start a new forward; clears any prior buffer.

    ``seqlen`` (optional) lets the caller suppress recording for a
    specific forward when ``MOEDBG_MAX_SEQ`` is set — used to skip the
    warmup query so MOEDBG=2 doesn't OOM under long sequences.
    """
    if not ENABLED:
        return
    if _DBG_MAX_SEQ > 0 and seqlen is not None and seqlen > _DBG_MAX_SEQ:
        _local.buf = None
        return
    _local.buf = []


def record(name: str, tensor: torch.Tensor) -> None:
    """Snapshot a tensor for this forward. No-op if MOEDBG=0 or no active buf."""
    if not ENABLED:
        return
    buf = _get_buf()
    if buf is None:
        return
    buf.append((name, tensor.detach().clone()))


def record_if_level(level: int, name: str, tensor: torch.Tensor) -> None:
    """Record only if MOEDBG >= level. Use level=2 for fine-grained per-layer detail."""
    if not ENABLED or LEVEL < level:
        return
    record(name, tensor)


def dump(*, step: int, extra: Optional[Dict[str, Any]] = None) -> None:
    """Sync, serialize the current buffer, and write to disk."""
    if not ENABLED:
        return
    buf = _get_buf()
    if not buf:
        return
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    out_dir = f"{_DBG_BASE}/{_DBG_CASE}"
    os.makedirs(out_dir, exist_ok=True)

    torch.cuda.synchronize()

    save: Dict[str, torch.Tensor] = {}
    hashes: Dict[str, str] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    for name, t in buf:
        cpu_t = t.detach().to(torch.float32).cpu()
        n = cpu_t.numel()
        hashes[name] = hashlib.md5(cpu_t.numpy().tobytes()).hexdigest()
        stats[name] = {
            "shape": tuple(cpu_t.shape),
            "dtype": str(t.dtype),
            "mean": cpu_t.mean().item() if n > 0 else 0.0,
            "std": cpu_t.std().item() if n > 1 else 0.0,
            "abs_max": cpu_t.abs().max().item() if n > 0 else 0.0,
            "n_nan": int(torch.isnan(cpu_t).sum().item()),
            "n_inf": int(torch.isinf(cpu_t).sum().item()),
            "numel": n,
        }
        if n <= _FULL_THRESHOLD:
            save[name] = cpu_t

    payload = {"tensors": save, "hashes": hashes, "stats": stats, "extra": extra or {}}
    fpath = f"{out_dir}/rank{rank}_step{step:03d}.pt"
    torch.save(payload, fpath)

    _local.buf = []
