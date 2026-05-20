"""Env-gated tensor recorder for DSV4 forward bisection.

MOEDBG=0 (default): zero overhead, no allocations.
MOEDBG=1: per-call buffer, dumped at end of forward via `dump`.
MOEDBG_STREAM=1: compute hash/stats at record time instead of keeping
    large GPU clones until dump. Useful for 64k+ prefill bisection.

Usage:
    from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
    _rt.begin()                        # call at start of forward
    _rt.record("embed_out", h)         # callable from anywhere in the call tree
    ...
    _rt.dump(step=N, extra={...})      # call at end of forward

Output: $MOEDBG_DIR/$MOEDBG_CASE/rank{R}_pid{P}_step{N}.pt with structure:
    {"tensors": {name: cpu_tensor (full, only if numel <= _FULL_THRESHOLD)},
     "hashes":  {name: md5},
     "stats":   {name: {shape, dtype, mean, std, abs_max, n_nan, n_inf, numel}},
     "extra":   {cp_size, cp_rank, global_positions, ...}}
"""

import hashlib
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

_MOEDBG = int(os.environ.get("MOEDBG", "0"))
_DBG_BASE = os.environ.get("MOEDBG_DIR", "/tmp/moedbg_runs")
_DBG_CASE = os.environ.get("MOEDBG_CASE", "default")
_STREAM = int(os.environ.get("MOEDBG_STREAM", "0")) > 0
_FULL_THRESHOLD = int(os.environ.get("MOEDBG_FULL_THRESHOLD", "1000000"))
_TAIL_TOKENS = int(os.environ.get("MOEDBG_TAIL_TOKENS", "0"))
# Skip recording for forward passes whose seqlen exceeds this (warmup
# fires a single max_seq_len pass; recording it at MOEDBG=2 OOMs / hangs
# health checks before the real query arrives).  0 disables the gate.
_DBG_MAX_SEQ = int(os.environ.get("MOEDBG_MAX_SEQ", "0"))
_DBG_NAME_REGEX = os.environ.get("MOEDBG_NAME_REGEX", "")
_DBG_NAME_RE = re.compile(_DBG_NAME_REGEX) if _DBG_NAME_REGEX else None
_DBG_GLOBAL_POS = int(os.environ.get("MOEDBG_GLOBAL_POS", "-1"))

ENABLED = _MOEDBG > 0
LEVEL = _MOEDBG  # 1 = top-level only, 2 = top + per-layer detail

_local = threading.local()


def _get_buf() -> Optional[List[Tuple[str, torch.Tensor]]]:
    return getattr(_local, "buf", None)


def should_record_layer(layer_id: int) -> bool:
    """Layer filter for detailed traces.

    Default keeps the historical first-three-layer trace. Set
    MOEDBG_ALL_LAYERS=1 for full model, or MOEDBG_LAYER=17 / MOEDBG_LAYER=17,18
    to focus level-2 operator dumps after a level-1 bisection.
    """
    if not ENABLED:
        return False
    if _get_buf() is None:
        return False
    layer_filter = os.environ.get("MOEDBG_LAYER")
    if layer_filter:
        wanted = {int(x) for x in layer_filter.split(",") if x.strip()}
        return layer_id in wanted
    if os.environ.get("MOEDBG_ALL_LAYERS", "0") == "1":
        return True
    return layer_id <= 2


def should_record_positions(positions: Any) -> bool:
    """Forward-level gate for targeted debug dumps.

    When MOEDBG_GLOBAL_POS is set, skip forwards that do not contain that
    absolute position.  This keeps CUDA-graph warmup from writing debug
    tensors while still allowing the real decode step to be captured.
    """
    if not ENABLED:
        return False
    if _DBG_GLOBAL_POS < 0:
        return True
    if positions is None:
        return False
    try:
        if isinstance(positions, torch.Tensor):
            pos = positions.detach().to(dtype=torch.long).view(-1)
            return bool((pos == int(_DBG_GLOBAL_POS)).any().item())
        return int(positions) == int(_DBG_GLOBAL_POS)
    except Exception:
        return False


def _trim_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _TAIL_TOKENS <= 0:
        return tensor
    if tensor.dim() >= 2 and tensor.shape[0] <= 8 and tensor.shape[1] > _TAIL_TOKENS:
        return tensor[:, -_TAIL_TOKENS:].contiguous()
    if tensor.dim() >= 1 and tensor.shape[0] > _TAIL_TOKENS:
        return tensor[-_TAIL_TOKENS:].contiguous()
    return tensor


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
    if _DBG_NAME_RE is not None and _DBG_NAME_RE.search(name) is None:
        return
    buf = _get_buf()
    if buf is None:
        return
    tensor = _trim_tensor(tensor)
    if _STREAM:
        buf.append((name, _snapshot_cpu(tensor)))
    else:
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
        if isinstance(t, dict):
            hashes[name] = t["hash"]
            stats[name] = t["stats"]
            if t["tensor"] is not None:
                save[name] = t["tensor"]
            continue

        snap = _snapshot_cpu(t)
        hashes[name] = snap["hash"]
        stats[name] = snap["stats"]
        if snap["tensor"] is not None:
            save[name] = snap["tensor"]

    payload = {"tensors": save, "hashes": hashes, "stats": stats, "extra": extra or {}}
    fpath = f"{out_dir}/rank{rank}_pid{os.getpid()}_step{step:03d}.pt"
    torch.save(payload, fpath)
    print(f"[MOEDBG] dumped {fpath} tensors={len(buf)}", flush=True)

    _local.buf = []


def _snapshot_cpu(t: torch.Tensor) -> Dict[str, Any]:
    """Build the serialized representation for one tensor."""
    cpu_t = t.detach().cpu()
    stats_t = cpu_t.to(torch.float32)
    n = stats_t.numel()
    stats = {
        "shape": tuple(stats_t.shape),
        "dtype": str(t.dtype),
        "mean": stats_t.mean().item() if n > 0 else 0.0,
        "std": stats_t.std().item() if n > 1 else 0.0,
        "abs_max": stats_t.abs().max().item() if n > 0 else 0.0,
        "n_nan": int(torch.isnan(stats_t).sum().item()),
        "n_inf": int(torch.isinf(stats_t).sum().item()),
        "numel": n,
    }
    raw_bytes = cpu_t.reshape(-1).contiguous().view(torch.uint8)
    return {
        "hash": hashlib.md5(raw_bytes.numpy().tobytes()).hexdigest(),
        "stats": stats,
        "tensor": cpu_t.contiguous() if n <= _FULL_THRESHOLD else None,
    }
