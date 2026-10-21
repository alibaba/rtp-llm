"""Experimental HY4 Q/KV norm + Q MXFP8 producer; allocate before capture."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class QKVNormOutputs:
    q_fp8: torch.Tensor
    q_scale: torch.Tensor
    q_bf16: torch.Tensor | None
    kv_bf16: torch.Tensor
    q_rank: int
    kv_rank: int


@lru_cache(maxsize=1)
def prepare_extension():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_extension must run before graph capture")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    sources = [root / "cuda" / name for name in ("qkv_norm.cpp", "qkv_norm.cu")]
    flags = ["-O3", "-std=c++17", "-lineinfo", "--fmad=false", "--ftz=false",
             "-gencode=arch=compute_100,code=sm_100",
             "-gencode=arch=compute_103,code=sm_103",
             "-gencode=arch=compute_100,code=compute_100"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources + [root / "cuda/qkv_norm.cuh"])
                            + repr(flags).encode() + torch.__version__.encode()).hexdigest()[:16]
    name = "hy4_qkv_norm_" + digest
    from torch.utils.cpp_extension import _get_build_directory
    build = Path(_get_build_directory(name, verbose=False))
    build.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(p) for p in sources], build_directory=str(build),
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags, verbose=True)


def _validate_input(qkv, q_rank, kv_rank):
    if (qkv.ndim != 2 or not qkv.is_cuda or qkv.dtype != torch.bfloat16
            or qkv.stride(1) != 1 or qkv.stride(0) < qkv.shape[1]):
        raise ValueError("qkv must be nonoverlapping CUDA BF16 rows with unit inner stride")
    if not 1 <= qkv.shape[0] <= 256:
        raise ValueError("M must be in 1..256")
    if q_rank not in (128, 256, 512, 1024, 2048, 4096) or kv_rank not in (128, 256, 512, 1024):
        raise ValueError("supported power-of-two Q ranks 128..4096, KV ranks 128..1024")
    if qkv.shape[1] < q_rank + kv_rank:
        raise ValueError("qkv width must include Q and KV; remaining columns are untouched k_pe")


def allocate_outputs(qkv, q_rank=2048, kv_rank=512, need_q_bf16=True):
    _validate_input(qkv, q_rank, kv_rank)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("allocate_outputs must run outside graph capture")
    import deep_gemm

    m = qkv.shape[0]
    aligned_m = deep_gemm.get_tma_aligned_size(m, 4)
    scale = torch.empty((q_rank // 128, aligned_m), device=qkv.device, dtype=torch.int32).T[:m]
    return QKVNormOutputs(
        torch.empty((m, q_rank), device=qkv.device, dtype=torch.float8_e4m3fn), scale,
        torch.empty((m, q_rank), device=qkv.device, dtype=torch.bfloat16) if need_q_bf16 else None,
        torch.empty((m, kv_rank), device=qkv.device, dtype=torch.bfloat16), q_rank, kv_rank,
    )


def validate_cache_metadata(mla_cache, slots, positions, cos_sin):
    """Explicit request-preparation check; call after each metadata update, outside capture.

    Launch checks layouts without synchronizing. Unique live slots and in-range
    values are a caller contract for every replay, including after preparation.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("validate_cache_metadata must run outside capture")
    capacity = mla_cache.shape[0] * (mla_cache.shape[1] if mla_cache.ndim == 3 else 1)
    if bool(((slots < -1) | (slots >= capacity)).any().item()):
        raise ValueError("slots must be -1 or an in-range cache slot")
    live = slots[slots >= 0]
    if live.unique().numel() != live.numel():
        raise ValueError("nonnegative slots must be unique")
    if bool(((positions < 0) | (positions >= cos_sin.shape[0])).any().item()):
        raise ValueError("positions outside cos_sin bounds")


def qkv_norm_quant(qkv, q_weight, kv_weight, *, out, q_eps, kv_eps, pdl=False, notify_event=None,
                   write_kv_cache=False, mla_cache=None, slots=None, positions=None, cos_sin=None,
                   release="late"):
    """Early release grants launch permission, never output completion.

    after_wait releases after the input GDC wait and before dynamic input loads.
    All dependent consumers must execute GDC before reading outputs. With
    pdl=False all release values use the unchanged ordinary launch path.
    """
    if release not in ("late", "early", "after_wait"):
        raise ValueError("release must be late, early or after_wait")
    _validate_input(qkv, out.q_rank, out.kv_rank)
    if not all(math.isfinite(e) and e > 0 for e in (q_eps, kv_eps)):
        raise ValueError("epsilons must be finite and positive")
    metadata = (mla_cache, slots, positions, cos_sin)
    if write_kv_cache:
        if any(t is None for t in metadata):
            raise ValueError("cache fusion requires mla_cache, slots, positions and cos_sin")
    elif any(t is not None for t in metadata):
        raise ValueError("cache metadata supplied without write_kv_cache=True")
    event = 0
    if notify_event is not None:
        event = int(notify_event.cuda_event)
        if not event or notify_event.device != qkv.device:
            raise ValueError("notify_event must be initialized on the input device")
    prepare_extension().qkv_norm_quant(qkv, q_weight, kv_weight, out.q_fp8, out.q_scale,
                                       out.q_bf16, out.kv_bf16, out.q_rank, out.kv_rank,
                                       q_eps, kv_eps, pdl, event, mla_cache, slots, positions, cos_sin, release)
    return out
