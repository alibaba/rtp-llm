"""Bounded candidate: vLLM's single-CTA small-batch MoE alignment.

Used by the opt-in SM90 indexed MoE. Callers fall back when this returns None.
Output is (sorted flat route slots, expert per block, padded route count), all
int32. Invalid global IDs are omitted, padding slots equal ids.numel(), and
inactive expert blocks are -1. Within-expert order is upstream's thread-strided
order, NOT global stable slot order. Output positions still encode original
route slots, so consumers must scatter rather than reduce in this order.

Warm up once outside CUDA Graph capture to build/load the extension. No tensor
value is read on the host; output capacities depend only on input shape.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch


UPSTREAM_REVISION = "de8291e24f1312da0239605e33a89fe9cc901e47"
UPSTREAM_SOURCE_SHA256 = "d21e3309a45fd2439ffcf08a01ff429663f3ef684a675f13e8ea82a2e96fb495"


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    sources = [root / "binding.cpp", root / "align.cu"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:16]
    return load(
        name=f"rtp_sm90_moe_align_{digest}",
        sources=[str(p) for p in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
        with_cuda=True,
        verbose=False,
    )


def align_small_batch(ids, num_experts, block_size, *, local_start=0):
    """Align contiguous CUDA int32/int64 [rows, topk], or return None.

    Bounded to SM90, 1 <= E <= 64, fewer than 1024 route slots, and block
    sizes 1..256. Valid IDs are [local_start, local_start + E); local_start=0
    is the local-ID API. The range test precedes narrowing/subtraction, even
    for arbitrary int64 invalid IDs. Empty rows and all-invalid routes work.
    Unsupported shapes/types are detected without GPU-to-CPU synchronization.
    """
    if (
        not isinstance(ids, torch.Tensor)
        or ids.device.type != "cuda"
        or ids.dtype not in (torch.int32, torch.int64)
        or ids.ndim != 2
        or not ids.is_contiguous()
        or ids.shape[1] == 0
        or ids.numel() >= 1024
        or type(num_experts) is not int
        or not 1 <= num_experts <= 64
        or type(block_size) is not int
        or not 1 <= block_size <= 256
        or type(local_start) is not int
        or not 0 <= local_start <= 2**31 - 1 - num_experts
    ):
        return None
    if torch.cuda.get_device_capability(ids.device) != (9, 0):
        return None
    # Avoid accidental compile/import during capture. Cached calls perform only
    # host metadata validation, graph-pool allocations, and one CUDA launch.
    if not _extension.cache_info().currsize and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up align_small_batch before CUDA Graph capture")
    return tuple(_extension().align(ids, num_experts, block_size, local_start))
