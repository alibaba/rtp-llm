"""Adapt GLM's flat physical candidate set to TRTLLM-GEN's two-pool ABI."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
import triton
import triton.language as tl
from packaging.version import Version

_WORKSPACES: dict[tuple[torch.device, int], torch.Tensor] = {}


@lru_cache(maxsize=None)
def flashinfer_sparse_supported(device: torch.device) -> bool:
    """Keep auto selection compatible with older wheels and other devices."""
    if torch.version.hip or torch.cuda.get_device_capability(device) not in (
        (10, 0),
        (10, 3),
    ):
        return False
    try:
        return Version(version("flashinfer-python")) >= Version("0.6.14")
    except PackageNotFoundError:
        return False


@triton.jit
def _compact_glm_candidates(
    IN,
    OUT,
    LENGTHS,
    SEQ,
    ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    KV_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    ids = tl.load(IN + row * ROW_STRIDE + col, col < WIDTH, other=-1)
    valid = (col < WIDTH) & (ids >= 0) & (ids < KV_ROWS)
    offset = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    # The TRTLLM-GEN ABI does NOT mask -1 entries within active lengths: they
    # load zero KV but still enter softmax. Compact holes, report exact lengths,
    # and mask short SWA rows through seq_lens. Empty rows use zero KV (-1).
    tl.store(OUT + row * CAPACITY + offset, ids, valid)
    tl.store(LENGTHS + row, tl.maximum(count, 128))
    tl.store(SEQ + row, tl.maximum(tl.minimum(count, 128), 1))


def flashinfer_sparse_fwd(q, kv, indices, sm_scale):
    import flashinfer

    if Version(flashinfer.__version__) < Version("0.6.14"):
        raise RuntimeError("GLM H8 TRTLLM-GEN requires FlashInfer >= 0.6.14")
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    if (
        q.shape[1:] != (8, 512)
        or q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
    ):
        raise ValueError("GLM TRTLLM-GEN path requires BF16 Q[T,8,512] and KV[N,1,512]")
    if kv.ndim != 3 or tuple(kv.shape[1:]) != (1, 512):
        raise ValueError("GLM small-head MLA requires KV[N,1,512]")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (q.shape[0], 1)
        or indices.dtype != torch.int32
        or indices.stride(-1) != 1
    ):
        raise ValueError(
            "GLM small-head MLA requires int32 indices[T,1,K] with contiguous K"
        )
    if not q.is_cuda or q.device != kv.device or q.device != indices.device:
        raise ValueError("Q, KV and indices must share a CUDA device")
    if not q.shape[0]:
        return torch.empty_like(q)
    if not kv.is_contiguous():
        raise ValueError("GLM TRTLLM-GEN requires a contiguous KV pool")
    if not flashinfer_sparse_supported(q.device):
        raise RuntimeError(
            "GLM H8 TRTLLM-GEN requires SM100/SM103 and FlashInfer >= 0.6.14"
        )
    tokens, width = q.shape[0], indices.shape[-1]
    capacity = max(128, (width + 63) // 64 * 64)
    padded = torch.full((tokens, capacity), -1, dtype=torch.int32, device=q.device)
    lengths = torch.empty((tokens,), dtype=torch.int32, device=q.device)
    seq_lens = torch.empty_like(lengths)
    _compact_glm_candidates[(tokens,)](
        indices,
        padded,
        lengths,
        seq_lens,
        indices.stride(0),
        width,
        capacity,
        kv.shape[0],
        triton.next_power_of_2(max(width, 1)),
    )
    # Both ABI segments alias the same GLM BF16 pool. The compacted candidate
    # set, order and live tail are preserved exactly; only padding moves.
    pool = kv.reshape(kv.shape[0], 1, 1, 512)
    key = (q.device, torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=q.device
        )
    out = trtllm_batch_decode_sparse_mla_dsv4(
        query=q.contiguous().unsqueeze(1),
        swa_kv_cache=pool,
        compressed_kv_cache=pool,
        workspace_buffer=_WORKSPACES[key],
        sparse_indices=padded,
        sparse_topk_lens=lengths,
        seq_lens=seq_lens,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        enable_pdl=False,
    ).squeeze(1)
    return out
