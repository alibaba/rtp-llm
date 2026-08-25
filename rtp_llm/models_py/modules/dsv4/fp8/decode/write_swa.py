"""DSv4 decode SWA FP8 pool write — extracted from ``Attention._forward_decode_body``.

Mirrors :meth:`rtp_llm.models_py.modules.dsv4.fp8.attention.AttentionFP8._prefill_write_swa_fp8_paged`
for the decode path. Dispatches the Triton writer that emits the
584B/slot FP8 SWA layout (fp8 NoPE 448 + bf16 RoPE 128 + ue8m0 scale 8).

BF16 SWA decode write is intentionally unsupported — :meth:`Attention.forward_decode`
asserts ``_kv_cache_is_fp8`` at entry (mirrors the prefill FP8-only gate).
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)


def decode_write_swa_fp8(
    kv: torch.Tensor,  # [B, q_len, head_dim] bf16  (head_dim == 512 for DSv4)
    slot_mapping: Optional[torch.Tensor],  # [T] int64 pool slot indices, -1 for skip
    swa_pool_3d: Optional[torch.Tensor],  # [num_blocks, entries_per_block, 584] uint8
    bsz: int,
    q_len: int,
    head_dim: int,
) -> None:
    """Write newly-computed SWA KV into the FP8 584B/slot pool.

    No-op when ``slot_mapping`` is empty or the pool view is unavailable
    (warmup forward before the framework allocates the pool).

    ``slot_mapping[i] == -1`` entries are skipped in-kernel, so CUDA-graph
    padding slots at the tail of max-bs buffers don't need a Python-side
    slice.
    """
    if slot_mapping is None or slot_mapping.numel() == 0 or swa_pool_3d is None:
        return
    kv_flat = kv.reshape(bsz * q_len, head_dim)
    if kv_flat.dtype != torch.bfloat16:
        kv_flat = kv_flat.to(torch.bfloat16)
    slot_mapping = slot_mapping[: bsz * q_len]

    assert kv_flat.stride(0) == head_dim and kv_flat.stride(1) == 1, (
        "DSV4 decode SWA Triton writer requires a row-contiguous "
        f"[T, {head_dim}] KV view, got shape={tuple(kv_flat.shape)} "
        f"stride={tuple(kv_flat.stride())}"
    )
    quantize_and_insert_k_cache(kv_flat, swa_pool_3d, slot_mapping)
