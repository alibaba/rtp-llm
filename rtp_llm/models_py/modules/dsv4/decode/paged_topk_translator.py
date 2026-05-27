"""Per-step decode topk → global pool slot id translator.

Hot-path module: runs once per (layer, decode-step). Wraps the local
``triton_convert_req_index_to_global_index`` Triton kernel (vendored
from vLLM) so the call site is a one-liner.

Design rationale (vs. an "index_select-into-packed-buffer" gather):
    * vLLM's flash_mla integration passes global slot ids straight
      into the attention kernel, which does the gather inside compute
      via indirect addressing. No intermediate ``[B, win+K, D]``
      buffer, no ``index_select``, no ``torch.where`` mask.
    * RTP-LLM's TileLang ``sparse_attn_kernel`` (dsv4/tilelang_kernels.py)
      already accepts ``kv[b, n, d]`` + ``topk_idxs[b, m, topk]`` with
      ``kv[by, idxs[i], j]`` indirect read inside the kernel. Setting
      ``b = 1`` and feeding the full pool as ``[1, num_global_slots, d]``
      reproduces the vLLM zero-copy pattern with the existing kernel —
      just reinterpret-shape on q/o (no actual copy).

Module returns the slot-id tensor; the call site does the one-line
shape gymnastics. We keep this module read-only (no allocs of its own
beyond what the kernel returns) so the hot path stays lean.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index,
)


def build_req_id_per_token(
    batch_size: int, q_len: int, device: torch.device
) -> torch.Tensor:
    """``[T = B * q_len]`` int32 mapping each query token to its request.

    For decode q_len is typically 1 (or 1+spec for MTP), so this is just
    ``arange(B)`` repeated ``q_len`` times. Kept as a helper so callers
    don't redo the same arange + repeat_interleave each step.
    """
    base = torch.arange(batch_size, device=device, dtype=torch.int32)
    if q_len == 1:
        return base
    return base.repeat_interleave(q_len)


def gather_dual_pool_kv_packed(
    swa_pool_view: torch.Tensor,  # [num_swa_slots, D] bf16
    cmp_pool_view: torch.Tensor,  # [num_cmp_slots, D] bf16
    swa_global: torch.Tensor,  # [T, win] int32, -1 sentinel
    cmp_global: torch.Tensor,  # [T, K] int32, -1 sentinel
    head_dim: int,
    batch_size: int,
    q_len: int,
) -> torch.Tensor:
    """Phase 2B-2b: vectorized dual-pool gather → packed scratch.

    Returns ``[B, q_len, win+K, D]`` bf16. Invalid (``-1``) slots are
    masked to zeros so the downstream sparse-attn kernel can treat the
    packed buffer as dense (identity topk = ``arange(win+K)``) — same
    output as if the kernel did the gather + ``-1`` skip itself.

    Hot-path note: this is the gather-into-packed fallback because our
    TileLang ``sparse_attn_kernel`` only takes a single ``kv`` tensor.
    Memory traffic is ~``(win+K)*D*bs`` bf16 per call (~10MB at typical
    bs=16, win+K=640, D=512). Phase 2B-2c may extend the kernel to take
    two pools (mirror vLLM ``flash_mla_with_kvcache(extra_k_cache,
    extra_indices)``) for true zero-copy.
    """
    T = batch_size * q_len
    win = swa_global.shape[1]
    K = cmp_global.shape[1]

    from rtp_llm.models_py.modules.dsv4._pool_triton import masked_gather_from_pool

    # Fused -1-safe gather + zero-fill for each pool.  CUDA avoids the
    # repeated torch.where -> index_select -> torch.where launch chain; CPU
    # falls back inside the helper.
    swa_kv = masked_gather_from_pool(
        swa_pool_view,
        swa_global,
        swa_global >= 0,
        out_shape=(T, win, head_dim),
        dtype=swa_pool_view.dtype,
    )

    cmp_kv = masked_gather_from_pool(
        cmp_pool_view,
        cmp_global,
        cmp_global >= 0,
        out_shape=(T, K, head_dim),
        dtype=cmp_pool_view.dtype,
    )

    return torch.cat([swa_kv, cmp_kv], dim=1).view(batch_size, q_len, win + K, head_dim)


def translate_local_to_global_slots(
    req_id_per_token: torch.Tensor,  # [T] int32
    block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    local_idx: torch.Tensor,  # [T, K] int32, -1 sentinel preserved
    entries_per_block: int,
    tokens_per_block_for_block_table: int,
) -> torch.Tensor:
    """[T, K] int32 global pool slot ids; -1 propagates as -1.

    ``entries_per_block`` is the pool's flat slot multiplier.
    ``tokens_per_block_for_block_table`` is the raw-token coverage of one
    block-table row.

    The Triton kernel asserts ``K % BLOCK_N == 0`` (default 128). For
    ``K`` that doesn't naturally divide 128 (e.g. SWA win=128 →
    ``BLOCK_N=128`` works; smaller K → use 64). We pick the largest
    power-of-2 divisor of K up to 128.
    """
    K = local_idx.shape[1]
    block_n = 128
    while block_n > 1 and K % block_n != 0:
        block_n //= 2
    return triton_convert_req_index_to_global_index(
        req_id_per_token,
        block_table,
        local_idx.contiguous(),
        ENTRIES_PER_BLOCK=int(entries_per_block),
        TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=int(tokens_per_block_for_block_table),
        NUM_TOPK_TOKENS=K,
        BLOCK_N=block_n,
    )
