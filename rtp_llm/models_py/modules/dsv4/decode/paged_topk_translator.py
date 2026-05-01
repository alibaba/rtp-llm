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

    # SWA half: index_select with -1-safe redirect, then mask to 0.
    swa_safe = torch.where(
        swa_global >= 0,
        swa_global,
        torch.zeros_like(swa_global),
    ).to(torch.long)
    swa_kv = swa_pool_view.index_select(0, swa_safe.view(-1)).view(T, win, head_dim)
    swa_mask = (swa_global >= 0).unsqueeze(-1)
    swa_kv = torch.where(swa_mask, swa_kv, torch.zeros_like(swa_kv))

    # Compressed half: same pattern, different pool.
    cmp_safe = torch.where(
        cmp_global >= 0,
        cmp_global,
        torch.zeros_like(cmp_global),
    ).to(torch.long)
    cmp_kv = cmp_pool_view.index_select(0, cmp_safe.view(-1)).view(T, K, head_dim)
    cmp_mask = (cmp_global >= 0).unsqueeze(-1)
    cmp_kv = torch.where(cmp_mask, cmp_kv, torch.zeros_like(cmp_kv))

    return torch.cat([swa_kv, cmp_kv], dim=1).view(batch_size, q_len, win + K, head_dim)


def translate_local_to_global_slots(
    req_id_per_token: torch.Tensor,  # [T] int32
    block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    local_idx: torch.Tensor,  # [T, K] int32, -1 sentinel preserved
    block_size: int,  # entries_per_block of the target pool
) -> torch.Tensor:
    """[T, K] int32 global pool slot ids; -1 propagates as -1.

    ``block_size`` is the *pool's* ``entries_per_block`` (e.g. 256 for
    SWA_KV, 64 for CSA_KV / INDEXER_KV, 2 for HCA_KV).

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
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=K,
        BLOCK_N=block_n,
    )
