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

from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._pool_handle import PoolHandle
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


def translate_local_to_global_slots(
    req_id_per_token: torch.Tensor,  # [T] int32
    block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    local_idx: torch.Tensor,  # [T, K] int32, -1 sentinel preserved
    block_size: int,  # entries_per_block of the target pool
    *,
    handle: Optional[PoolHandle] = None,
    q_len: int = 1,
) -> torch.Tensor:
    """``[T, K]`` int32 global pool slot ids; -1 propagates as -1.

    ``block_size`` is the *kernel-visible* per-pool ``entries_per_block``
    (``E_pool``) — matches the vendored kernel's ``BLOCK_SIZE`` constexpr
    (default 256 for SWA; callers for HCA/CSA/INDEXER override). Any
    per-pool divergence (``E < 256``) lives ONLY in ``block_table`` values,
    never in the kernel divisor.

    The Triton kernel asserts ``K % BLOCK_N == 0`` (default 128). For
    ``K`` that doesn't naturally divide 128 (e.g. SWA win=128 →
    ``BLOCK_N=128`` works; smaller K → use 64). We pick the largest
    power-of-2 divisor of K up to 128.

    M09 §3.3 (Fix 143 — C03 05-2 / 19-3 / 19-4 / 47-3, E06 24-1 / 38-5):

    * ``q_len`` is the per-request decode width (1 for decode, 1+spec for
      MTP-verify). When ``q_len > 1``, the output reshapes to
      ``[B, q_len, K]`` so downstream FlashMLA ``indices`` consumption is
      shape-stable; the per-token recompute in the decode metadata builder
      then consumes the ``[B, q_len, *]`` shape without an extra view.
    * For ``q_len == 1`` the reshape is a no-op view — bit-equal to the
      pre-M09 return.

    Per M06 §3.6 / B04 04-1: when ``handle`` is provided, asserts
    ``block_size == handle.eb`` so kernel-side arithmetic uses the
    owning region's ``entries_per_block`` from the descriptor — not a
    loose caller-supplied integer. Closes C03 §8 Risk 8 (missing pool-id
    check). ``handle=None`` preserves the legacy bare-int contract.
    """
    if handle is not None:
        assert block_size == handle.eb, (
            f"translate_local_to_global_slots: block_size={block_size} "
            f"must match handle.eb={handle.eb} for region "
            f"{handle.region_id}"
        )
    K = local_idx.shape[1]
    block_n = 128
    while block_n > 1 and K % block_n != 0:
        block_n //= 2
    flat = triton_convert_req_index_to_global_index(
        req_id_per_token,
        block_table,
        local_idx.contiguous(),
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=K,
        BLOCK_N=block_n,
    )
    if int(q_len) > 1:
        # Optional post-process for region remap (Option (b) per M09 §3.3 doc):
        # the kernel writes a flat ``[T, K]`` block where ``T = B * q_len``;
        # the per-token recompute downstream prefers a ``[B, q_len, K]`` shape.
        T = int(flat.shape[0])
        assert T % int(q_len) == 0, (
            f"translate_local_to_global_slots: T={T} not divisible by "
            f"q_len={q_len}"
        )
        B = T // int(q_len)
        return flat.view(B, int(q_len), K)
    return flat
