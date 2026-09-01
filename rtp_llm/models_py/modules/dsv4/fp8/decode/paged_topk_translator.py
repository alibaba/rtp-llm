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


def translate_local_to_global_slots(
    req_id_per_token: torch.Tensor,  # [T] int32
    block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    local_idx: torch.Tensor,  # [T, K] int32, -1 sentinel preserved
    entries_per_block: int,
    tokens_per_block_for_block_table: int,
) -> torch.Tensor:
    """[T, K] int32 global pool slot ids; -1 propagates as -1.

    ``entries_per_block`` is the pool's flat slot multiplier / tensor
    second dimension. ``tokens_per_block_for_block_table`` is the raw-token
    coverage of one block-table row.

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
