"""DSV4 SWA prefill ops — vLLM-aligned Triton kernels.

Two kernels vendored verbatim (signature + math) from vLLM:

* ``compute_prefill_gather_lens`` — vLLM
  ``vllm/v1/attention/backends/mla/sparse_swa.py:_compute_prefill_metadata_kernel``.
  Per request: ``gather_len = query_len + min(prefix_len, window_size - 1)``.
  Drives ``dequantize_and_gather_k_cache``'s gather amount so every prefill
  query has its full SWA window in the BF16 workspace.

* ``combine_topk_swa_indices`` — vLLM ``cache_utils.py:445``. Per query
  token, lays out a ``[combined_topk]`` int32 row of:
    - first ``topk_len`` entries: compressed-attn indices, offset by
      ``M * batch_idx`` so they land in this batch's slice of the gathered
      workspace
    - next ``swa_len`` entries: SWA window indices into the same workspace
      (shifted by ``N`` to skip the compressed segment)
  + a parallel ``[num_tokens]`` int32 ``combined_lens`` so
  ``flash_mla_sparse_fwd``'s ``topk_length`` masks the right tail.

These are the only two NEW Triton kernels needed for vLLM-style FP8 SWA
prefill — ``dequantize_and_gather_k_cache`` already lives in
``_swa_fp8_dequant_triton.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Mirror vLLM's combined-topk row alignment (kernel tile width).
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


# ---------------------------------------------------------------------------
# 1) Per-request gather_len = query_len + min(prefix_len, window_size - 1)
# ---------------------------------------------------------------------------


@triton.jit
def _compute_prefill_gather_lens_kernel(
    # Outputs
    prefill_gather_lens_ptr,
    # Inputs
    seq_lens_ptr,
    query_start_loc_ptr,
    num_prefills,
    num_decodes,
    window_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass compute of per-prefill ``gather_len``.

    For prefill request ``i`` (offset by ``num_decodes`` in the merged
    decode+prefill batch view):
        query_len  = qsl[num_decodes+i+1] - qsl[num_decodes+i]
        prefix_len = seq_len - query_len             # already-cached tokens
        gather_len = query_len + min(prefix_len, window_size - 1)

    The ``window_size - 1`` cap is the SWA invariant: the earliest prefill
    query at sequence position p reuses at most win-1 prefix tokens
    (positions p-win+1 .. p-1), so gathering more than that is wasted.
    """
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < num_prefills

    seq_len = tl.load(seq_lens_ptr + num_decodes + offset, mask=mask)
    qsl_start = tl.load(query_start_loc_ptr + num_decodes + offset, mask=mask)
    qsl_end = tl.load(query_start_loc_ptr + num_decodes + offset + 1, mask=mask)

    query_len = qsl_end - qsl_start
    prefix_len = seq_len - query_len
    gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

    tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)


def compute_prefill_gather_lens(
    seq_lens: torch.Tensor,  # [num_decodes + num_prefills] int32 — total seq lens
    query_start_loc: torch.Tensor,  # [num_decodes + num_prefills + 1] int32
    num_prefills: int,
    num_decodes: int,
    window_size: int,
) -> torch.Tensor:
    """Return ``[num_prefills]`` int32 ``gather_lens``.

    Args:
      seq_lens: total token count per request (decodes + prefills, in that order).
      query_start_loc: cumulative new-token offsets; length = batch_size + 1.
      num_prefills:  count of prefill requests in the merged batch.
      num_decodes:   count of decode requests preceding the prefills.
      window_size:   SWA window (≥1).
    """
    assert seq_lens.dtype == torch.int32 and query_start_loc.dtype == torch.int32, (
        f"seq_lens and query_start_loc must be int32, got {seq_lens.dtype} / "
        f"{query_start_loc.dtype}"
    )
    assert num_prefills >= 0 and num_decodes >= 0 and window_size >= 1
    out = torch.empty(num_prefills, dtype=torch.int32, device=seq_lens.device)
    if num_prefills == 0:
        return out
    # Triton requires the block size to be a power of 2 ≥ num_prefills.
    block_size = max(1, triton.next_power_of_2(num_prefills))
    _compute_prefill_gather_lens_kernel[(1,)](
        out,
        seq_lens,
        query_start_loc,
        num_prefills,
        num_decodes,
        window_size,
        BLOCK_SIZE=block_size,
    )
    return out


# ---------------------------------------------------------------------------
# 2) Per-token SWA paged-tail slot_mapping (write side)
# ---------------------------------------------------------------------------


@triton.jit
def _compute_swa_slot_mapping_kernel(
    # Output
    slot_mapping_ptr,  # [num_tokens] int64
    # Inputs
    block_table_ptr,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc_ptr,  # [num_reqs+1] int32 — chunk-local cumulative
    seq_lens_ptr,  # [num_reqs] int32 — total seq len per req (sp + S_local)
    # Constants
    num_reqs,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,  # eb
    BLOCK_M: tl.constexpr,  # tokens per program
):
    """One program per (request, BLOCK_M chunk-of-tokens).

    For request ``b``, chunk-local token index ``i`` (= absolute token
    in flattened ``[T]`` view at ``query_start[b] + i``):

        global_pos    = (seq_len[b] - query_len[b]) + i
        block_in_seq  = global_pos // block_size
        in_block      = global_pos % block_size
        block_id      = block_table[b, block_in_seq]
        slot          = -1                       if block_id <= 0
                        block_id * eb + in_block otherwise

    SWA paged-tail semantics: ``block_table`` has the last ``2``
    entries valid, earlier entries are ``-1`` (segment evicted from
    the SWA pool). Tokens whose segment is evicted get ``slot = -1``
    so the FP8 insert kernel skips them — matches the sliding-window
    contract that only the last ``2*eb`` tokens persist.
    """
    batch_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    if batch_idx >= num_reqs:
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    sp = seq_len - query_len  # absolute pos of first new token

    chunk_start = chunk_idx * BLOCK_M
    i = chunk_start + tl.arange(0, BLOCK_M)  # [BLOCK_M] chunk-local
    valid_token = i < query_len
    global_pos = sp + i  # [BLOCK_M] absolute

    block_in_seq = global_pos // block_size
    in_block = global_pos % block_size

    in_capacity = block_in_seq < max_blocks_per_seq
    safe_block_in_seq = tl.where(in_capacity, block_in_seq, 0)

    bt_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
    block_id = tl.load(bt_row_ptr + safe_block_in_seq, mask=valid_token, other=-1)

    valid = (block_id > 0) & in_capacity & valid_token
    slot = tl.where(
        valid,
        block_id.to(tl.int64) * block_size + in_block.to(tl.int64),
        tl.full((BLOCK_M,), -1, dtype=tl.int64),
    )

    out_ptr = slot_mapping_ptr + query_start + i
    tl.store(out_ptr, slot, mask=valid_token)


def compute_swa_slot_mapping(
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq] int32
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32 — chunk-local cumulative
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len = sp + query_len
    block_size: int,  # eb
    num_tokens: int,  # total tokens across all reqs (= query_start_loc[-1])
) -> torch.Tensor:
    """Build ``[num_tokens]`` int64 slot_mapping for SWA paged-tail FP8 write.

    Mirrors the slot formula in ``DSV4_CACHE_LAYOUT.md §6`` / what
    ``_dequantize_and_gather_k_kernel`` expects: ``block_table[b, pos //
    block_size] * block_size + (pos % block_size)``. Tokens whose segment
    has ``block_id <= 0`` (evicted by SWA tail-only allocation) emit
    ``-1`` so the FP8 insert kernel skips them.
    """
    assert (
        block_table.dtype == torch.int32
    ), f"block_table must be int32, got {block_table.dtype}"
    assert (
        query_start_loc.dtype == torch.int32 and seq_lens.dtype == torch.int32
    ), "query_start_loc and seq_lens must be int32"
    assert block_size >= 1

    device = block_table.device
    slot_mapping = torch.empty(num_tokens, dtype=torch.long, device=device)
    if num_tokens == 0:
        return slot_mapping

    num_reqs = int(seq_lens.shape[0])
    max_blocks_per_seq = int(block_table.shape[1])
    BLOCK_M = 256
    num_chunks = max(1, (num_tokens + BLOCK_M - 1) // BLOCK_M)
    grid = (num_reqs, num_chunks)
    _compute_swa_slot_mapping_kernel[grid](
        slot_mapping,
        block_table,
        query_start_loc,
        seq_lens,
        num_reqs,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        BLOCK_M=BLOCK_M,
    )
    return slot_mapping


# ---------------------------------------------------------------------------
# 3) Combine compressed-attn topk + SWA window indices into one row per query
# ---------------------------------------------------------------------------


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
):
    """Per-query layout of ``[combined_topk]`` row in the gathered workspace.

    Workspace layout for a single batch slice (length ``M``):
        [0, N)              — compressed-attn region
        [N, N + gather_len) — SWA region (last ``gather_len`` tokens)
        [N + gather_len, M) — pad / unused

    For query at absolute sequence position ``pos``:
        topk_len = min((pos+1) // COMPRESS_RATIO, TOP_K)   # 0 for SWA-only
        swa_len  = min(pos+1, WINDOW_SIZE)
        row[:topk_len]                   = topk_indices[token, :topk_len] + M*batch
        row[topk_len:topk_len+swa_len]   = M*batch + N + (pos - swa_len + 1
                                                          - gather_start) + arange
            where gather_start = seq_len - gather_len
    """
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    # query_start_loc is a global tensor (across all decodes + prefills);
    # rebase to chunk-local offsets so the index math works on the local
    # ``topk_indices_ptr`` slice.
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len  # absolute sequence pos of first new query
    gather_start = seq_len - gather_len  # first absolute pos in gathered region

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        # Both the C4A indexer and the C128A metadata builder emit
        # min((pos + 1) // compress_ratio, topk_tokens) valid entries —
        # this matches that. Caller passes TOP_K=0 for SWA-only layers
        # to zero out the compressed contribution.
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            topk_indices + M * batch_idx,
            mask=mask,
        )

        # Triton ``arange`` requires a power-of-2 range; use the padded
        # tile but mask down to the real ``swa_len`` (≤ WINDOW_SIZE) so
        # callers can pass any window_size.
        offset = tl.arange(0, PADDED_WINDOW_SIZE)
        # SWA workspace indices for positions [pos - swa_len + 1, pos]:
        # buffer index = N + (position - gather_start), then offset by
        # this batch's slice (M * batch_idx).
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + offset,
            M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        combined_len = topk_len + swa_len
        tl.store(combined_lens_ptr + token_idx, combined_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,  # [num_tokens, topk] int32 — compressed-attn indices
    query_start_loc: torch.Tensor,  # [num_reqs + 1] int32 — chunk-local start locs
    seq_lens: torch.Tensor,  # [num_reqs] int32 — total seq len per request
    gather_lens: torch.Tensor,  # [num_reqs] int32 — from compute_prefill_gather_lens
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(combined_indices, combined_lens)`` for ``flash_mla_sparse_fwd``.

    Returns:
      combined_indices: ``[num_tokens, combined_topk]`` int32; ``combined_topk
        = align(topk + window_size, 128)``. Sentinel ``-1`` in unused tail.
      combined_lens:   ``[num_tokens]`` int32 — pass to ``flash_mla_sparse_fwd``
        as ``topk_length``.

    Args mirror vLLM signature exactly. ``M`` is the per-batch workspace
    stride (``N + window_size + max_num_batched_tokens``). ``N`` is the
    compressed-region size (``ceil(max_model_len / compress_ratio)``); pass
    ``N=0`` and ``topk=0`` for SWA-only layers.
    """
    assert (
        topk_indices.dtype == torch.int32
    ), f"topk_indices must be int32, got {topk_indices.dtype}"
    assert (
        query_start_loc.dtype == torch.int32 and seq_lens.dtype == torch.int32
    ), "query_start_loc and seq_lens must be int32"
    assert gather_lens.dtype == torch.int32, "gather_lens must be int32"
    assert window_size >= 1 and compress_ratio >= 1

    num_tokens = int(topk_indices.shape[0])
    num_reqs = int(seq_lens.shape[0])
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    if num_tokens == 0 or num_reqs == 0:
        return combined_indices, combined_lens

    # ``PADDED_TOP_K`` is the kernel-side load tile for the compressed-attn
    # part; pad to power-of-2 ≥ topk_indices.shape[-1] so the SIMD load mask
    # lines up. SWA-only layers pass topk_indices.shape[-1] == 0 → use 1.
    padded_top_k = max(1, triton.next_power_of_2(int(topk_indices.shape[-1])))
    # Same constraint for the SWA arange tile — Triton requires arange ranges
    # to be power-of-2; mask inside the kernel handles the real ``window_size``.
    padded_window_size = max(1, triton.next_power_of_2(int(window_size)))
    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=padded_top_k,
        PADDED_WINDOW_SIZE=padded_window_size,
    )
    return combined_indices, combined_lens
