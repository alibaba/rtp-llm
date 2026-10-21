"""DeepSeek-V4 Attention with HCA / CSA / SWA-only path selection.

Direct port of `inference/model.py:Attention` (BF16-only, mock per-layer
KV cache via register_buffer). Skips Hadamard rotate / FP4 / FP8 quant.

Layer schedule via `compress_ratio`:
  0   -> SWA-only (no Compressor, no Indexer)
  4   -> CSA (Compressor with overlap=True + Indexer for sparse top-k)
  128 -> HCA (Compressor with overlap=False, dense compressed MQA)

Sparse attention reference uses `gather`-based PyTorch implementation —
slow but correct. M6 will swap in FlashMLA sparse impl.
"""

import json
import logging
import os
import threading
import weakref
from contextlib import contextmanager, suppress
from functools import lru_cache
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.modules.dsv41._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv41._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)
from rtp_llm.models_py.modules.dsv41._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv41.chunk_env import (
    FLASH_MLA_SPARSE_Q_CHUNK as _FLASH_MLA_SPARSE_Q_CHUNK,
)
from rtp_llm.models_py.modules.dsv41.cp import (
    _CP_ROLE_MAIN,
    CPContext,
    build_cp_full_prefill_positions,
    cp_actual_owned_kv_lens,
    cp_all_gather_full_varlen,
    cp_freqs_cis_local,
    cp_padded_local_kv_lens,
    cp_swa_replay_starts,
)
from rtp_llm.models_py.modules.dsv41.fp8._cp_attention_merge import merge_lse_output
from rtp_llm.models_py.modules.dsv41.fp8._cp_attention_shard import (
    build_swa_cp_local_indices,
    prefer_raw_q_merge_attention_conservative,
    remap_topk_to_cp_local,
)
from rtp_llm.models_py.modules.dsv41.fp8._pool_reader import (
    CompressedKPoolReader,
    LocalPoolReader,
    make_compressed_k_pool_reader,
)
from rtp_llm.models_py.modules.dsv41.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv41.fp8.compressor import CompressorFP8, CompressorMeta
from rtp_llm.models_py.modules.dsv41.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv41.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.modules.dsv41.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv41.utils import (
    V41MXFP8Linear,
    _v4_fp8_linear,
    merge_v41_qkv_weights,
)
from rtp_llm.models_py.utils.memory import dispose_tensor


@lru_cache(maxsize=1)
def _configure_flash_mla_l2_persist() -> None:
    """Configure the process-wide FlashMLA policy once, before inference.

    Per-call persisting-L2 setup invokes cudaDeviceSetLimit and can block the
    host behind previously queued layer work. Disable it by default. An
    explicit FLASH_MLA_NO_L2_PERSIST=0 restores the wheel's original policy;
    the wheel tests presence rather than value, so remove the native flag in
    that case. Cache this initialization so later layers do not reset it.
    """
    if os.environ.get("FLASH_MLA_NO_L2_PERSIST", "1") == "0":
        os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
    else:
        os.environ.setdefault("FLASH_MLA_NO_L2_PERSIST", "1")


from rtp_llm.models_py.modules.dsv41.attn_type import (
    CSA_KV,
    CSA_STATE,
    DECODER_SWA_KV,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)

_CACHE_TAG_BY_NAME = {
    tag: tag
    for tag in (
        SWA_KV,
        DECODER_SWA_KV,
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        INDEXER_STATE,
        CSA_STATE,
        HCA_STATE,
    )
}


def _use_read_from_pool() -> bool:
    return os.environ.get("DSV4_READ_FROM_POOL", "1") != "0"


def _use_cp_cache_hit_raw_q_merge() -> bool:
    return _force_cp_cache_hit_raw_q_merge() or _force_all_cp_raw_q_merge()


def _force_cp_cache_hit_raw_q_merge() -> bool:
    return os.environ.get("DSV4_CP_CACHE_HIT_RAW_Q_MERGE", "0").lower() == "force"


def _force_all_cp_raw_q_merge() -> bool:
    return os.environ.get("DSV4_CP_CACHE_HIT_RAW_Q_MERGE", "0").lower() in (
        "force_all",
        "all",
    )


from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import (
    require_pool_tokens_per_block as _dsv4_pool_tokens_per_block,
)


def _prefill_cp_overlap_enabled() -> bool:
    return os.environ.get("DSV4_PREFILL_CP_OVERLAP", "0") == "1"


def _prefill_cp_async_workspace_reads_enabled() -> bool:
    return os.environ.get("DSV4_PREFILL_CP_ASYNC_WORKSPACE_READS", "1") != "0"


_CP_POST_GATHER_STREAMS: Dict[int, torch.cuda.Stream] = {}
_CP_GATHER_STREAMS: Dict[int, torch.cuda.Stream] = {}
_CP_STREAM_CACHE_LOCK = threading.Lock()


def _cuda_device_index(device: torch.device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _get_process_cuda_stream(
    cache: Dict[int, torch.cuda.Stream], device: torch.device
) -> torch.cuda.Stream:
    """Return one process-local CUDA stream per role and device."""
    device_index = _cuda_device_index(device)
    with _CP_STREAM_CACHE_LOCK:
        stream = cache.get(device_index)
        if stream is None or stream.device.index != device_index:
            stream = torch.cuda.Stream(device=torch.device("cuda", device_index))
            cache[device_index] = stream
        return stream


def _get_cp_comm_stream(device: torch.device) -> torch.cuda.Stream:
    """Serialized CP communication stream shared by all layers on one device."""
    return _get_process_cuda_stream(_CP_GATHER_STREAMS, device)


def _get_cp_post_gather_stream(device: torch.device) -> torch.cuda.Stream:
    """Post-NCCL local work stream shared by CP prefetch helpers."""
    return _get_process_cuda_stream(_CP_POST_GATHER_STREAMS, device)


def _flat_1d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1 and t.is_contiguous():
        return t
    return t.reshape(-1).contiguous()


def _build_suffix_pool_slot_mapping(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    entries_per_block: int,
    tokens_per_block_for_block_table: int,
    ring_entries: int,
) -> torch.Tensor:
    """Build request-major flat slots for a suffix gather.

    ``seq_lens`` and ``gather_lens`` follow ``dequantize_and_gather_k_cache``:
    request ``b`` gathers absolute positions
    ``[seq_lens[b] - gather_lens[b], seq_lens[b])``. The block-table row
    token coverage is intentionally separate from the in-block ring modulo.
    """
    device = block_table.device
    B = int(seq_lens.numel())
    if B == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)
    gather_lens_l = gather_lens.to(device=device, dtype=torch.long).reshape(-1)
    seq_lens_l = seq_lens.to(device=device, dtype=torch.long).reshape(-1)
    max_gather = int(gather_lens_l.max().item()) if gather_lens_l.numel() else 0
    if max_gather <= 0:
        return torch.empty((B, 0), dtype=torch.long, device=device)
    step = torch.arange(max_gather, device=device, dtype=torch.long)
    start = seq_lens_l - gather_lens_l
    abs_pos = start.unsqueeze(1) + step.unsqueeze(0)
    valid_pos = (step.unsqueeze(0) < gather_lens_l.unsqueeze(1)) & (abs_pos >= 0)
    block_in_seq = abs_pos // int(tokens_per_block_for_block_table)
    in_block = abs_pos % int(ring_entries)
    max_blocks = int(block_table.shape[1])
    in_capacity = valid_pos & (block_in_seq >= 0) & (block_in_seq < max_blocks)
    safe_block = torch.where(in_capacity, block_in_seq, torch.zeros_like(block_in_seq))
    bt_long = block_table[:B].to(device=device, dtype=torch.long)
    req = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1)
    block_id = bt_long[req, safe_block]
    valid = in_capacity & (block_id > 0)
    slot = block_id * int(entries_per_block) + in_block
    return torch.where(valid, slot, torch.full_like(slot, -1)).contiguous()


def _build_suffix_cp_sliced_slot_mapping(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    local_entries_per_block: int,
    tokens_per_block_for_block_table: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Build suffix slots for CP-sliced SWA_KV local blocks.

    The block table is still indexed by the logical/cache-key block size. The
    physical local SWA block stores only this rank's slice of the full SWA ring,
    whose size is independent of the logical block-table row size.
    """
    full_entries_per_block = int(local_entries_per_block) * int(cp_size)
    device = block_table.device
    B = int(seq_lens.numel())
    if B == 0:
        return torch.empty((0, 0), dtype=torch.long, device=device)
    gather_lens_l = gather_lens.to(device=device, dtype=torch.long).reshape(-1)
    seq_lens_l = seq_lens.to(device=device, dtype=torch.long).reshape(-1)
    max_gather = int(gather_lens_l.max().item()) if gather_lens_l.numel() else 0
    if max_gather <= 0:
        return torch.empty((B, 0), dtype=torch.long, device=device)
    step = torch.arange(max_gather, device=device, dtype=torch.long)
    start = seq_lens_l - gather_lens_l
    abs_pos = start.unsqueeze(1) + step.unsqueeze(0)
    valid_pos = (step.unsqueeze(0) < gather_lens_l.unsqueeze(1)) & (abs_pos >= 0)
    block_in_seq = abs_pos // int(tokens_per_block_for_block_table)
    ring_offset = abs_pos % full_entries_per_block
    owner_rank = ring_offset // int(local_entries_per_block)
    local_offset = ring_offset - owner_rank * int(local_entries_per_block)
    max_blocks = int(block_table.shape[1])
    in_capacity = valid_pos & (block_in_seq >= 0) & (block_in_seq < max_blocks)
    safe_block = torch.where(in_capacity, block_in_seq, torch.zeros_like(block_in_seq))
    bt_long = block_table[:B].to(device=device, dtype=torch.long)
    req = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1)
    block_id = bt_long[req, safe_block]
    block_end = (block_in_seq + 1) * int(tokens_per_block_for_block_table)
    effective_end = torch.minimum(block_end, seq_lens_l.unsqueeze(1))
    tail_write = abs_pos + full_entries_per_block >= effective_end
    valid = in_capacity & (block_id > 0) & (owner_rank == int(cp_rank)) & tail_write
    slot = block_id * int(local_entries_per_block) + local_offset
    return torch.where(valid, slot, torch.full_like(slot, -1)).contiguous()


_DSV4_FP8_KV_ENTRY_BYTES = 584
_SWA_CP_RR_LOGGED_SITES: set = set()
BIND_KEEP = object()


@contextmanager
def bind_attn_cache(attn, kv_cache=None, block_tables_by_type=None, cp_ctx=BIND_KEEP):
    """Temporarily bind a kv-cache / block-table (and optionally a CP context)
    view onto ``attn``, restoring the previous binding on exit.  ``None`` for
    the cache/table arguments keeps the current binding; pass ``cp_ctx`` only
    when it should be replaced."""
    from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
        bind_swa_table,
        cached_swa_region,
    )

    prev_kv = attn._kv_cache
    prev_bt = attn._block_tables_by_type
    prev_cp = attn._cp_ctx
    prev_region = getattr(attn, "_swa_cache_region", SWA_KV)
    try:
        if kv_cache is not None:
            attn._kv_cache = kv_cache
        attn._swa_cache_region = cached_swa_region(attn, attn._kv_cache, attn.layer_id)
        if block_tables_by_type is not None:
            attn._block_tables_by_type = bind_swa_table(
                block_tables_by_type, attn._swa_cache_region
            )
        if cp_ctx is not BIND_KEEP:
            attn._cp_ctx = cp_ctx
        yield attn
    finally:
        attn._kv_cache = prev_kv
        attn._block_tables_by_type = prev_bt
        attn._cp_ctx = prev_cp
        attn._swa_cache_region = prev_region


_DSV4_FP8_INDEXER_ENTRY_BYTES = 132


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` → DeepGEMM ``[N, K/128]`` UE8M0
    int32-packed TMA-aligned scale.  Row-repeats by 128 along N so each
    weight row gets its own scale row, then hands off to DeepGEMM's
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (column-major,
    int32-packed).  Must be called on-device (DeepGEMM helper is CUDA)."""
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    N_blk, _ = scale.shape
    N = N_blk * 128
    idx = torch.arange(N, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _prepare_wo_a_stacked(
    weight_fp8: torch.Tensor, scale_raw: torch.Tensor, G: int, R: int, K: int
) -> tuple:
    """Stack V4 wo_a ckpt (``[G*R, K]`` fp8 + ``[G*R/128, K/128]`` e8m0fnu)
    into the ``fp8_einsum``-expected layout, computed once at init:

    - weight: ``[G, R, K]`` fp8 contiguous (free view)
    - scale : ``[G, R, K/512]`` int32 UE8M0 packed, MN-major TMA-aligned
      (stride ``(K/512 * tma_R, 1, tma_R)``)

    Cast e8m0fnu → fp32, row-repeat by 128 along R so
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (which operates on
    fp32 ``[*, mn, k/128]``) sees the full [G, R, K/128] grid.  The helper
    floor-log2 bitcasts each block scale and packs 4 UE8M0 bytes per
    int32; output shape ``[G, R, K/512]`` matches
    ``deep_gemm.fp8_einsum(..., recipe=(1, 1, 128))`` expectations."""
    w_stk = weight_fp8.view(G, R, K).contiguous()
    scale_block = K // scale_raw.shape[-1]
    scale_fp32 = scale_raw.float().view(G, R // scale_block, K // scale_block)
    idx = torch.arange(R, device=scale_raw.device) // scale_block
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return (w_stk, s_stk)


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for callers that
    still pass a flat dict + keys.  Mutates ``weights[scale_key]`` to the
    packed form so subsequent callers don't repack."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)


def _get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device
) -> torch.Tensor:
    """Returns int64 [bsz, seqlen, window_size] with linear absolute KV indices."""
    if start_pos > 0 and seqlen > 1:
        base = torch.arange(start_pos, start_pos + seqlen, device=device).unsqueeze(1)
        offs = torch.arange(window_size, device=device)
        window_start = (base - window_size + 1).clamp_min(0)
        matrix = window_start + offs
        matrix = torch.where(matrix > base, -1, matrix)
    elif start_pos >= window_size - 1:
        sp = start_pos % window_size
        matrix = torch.cat(
            [
                torch.arange(sp + 1, window_size, device=device),
                torch.arange(0, sp + 1, device=device),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
        if matrix.size(1) < window_size:
            matrix = F.pad(matrix, (0, window_size - matrix.size(1)), value=-1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_window_topk_idxs_cp(
    window_size: int,
    bsz: int,
    seq_len_total: int,
    global_positions: torch.Tensor,
    use_ring_layout: bool = False,
) -> torch.Tensor:
    """CP-prefill variant: each rank-local Q token at local index i sits
    at GLOBAL position g = global_positions[i].  Its sliding window
    reads KV at global positions [max(0, g-win+1), g+1).

    Fresh CP prefill uses the all-gathered linear ``kv_full`` layout.  CP
    continuation prefill reconstructs the same linear absolute view from the
    paged SWA pool; ``use_ring_layout`` is kept for legacy ring callers.

    Returns [bsz, S_local, window_size] int64 with valid indices at the
    START of each row (slots 0..k-1) and -1 padding at the END — matching
    the non-CP ``_get_window_topk_idxs`` layout.  This slot ordering is
    LOAD-BEARING for sparse-attn numerical equivalence: the TileLang
    sparse_attn kernel's per-block fp32 reductions are not invariant to
    the position of valid vs masked slots within a 64-wide block, so any
    deviation from the non-CP layout leaks ~1 BF16 ULP per layer of
    noise that compounds across 43 layers and shifts greedy decode onto
    OOV vocab indices.  See `project_dsv4_cp_ep_wrong_output` memory.

    Entries beyond each row's valid window are -1 so the sparse_attn
    kernel masks them out.
    """
    device = global_positions.device
    S_local = int(global_positions.shape[0])
    if use_ring_layout:
        offsets = torch.arange(window_size, device=device)
        base = global_positions.unsqueeze(1)
        idxs = (base % window_size + 1 + offsets.unsqueeze(0)) % window_size
        valid_count = torch.clamp(global_positions + 1, max=window_size)
        invalid = offsets.unsqueeze(0) < window_size - valid_count.unsqueeze(1)
        matrix = torch.where(invalid, torch.full_like(idxs, -1), idxs)
        return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()
    W = min(window_size, max(seq_len_total, 1))
    base = global_positions.unsqueeze(1)
    window_start = (base - W + 1).clamp_min(0)
    offs = torch.arange(W, device=device)
    matrix = window_start + offs
    invalid = (matrix > base) | (matrix >= seq_len_total)
    matrix = torch.where(invalid, torch.full_like(matrix, -1), matrix)
    if W < window_size:
        pad = torch.full(
            (S_local, window_size - W), -1, dtype=matrix.dtype, device=device
        )
        matrix = torch.cat([matrix, pad], dim=1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _get_window_topk_idxs_varlen(
    window_size: int,
    cu_seqlens: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    req_id_per_token: torch.Tensor,
) -> torch.Tensor:
    """Returns ``[T_total, window_size]`` **int32** flat-KV indices.

    For request b, token t at local pos p = position_ids[t] - prefix_lengths[b]:
      * window covers local positions [max(0, p - win + 1), p]
      * flat indices = cu_seqlens[b] + (those local positions), all within
        [cu_seqlens[b], cu_seqlens[b] + S_b)
      * tail slots beyond the valid window get -1 (kernel masks them out)

    Only consumed by the cold ``_attn_fp8_swa_via_kv_full`` path. Continuation
    prefill uses ``combined_indices`` (workspace coordinates, M*batch_idx+slot)
    via ``_attn_fp8_swa_via_concat``, so prefix-tail KV does NOT need to be
    represented here.

    **CP alignment:** under cp_size > 1 the caller passes GLOBAL per-request
    positions for each rank-local token plus the full per-request
    ``cu_seqlens`` view. The formula therefore emits row indices into the
    all-gathered ``kv_full[seq_len_full]`` while preserving request
    boundaries for B>=1.

    Mirrors the right-pad slot ordering of ``_get_window_topk_idxs`` (sparse_attn
    kernel block reductions are not invariant to slot ordering — see comment
    on ``_get_window_topk_idxs_cp`` line 281+).

    **Dtype:** internal math is int32 — every value (T_total, max_seq_len,
    batch_size, accumulated cu_seqlens) is bounded by max_seq_len * batch_size
    which is well within int32 (≪ 2^31). Only ``req_id_per_token`` is cast
    to int64 because ``torch.gather`` requires int64 indices. The downstream
    ``_attn_fp8_swa_via_kv_full`` casts to int32 anyway, so int32 here saves
    a 64MB → 32MB allocation at T=16K, win=512.
    """
    position_ids = _flat_1d(position_ids)
    req_id_per_token = _flat_1d(req_id_per_token)
    cu_seqlens = _flat_1d(cu_seqlens)
    prefix_lengths = _flat_1d(prefix_lengths)
    device = position_ids.device
    req_id_idx = req_id_per_token.to(device=device, dtype=torch.long)
    cu_seqlens_i32 = cu_seqlens.to(device=device, dtype=torch.int32)
    prefix_lengths_i32 = prefix_lengths.to(device=device, dtype=torch.int32)
    positions_i32 = position_ids.to(device=device, dtype=torch.int32)
    prefix_per_token = prefix_lengths_i32.gather(0, req_id_idx)
    req_start_in_flat = cu_seqlens_i32.gather(0, req_id_idx)
    local_query_pos = positions_i32 - prefix_per_token
    window_offset = torch.arange(window_size, device=device, dtype=torch.int32)
    query_pos_col = local_query_pos.unsqueeze(1)
    window_local_start = (query_pos_col - window_size + 1).clamp_min(0)
    window_local_idx = window_local_start + window_offset
    is_causal_pad = window_local_idx > query_pos_col
    window_flat_idx = req_start_in_flat.unsqueeze(1) + window_local_idx
    return torch.where(
        is_causal_pad, torch.full_like(window_flat_idx, -1), window_flat_idx
    ).contiguous()


class SwaPrefillMeta(NamedTuple):
    """FP8 prefill metadata bundle — built once per ``_prefill_common_setup``
    call for **all FP8 KV-cache layers** (compress_ratio 0/4/128 alike).

    Two field groups with different lifecycles:

    1. **FP8 KV cache write metadata** (used by ``_prefill_write_swa_fp8_paged``)
       — built for every FP8 layer regardless of ``compress_ratio``,
       because CSA/HCA layers still need to populate the SWA pool for
       downstream decode reads.

       Fields: ``slot_mapping``, ``query_start_loc``, ``combined_seq_lens``.
       ``None`` on warmup forward (``self._kv_cache is None``).

    2. **SWA-only attention metadata** (used by ``_attn_fp8_swa_via_kv_full``
       / ``_attn_fp8_swa_via_concat``) — built only when
       ``compress_ratio == 0``. CSA/HCA layers don't read the SWA pool
       directly during attention (they go through compressor/indexer).

       ``topk_length_kv_full`` is cache-independent so it's also set on
       warmup. ``cache_*`` / ``combined_indices`` etc. are skipped on
       warmup or non-SWA-only layers.
    """

    slot_mapping: Optional[torch.Tensor]
    query_start_loc: Optional[torch.Tensor]
    combined_seq_lens: Optional[torch.Tensor]
    topk_length_kv_full: Optional[torch.Tensor]
    combined_gather_lens: Optional[torch.Tensor]
    combined_gather_len_max: int
    M: int
    cache_seq_lens: Optional[torch.Tensor]
    cache_gather_lens: Optional[torch.Tensor]
    prefix_len_max: int
    combined_indices: Optional[torch.Tensor]
    combined_lens: Optional[torch.Tensor]
    slot_in_flat: Optional[torch.Tensor]
    cache_slot_mapping: Optional[torch.Tensor] = None
    slot_compaction: Optional[CPByteSlicedSlotCompaction] = None
    cache_compaction: Optional[CPByteSlicedSlotCompaction] = None


class WorkspaceMeta(NamedTuple):
    """Static index/dim metadata for the vLLM-style workspace + dual-gather
    + ``combine_topk_swa_indices`` + ``flash_mla_sparse_fwd`` flow used by
    both CSA and HCA paths. Built once per (forward, ratio) by
    :meth:`Attention._build_workspace_meta`.

    Workspace layout under varlen B>=1 — **N_max-padded** so the per-request
    compressed and SWA regions land at the same column offset across the
    batch (lets ``combine_topk_swa_indices`` keep its scalar ``M`` / ``N``
    contract). For each request b in ``workspace[b, :, :]``:

        ``[0,             N_b              )`` — request b compressed
        ``[N_b,           N_max            )`` — zero pad
        ``[N_max,         N_max + gather_b )`` — request b SWA stream
                  (first ``P_b = min(sp_b, win-1)`` rows are prefix tail
                  dequant'd from pool; next ``S_b`` rows are overwritten
                  by fresh BF16 new K via ``new_k_slot_in_flat``)
        ``[N_max+gather_b, M               )`` — zero pad

    with ``N_max = max_b N_b``, ``gather_len_max = max_b gather_b``,
    ``M = N_max + gather_len_max``.

    Every elementwise operation needed by ``_attn_via_workspace`` is
    pre-baked here so the hot path stays kernel-only (dequant ×2 +
    ``index_copy_`` + ``combine_topk`` + ``flash_mla_sparse_fwd``).
    """

    M: int
    N: int
    swa_eb: int
    cmp_eb: int
    swa_bt_int32: torch.Tensor
    cmp_bt_int32: torch.Tensor
    swa_seq_lens: torch.Tensor
    cmp_seq_lens: torch.Tensor
    swa_gather_lens: torch.Tensor
    swa_cache_seq_lens: torch.Tensor
    swa_cache_gather_lens: torch.Tensor
    qsl: torch.Tensor
    dense_cmp_topk: Optional[torch.Tensor]
    new_k_slot_in_flat: torch.Tensor
    cmp_reader: Optional["CompressedKPoolReader"] = None
    use_cp_raw_q_merge: bool = False
    swa_cache_slot_mapping: Optional[torch.Tensor] = None
    swa_cache_compaction: Optional[CPByteSlicedSlotCompaction] = None


class CsaPrefillMeta(NamedTuple):
    """CSA-layer prefill metadata (compress_ratio == 4). Carries the
    nested indexer metadata + the main CSA compressor write metadata so
    the per-layer ``_forward_prefill_csa`` is kernel-only."""

    indexer_meta: Any
    compressor_meta: Any
    workspace_meta: Optional[WorkspaceMeta]


class HcaPrefillMeta(NamedTuple):
    """HCA-layer prefill metadata (compress_ratio == 128). Carries the
    main HCA compressor write metadata. HCA generates dense compressed
    indices in-line so no indexer is involved."""

    compressor_meta: Any
    workspace_meta: Optional[WorkspaceMeta]


class PrefillMeta(NamedTuple):
    """Per-call prefill metadata, layer-invariant within a
    ``compress_ratio`` bucket. Built once per (forward, ratio) by
    :meth:`Attention._build_shared_prefill_meta` and broadcast to every
    same-ratio layer via :meth:`Attention._set_prefill_meta_shared`.

    The three sub-metadata fields are mutually exclusive — exactly one
    is non-None per layer, gated by ``compress_ratio``:
      * ``compress_ratio == 0``   → ``swa_meta`` only (SWA-only path)
      * ``compress_ratio == 4``   → ``swa_meta`` + ``csa_meta``
      * ``compress_ratio == 128`` → ``swa_meta`` + ``hca_meta``

    ``swa_meta`` is set on every FP8 layer because all paths still
    write the SWA pool for downstream decode.
    """

    seqlen: int
    seqlen_full: int
    rd: int
    device: torch.device
    cp_ctx: Optional[CPContext]
    cp_on: bool
    freqs_cis: torch.Tensor
    topk_idxs: torch.Tensor
    sp_int: int
    any_cont: bool
    row_seqlens_full: torch.Tensor
    use_varlen: bool = False
    sp_per_req: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    batch_size: int = 1
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    req_id_per_token: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    swa_meta: Optional[SwaPrefillMeta] = None
    csa_meta: Optional[CsaPrefillMeta] = None
    hca_meta: Optional[HcaPrefillMeta] = None
    workspace: Optional[PrefillWorkspace] = None
    freqs_cis_source_id: int = 0
    request_row_slices: Optional[Tuple[slice, ...]] = None


class PrefillQKV(NamedTuple):
    """Q/KV intermediate produced by ``_prefill_compute_qkv``.

    ``qr`` is fed to the indexer (CSA layers); ``q`` is the dense Q.
    ``kv_full`` is the all-gathered KV under CP; equals ``kv`` otherwise.
    Current-layer SWA KV all-gather intentionally stays synchronous: it is not
    part of the prefill overlap feature because the resulting tensor remains
    live across Q materialization. The CP-aware sequence length lives on
    ``PrefillMeta.seqlen_full``.

    ``q`` starts ``None``: its ``q_lora_b`` + RoPE are DEFERRED to
    :meth:`AttentionFP8._materialize_prefill_q` (called just before the
    ``flash_mla_sparse_fwd`` consumers) so the 16 GiB Q buffer can share the
    union ``PrefillWorkspace`` storage with the compressor gather/restore
    buffers, which are dead by then. Consumers enter only after materialization.
    """

    qr: torch.Tensor
    q: Optional[torch.Tensor]
    kv_full: torch.Tensor


class AttentionFP8(nn.Module):
    _prefill_cp_overlap_hard_on: bool = False

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-06,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Reads ``W.v4_attn_*`` for dense attention weights, ``W.v4_compressor_*``
        for the outer compressor, ``W.v4_indexer_*`` (forwarded) for the
        indexer."""
        super().__init__()
        _configure_flash_mla_l2_persist()
        self.layer_id = layer_id
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.eps = norm_eps
        self.softmax_scale = head_dim ** (-0.5)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_heads = n_heads // tp_size
        self.n_groups = o_groups // tp_size
        n_heads_local = self.n_heads
        n_groups_local = self.n_groups
        wq_b_row_slice = slice(
            tp_rank * n_heads_local * head_dim, (tp_rank + 1) * n_heads_local * head_dim
        )
        wo_a_row_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        wo_b_col_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        attn_sink_slice = slice(tp_rank * n_heads_local, (tp_rank + 1) * n_heads_local)
        from rtp_llm.utils.model_weight import W

        def _fp8_w_s(w_tag, s_tag, row_slice=None, col_slice=None):
            """Pull (weight, scale) by W tag with optional TP slicing,
            then build a CudaFp8DeepGEMMLinear via ``_v4_fp8_linear``.

            Scale slicing depends on layout:
              * legacy raw UE8M0 [N//128, K//128]: row/col strides are
                block-128 → ``slice.start // 128``.
              * framework packed int32 [N, K//128//4]: N is fully
                expanded (slice by full N stride) and K is packed 4×
                (slice by ``slice.start // 512``)."""
            raw_w = layer_weights[w_tag]
            raw_s = layer_weights[s_tag]
            w = raw_w
            s = raw_s
            scale_is_packed_int32 = s.dtype == torch.int32
            if row_slice is not None:
                w = w[row_slice]
                if scale_is_packed_int32:
                    s = s[row_slice]
                else:
                    scale_rows = raw_w.shape[0] // raw_s.shape[0]
                    s = s[row_slice.start // scale_rows : row_slice.stop // scale_rows]
            if col_slice is not None:
                w = w[:, col_slice]
                if scale_is_packed_int32:
                    s = s[:, col_slice.start // 512 : col_slice.stop // 512]
                else:
                    scale_cols = raw_w.shape[1] // raw_s.shape[1]
                    s = s[
                        :, col_slice.start // scale_cols : col_slice.stop // scale_cols
                    ]
            if row_slice is not None or col_slice is not None:
                w = w.contiguous()
                s = s.contiguous()
            linear = _v4_fp8_linear(w, s)
            return linear

        self.wq_a_wkv = merge_v41_qkv_weights(
            layer_weights,
            W.v4_attn_wq_a_w,
            W.v4_attn_wq_a_s,
            W.v4_attn_wkv_w,
            W.v4_attn_wkv_s,
        )
        self.wq_a = _fp8_w_s(W.v4_attn_wq_a_w, W.v4_attn_wq_a_s)
        self.wq_b = _fp8_w_s(
            W.v4_attn_wq_b_w,
            W.v4_attn_wq_b_s,
            row_slice=wq_b_row_slice if tp_size > 1 else None,
        )
        self.wkv = _fp8_w_s(W.v4_attn_wkv_w, W.v4_attn_wkv_s)
        wo_a_w = layer_weights[W.v4_attn_wo_a_w]
        wo_a_s = layer_weights[W.v4_attn_wo_a_s]
        wo_a_raw_w = wo_a_w
        wo_a_raw_s = wo_a_s
        if tp_size > 1:
            wo_a_w = wo_a_w[wo_a_row_slice].contiguous()
            if wo_a_s.dtype == torch.int32:
                wo_a_s = wo_a_s[wo_a_row_slice].contiguous()
            else:
                scale_rows = wo_a_raw_w.shape[0] // wo_a_raw_s.shape[0]
                wo_a_s = wo_a_s[
                    wo_a_row_slice.start
                    // scale_rows : wo_a_row_slice.stop
                    // scale_rows
                ].contiguous()
        self.wo_a_w = wo_a_w
        self.wo_a_s = wo_a_s
        K_local = n_heads_local * head_dim // n_groups_local
        _stk_w, _stk_s = _prepare_wo_a_stacked(
            wo_a_w, wo_a_s, n_groups_local, o_lora_rank, K_local
        )
        self.register_buffer("_wo_a_stk_w", _stk_w, persistent=False)
        self.register_buffer("_wo_a_stk_s", _stk_s, persistent=False)
        self.wo_b = _fp8_w_s(
            W.v4_attn_wo_b_w,
            W.v4_attn_wo_b_s,
            col_slice=wo_b_col_slice if tp_size > 1 else None,
        )
        self.q_norm = layer_weights[W.v4_attn_q_norm]
        self.kv_norm = layer_weights[W.v4_attn_kv_norm]
        attn_sink_full = layer_weights[W.v4_attn_sink]
        self.attn_sink = (
            attn_sink_full[attn_sink_slice].contiguous()
            if tp_size > 1
            else attn_sink_full
        )
        if compress_ratio:
            outer_cmp_weights = {
                "ape": layer_weights[W.v4_compressor_ape],
                "wkv": layer_weights[W.v4_compressor_wkv],
                "wgate": layer_weights[W.v4_compressor_wgate],
                "norm": layer_weights[W.v4_compressor_norm],
            }
            from rtp_llm.models_py.modules.dsv41.fp8.compressor import CompressorFP8

            self.compressor = CompressorFP8(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=max_batch_size,
                cp_role=_CP_ROLE_MAIN,
                norm_eps=norm_eps,
                compressor_weights=outer_cmp_weights,
            )
            self.compressor._profile_label = (
                f"L{layer_id:02d}.csa_main"
                if compress_ratio == 4
                else f"L{layer_id:02d}.hca_main"
            )
            self.compressor.configure_kv_cache_shape(max_seq_len // compress_ratio)
            if compress_ratio == 4:
                from rtp_llm.models_py.modules.dsv41.fp8.indexer import IndexerFP8

                self.indexer = IndexerFP8(
                    dim=dim,
                    q_lora_rank=q_lora_rank,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    rope_head_dim=rope_head_dim,
                    index_topk=index_topk,
                    compress_ratio=compress_ratio,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    layer_weights=layer_weights,
                )
                if self.indexer.compressor is not None:
                    self.indexer.compressor._profile_label = (
                        f"L{layer_id:02d}.csa_nested_indexer"
                    )
                    self.indexer.compressor.configure_kv_cache_shape(
                        max_seq_len // compress_ratio
                    )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None
        kv_cache_size = window_size + (
            max_seq_len // compress_ratio if compress_ratio else 0
        )
        if compress_ratio:
            self._rope_base = compress_rope_theta
            self._rope_o_seq_len = original_seq_len
        else:
            self._rope_base = rope_theta
            self._rope_o_seq_len = 0
        self._rope_factor = rope_factor
        self._rope_beta_fast = beta_fast
        self._rope_beta_slow = beta_slow
        self._rope_dim = rope_head_dim
        self._rope_max_seq_len = max_seq_len
        freqs_cis = precompute_freqs_cis(
            rope_head_dim,
            max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        self.freqs_cis = freqs_cis
        self._fp8_decode_op: Optional[Any] = None
        self._cp_ctx: Optional[CPContext] = None
        self._prefill_meta_shared: Optional["PrefillMeta"] = None
        self._kv_cache: Optional[Any] = None
        self._block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        idx_hd = index_head_dim
        coff_csa = 2
        coff_idx = 2
        kv_spec = (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
        indexer_kv_spec = (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES)
        self._pool_spec: Dict[int, tuple] = {
            SWA_KV: kv_spec,
            CSA_KV: kv_spec,
            HCA_KV: kv_spec,
            INDEXER_KV: indexer_kv_spec,
            CSA_STATE: (torch.float32, 2 * coff_csa * head_dim),
            HCA_STATE: (torch.float32, 2 * head_dim),
            INDEXER_STATE: (torch.float32, 2 * coff_idx * idx_hd),
        }

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind CP context.  When active on a prefill call, ``forward``
        does rank-local Q × FULL-KV attention: RoPE uses global
        positions; the rank-local KV is all-gathered + padding-stripped
        so every rank sees the same full sliding-window KV; the
        sliding-window + compressed topk indices are computed relative
        to that full-KV layout; sparse_attn runs on rank-local Q rows
        only so the output is ``[B, chunk_length, H, D]`` — the frame-
        work then all-gathers across ranks and strips padding."""
        self._cp_ctx = cp_ctx

    def _pool_view(self, attn_type: int) -> Optional[torch.Tensor]:
        """Return a flat ``[total_slots, vec_dim]`` typed view of the
        framework BlockPool for this layer + attn_type, or ``None`` if
        the pool isn't allocated (e.g. SWA-only layer has no CSA/HCA
        pool).  Delegates to ``KVCache.get_layer_cache(layer_id,
        attn_type)`` — no Python-side descriptor cache."""
        if self._kv_cache is None:
            return None
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return None
        attn_type_enum = _CACHE_TAG_BY_NAME.get(attn_type)
        if attn_type_enum is None:
            return None
        try:
            layer_kv = self._kv_cache.get_layer_cache(
                self.layer_id,
                (
                    getattr(self, "_swa_cache_region", SWA_KV)
                    if attn_type_enum == SWA_KV
                    else attn_type_enum
                ),
            )
        except RuntimeError:
            return None
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return None
        vec_dtype, vec_dim = spec
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim * vec_dtype.itemsize
        if bytes_per_entry <= 0 or stride_bytes < bytes_per_entry:
            return None
        eb = self._pool_entries_per_block(attn_type)
        useful_bytes = eb * bytes_per_entry
        raw_u8 = base.view(torch.uint8)
        if raw_u8.shape[1] < useful_bytes:
            return None
        if vec_dtype == torch.uint8 and stride_bytes > useful_bytes:
            return None
        return raw_u8[:, :useful_bytes].view(vec_dtype).view(-1, vec_dim)

    def _pool_view_3d_fp8(self, attn_type: int) -> Optional[torch.Tensor]:
        """Return ``[num_blocks, eb, ENTRY_BYTES]`` uint8 view of an FP8 KV
        pool, respecting C++-side TMA padding (per-block stride may exceed
        ``eb * ENTRY_BYTES``). The flat 2D form ``_pool_view`` returns is
        invalid here because the slice it produces is non-contiguous and
        can't be ``.view()``'d through the dtype/shape chain.
        """
        if self._kv_cache is None:
            return None
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return None
        attn_type_enum = _CACHE_TAG_BY_NAME.get(attn_type)
        if attn_type_enum is None:
            return None
        try:
            layer_kv = self._kv_cache.get_layer_cache(
                self.layer_id,
                (
                    getattr(self, "_swa_cache_region", SWA_KV)
                    if attn_type_enum == SWA_KV
                    else attn_type_enum
                ),
            )
        except RuntimeError:
            return None
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return None
        vec_dtype, vec_dim = spec
        if vec_dtype != torch.uint8:
            return None
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim
        if bytes_per_entry <= 0 or stride_bytes < bytes_per_entry:
            return None
        eb = self._pool_entries_per_block(attn_type)
        raw_u8 = base.view(torch.uint8)
        num_blocks = int(raw_u8.shape[0])
        return raw_u8.as_strided(
            (num_blocks, eb, bytes_per_entry), (stride_bytes, bytes_per_entry, 1)
        )

    def _pool_raw_u8(self, attn_type: int) -> Optional[torch.Tensor]:
        if self._kv_cache is None:
            return None
        attn_type_enum = _CACHE_TAG_BY_NAME.get(attn_type)
        if attn_type_enum is None:
            return None
        try:
            layer_kv = self._kv_cache.get_layer_cache(
                self.layer_id,
                (
                    getattr(self, "_swa_cache_region", SWA_KV)
                    if attn_type_enum == SWA_KV
                    else attn_type_enum
                ),
            )
        except RuntimeError:
            return None
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return None
        return base.view(torch.uint8)

    def _swa_cp_byte_sliced(self) -> bool:
        cp_ctx = getattr(self, "_cp_ctx", None)
        return (
            cp_ctx is not None
            and int(getattr(cp_ctx, "cp_size", 1)) > 1
            and bool(getattr(cp_ctx, "kv_cache_sharded", False))
        )

    def _swa_entries_per_block(self) -> int:
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        if self._swa_cp_byte_sliced():
            raw = self._pool_raw_u8(SWA_KV)
            cp_ctx = getattr(self, "_cp_ctx", None)
            if raw is not None and cp_ctx is not None:
                return (
                    int(raw.shape[1]) * int(cp_ctx.cp_size) // _DSV4_FP8_KV_ENTRY_BYTES
                )
        return self._pool_entries_per_block(SWA_KV)

    def _build_swa_cp_byte_compaction(
        self,
        slot_mapping: torch.Tensor,
        full_entries_per_block: int,
        validation_site: str,
        negative_mode: str,
        gather_lens: Optional[torch.Tensor] = None,
    ) -> Optional[CPByteSlicedSlotCompaction]:
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        if not self._swa_cp_byte_sliced():
            return None
        raw = self._pool_raw_u8(SWA_KV)
        if validation_site not in _SWA_CP_RR_LOGGED_SITES:
            _SWA_CP_RR_LOGGED_SITES.add(validation_site)
            cp_ctx = getattr(self, "_cp_ctx", None)
            logging.info(
                "[dsv4-cp-rr] byte-sliced SWA cache path engaged at %s (cp_size=%s, cp_rank=%s, entries_per_block=%d)",
                validation_site,
                getattr(cp_ctx, "cp_size", None),
                getattr(cp_ctx, "cp_rank", None),
                int(full_entries_per_block),
            )
        return build_cp_byte_sliced_slot_compaction(
            slot_mapping,
            full_entries_per_block=full_entries_per_block,
            num_blocks=int(raw.shape[0]),
            validation_site=validation_site,
            negative_mode=negative_mode,
            gather_lens=gather_lens,
        )

    def _pool_entries_per_block(self, attn_type: int) -> int:
        """Derive ``entries_per_block`` from the framework pool tensor for
        this layer + attn_type.  Returns 0 if pool unavailable."""
        if self._kv_cache is None:
            return 0
        spec = self._pool_spec.get(attn_type)
        if spec is None:
            return 0
        attn_type_enum = _CACHE_TAG_BY_NAME.get(attn_type)
        if attn_type_enum is None:
            return 0
        try:
            layer_kv = self._kv_cache.get_layer_cache(
                self.layer_id,
                (
                    getattr(self, "_swa_cache_region", SWA_KV)
                    if attn_type_enum == SWA_KV
                    else attn_type_enum
                ),
            )
        except RuntimeError:
            return 0
        base = layer_kv.kv_cache_base
        if base is None or base.numel() == 0 or base.dim() != 2:
            return 0
        vec_dtype, vec_dim = spec
        stride_bytes = int(base.shape[1]) * int(base.element_size())
        bytes_per_entry = vec_dim * vec_dtype.itemsize
        if bytes_per_entry <= 0:
            return 0
        from ._kv_cache_utils import pool_entry_count

        tag = (
            getattr(self, "_swa_cache_region", SWA_KV)
            if attn_type == SWA_KV
            else attn_type
        )
        entries = pool_entry_count(self._kv_cache, tag)
        capacity = stride_bytes // bytes_per_entry
        # A byte-sliced SWA tensor is only a partial page. Its full logical
        # ring is read by _swa_entries_per_block after CP reconstruction.
        return min(entries, capacity) if entries is not None else capacity

    def _prefill_paged_write_kv(
        self, attn_type: int, source_buf: torch.Tensor, bsz: int
    ) -> None:
        """Phase F generic dual-write: mirror ``source_buf[:bsz, :T]`` into
        the framework BlockPool of ``attn_type``. No-op when no KVCache
        handle / block table bound, or the pool isn't allocated for this
        layer. Sentinel block_id ≤ 0 entries are skipped via
        ``mask_negative=True``.

        Supports ``bsz >= 1``.  ``self._block_tables_by_type[attn_type]``
        must carry at least ``bsz`` rows; each row's block_id list addresses
        that request's own pool slots.  slot_mapping is built as ``[B, T]``
        via double-axis ``bt[b_idx, block_in_seq]`` so per-row block
        assignment is respected.  bsz==1 produces byte-equal slot_mapping
        to the historical scalar-row implementation."""
        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        bt = self._block_tables_by_type.get(attn_type)
        if bt is None or bt.numel() == 0:
            return
        pool_view = self._pool_view(attn_type)
        eb = self._pool_entries_per_block(attn_type)
        if pool_view is None or eb <= 0:
            return
        T = int(source_buf.shape[1])
        D = int(source_buf.shape[2])
        if T == 0:
            return
        device = source_buf.device
        max_blocks = bt.shape[1]
        pool_capacity = max_blocks * eb
        pos = torch.arange(T, device=device, dtype=torch.long)
        in_capacity_row = pos < pool_capacity
        safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
        block_in_seq = safe_pos // eb
        in_block = safe_pos % eb
        bt_long = bt.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]
        in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)
        valid = (block_id >= 0) & in_capacity
        slot_per = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.full_like(block_id, -1)
        )
        slot_mapping = slot_per.reshape(-1)
        buf_flat = source_buf[:bsz].reshape(bsz * T, D)
        write_kv_to_pool(buf_flat, slot_mapping, pool_view, mask_negative=True)

    def _prefill_paged_write_kv_range(
        self, attn_type: int, source_buf: torch.Tensor, bsz: int, write_start: int
    ) -> None:
        if self._kv_cache is None or self._block_tables_by_type is None:
            return
        from rtp_llm.models_py.modules.dsv4.fp8.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        bt = self._block_tables_by_type.get(attn_type)
        pool_view = self._pool_view(attn_type)
        eb = self._pool_entries_per_block(attn_type)
        if bt is None or bt.numel() == 0 or pool_view is None or (eb <= 0):
            return
        T = int(source_buf.shape[1])
        D = int(source_buf.shape[2])
        if T <= 0:
            return
        device = source_buf.device
        pos = torch.arange(
            write_start, write_start + T, device=device, dtype=torch.long
        )
        block_in_seq = pos // eb
        in_block = pos % eb
        in_capacity = block_in_seq < int(bt.shape[1])
        safe_block = torch.where(
            in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[b_idx, safe_block.unsqueeze(0).expand(bsz, -1)]
        valid = in_capacity.unsqueeze(0) & (block_id >= 0)
        slot_per = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.full_like(block_id, -1)
        )
        write_kv_to_pool(
            source_buf[:bsz].reshape(bsz * T, D),
            slot_per.reshape(-1),
            pool_view,
            mask_negative=True,
        )

    def _prefill_read_swa_from_pool(
        self,
        bsz: int,
        sp: Union[int, torch.Tensor],
        row_seqlens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Reconstruct the dense SWA ring ``[B, window_size, head_dim]`` from
        sparse absolute-token block tables.

        ``topk_idxs`` indexes SWA by ring slot (``global_pos % window_size``),
        while allocator block tables retain only the last absolute token
        blocks.  For each ring slot we locate the latest global position that
        maps to it and gather that absolute pool slot.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return None
        pool_view = self._pool_view(SWA_KV)
        eb = self._pool_entries_per_block(SWA_KV)
        if pool_view is None or eb <= 0:
            return None
        swa_tokens_per_block = _dsv4_pool_tokens_per_block(
            self._kv_cache, region=SWA_KV
        )
        win = self.window_size
        if win <= 0:
            return None
        device = pool_view.device
        dtype = torch.bfloat16
        max_blocks = bt.shape[1]
        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=torch.long)
            if sp_t.dim() == 0:
                sp_t = sp_t.unsqueeze(0)
            if sp_t.numel() == 1 and bsz > 1:
                sp_t = sp_t.expand(bsz)
        else:
            sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)
        if row_seqlens is None:
            seq_t = torch.full((bsz,), win, device=device, dtype=torch.long)
        else:
            seq_t = row_seqlens.to(device=device, dtype=torch.long)
            if seq_t.dim() == 0:
                seq_t = seq_t.unsqueeze(0)
            if seq_t.numel() == 1 and bsz > 1:
                seq_t = seq_t.expand(bsz)
        last_global = sp_t + seq_t - 1
        window_start = torch.clamp(last_global - win + 1, min=0)
        ring_pos = torch.arange(win, device=device, dtype=torch.long)
        delta = torch.remainder(last_global.unsqueeze(1) - ring_pos.unsqueeze(0), win)
        global_pos = last_global.unsqueeze(1) - delta
        valid_pos = (seq_t.unsqueeze(1) > 0) & (global_pos >= window_start.unsqueeze(1))
        block_in_seq = global_pos // int(swa_tokens_per_block)
        in_block = global_pos % eb
        in_capacity = (block_in_seq >= 0) & (block_in_seq < max_blocks)
        safe_block = torch.where(
            in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt.to(torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[:bsz][b_idx, safe_block]
        valid = valid_pos & in_capacity & (block_id >= 0)
        safe_slot = torch.where(
            valid, block_id * eb + in_block, torch.zeros_like(block_id)
        )
        gathered = pool_view.index_select(0, safe_slot.reshape(-1))
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        return out.view(bsz, win, self.head_dim).contiguous()

    def _prefill_read_swa_dense_abs_from_pool(
        self,
        bsz: int,
        sp: Union[int, torch.Tensor],
        row_seqlens: torch.Tensor,
        dense_len: int,
        current_kv_full: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build a dense absolute SWA view for continuation prefill.

        Continuation prefill attention needs the SWA window as it existed at
        each query position, not the final ring after this suffix has been
        written.  The pool contains the prefix tail at entry; overlay the
        current suffix KV from ``current_kv_full`` so absolute topk indices
        can read ``[prefix tail | current suffix]`` by token position.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(SWA_KV)
        if bt is None or bt.numel() == 0:
            return None
        pool_view = self._pool_view(SWA_KV)
        eb = self._pool_entries_per_block(SWA_KV)
        if pool_view is None or eb <= 0 or dense_len <= 0:
            return None
        swa_tokens_per_block = _dsv4_pool_tokens_per_block(
            self._kv_cache, region=SWA_KV
        )
        device = pool_view.device
        dtype = torch.bfloat16
        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=torch.long)
            if sp_t.dim() == 0:
                sp_t = sp_t.unsqueeze(0)
            if sp_t.numel() == 1 and bsz > 1:
                sp_t = sp_t.expand(bsz)
        else:
            sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)
        seq_t = row_seqlens.to(device=device, dtype=torch.long)
        if seq_t.dim() == 0:
            seq_t = seq_t.unsqueeze(0)
        if seq_t.numel() == 1 and bsz > 1:
            seq_t = seq_t.expand(bsz)
        pos = torch.arange(dense_len, device=device, dtype=torch.long)
        block_in_seq = pos // int(swa_tokens_per_block)
        in_block = pos % eb
        max_blocks = bt.shape[1]
        in_capacity_row = block_in_seq < max_blocks
        safe_block = torch.where(
            in_capacity_row, block_in_seq, torch.zeros_like(block_in_seq)
        )
        bt_long = bt[:bsz].to(device=device, dtype=torch.long)
        b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
        block_id = bt_long[b_idx, safe_block.unsqueeze(0).expand(bsz, -1)]
        valid = in_capacity_row.unsqueeze(0) & (block_id >= 0)
        safe_slot = torch.where(
            valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
        )
        gathered = pool_view.index_select(0, safe_slot.reshape(-1))
        if gathered.dtype != dtype:
            gathered = gathered.to(dtype)
        zero_row = torch.zeros((), dtype=dtype, device=device)
        out = torch.where(valid.reshape(-1).unsqueeze(-1), gathered, zero_row)
        out = out.view(bsz, dense_len, self.head_dim).contiguous()
        for b in range(bsz):
            sp_b = int(sp_t[b].item())
            seq_b = int(seq_t[b].item())
            if current_kv_full is not None and seq_b > 0 and (sp_b < dense_len):
                dst_end = min(sp_b + seq_b, dense_len)
                copy_len = dst_end - sp_b
                if copy_len > 0:
                    src = current_kv_full[b, :copy_len]
                    if src.dtype != dtype:
                        src = src.to(dtype)
                    out[b, sp_b:dst_end] = src
        return out.contiguous()

    def _set_compressor_pool_context(self) -> None:
        """#50: resolve CSA/HCA + INDEXER pool views + per-request block
        tables from ``self._kv_cache`` + ``self._block_tables_by_type`` and
        hand them to Compressor / Indexer via ``set_pool_context``.  Called
        once at the top of every forward/forward_decode; paired with
        :meth:`_clear_compressor_pool_context` in a try/finally so stale
        pool views don't leak across forwards."""
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
        )

        bt_by_type = self._block_tables_by_type
        if self.compressor is not None:
            if self.compress_ratio == 4:
                kv_at, state_at = (CSA_KV, CSA_STATE)
            elif self.compress_ratio == 128:
                kv_at, state_at = (HCA_KV, HCA_STATE)
            else:
                kv_at, state_at = (None, None)
            if kv_at is not None:
                kv_view = self._pool_view_3d_fp8(kv_at)
            else:
                kv_view = self._pool_view(kv_at) if kv_at is not None else None
            kv_bt = (
                bt_by_type.get(kv_at)
                if bt_by_type is not None and kv_at is not None
                else None
            )
            kv_eb = self._pool_entries_per_block(kv_at) if kv_at is not None else 0
            state_view = self._pool_view(state_at) if state_at is not None else None
            state_bt = (
                bt_by_type.get(state_at)
                if bt_by_type is not None and state_at is not None
                else None
            )
            state_eb = (
                self._pool_entries_per_block(state_at) if state_at is not None else 0
            )
            kv_tpb = (
                _dsv4_pool_tokens_per_block(self._kv_cache, region=kv_at)
                if kv_at is not None
                else 0
            )
            kv_owner_tpb = (
                self._kv_cache.get_seq_size_per_block(
                    kv_at if kv_at is not None else SWA_KV
                )
                if kv_at is not None and self._kv_cache is not None
                else 0
            )
            state_tpb = (
                _dsv4_pool_tokens_per_block(self._kv_cache, region=state_at)
                if state_at is not None
                else 0
            )
            self.compressor.set_pool_context(
                kv_view,
                kv_bt,
                kv_eb,
                state_view,
                state_bt,
                state_eb,
                state_tokens_per_block=state_tpb,
                kv_tokens_per_block=kv_tpb,
                kv_owner_tokens_per_block=kv_owner_tpb,
            )
        if self.indexer is not None:
            kv_view = self._pool_view_3d_fp8(INDEXER_KV)
            kv_bt = bt_by_type.get(INDEXER_KV) if bt_by_type is not None else None
            kv_eb = self._pool_entries_per_block(INDEXER_KV)
            state_view = self._pool_view(INDEXER_STATE)
            state_bt = bt_by_type.get(INDEXER_STATE) if bt_by_type is not None else None
            state_eb = self._pool_entries_per_block(INDEXER_STATE)
            kv_tpb = _dsv4_pool_tokens_per_block(self._kv_cache, region=INDEXER_KV)
            kv_owner_tpb = self._kv_cache.get_seq_size_per_block(
                kv_at if kv_at is not None else SWA_KV
            )
            state_tpb = _dsv4_pool_tokens_per_block(
                self._kv_cache, region=INDEXER_STATE
            )
            self.indexer.set_pool_context(
                kv_view,
                kv_bt,
                kv_eb,
                state_view,
                state_bt,
                state_eb,
                state_tokens_per_block=state_tpb,
                kv_tokens_per_block=kv_tpb,
                kv_owner_tokens_per_block=kv_owner_tpb,
            )

    def _clear_compressor_pool_context(self) -> None:
        if self.compressor is not None:
            self.compressor.clear_pool_context()
        if self.indexer is not None:
            self.indexer.clear_pool_context()

    def _prefill_paged_read_kv(
        self,
        attn_type: int,
        bsz: int,
        T: int,
        vec_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Phase E1 read-path counterpart to ``_prefill_paged_write_kv``.

        Gathers a ``[bsz, T, vec_dim]`` dense tensor from the framework
        BlockPool of ``attn_type`` using the same slot_mapping formula as
        the writer, so the write-then-read round trip is byte-equal on
        valid positions.  Sentinel positions (pos ≥ pool_capacity or
        unallocated block_id) are zero-filled.

        Supports ``bsz >= 1``.  Per-row block_id lookup uses
        ``bt[b_idx, block_in_seq]`` so each row reads from its own
        block allocation.  bsz==1 produces byte-equal output to the
        historical scalar-row implementation.

        Returns ``None`` when the ctx is unbound or the pool isn't
        registered for this layer.
        """
        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type.get(attn_type)
        if bt is None or bt.numel() == 0 or T == 0:
            return None
        eb = self._pool_entries_per_block(attn_type)
        if eb <= 0:
            return None
        pool_3d = self._pool_view_3d_fp8(attn_type)
        if pool_3d is None or pool_3d.shape[-1] != _DSV4_FP8_KV_ENTRY_BYTES:
            return None
        from rtp_llm.models_py.modules.dsv41.fp8._swa_dequant_triton import (
            dequantize_and_gather_k_cache,
        )

        HD_DEQUANT = 512
        out = torch.zeros((bsz, T, HD_DEQUANT), dtype=torch.bfloat16, device=device)
        seq_lens = torch.full((bsz,), T, dtype=torch.int32, device=device)
        bt_for_kernel = bt[:bsz].to(torch.int32).contiguous()
        dequantize_and_gather_k_cache(
            out, pool_3d, seq_lens, None, bt_for_kernel, eb, 0
        )
        return out

    def _gather_kv_cache_dense_from_pool(
        self,
        bsz: int,
        sp: Optional[Union[int, torch.Tensor]] = None,
        row_seqlens: Optional[torch.Tensor] = None,
        swa_dense_len: Optional[int] = None,
        swa_dense_override: Optional[torch.Tensor] = None,
        swa_T: Optional[int] = None,
        cmp_T: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Phase E1: reconstruct the ``[bsz, kv_cache_size, head_dim]``
        dense tensor that ``self.kv_cache[:bsz]`` presents, but sourced
        from the framework pools instead of the register_buffer mirror.

        Layout (matches register_buffer):
          ``[:, :swa_T, :]``             -- SWA_KV absolute-position stream
          ``[:, swa_T:swa_T+cmp_T, :]``  -- CSA_KV or HCA_KV compressed stream

        Returns ``None`` when ctx not bound — caller falls back to
        register_buffer.  SWA-only layers (compress_ratio == 0) get a bare
        ``[bsz, swa_T, hd]`` read.  ``swa_T`` defaults to ``window_size`` for
        decode-like ring callers; continuation prefill passes the absolute
        sequence end so every query can use linear absolute topk indices.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import CSA_KV, HCA_KV, SWA_KV

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        win = self.window_size
        hd = self.head_dim
        dtype = torch.bfloat16
        device = self.freqs_cis.device
        T_swa = int(swa_T) if swa_T is not None else win
        T_cmp = (
            int(cmp_T)
            if cmp_T is not None
            else (
                self.compressor._kv_cache_t
                if self.compressor is not None and self.compress_ratio
                else 0
            )
        )
        if swa_dense_override is not None:
            swa_dense = swa_dense_override
        elif swa_dense_len is not None:
            swa_dense = self._prefill_read_swa_dense_abs_from_pool(
                bsz, sp, row_seqlens, int(swa_dense_len)
            )
        elif sp is not None:
            swa_dense = self._prefill_read_swa_from_pool(bsz, sp, row_seqlens)
        else:
            swa_dense = self._prefill_paged_read_kv(
                SWA_KV, bsz, T_swa, hd, dtype, device
            )
        if swa_dense is None:
            return None
        if T_cmp <= 0 or self.compress_ratio == 0:
            return swa_dense
        cmp_at = CSA_KV if self.compress_ratio == 4 else HCA_KV
        cmp_dense = self._prefill_paged_read_kv(cmp_at, bsz, T_cmp, hd, dtype, device)
        if cmp_dense is None:
            return None
        return torch.cat([swa_dense, cmp_dense], dim=1)

    def init_rope_cache(self, device=None):
        """Recompute `freqs_cis` on the actual device — MUST be called after
        `model.to_empty(device=...)` since meta-tensor construction leaves the
        cached freqs as zeros. Pass ``device`` so the memoized
        ``precompute_freqs_cis`` returns the shared (params, device) tensor;
        all layers with identical rope params now point at the same object,
        which lets compressors share one prebuilt cos_sin_cache."""
        freqs_cis = precompute_freqs_cis(
            self._rope_dim,
            self._rope_max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            self._rope_factor,
            self._rope_beta_fast,
            self._rope_beta_slow,
            device=device,
        )
        self.freqs_cis = freqs_cis
        if self.compressor is not None:
            self.compressor.init_rope_cache(freqs_cis)
        if self.indexer is not None:
            self.indexer.freqs_cis = freqs_cis
            if self.indexer.compressor is not None:
                self.indexer.compressor.init_rope_cache(freqs_cis)

    def _get_fp8_decode_op(self):
        """Lazy-build the persistent ``SparseAttnV4DecodeFp8Op`` so its
        ``sched_meta`` cache survives across decode steps.

        Iter1' instantiated the op per call inside ``_forward_decode_body``
        which threw away the FlashMLA planner state on every call (60 layers
        × per step = 60 planner setups). Caching here cuts that to one setup
        per layer-type per process.
        """
        if self._fp8_decode_op is None:
            from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_sparse_attn_decode_op import (
                SparseAttnV4DecodeFp8Op,
            )

            self._fp8_decode_op = SparseAttnV4DecodeFp8Op(
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                softmax_scale=self.softmax_scale,
            )
        return self._fp8_decode_op

    def _rmsnorm_weighted(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # The historical native rtp_llm_ops.rmsnorm binding left the tree with the
        # bundled C++ FlashInfer; flashinfer-python owns the same kernel now.
        import flashinfer

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        return flashinfer.norm.rmsnorm(x_2d, weight, eps=self.eps).view(orig_shape)

    def _try_fused_qr_kv(
        self,
        x: torch.Tensor,
        shared_input_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        linear = getattr(self, "wq_a_wkv", None)
        if linear is None:
            return None
        from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import try_project_qr_kv

        return try_project_qr_kv(
            linear,
            x,
            self.q_norm,
            self.q_lora_rank,
            self.eps,
            quantized_input=shared_input_quant,
        )

    def _lin(
        self, layer: nn.Module, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            x_2d = x.reshape(-1, shape[-1])
            y = layer(x_2d, out=out) if out is not None else layer(x_2d)
            return y.view(*shape[:-1], y.shape[-1])
        return layer(x, out=out) if out is not None else layer(x)

    def _can_reuse_qkv_input_quant(self) -> bool:
        return (
            hasattr(self.wq_a, "quantize_input")
            and hasattr(self.wq_a, "forward_quantized")
            and hasattr(self.wkv, "forward_quantized")
            and (getattr(self.wq_a, "K", None) == getattr(self.wkv, "K", None))
            and (
                getattr(self.wq_a, "scale_ue8m0", None)
                == getattr(self.wkv, "scale_ue8m0", None)
            )
        )

    def can_fuse_prefill_attn_norm_input_quant(
        self, x: torch.Tensor, norm_weight: torch.Tensor
    ) -> bool:
        return (
            self._can_reuse_qkv_input_quant()
            and getattr(self.wq_a, "scale_ue8m0", False)
            and (x.dim() == 2)
            and (x.dtype == torch.bfloat16)
            and x.is_cuda
            and x.is_contiguous()
            and (norm_weight.dtype == torch.bfloat16)
            and norm_weight.is_cuda
            and norm_weight.is_contiguous()
            and (norm_weight.shape == (x.shape[-1],))
            and (x.shape[-1] % 128 == 0)
        )

    def prefill_fused_attn_norm_input_quant(
        self, x: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with record_function_range("dsv4.fp8.attn.qkv.fused_attn_norm_input_quant"):
            x_norm, x_fp8, x_scale = rmsnorm_fp8_quant_ue8m0(
                x,
                norm_weight,
                eps=norm_eps,
                group_size=128,
                clamp_eps=0.0001,
                out_norm=x,
            )
        return (x_norm, (x_fp8, x_scale))

    def _lin_from_shared_quant(
        self,
        layer: nn.Module,
        quantized_input: Tuple[torch.Tensor, torch.Tensor],
        shape: torch.Size,
    ) -> torch.Tensor:
        y = layer.forward_quantized(*quantized_input)
        return y.view(*shape[:-1], y.shape[-1])

    def _wo_a_einsum_from_fp8(
        self, o_fp8: torch.Tensor, o_scale: torch.Tensor, B: int, S: int
    ) -> torch.Tensor:
        """One ``fp8_einsum`` call on pre-quantized activations.

        ``fused_inv_rope_fp8_quant`` emits ``(o_fp8 [M, G, K], o_scale
        [M, G, K/512])`` in the exact layout ``deep_gemm.fp8_einsum``
        consumes, so the wo_a projection is a single einsum launch.
        Matches vLLM ``deepseek_v4_attention.py:325`` (same
        ``"bhr,hdr->bhd"`` + recipe ``(1, 1, 128)`` for SM100 UE8M0)."""
        M, G, _K = o_fp8.shape
        R = self.o_lora_rank
        out = torch.empty(M, G, R, dtype=torch.bfloat16, device=o_fp8.device)
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (self._wo_a_stk_w, self._wo_a_stk_s),
            out,
            recipe=(1, 1, 128),
        )
        return out.view(B, S, G, R)

    def forward_decode(
        self,
        x: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
        kv_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """qwen3-style: ``kv_cache`` (framework KVCache handle) flows in
        as a kwarg. Stashed for the duration of this call so pool views
        resolve via ``self._pool_view(...)``; block tables for decode
        come from ``attn_metadata.pool_block_tables`` — stashed onto
        ``self._block_tables_by_type`` here so Compressor / Indexer pool
        context resolution shares one code path with prefill."""
        with bind_attn_cache(self, kv_cache, attn_metadata.pool_block_tables):
            if self._swa_cache_region != SWA_KV:
                attn_metadata = attn_metadata.decoder_swa_metadata
            self._set_compressor_pool_context()
            try:
                return self._forward_decode_body(x, attn_metadata)
            finally:
                self._clear_compressor_pool_context()

    def _forward_decode_body(
        self, x: torch.Tensor, attn_metadata: "DSv4DecodeAttnMetadataFP8"
    ) -> torch.Tensor:
        """Decode attention body — thin dispatcher mirroring ``_forward_prefill``.

        Pipeline:
          1. Q/KV + per-request partial RoPE (``decode_compute_qkv``).
          2. FP8 SWA pool write (``decode_write_swa_fp8``).
          3. Per-``compress_ratio`` body:
             * ``0``   → :meth:`_forward_decode_swa_only`
             * ``4``   → :meth:`_forward_decode_csa`   (indexer + compressor)
             * ``128`` → :meth:`_forward_decode_hca`   (compressor, dense idx)
          4. Output projection (``decode_output_proj``).

        Compressor / Indexer freqs_cis is bound lazily on first call
        (pool context is set in :meth:`forward_decode`'s try/finally).
        """
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.output_proj import (
            decode_output_proj,
        )

        bsz, q_len, _ = x.size()
        T = bsz * q_len
        start_pos = attn_metadata.start_pos[:bsz]
        position_ids = attn_metadata.position_ids[:T]
        self._ensure_freqs_cis_bound()
        qkv = decode_compute_qkv(self, x, position_ids)
        with suppress(Exception):
            from rtp_llm.models_py.modules.dsv41.attn_type import (
                CSA_KV,
                HCA_KV,
                INDEXER_KV,
                SWA_KV,
            )
        self._decode_write_swa_fp8(qkv.kv, bsz, q_len, attn_metadata)
        if self.compress_ratio == 0:
            o = self._forward_decode_swa_only(qkv.q, bsz, q_len, attn_metadata)
        elif self.compress_ratio == 4:
            o = self._forward_decode_csa(
                x, qkv, bsz, q_len, start_pos, position_ids, attn_metadata
            )
        elif self.compress_ratio == 128:
            o = self._forward_decode_hca(
                x, qkv, bsz, q_len, start_pos, position_ids, attn_metadata
            )
        return decode_output_proj(self, o, qkv.freqs_cis, bsz, q_len)

    def _decode_write_swa_fp8(
        self,
        kv: torch.Tensor,
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
    ) -> None:
        """Write newly computed SWA KV into the FP8 584B/slot pool.

        Mirrors :meth:`_prefill_write_swa_fp8_paged` for decode — uses
        the framework-populated ``pool_write_slot_mappings[SWA_KV]``
        plus the CUDA ``concat_and_cache_mla("fp8_model1_mla", ...)``
        kernel dispatched by ``quantize_v4_kv_decode``.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8.decode.write_swa import (
            decode_write_swa_fp8,
        )

        slot_mapping = attn_metadata.pool_write_slot_mappings.get(SWA_KV)
        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        decode_write_swa_fp8(
            kv=kv,
            slot_mapping=slot_mapping,
            swa_pool_3d=swa_pool_3d,
            bsz=bsz,
            q_len=q_len,
            head_dim=self.head_dim,
        )

    def _decode_compressor_meta_from_metadata(
        self,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
        *,
        state_attn_type: int,
        kv_attn_type: int,
        bsz: int,
        q_len: int,
    ) -> CompressorMeta:
        """Return a CompressorMeta view backed by step-level buildmeta.

        The slot tensors are prepared once per decode step. Per-layer code
        only slices stable prefixes and then launches the compressor kernels.
        """
        state_slots = attn_metadata.compressor_state_slot_mappings.get(state_attn_type)
        kv_slots = attn_metadata.pool_write_slot_mappings.get(kv_attn_type)
        from rtp_llm.models_py.modules.dsv41.attn_type import CSA_KV, HCA_KV, INDEXER_KV

        ratio_by_kv = {CSA_KV: 4, INDEXER_KV: 4, HCA_KV: 128}
        ratio = ratio_by_kv.get(kv_attn_type)
        compressed_lens_per_token = (
            attn_metadata.compressed_lens_per_token[ratio][:bsz, :q_len]
            if ratio in attn_metadata.compressed_lens_per_token
            else None
        )
        T = bsz * q_len
        positions = attn_metadata.position_ids_long[:T]
        b_idx = attn_metadata.req_id_per_token_long[:T]
        return CompressorMeta(
            positions=positions,
            b_idx=b_idx,
            state_slots=state_slots[:T],
            kv_slots=kv_slots[:T],
            token_to_req=attn_metadata.req_id_per_token[:T],
            has_prefix=True,
            is_batched=q_len > 1,
            seq_start_per_req=attn_metadata.decode_seq_start_per_req[:bsz],
            cu_seq_per_req=attn_metadata.decode_cu_seq_per_req[: bsz + 1],
            compressed_lens_per_token=compressed_lens_per_token,
        )

    def _forward_decode_swa_only(
        self,
        q: torch.Tensor,
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
    ) -> torch.Tensor:
        """SWA-only layer (compress_ratio == 0) — one FlashMLA call over
        the FP8 SWA pool using per-request global slot ids translated
        from ``swa_abs_idx`` through the SWA block table."""
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8.decode.attention_kernels import (
            attn_fp8_swa_paged,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )

        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = (
            attn_metadata.pool_block_tables.get(SWA_KV)
            if attn_metadata.pool_block_tables
            else None
        )
        win = self.window_size
        T = bsz * q_len
        swa_global = attn_metadata.swa_global_slots[:T]
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()
        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=self.window_size,
            extra_attn_type=None,
        )
        swa_topk_length = (
            attn_metadata.swa_topk_length[:bsz]
            if attn_metadata.swa_topk_length is not None
            else None
        )
        return attn_fp8_swa_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
            topk_length=swa_topk_length,
        )

    def _forward_decode_csa(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
    ) -> torch.Tensor:
        """CSA layer (compress_ratio == 4). Indexer + main compressor
        both scatter into their pools; the indexer's topk buffer holds
        raw (pre +win) compressed local indices which the shared
        dual-pool epilogue consumes."""
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
        )

        indexer_compressor_meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=INDEXER_STATE,
            kv_attn_type=INDEXER_KV,
            bsz=bsz,
            q_len=q_len,
        )
        self.indexer.forward_decode_vectorized(
            x,
            qkv.qr,
            start_pos,
            attn_metadata.topk_buffer_compressed[:bsz],
            position_ids=position_ids,
            compressor_meta=indexer_compressor_meta,
        )
        csa_compressor_meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=CSA_STATE,
            kv_attn_type=CSA_KV,
            bsz=bsz,
            q_len=q_len,
        )
        self.compressor.forward_decode_vectorized(
            x, start_pos, meta=csa_compressor_meta, position_ids=position_ids
        )
        cmp_local_raw = attn_metadata.topk_buffer_compressed[:bsz]
        return self._forward_decode_compressed(
            qkv.q, cmp_local_raw, bsz, q_len, attn_metadata, cmp_attn_type=CSA_KV
        )

    def _forward_decode_hca(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
    ) -> torch.Tensor:
        """HCA layer (compress_ratio == 128). Compressor writes
        HCA_KV / HCA_STATE (no indexer). ``cmp_local_raw`` is the dense
        idx precomputed once per step by
        ``update_decode_metadata_in_place._build_dense_compressed_idxs``
        (reused across all HCA layers via ``topk_total_by_ratio[128]``)."""
        from rtp_llm.models_py.modules.dsv41.attn_type import HCA_KV, HCA_STATE

        hca_compressor_meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=HCA_STATE,
            kv_attn_type=HCA_KV,
            bsz=bsz,
            q_len=q_len,
        )
        self.compressor.forward_decode_vectorized(
            x, start_pos, meta=hca_compressor_meta, position_ids=position_ids
        )
        win = self.window_size
        tt_h = attn_metadata.topk_total_by_ratio.get(int(self.compress_ratio))
        cmp_local_raw = tt_h[:bsz, :, win:]
        return self._forward_decode_compressed(
            qkv.q, cmp_local_raw, bsz, q_len, attn_metadata, cmp_attn_type=HCA_KV
        )

    def _forward_decode_compressed(
        self,
        q: torch.Tensor,
        cmp_local_raw: torch.Tensor,
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",
        cmp_attn_type: int,
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: translate pool-local → global slots
        for both SWA and compressed pools, then one dual-pool FlashMLA
        call (``extra_k_cache`` + ``extra_indices_in_kvcache`` merges
        softmax in-kernel; mirrors vLLM ``deepseek_v4_attention.py:849-865``)."""
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8.decode.attention_kernels import (
            attn_fp8_dual_paged,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
            get_or_build_sched_meta,
        )

        win = self.window_size
        T = bsz * q_len
        K_cmp = cmp_local_raw.shape[-1]
        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = attn_metadata.pool_block_tables.get(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_attn_type)
        cmp_pool_bt = attn_metadata.pool_block_tables.get(cmp_attn_type)
        req_id = attn_metadata.req_id_per_token[:T]
        swa_global = attn_metadata.swa_global_slots[:T]
        if self.indexer is None and attn_metadata.hca_cmp_global_slots is not None:
            cmp_global = attn_metadata.hca_cmp_global_slots[:T]
        else:
            cmp_tokens_per_block = int(
                attn_metadata.paged_pool_tokens_per_block[cmp_attn_type]
            ) // int(self.compress_ratio)
            cmp_global = translate_local_to_global_slots(
                req_id,
                cmp_pool_bt[:bsz],
                cmp_local_raw.reshape(T, K_cmp),
                entries_per_block=self._pool_entries_per_block(cmp_attn_type),
                tokens_per_block_for_block_table=cmp_tokens_per_block,
            )
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()
        cmp_topk_3d = cmp_global.view(bsz, q_len, K_cmp).contiguous()
        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=win,
            extra_attn_type=cmp_attn_type,
        )
        swa_topk_length = (
            attn_metadata.swa_topk_length[:bsz]
            if attn_metadata.swa_topk_length is not None
            else None
        )
        cmp_len_buf = attn_metadata.compressed_topk_length_by_ratio.get(
            int(self.compress_ratio)
        )
        extra_topk_length = cmp_len_buf[:bsz] if cmp_len_buf is not None else None
        return attn_fp8_dual_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            cmp_pool_3d=cmp_pool_3d,
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            cmp_topk_3d=cmp_topk_3d,
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
            topk_length=swa_topk_length,
            extra_topk_length=extra_topk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Any] = None,
        block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill entry point.

        ``x``: flat ``[T, dim]`` (single-request, B==1 — enforced by
        the FIFO scheduler's ``max_context_batch_size=1`` setting and
        ``DeepSeekV4Model.forward``). ``positions``: ``[T]`` int64 of
        absolute token positions; ``positions[0]`` is the prefill
        start position. We don't read it eagerly — under broadcast
        meta the sp_int is already on ``self._prefill_meta_shared``
        (synced once in ``forward.py`` for all layers); standalone
        path syncs once inside ``_build_shared_prefill_meta``.
        ``kv_cache`` and ``block_tables_by_type`` are stashed on
        ``self`` for the duration of the call so the many
        ``_prefill_*`` / pool helpers can resolve via
        ``self._kv_cache`` without threading the handles through
        every signature.
        """
        with bind_attn_cache(self, kv_cache, block_tables_by_type):
            with record_function_range("dsv4.fp8.attn.set_pool_context"):
                self._set_compressor_pool_context()
            try:
                with record_function_range(
                    f"dsv4.fp8.attn.L{self.layer_id:02d}.prefill"
                ):
                    return self._forward_prefill(x, positions)
            finally:
                with record_function_range("dsv4.fp8.attn.clear_pool_context"):
                    self._clear_compressor_pool_context()

    def forward_with_shared_input_quant(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        shared_input_quant: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[Any] = None,
        block_tables_by_type: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        with bind_attn_cache(self, kv_cache, block_tables_by_type):
            with record_function_range("dsv4.fp8.attn.set_pool_context"):
                self._set_compressor_pool_context()
            try:
                with record_function_range(
                    f"dsv4.fp8.attn.L{self.layer_id:02d}.prefill"
                ):
                    return self._forward_prefill(
                        x, positions, shared_input_quant=shared_input_quant
                    )
            finally:
                with record_function_range("dsv4.fp8.attn.clear_pool_context"):
                    self._clear_compressor_pool_context()

    def _should_overlap_cp_for_prefill(self, common: PrefillMeta) -> bool:
        """Per-call gate for the CP-overlap orchestrator.

        All conditions must hold:
          * overlap enabled — env ``DSV4_PREFILL_CP_OVERLAP=1`` or a
            subclass hard-on (``_prefill_cp_overlap_hard_on``);
          * CP is actually active (``cp_size > 1``); no NCCL gather to
            overlap with otherwise;
          * prefill tensors are CUDA-backed — CPU / sync-reference CP
            should keep using the baseline path;
          * not inside a CUDA-graph capture — NCCL collectives are not
            capturable on this branch and ``cp_all_gather_full_async``
            calls ``work.wait()``;
          * the layer has a compressor (``compress_ratio > 0``) —
            SWA-only layers (ratio == 0) have nothing to overlap.
        """
        if not (self._prefill_cp_overlap_hard_on or _prefill_cp_overlap_enabled()):
            return False
        if not common.cp_on or common.cp_ctx is None or common.cp_ctx.cp_size <= 1:
            return False
        if self.compress_ratio == 0:
            return False
        if self.compress_ratio not in (1, 2, 4, 128):
            return False
        if common.device.type != "cuda":
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _cp_kv_cache_sharded(self, common: Optional[PrefillMeta] = None) -> bool:
        cp_ctx = getattr(self, "_cp_ctx", None) or getattr(common, "cp_ctx", None)
        return bool(getattr(cp_ctx, "kv_cache_sharded", False))

    def _should_async_workspace_reads_for_prefill(self, common: PrefillMeta) -> bool:
        """Whether workspace cache reads may use side-stream async restore.

        This shares the main prefill-CP-overlap gate and only applies to the
        page-rr / KV-cache-sharded path. With ``DSV4_PREFILL_CP_OVERLAP=0``,
        workspace reads keep the original synchronous gather/restore/write
        ordering even when ``--prefill_cp_kv_cache_sharded=1`` is set.
        """
        if not _prefill_cp_async_workspace_reads_enabled():
            return False
        return self._should_overlap_cp_for_prefill(
            common
        ) and self._cp_kv_cache_sharded(common)

    def _get_cp_gather_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the process-local serialized CP communication stream.

        Compressor, state-read prefetch, and delayed SWA-prefix gathers share
        this one stream on each CUDA device. That keeps NCCL launch ordering
        explicit and avoids a per-layer stream fan-out in profiler traces.
        """
        return _get_cp_comm_stream(device)

    def _cleanup_pending_prefill_gather(
        self, compressor: Any, pending: Optional[Any]
    ) -> None:
        """Best-effort exception cleanup for already-launched CP gathers."""
        if pending is None:
            return
        wait_fn = getattr(compressor, "wait_prefill_gather", None)
        if wait_fn is None:
            return
        with suppress(Exception):
            wait_fn(pending)

    def _forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        shared_input_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill body. ``x`` is flat ``[T, dim]``; output flat ``[T, dim]``.

        Three mutually-exclusive paths gated by ``compress_ratio``:
          * 0   → :meth:`_forward_prefill_swa_only`
          * 4   → :meth:`_forward_prefill_csa`   (indexer + compressor)
          * 128 → :meth:`_forward_prefill_hca`   (compressor, dense idx)

        All three share the same prologue: ``_prefill_common_setup`` →
        ``_prefill_compute_qkv`` → ``_prefill_write_swa_fp8_paged``.
        ``csa`` / ``hca`` additionally share the ``[sliding | compressed]``
        kv_cat + sparse_attn epilogue via :meth:`_forward_prefill_compressed`.

        Under ``DSV4_PREFILL_CP_OVERLAP=1`` + CP-active, CSA/HCA layers
        instead dispatch to the overlap orchestrators (which hoist the
        compressor's CP all-gather ahead of the SWA write so they can
        overlap on the default stream vs the CP communication stream). The baseline sequential path
        below stays byte-equal otherwise.

        The public model path establishes the FP8 KV-cache contract before
        entering this per-layer body.
        """
        with record_function_range("dsv4.fp8.attn.prefill.common_setup"):
            common = self._prefill_common_setup(x, positions)
        with record_function_range("dsv4.fp8.attn.prefill.compute_qkv"):
            qkv = self._prefill_compute_qkv(
                x, common, shared_input_quant=shared_input_quant
            )
        if self._should_overlap_cp_for_prefill(common):
            if self.compress_ratio == 128:
                with record_function_range("dsv4.fp8.attn.prefill.path_hca_overlap"):
                    return self._forward_prefill_hca_overlapped(x, qkv, common)
            if self.compress_ratio == 4:
                with record_function_range("dsv4.fp8.attn.prefill.path_csa_overlap"):
                    return self._forward_prefill_csa_overlapped(x, qkv, common)
        with record_function_range("dsv4.fp8.attn.prefill.swa_write"):
            self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
        if self.compress_ratio == 0:
            with record_function_range("dsv4.fp8.attn.prefill.path_swa"):
                out = self._forward_prefill_swa_only(qkv, common)
        elif self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.attn.prefill.path_csa"):
                out = self._forward_prefill_csa(x, qkv, common)
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.attn.prefill.path_hca"):
                out = self._forward_prefill_hca(x, qkv, common)
        return out

    def _forward_prefill_swa_only(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """SWA-only path (compress_ratio == 0). Skips kv_cat + sparse_attn.
        Cold/warmup attends over BF16 ``kv_full`` directly; continuation
        builds ``[prefix_tail | new_K_bf16]`` in a workspace and runs
        chunked ``flash_mla_sparse_fwd`` over it. Each attention chunk is
        output-projected immediately, so the full ``[T, H, D]`` attention
        output is never materialized alongside Q. ``any_cont`` is
        varlen-aware (set from ``prefix_lengths.any()`` under varlen,
        ``sp_int > 0`` otherwise) so a B>1 batch with any continuation
        request takes the workspace path."""
        qkv = self._materialize_prefill_q(qkv, common)
        if not common.any_cont or self._kv_cache is None:
            with record_function_range("dsv4.fp8.attn.swa.via_kv_full"):
                out = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.swa.via_concat"):
                out = self._attn_fp8_swa_via_concat(qkv, common)
        return out

    def _forward_prefill_csa(
        self, x: torch.Tensor, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """CSA path (compress_ratio == 4). Sparse compress topk via the
        IndexerFP8 lightning indexer; main compressor writes the CSA
        pool with hoisted meta. Attention runs through the vLLM-style
        workspace path (dual FP8 dequant + BF16 overlay + flash_mla_sparse_fwd).
        Falls back to BF16 ``kv_full`` attention on warmup (workspace_meta None)."""
        with record_function_range("dsv4.fp8.attn.csa.indexer"):
            raw = self.indexer(
                x, qkv.qr, common.csa_meta.indexer_meta, workspace=common.workspace
            )
        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=raw,
            compressor_meta=common.csa_meta.compressor_meta,
            workspace_meta=common.csa_meta.workspace_meta,
        )

    def _forward_prefill_csa_overlapped(
        self, x: torch.Tensor, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """CSA path with two CP all-gathers overlapped onto the SWA write
        + the safe prefix of indexer work.

        Phase-Z orchestrator. CSA layers issue TWO independent NCCL
        collectives per step (nested indexer compressor + main CSA
        compressor); both must share the same ``cp_gather_stream`` so
        NCCL's per-stream FIFO ordering keeps the two collectives
        rank-consistent within the ProcessGroup. Sequence:

          1. ``indexer.start_prefill_nested_compressor`` enqueues NCCL
             #1 (nested indexer compressor's fused-KV gather) on
             ``cp_gather_stream``;
          2. ``compressor.start_prefill`` enqueues NCCL #2 (main CSA
             compressor's fused-KV gather) on the SAME stream — runs after
             NCCL #1 (FIFO);
          3. ``_prefill_write_swa_fp8_paged`` on the default stream
             overlaps with both gathers above;
          4. ``indexer.forward_with_pending_nested`` waits only NCCL #1,
             writes the indexer-side pool, then runs compute_q/weights_proj;
          5. before indexer ``gather_k_cache`` / score / topk, wait only
             NCCL #2. This keeps main NCCL from running concurrently with the
             numerically sensitive indexer score/topk chain while preserving
             the baseline-visible CSA pool write order;
          6. after indexer topk, ``compressor.finish_prefill`` writes the
             CSA pool;
          7. ``_forward_prefill_compressed(_skip_compressor_write=True,
             cmp_topk_runtime=raw)`` runs workspace_attn over the
             just-written CSA pool + the indexer topk.

        Bit-equal to :meth:`_forward_prefill_csa` (sequential baseline):
        same kernel inputs, same compressor_meta, same indexer chain,
        same workspace path — only the launch ordering differs.
        """
        cp_stream = self._get_cp_gather_stream(x.device)
        csa_meta = common.csa_meta
        layer_label = f"L{int(getattr(self, 'layer_id', 0)):02d}"
        nested_pending = None
        main_pending = None
        try:
            with record_function_range(
                "dsv4.fp8.attn.csa_overlap.start_nested_compressor"
            ):
                nested_pending = self.indexer.start_prefill_nested_compressor(
                    x,
                    csa_meta.indexer_meta.sp_int,
                    meta=csa_meta.indexer_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.csa_nested_indexer",
                    workspace=common.workspace,
                )
            with record_function_range(
                "dsv4.fp8.attn.csa_overlap.start_main_compressor"
            ):
                main_pending = self.compressor.start_prefill(
                    x,
                    common.sp_int,
                    meta=csa_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.csa_main",
                    workspace=common.workspace,
                )
            with record_function_range("dsv4.fp8.attn.csa_overlap.swa_write"):
                self._prefill_write_swa_fp8_paged(common, qkv.kv_full)

            def wait_main_before_indexer_k() -> None:
                if main_pending is None:
                    return
                with record_function_range(
                    "dsv4.fp8.attn.csa_overlap.wait_main_before_indexer_k"
                ):
                    self.compressor.wait_prefill_gather(main_pending)

            with record_function_range("dsv4.fp8.attn.csa_overlap.indexer"):
                indexer_post_stream = (
                    _get_cp_post_gather_stream(x.device) if x.is_cuda else None
                )
                raw = self.indexer.forward_with_pending_nested(
                    x,
                    qkv.qr,
                    csa_meta.indexer_meta,
                    nested_pending,
                    before_gather_k=wait_main_before_indexer_k,
                    cp_gather_stream=cp_stream,
                    post_gather_stream=indexer_post_stream,
                )
                nested_pending = None
            if main_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.csa_overlap.finish_main_compressor"
                ):
                    self.compressor.finish_prefill(main_pending)
                main_pending = None
            return self._forward_prefill_compressed(
                x,
                qkv,
                common,
                cmp_topk_runtime=raw,
                compressor_meta=csa_meta.compressor_meta,
                workspace_meta=csa_meta.workspace_meta,
                _skip_compressor_write=True,
            )
        except Exception:
            self._cleanup_pending_prefill_gather(self.compressor, main_pending)
            nested_compressor = getattr(self.indexer, "compressor", None)
            self._cleanup_pending_prefill_gather(nested_compressor, nested_pending)
            clear_nested = getattr(self.indexer, "_clear_nested_pool", None)
            if clear_nested is not None:
                with suppress(Exception):
                    clear_nested()
            raise

    def _forward_prefill_hca(
        self, x: torch.Tensor, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """HCA path (compress_ratio == 128). Dense compressed indices live
        in ``workspace_meta.dense_cmp_topk``; runtime cmp_topk is None.
        Main compressor writes the HCA pool with hoisted meta."""
        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=common.hca_meta.workspace_meta,
        )

    def _forward_prefill_hca_overlapped(
        self, x: torch.Tensor, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """HCA path with CP all-gather overlapped onto the SWA pool write.

        Phase-Z orchestrator. Reachable only when
        ``_should_overlap_cp_for_prefill`` returned True (env on + CP
        active + not capturing). Sequence:

          1. ``compressor.start_prefill`` enqueues the fused-KV CP gather
             on ``cp_gather_stream`` (side stream — no default-stream
             dependency yet);
          2. ``_prefill_write_swa_fp8_paged`` runs on the default stream
             in parallel with the NCCL gather above (disjoint pool +
             independent input ``qkv.kv_full``);
          3. ``compressor.finish_prefill`` waits the gather + writes the
             HCA pool;
          4. ``_forward_prefill_compressed(_skip_compressor_write=True)``
             runs the workspace path over the just-written HCA pool.

        Bit-equal to :meth:`_forward_prefill_hca` (sequential baseline):
        same kernel inputs, same compressor_meta, same workspace path —
        only the launch ordering differs.
        """
        cp_stream = self._get_cp_gather_stream(x.device)
        layer_label = f"L{int(getattr(self, 'layer_id', 0)):02d}"
        main_pending = None
        try:
            with record_function_range("dsv4.fp8.attn.hca_overlap.start_compressor"):
                main_pending = self.compressor.start_prefill(
                    x,
                    common.sp_int,
                    meta=common.hca_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.hca_main",
                    workspace=common.workspace,
                )
            with record_function_range("dsv4.fp8.attn.hca_overlap.swa_write"):
                self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
            with record_function_range("dsv4.fp8.attn.hca_overlap.finish_compressor"):
                self.compressor.finish_prefill(main_pending)
            main_pending = None
            return self._forward_prefill_compressed(
                x,
                qkv,
                common,
                cmp_topk_runtime=None,
                compressor_meta=common.hca_meta.compressor_meta,
                workspace_meta=common.hca_meta.workspace_meta,
                _skip_compressor_write=True,
            )
        except Exception:
            self._cleanup_pending_prefill_gather(self.compressor, main_pending)
            raise

    def _forward_prefill_compressed(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
        cmp_topk_runtime: Optional[torch.Tensor],
        compressor_meta,
        workspace_meta: Optional[WorkspaceMeta],
        *,
        _skip_compressor_write: bool = False,
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: write compressed-K via main compressor
        (with hoisted ``compressor_meta``), then run the workspace-path
        attention. The workspace path streams each attention chunk directly
        through output projection; warmup falls back to
        ``_attn_fp8_swa_via_kv_full`` when ``workspace_meta`` is None (pool
        context unbound).

        ``_skip_compressor_write`` is the Phase-Z overlap escape hatch:
        the orchestrator (HCA/CSA) has already drained the compressor's
        gather via ``finish_prefill``, so this method must NOT issue a
        second synchronous compressor call (which would re-do the work
        and break correctness). Baseline (non-overlap) callers leave
        the default ``False`` and the historical sequential path runs.
        """
        if not _skip_compressor_write:
            with record_function_range("dsv4.fp8.attn.compressed.compressor"):
                self.compressor(
                    x, common.sp_int, meta=compressor_meta, workspace=common.workspace
                )
        qkv = self._materialize_prefill_q(qkv, common)
        if workspace_meta is None:
            with record_function_range("dsv4.fp8.attn.compressed.warmup_attn"):
                out = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.compressed.workspace_attn"):
                out = self._attn_via_workspace(
                    qkv, common, workspace_meta, cmp_topk_runtime
                )
        return out

    def _attn_via_workspace(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
        workspace_meta: "WorkspaceMeta",
        cmp_topk_runtime: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """vLLM-style workspace path for CSA/HCA prefill (varlen B>=1, non-CP).

        Pipeline (kernel-only — every elementwise op pre-baked in
        :meth:`_build_workspace_meta`):
          1. Allocate ``workspace [B, M, head_dim]`` BF16 zeros.
          2. ``dequantize_and_gather_k_cache(cmp_pool, offset=0)`` →
             ``workspace[b, 0:N_b, :]`` per request (per-req ``cmp_seq_lens``
             handle the per-row variable length; ``[N_b, N_max)`` stays zero).
          3. ``dequantize_and_gather_k_cache(swa_pool, offset=N_max)`` →
             ``workspace[b, N_max:N_max+P_b, :]`` per request. SWA only
             stores the tail blocks; fresh ``S_b`` rows are supplied by step 4.
          4. BF16 overlay freshly computed new K via single ``index_copy_``
             over ``workspace.view(B*M, D)`` using ``wm.new_k_slot_in_flat``
             (already encodes ``M*req_id + N + P_b + local_pos``).
          5. ``combine_topk_swa_indices`` packs the per-query
             ``[compressed_valid | swa_valid]`` index list. The kernel
             internally computes per-token ``min((pos+1)//ratio, TOP_K)``
             so HCA's full ``arange(N_max)`` row gets masked even for tokens
             whose request has ``N_b < N_max``.
          6. ``flash_mla_sparse_fwd`` over the ``[B*M, 1, D]`` workspace view,
             chunked on Q; each chunk is immediately fed into
             :meth:`_prefill_output_proj_into` and written to the final contiguous
             ``[T, dim]`` output buffer, so no full ``[T, H, D]`` attention
             output is materialized.

        Mirrors vLLM ``DeepseekV4MultiHeadLatentAttentionWrapper._forward_prefill``.

        ``cmp_topk_runtime`` is the indexer output (CSA path); for HCA it's
        ignored and ``workspace_meta.dense_cmp_topk`` (precomputed
        ``arange(N_max)`` per token) is used instead.
        """
        q = qkv.q
        from rtp_llm.models_py.modules.dsv41.attn_type import CSA_KV, HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_dequant_triton as _swa_dq
        from rtp_llm.models_py.modules.dsv41.fp8._swa_ops_triton import (
            combine_topk_swa_indices,
            combine_topk_swa_indices_cp_prepared,
        )

        ratio = self.compress_ratio
        ratio_tag = "csa" if ratio == 4 else "hca"
        cmp_at = CSA_KV if ratio == 4 else HCA_KV
        swa_byte_sliced = self._swa_cp_byte_sliced()
        swa_pool_raw = self._pool_raw_u8(SWA_KV) if swa_byte_sliced else None
        swa_pool_3d = None if swa_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_at)
        wm = workspace_meta
        B = int(wm.swa_seq_lens.shape[0])
        D = self.head_dim
        if wm.use_cp_raw_q_merge:
            with record_function_range("dsv4.fp8.attn.workspace.cp_raw_q_merge"):
                o = self._attn_via_workspace_cp_raw_q_merge(
                    qkv=qkv,
                    common=common,
                    workspace_meta=wm,
                    cmp_topk_runtime=cmp_topk_runtime,
                    cmp_pool_3d=cmp_pool_3d,
                    swa_pool_3d=swa_pool_3d,
                )
            with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
                out = self._prefill_output_proj(o, common.freqs_cis)
            self._prefill_output_all_reduce(out)
            return out
        async_workspace_reads = self._should_async_workspace_reads_for_prefill(common)
        with record_function_range("dsv4.fp8.attn.workspace.alloc"):
            workspace = torch.empty((B, wm.M, D), dtype=torch.bfloat16, device=q.device)
        cmp_pending = None
        cmp_reader_for_pending = None
        cmp_prepare_for_pending = None
        swa_prefix_pending = None
        try:
            if wm.N > 0:
                with record_function_range("dsv4.fp8.attn.workspace.gather_cmp"):
                    cmp_reader = (
                        wm.cmp_reader
                        if wm.cmp_reader is not None
                        else LocalPoolReader()
                    )
                    start_cmp = (
                        getattr(cmp_reader, "start_fill_async", None)
                        if async_workspace_reads
                        else None
                    )
                    prepare_cmp = (
                        getattr(cmp_reader, "prepare_fill_async", None)
                        if async_workspace_reads
                        else None
                    )
                    if start_cmp is not None and prepare_cmp is not None:
                        with record_function_range(
                            "dsv4.fp8.attn.workspace.gather_cmp.prefetch_start"
                        ):
                            cmp_pending = start_cmp(
                                out=workspace,
                                k_cache=cmp_pool_3d,
                                seq_lens=wm.cmp_seq_lens,
                                gather_lens=None,
                                block_table=wm.cmp_bt_int32,
                                block_size=wm.cmp_eb,
                                offset=0,
                                stream=self._get_cp_gather_stream(q.device),
                            )
                        if cmp_pending is not None:
                            cmp_reader_for_pending = cmp_reader
                            cmp_prepare_for_pending = prepare_cmp
                    if cmp_pending is None:
                        cmp_reader.fill(
                            out=workspace,
                            k_cache=cmp_pool_3d,
                            seq_lens=wm.cmp_seq_lens,
                            gather_lens=None,
                            block_table=wm.cmp_bt_int32,
                            block_size=wm.cmp_eb,
                            offset=0,
                        )
            if common.any_cont:
                with record_function_range("dsv4.fp8.attn.workspace.gather_swa_prefix"):
                    if swa_byte_sliced:
                        if async_workspace_reads:
                            with record_function_range(
                                "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_start"
                            ):
                                swa_prefix_pending = _swa_dq.start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                                    k_cache_raw=swa_pool_raw,
                                    slot_mapping=wm.swa_cache_slot_mapping,
                                    gather_lens=wm.swa_cache_gather_lens,
                                    offset=wm.N,
                                    full_entries_per_block=wm.swa_eb,
                                    cp_rank=int(common.cp_ctx.cp_rank),
                                    cp_size=int(common.cp_ctx.cp_size),
                                    compaction=wm.swa_cache_compaction,
                                    stream=self._get_cp_gather_stream(q.device),
                                    profile_name=f"dsv4.cp.all_gather.L{self.layer_id:02d}.swa_prefix",
                                )
                        if swa_prefix_pending is None:
                            _swa_dq.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                                out=workspace,
                                k_cache_raw=swa_pool_raw,
                                slot_mapping=wm.swa_cache_slot_mapping,
                                gather_lens=wm.swa_cache_gather_lens,
                                offset=wm.N,
                                full_entries_per_block=wm.swa_eb,
                                cp_rank=int(common.cp_ctx.cp_rank),
                                cp_size=int(common.cp_ctx.cp_size),
                                compaction=wm.swa_cache_compaction,
                            )
                    else:
                        _swa_dq.dequantize_and_gather_k_cache_slots(
                            out=workspace,
                            k_cache=swa_pool_3d,
                            slot_mapping=wm.swa_cache_slot_mapping,
                            gather_lens=wm.swa_cache_gather_lens,
                            offset=wm.N,
                        )
            with record_function_range("dsv4.fp8.attn.workspace.overlay_new_k"):
                kv_owner_is_ready = (
                    qkv.kv_full.dim() == 2
                    and qkv.kv_full.shape[-1] == D
                    and (qkv.kv_full.dtype == torch.bfloat16)
                )
                kv_source = (
                    qkv.kv_full
                    if kv_owner_is_ready
                    else qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
                )
                workspace.view(B * wm.M, D).index_copy_(
                    0, wm.new_k_slot_in_flat, kv_source
                )
                if not kv_owner_is_ready:
                    dispose_tensor(kv_source)
                dispose_tensor(qkv.kv_full)
            if wm.dense_cmp_topk is not None:
                cmp_topk = wm.dense_cmp_topk
            else:
                cmp_topk = cmp_topk_runtime
            kv_view = workspace.view(B * wm.M, 1, D)
            projected_out = torch.empty(
                q.shape[0], self.dim, dtype=torch.bfloat16, device=q.device
            )
            if common.cp_on:
                cp_ctx_local = common.cp_ctx
                legacy_prefix_length = int(cp_ctx_local.prefix_length)
                global_positions = cp_ctx_local.global_positions
                req_id_per_token = common.req_id_per_token
                prefix_lengths = common.prefix_lengths
                with record_function_range("dsv4.fp8.attn.workspace.combine_topk_cp"):
                    combined_indices, combined_lens = (
                        combine_topk_swa_indices_cp_prepared(
                            topk_indices=cmp_topk,
                            global_positions=global_positions,
                            sp_int=legacy_prefix_length,
                            window_size=self.window_size,
                            compress_ratio=ratio,
                            topk=int(cmp_topk.shape[-1]),
                            M=wm.M,
                            N=wm.N,
                            req_id_per_token=req_id_per_token,
                            prefix_lengths=prefix_lengths,
                            flash_mla_indices=True,
                        )
                    )
            else:
                with record_function_range("dsv4.fp8.attn.workspace.combine_topk"):
                    combined_indices, combined_lens = combine_topk_swa_indices(
                        topk_indices=cmp_topk,
                        query_start_loc=wm.qsl,
                        seq_lens=wm.swa_seq_lens,
                        gather_lens=wm.swa_gather_lens,
                        window_size=self.window_size,
                        compress_ratio=ratio,
                        topk=int(cmp_topk.shape[-1]),
                        M=wm.M,
                        N=wm.N,
                        flash_mla_indices=True,
                    )
            post_gather_stream = None
            if cmp_pending is not None or swa_prefix_pending is not None:
                post_gather_stream = _get_cp_post_gather_stream(q.device)
            if cmp_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_cmp.prefetch_prepare"
                ):
                    cmp_prepare_for_pending(cmp_pending, stream=post_gather_stream)
            if swa_prefix_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_prepare"
                ):
                    _swa_dq.prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        swa_prefix_pending, out=workspace, stream=post_gather_stream
                    )
            if cmp_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_cmp.prefetch_wait"
                ):
                    cmp_reader_for_pending.wait_fill_async(cmp_pending)
                cmp_pending = None
            if swa_prefix_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_wait"
                ):
                    _swa_dq.wait_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        swa_prefix_pending
                    )
                swa_prefix_pending = None
            return self._flash_mla_sparse_fwd_chunked_projected(
                q=q,
                kv=kv_view,
                indices=combined_indices,
                topk_length=combined_lens,
                freqs_cis=common.freqs_cis,
                profile_name="dsv4.fp8.attn.workspace.flash_mla_sparse_fwd",
                out=projected_out,
            )
        finally:
            if cmp_pending is not None and cmp_reader_for_pending is not None:
                cmp_reader_for_pending.discard_fill_async(cmp_pending)
            if swa_prefix_pending is not None:
                _swa_dq.discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    swa_prefix_pending
                )

    @staticmethod
    def _cp_full_req_ids_and_positions(
        common: PrefillMeta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_ctx = common.cp_ctx
        lengths = cp_ctx.input_lengths_global
        if lengths is None:
            lengths = torch.tensor(
                [int(cp_ctx.seq_len_full)], device=common.device, dtype=torch.int32
            )
        prefix = cp_ctx.prefix_lengths
        if prefix is None:
            prefix = torch.tensor(
                [int(cp_ctx.prefix_length)], device=common.device, dtype=torch.long
            )
        device = common.device
        lengths_l = lengths.to(device=device, dtype=torch.long).reshape(-1)
        prefix_l = prefix.to(device=device, dtype=torch.long).reshape(-1)
        total = int(lengths_l.sum().item())
        if total == 0:
            return (
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )
        if int(lengths_l.numel()) == 1:
            req_full = torch.zeros(total, device=device, dtype=torch.long)
        else:
            req_full = torch.repeat_interleave(
                torch.arange(int(lengths_l.numel()), device=device, dtype=torch.long),
                lengths_l,
                output_size=total,
            )
        cu_starts = torch.zeros_like(lengths_l)
        cu_starts[1:] = torch.cumsum(lengths_l[:-1], dim=0)
        local_pos = torch.arange(
            total, device=device, dtype=torch.long
        ) - cu_starts.index_select(0, req_full)
        pos_full = local_pos + prefix_l.index_select(0, req_full)
        return (req_full.contiguous(), pos_full.contiguous())

    @staticmethod
    def _cp_local_full_row_indices(common: PrefillMeta) -> torch.Tensor:
        cp_ctx = common.cp_ctx
        device = common.device
        req = cp_ctx.req_id_per_token.to(device=device, dtype=torch.long).reshape(-1)
        if cp_ctx.prefix_lengths is not None:
            prefix = cp_ctx.prefix_lengths.to(device=device, dtype=torch.long).reshape(
                -1
            )
        else:
            prefix = torch.tensor(
                [int(cp_ctx.prefix_length)], device=device, dtype=torch.long
            )
        if cp_ctx.cu_seqlens_global is not None:
            cu = cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.long).reshape(
                -1
            )
        else:
            cu = torch.tensor(
                [0, int(cp_ctx.seq_len_full)], device=device, dtype=torch.long
            )
        pos = cp_ctx.global_positions.to(device=device, dtype=torch.long).reshape(-1)
        local_pos = pos - prefix.index_select(0, req)
        idx = cu.index_select(0, req) + local_pos
        return idx.clamp_(min=0, max=max(int(cp_ctx.seq_len_full) - 1, 0)).contiguous()

    @staticmethod
    def _compact_indices(
        parts: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row compact: for each row r, concat parts' valid (>=0) entries.

        Vectorized: for each part p of shape [T, w_p], find valid (row, col)
        pairs via ``nonzero``, compute the per-row rank-within-part using
        ``arange − exclusive_cumsum(counts)[row]``, then ``index_put_`` to
        the target columns. Replaces the original O(T·P_count) Python double
        loop with O(P_count) GPU ops — catastrophic at T=1M, fine at 64k.
        """
        T = int(parts[0].shape[0])
        device = parts[0].device
        max_width = sum((int(p.shape[1]) for p in parts))
        aligned_width = (max_width + 127) // 128 * 128
        out = torch.full((T, aligned_width), -1, dtype=torch.int32, device=device)
        cursor_per_row = torch.zeros((T,), dtype=torch.int64, device=device)
        for part in parts:
            valid_mask = part >= 0
            counts = valid_mask.sum(dim=1).to(torch.int64)
            row_idx, col_idx = valid_mask.nonzero(as_tuple=True)
            n_valid = int(row_idx.numel())
            if n_valid == 0:
                cursor_per_row = cursor_per_row + counts
                continue
            cumsum_excl = torch.zeros_like(counts)
            if T > 1:
                cumsum_excl[1:] = counts.cumsum(0)[:-1]
            rank_in_row = torch.arange(
                n_valid, device=device, dtype=torch.int64
            ) - cumsum_excl.index_select(0, row_idx)
            target_col = cursor_per_row.index_select(0, row_idx) + rank_in_row
            values = part[row_idx, col_idx].to(torch.int32)
            out.index_put_((row_idx, target_col), values, accumulate=False)
            cursor_per_row = cursor_per_row + counts
        lens = cursor_per_row.to(torch.int32)
        return (out, lens)

    @staticmethod
    def _compact_local_topk_to_workspace(
        local_topk: torch.Tensor,
        *,
        req_id_per_token: torch.Tensor,
        per_req_total_kv_lens: torch.Tensor,
        cp_size: int,
        block_size: int,
        M: int,
    ) -> torch.Tensor:
        device = local_topk.device
        per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64)
        local_lens = cp_padded_local_kv_lens(per_req, cp_size, block_size).to(
            device=device, dtype=torch.int64
        )
        cu_local = torch.zeros(
            int(local_lens.numel()) + 1, dtype=torch.int64, device=device
        )
        cu_local[1:] = torch.cumsum(local_lens, dim=0)
        req = req_id_per_token.to(device=device, dtype=torch.int64).reshape(-1)
        req_base = cu_local.index_select(0, req).unsqueeze(1)
        local_pos = local_topk.to(torch.int64) - req_base
        req_workspace_base = (req * int(M)).unsqueeze(1)
        valid = local_topk >= 0
        workspace_idx = req_workspace_base + local_pos
        return torch.where(valid, workspace_idx, torch.full_like(workspace_idx, -1)).to(
            torch.int32
        )

    def _raw_q_merge_apply_sink(
        self, out: torch.Tensor, lse: torch.Tensor
    ) -> torch.Tensor:
        if self.attn_sink is None:
            return out
        sink = self.attn_sink.to(device=out.device, dtype=torch.float32).view(1, -1)
        factor = torch.sigmoid(lse.float() - sink)
        factor = torch.where(torch.isfinite(lse), factor, torch.zeros_like(factor))
        return (out.float() * factor.unsqueeze(-1)).to(out.dtype)

    def _attn_via_workspace_cp_raw_q_merge(
        self,
        *,
        qkv: PrefillQKV,
        common: PrefillMeta,
        workspace_meta: WorkspaceMeta,
        cmp_topk_runtime: Optional[torch.Tensor],
        cmp_pool_3d: torch.Tensor,
        swa_pool_3d: torch.Tensor,
    ) -> torch.Tensor:
        q = qkv.q
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
        from rtp_llm.models_py.modules.dsv41.flash_mla import flash_mla_sparse_fwd
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_dequant_triton as _swa_dq

        cp_ctx = common.cp_ctx
        wm = workspace_meta
        D = self.head_dim
        B = int(wm.swa_seq_lens.shape[0])
        if (
            os.environ.get("DSV4_FORWARD_TENSOR_DEBUG", "0") != "0"
            or os.environ.get("DSV4_CP_RAW_Q_MERGE_LOG", "0") != "0"
        ):
            prefix_src = (
                cp_ctx.prefix_lengths
                if cp_ctx.prefix_lengths is not None
                else common.prefix_lengths
            )
            input_src = (
                cp_ctx.input_lengths_global
                if cp_ctx.input_lengths_global is not None
                else common.input_lengths
            )
            payload = {
                "tag": "DSV4_RAW_Q_MERGE",
                "layer_id": int(self.layer_id),
                "compress_ratio": int(self.compress_ratio),
                "cp_rank": int(cp_ctx.cp_rank),
                "cp_size": int(cp_ctx.cp_size),
                "B": B,
                "prefix_lengths": (
                    prefix_src.detach().cpu().reshape(-1).tolist()
                    if prefix_src is not None
                    else None
                ),
                "input_lengths": (
                    input_src.detach().cpu().reshape(-1).tolist()
                    if input_src is not None
                    else None
                ),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
        local_cmp_lens = cp_actual_owned_kv_lens(
            wm.cmp_seq_lens.to(torch.int64), cp_ctx.cp_size, wm.cmp_eb, cp_ctx.cp_rank
        ).to(device=q.device, dtype=torch.int32)
        local_N = int(local_cmp_lens.max().item()) if local_cmp_lens.numel() else 0
        gather_len_max = (
            int(wm.swa_gather_lens.max().item()) if wm.swa_gather_lens.numel() else 0
        )
        local_M = local_N + gather_len_max
        workspace = torch.empty((B, local_M, D), dtype=torch.bfloat16, device=q.device)
        if local_N > 0:
            LocalPoolReader().fill(
                out=workspace,
                k_cache=cmp_pool_3d,
                seq_lens=local_cmp_lens,
                gather_lens=None,
                block_table=wm.cmp_bt_int32,
                block_size=wm.cmp_eb,
                offset=0,
            )
        if common.any_cont:
            _swa_dq.dequantize_and_gather_k_cache(
                out=workspace,
                k_cache=swa_pool_3d,
                seq_lens=wm.swa_cache_seq_lens,
                gather_lens=wm.swa_cache_gather_lens,
                block_table=wm.swa_bt_int32,
                block_size=wm.swa_eb,
                offset=local_N,
            )
        req_full, pos_full = self._cp_full_req_ids_and_positions(common)
        if common.cp_ctx.prefix_lengths is not None:
            prefix_lens = common.cp_ctx.prefix_lengths.to(
                device=q.device, dtype=torch.long
            )
        else:
            prefix_lens = torch.tensor(
                [int(common.cp_ctx.prefix_length)], device=q.device, dtype=torch.long
            )
        P_per_req = torch.clamp_max(prefix_lens, self.window_size - 1)
        local_pos = pos_full - prefix_lens.index_select(0, req_full)
        new_k_slots = (
            req_full * local_M
            + local_N
            + P_per_req.index_select(0, req_full)
            + local_pos
        ).contiguous()
        workspace.view(B * local_M, D).index_copy_(
            0, new_k_slots, qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
        )
        q_full = cp_all_gather_full_varlen(q, cp_ctx)
        if wm.dense_cmp_topk is not None:
            if wm.N > 0:
                dense = (
                    torch.arange(wm.N, device=q.device, dtype=torch.int64)
                    .view(1, wm.N)
                    .expand(int(q_full.shape[0]), wm.N)
                )
                dense_len = torch.clamp(
                    (pos_full + 1) // int(self.compress_ratio), max=wm.N
                ).unsqueeze(1)
                cmp_topk_full = (
                    torch.where(dense < dense_len, dense, torch.full_like(dense, -1))
                    .to(torch.int32)
                    .contiguous()
                )
            else:
                cmp_topk_full = torch.empty(
                    (int(q_full.shape[0]), 0), device=q.device, dtype=torch.int32
                )
        else:
            cmp_topk_full = cp_all_gather_full_varlen(cmp_topk_runtime, cp_ctx)
        local_topk_compact = remap_topk_to_cp_local(
            cmp_topk_full,
            per_req_total_kv_lens=wm.cmp_seq_lens.to(torch.int64),
            cp_size=cp_ctx.cp_size,
            cp_rank=cp_ctx.cp_rank,
            block_size=wm.cmp_eb,
            req_id_per_token=req_full,
        )
        local_topk = self._compact_local_topk_to_workspace(
            local_topk_compact,
            req_id_per_token=req_full,
            per_req_total_kv_lens=wm.cmp_seq_lens.to(torch.int64),
            cp_size=cp_ctx.cp_size,
            block_size=wm.cmp_eb,
            M=local_M,
        )
        local_swa, _ = build_swa_cp_local_indices(
            pos_full,
            prefix_lengths=prefix_lens,
            cp_size=cp_ctx.cp_size,
            cp_rank=cp_ctx.cp_rank,
            window_size=self.window_size,
            M=local_M,
            N=local_N,
            req_id_per_token=req_full,
        )
        combined_indices, combined_lens = self._compact_indices([local_topk, local_swa])
        local_o, _, local_lse = flash_mla_sparse_fwd(
            q=q_full,
            kv=workspace.view(B * local_M, 1, D),
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            attn_sink=None,
            topk_length=combined_lens,
        )
        with record_function_range(
            f"dsv4.cp.all_gather.L{self.layer_id:02d}.workspace_attn.o.launch"
        ):
            gathered_o = all_gather(local_o.contiguous(), group=Group.TP).view(
                cp_ctx.cp_size, int(q_full.shape[0]), self.n_heads, D
            )
        with record_function_range(
            f"dsv4.cp.all_gather.L{self.layer_id:02d}.workspace_attn.lse.launch"
        ):
            gathered_lse = all_gather(local_lse.contiguous(), group=Group.TP).view(
                cp_ctx.cp_size, int(q_full.shape[0]), self.n_heads
            )
        merged_o, merged_lse = merge_lse_output(gathered_o, gathered_lse, dim=0)
        merged_o = self._raw_q_merge_apply_sink(merged_o, merged_lse)
        local_rows = self._cp_local_full_row_indices(common)
        return merged_o.index_select(0, local_rows).unsqueeze(0)

    def _set_prefill_meta_shared(self, meta: Optional["PrefillMeta"]) -> None:
        """Inject the (compress_ratio bucket) shared prefill meta built by
        the upper layer (V4Transformer.forward_layers). The shared meta is
        layer-invariant within a ratio, so the upper layer builds it once
        per ratio and broadcasts to every layer attention sharing that
        ratio. ``None`` clears the binding (used at end of forward).
        """
        self._prefill_meta_shared = meta

    def _ensure_freqs_cis_bound(self) -> None:
        """Bind ``self.freqs_cis`` onto this layer's compressor / indexer
        chain so their forward()s can read ``self.freqs_cis`` without an
        extra parameter. Idempotent — safe to call many times. Required
        on every layer (not just the meta-build rep) because each layer
        owns its own compressor / indexer instance.
        """
        if not self.compress_ratio:
            return
        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            if self.indexer.compressor.freqs_cis is None:
                self.indexer.compressor.freqs_cis = self.freqs_cis

    def _build_shared_prefill_meta(
        self,
        x: torch.Tensor,
        positions: Union[int, torch.Tensor],
        sp_per_req: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        reuse_common_meta: Optional["PrefillMeta"] = None,
        reuse_freqs_meta: Optional["PrefillMeta"] = None,
        reuse_swa_write_meta: Optional["SwaPrefillMeta"] = None,
        host_swa_table: Optional[torch.Tensor] = None,
        swa_write_only: bool = False,
    ) -> "PrefillMeta":
        """Build the layer-invariant (within compress_ratio bucket) part
        of per-call prefill metadata. All host-side prep work that
        doesn't depend on ``self.layer_id`` lives here so the upper layer
        can run it once per ratio and broadcast to every same-ratio
        attention via :meth:`_set_prefill_meta_shared`. Standalone path
        falls back to running this per-layer.

        ``positions`` accepts either a ``[T]`` int64 tensor or a host scalar.
        CP metadata carries the matching host position so tensor callers
        also avoid reading a CUDA scalar.

        ``reuse_common_meta`` is supplied only by the upper-layer broadcast
        builder. It reuses the first ratio's top-k/continuation tensors and
        SWA Group-1 write metadata while this call still builds its own
        ratio-specific CSA/HCA metadata. ``reuse_freqs_meta`` is separate:
        ratio0 uses base RoPE while ratio4/128 share compressed RoPE, so only
        a metadata object produced from the identical source table may supply
        the gathered frequencies. Omitting either input preserves the full or
        partial standalone fallback.
        """
        seqlen = int(x.shape[0])
        rd = self.rope_head_dim
        device = x.device
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1
        if isinstance(positions, torch.Tensor):
            first_position_host = getattr(cp_ctx, "first_position_host", None)
            sp_int = (
                first_position_host
                if first_position_host is not None
                else int(positions.reshape(-1)[0].item())
            )
        else:
            sp_int = int(positions)
        use_varlen = True
        win = self.window_size
        seqlen_full = cp_ctx.seq_len_full if cp_on else seqlen
        cu_seqlens = _flat_1d(cu_seqlens)
        input_lengths = _flat_1d(input_lengths)
        prefix_lengths = _flat_1d(prefix_lengths)
        position_ids = _flat_1d(position_ids)
        req_id_per_token = _flat_1d(req_id_per_token)
        sp_per_req = _flat_1d(sp_per_req)
        can_reuse_freqs = (
            reuse_freqs_meta is not None
            and reuse_freqs_meta.freqs_cis_source_id == id(self.freqs_cis)
        )
        position_ids_eff: Optional[torch.Tensor] = None
        if reuse_common_meta is None or not can_reuse_freqs:
            position_ids_eff = position_ids
            if cp_on:
                position_ids_eff = _flat_1d(
                    cp_ctx.global_positions.to(device=device, dtype=torch.long)
                )
        if reuse_common_meta is None:
            with record_function_range("dsv4.fp8.meta.varlen.freqs_topk"):
                freqs_cis = self.freqs_cis.index_select(
                    0,
                    position_ids_eff.to(device=self.freqs_cis.device, dtype=torch.long),
                )
                from rtp_llm.models_py.modules.dsv41.fp8 import (
                    _swa_ops_triton as _swa_ops,
                )

                cu_seqlens_for_k = cu_seqlens
                if cp_on:
                    if cp_ctx.cu_seqlens_global is not None:
                        cu_seqlens_for_k = _flat_1d(
                            cp_ctx.cu_seqlens_global.to(
                                device=device, dtype=torch.int32
                            )
                        )
                topk_idxs, topk_length_kv_full = (
                    _swa_ops.compute_window_topk_and_length_varlen(
                        win,
                        cu_seqlens_for_k,
                        position_ids_eff,
                        prefix_lengths,
                        req_id_per_token,
                    )
                )
                host_prefixes = getattr(cp_ctx, "prefix_lengths_host", None)
                any_cont = (
                    any((prefix > 0 for prefix in host_prefixes))
                    if host_prefixes is not None
                    else bool((prefix_lengths > 0).any().item())
                )
            with record_function_range("dsv4.fp8.meta.swa_varlen"):
                swa_meta = self._build_swa_prefill_meta_varlen(
                    seqlen=seqlen,
                    device=device,
                    any_cont=any_cont,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    topk_length_kv_full=topk_length_kv_full,
                    reuse_write_meta=reuse_swa_write_meta,
                    host_swa_table=host_swa_table,
                    write_only=swa_write_only,
                )
            row_seqlens_full = torch.full(
                (1,), seqlen_full, device=device, dtype=torch.long
            )
        else:
            if can_reuse_freqs:
                freqs_cis = reuse_freqs_meta.freqs_cis
            else:
                with record_function_range("dsv4.fp8.meta.varlen.freqs"):
                    freqs_cis = self.freqs_cis.index_select(
                        0,
                        position_ids_eff.to(
                            device=self.freqs_cis.device, dtype=torch.long
                        ),
                    )
            topk_idxs = reuse_common_meta.topk_idxs
            any_cont = reuse_common_meta.any_cont
            row_seqlens_full = reuse_common_meta.row_seqlens_full
            source_swa = reuse_common_meta.swa_meta
            swa_meta = SwaPrefillMeta(
                slot_mapping=source_swa.slot_mapping,
                query_start_loc=source_swa.query_start_loc,
                combined_seq_lens=source_swa.combined_seq_lens,
                topk_length_kv_full=source_swa.topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                cache_slot_mapping=None,
                slot_compaction=source_swa.slot_compaction,
                cache_compaction=None,
            )
        self._ensure_freqs_cis_bound()
        csa_meta: Optional[CsaPrefillMeta] = None
        hca_meta: Optional[HcaPrefillMeta] = None
        if self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.meta.csa"):
                csa_meta = self._build_csa_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                    has_prefix=any_cont,
                )
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.meta.hca"):
                hca_meta = self._build_hca_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                    has_prefix=any_cont,
                )
        return PrefillMeta(
            seqlen=seqlen,
            seqlen_full=seqlen_full,
            rd=rd,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=cp_on,
            freqs_cis=freqs_cis,
            topk_idxs=topk_idxs,
            sp_int=sp_int,
            any_cont=any_cont,
            row_seqlens_full=row_seqlens_full,
            use_varlen=use_varlen,
            sp_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            max_seqlen_q=max_seqlen_q,
            swa_meta=swa_meta,
            csa_meta=csa_meta,
            hca_meta=hca_meta,
            freqs_cis_source_id=id(self.freqs_cis),
        )

    def _build_csa_prefill_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        has_prefix: bool,
    ) -> CsaPrefillMeta:
        """Build CSA-layer per-call metadata: indexer prepare + main CSA
        compressor prepare_metadata.

        Pool context binding: this method runs from the broadcast-meta
        path (``forward_layers`` → ``_build_and_propagate_prefill_meta``)
        BEFORE the per-layer ``_set_compressor_pool_context`` would
        otherwise fire inside ``_forward_prefill_internal_wrapper``, so we
        bind here ourselves. Without this bind the indexer's
        ``self._kv_block_table`` / ``_state_block_table`` are still None
        and the hoist inside ``IndexerFP8.prepare`` silently no-ops —
        ``compressor_meta`` comes back as None and the per-call slot
        mapping (~20 small kernels per CSA layer) ends up rebuilt on the
        hot path between the FP32 SGEMM and ``_save_partial_states_kernel``.

        One bind covers both halves: ``_set_compressor_pool_context``
        wires the indexer AND the host compressor in the same call, and
        ``_build_compressor_meta`` is inlined here so we don't redundantly
        re-bind for the second meta. No try/finally — if something below
        raises we want the process to die instead of leaking a stale
        binding into the next call.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV
        from rtp_llm.models_py.modules.dsv41.fp8.compressor import (
            build_prepare_metadata_args,
        )

        idx_bt = (
            self._block_tables_by_type.get(INDEXER_KV)
            if self._block_tables_by_type is not None
            else None
        )
        idx_eb = self._pool_entries_per_block(INDEXER_KV)
        with record_function_range("dsv4.fp8.meta.csa.bind_pool"):
            self._set_compressor_pool_context()
        with record_function_range("dsv4.fp8.meta.csa.indexer_prepare"):
            indexer_meta = self.indexer.prepare(
                bsz=1,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=idx_bt,
                kv_eb=idx_eb,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
                has_prefix=has_prefix,
            )
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
        if cp_active:
            with record_function_range("dsv4.fp8.meta.csa.cp_compressor_prepare"):
                cp_positions, cp_b_idx, cp_seq_start_per_req, cp_cu_seq_per_req = (
                    build_cp_full_prefill_positions(cp_ctx_local, device)
                )
                compressor_meta = self.compressor.prepare_metadata(
                    cp_positions,
                    cp_b_idx,
                    has_prefix=has_prefix,
                    is_batched=True,
                    seq_start_per_req=cp_seq_start_per_req,
                    cu_seq_per_req=cp_cu_seq_per_req,
                )
        else:
            with record_function_range("dsv4.fp8.meta.csa.compressor_prepare"):
                cmp_args = build_prepare_metadata_args(
                    use_varlen=use_varlen,
                    has_prefix=has_prefix,
                    device=device,
                    sp_int=sp_int,
                    seqlen=seqlen,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    seq_start_per_req=sp_per_req,
                    cu_seqlens=cu_seqlens,
                )
                compressor_meta = self.compressor.prepare_metadata(**cmp_args)
        with record_function_range("dsv4.fp8.meta.csa.clear_pool"):
            self._clear_compressor_pool_context()
        with record_function_range("dsv4.fp8.meta.csa.workspace"):
            workspace_meta = self._build_workspace_meta(
                seqlen,
                sp_int,
                device,
                with_dense_cmp_topk=False,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sp_per_req=sp_per_req,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        return CsaPrefillMeta(
            indexer_meta=indexer_meta,
            compressor_meta=compressor_meta,
            workspace_meta=workspace_meta,
        )

    def _build_hca_prefill_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        has_prefix: bool,
    ) -> HcaPrefillMeta:
        """Build HCA-layer per-call metadata: main HCA compressor
        prepare_metadata."""
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        if cp_ctx_local is not None and cp_ctx_local.cp_size > 1:
            with record_function_range("dsv4.fp8.meta.hca.cp_compressor_prepare"):
                self._set_compressor_pool_context()
                try:
                    cp_positions, cp_b_idx, cp_seq_start_per_req, cp_cu_seq_per_req = (
                        build_cp_full_prefill_positions(cp_ctx_local, device)
                    )
                    compressor_meta = self.compressor.prepare_metadata(
                        cp_positions,
                        cp_b_idx,
                        has_prefix=has_prefix,
                        is_batched=True,
                        seq_start_per_req=cp_seq_start_per_req,
                        cu_seq_per_req=cp_cu_seq_per_req,
                    )
                finally:
                    self._clear_compressor_pool_context()
        else:
            with record_function_range("dsv4.fp8.meta.hca.compressor_prepare"):
                compressor_meta = self._build_compressor_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                    has_prefix=has_prefix,
                )
        with record_function_range("dsv4.fp8.meta.hca.workspace"):
            workspace_meta = self._build_workspace_meta(
                seqlen,
                sp_int,
                device,
                with_dense_cmp_topk=True,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sp_per_req=sp_per_req,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        return HcaPrefillMeta(
            compressor_meta=compressor_meta, workspace_meta=workspace_meta
        )

    def _build_workspace_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        with_dense_cmp_topk: bool,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
    ) -> Optional[WorkspaceMeta]:
        """Static index/dim metadata for the vLLM-style workspace + dual-
        gather + ``combine_topk_swa_indices`` flow. Returns ``None`` when
        pool context isn't bound (warmup) so callers fall through to BF16
        ``kv_full`` fast paths.

        Per-request ``N_b`` / ``gather_b`` / ``M_b`` are derived from
        ``prefix_lengths`` + ``input_lengths`` + ``cu_seqlens`` +
        ``position_ids`` + ``req_id_per_token``. ``new_k_slot_in_flat`` is
        the per-token target into ``workspace.view(B*M, D)`` for the BF16
        overlay.

        ``with_dense_cmp_topk=True`` precomputes the dense ``arange(N_max)``
        topk grid HCA needs (``[T_total, N_max]`` int32 contiguous; the
        ``_combine_topk_swa_indices_kernel`` masks per-token validity via
        ``COMPRESS_RATIO`` so no per-token mask is required up here).
        CSA passes ``False`` and feeds runtime indexer output to combine_topk.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_ops_triton as _swa_ops

        if self._kv_cache is None or self._block_tables_by_type is None:
            return None
        ratio = self.compress_ratio
        if ratio not in (4, 128):
            return None
        cmp_at = CSA_KV if ratio == 4 else HCA_KV
        state_at = CSA_STATE if ratio == 4 else HCA_STATE
        swa_bt = self._block_tables_by_type.get(SWA_KV)
        cmp_bt = self._block_tables_by_type.get(cmp_at)
        if (
            swa_bt is None
            or swa_bt.numel() == 0
            or cmp_bt is None
            or (cmp_bt.numel() == 0)
        ):
            return None
        swa_eb = self._swa_entries_per_block()
        cmp_eb = self._pool_entries_per_block(cmp_at)
        if swa_eb <= 0 or cmp_eb <= 0:
            return None
        swa_tokens_per_block = _dsv4_pool_tokens_per_block(
            self._kv_cache, region=SWA_KV
        )
        win = self.window_size
        if use_varlen:
            cu_seqlens = _flat_1d(cu_seqlens)
            input_lengths = _flat_1d(input_lengths)
            prefix_lengths = _flat_1d(prefix_lengths)
            position_ids = _flat_1d(position_ids)
            req_id_per_token = _flat_1d(req_id_per_token)
            B = batch_size
            sp_i32 = prefix_lengths.to(device=device, dtype=torch.int32)
            S_i32 = input_lengths.to(device=device, dtype=torch.int32)
            cp_ctx_local = getattr(self, "_cp_ctx", None)
            cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
            if cp_active:
                S_i32 = cp_ctx_local.input_lengths_global.to(
                    device=device, dtype=torch.int32
                )
                B = int(S_i32.shape[0])
                seq_len_full = int(cp_ctx_local.seq_len_full)
                if cp_ctx_local.cu_seqlens_global is not None:
                    cu_seqlens_full_eff = _flat_1d(
                        cp_ctx_local.cu_seqlens_global.to(
                            device=device, dtype=torch.int32
                        )
                    ).contiguous()
                else:
                    cum_after = torch.cumsum(S_i32, 0).to(torch.int32)
                    cu_seqlens_full_eff = torch.cat(
                        [torch.zeros(1, dtype=torch.int32, device=device), cum_after]
                    ).contiguous()
            else:
                position_ids_eff = _flat_1d(
                    position_ids.to(device=device, dtype=torch.int64)
                )
                req_id_per_token_eff = _flat_1d(
                    req_id_per_token.to(device=device, dtype=torch.int64)
                )
            seq_total_per_req = sp_i32 + S_i32
            N_per_req = seq_total_per_req // ratio
            P_per_req = torch.clamp_max(sp_i32, win - 1)
            gather_len_per_req = S_i32 + P_per_req
            host_input_lengths = getattr(
                cp_ctx_local, "input_lengths_global_host", None
            )
            host_prefix_lengths = getattr(cp_ctx_local, "prefix_lengths_host", None)
            if (
                cp_active
                and host_input_lengths is not None
                and (host_prefix_lengths is not None)
                and (len(host_input_lengths) == B)
                and (len(host_prefix_lengths) == B)
            ):
                seq_total_host = [
                    int(prefix) + int(length)
                    for prefix, length in zip(host_prefix_lengths, host_input_lengths)
                ]
                n_per_req_host = [total // ratio for total in seq_total_host]
                gather_per_req_host = [
                    int(length) + min(int(prefix), win - 1)
                    for prefix, length in zip(host_prefix_lengths, host_input_lengths)
                ]
                N_max = max(n_per_req_host, default=0)
                gather_len_max = max(gather_per_req_host, default=0)
                total_compressed_kv = sum(n_per_req_host)
            else:
                stats = torch.stack(
                    [N_per_req.max(), gather_len_per_req.max(), N_per_req.sum()]
                )
                N_max, gather_len_max, total_compressed_kv = (
                    int(v) for v in stats.tolist()
                )
            N = N_max
            M = N_max + gather_len_max
            swa_seq_lens = seq_total_per_req.contiguous()
            cmp_seq_lens = N_per_req.contiguous()
            swa_gather_lens = gather_len_per_req.contiguous()
            swa_cache_seq_lens = sp_i32.contiguous()
            swa_cache_gather_lens = P_per_req.contiguous()
            qsl = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
            swa_bt_int32 = swa_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            cmp_bt_int32 = cmp_bt[:B].to(device=device, dtype=torch.int32).contiguous()
            if cp_active:
                new_k_slot_in_flat = _swa_ops.compute_swa_slot_in_flat_from_cu(
                    cu_seqlens_full_eff,
                    prefix_lengths.to(device=device)[:B],
                    num_tokens=seq_len_full,
                    M=M,
                    window_size=win,
                    base_offset=N,
                )
            else:
                new_k_slot_in_flat = _swa_ops.compute_swa_slot_in_flat(
                    position_ids_eff,
                    req_id_per_token_eff,
                    prefix_lengths.to(device=device),
                    M=M,
                    window_size=win,
                    base_offset=N,
                )
            T_total = seqlen
        dense_cmp_topk: Optional[torch.Tensor] = None
        if with_dense_cmp_topk:
            if N > 0:
                dense_cmp_topk = (
                    torch.arange(N, device=device, dtype=torch.int32)
                    .view(1, N)
                    .expand(T_total, N)
                )
            else:
                dense_cmp_topk = torch.empty(
                    (T_total, 0), device=device, dtype=torch.int32
                )
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        kv_cache_sharded = bool(getattr(cp_ctx_local, "kv_cache_sharded", False))
        per_req_total_kv_lens: Optional[torch.Tensor] = None
        ratio = self.compress_ratio
        if (
            kv_cache_sharded
            and ratio > 0
            and (prefix_lengths is not None)
            and (prefix_lengths.numel() > 0)
            and (cmp_eb > 0)
        ):
            per_req_total_kv_lens = cmp_seq_lens.to(
                device=device, dtype=torch.int64
            ).contiguous()
        cmp_owner_block_size: Optional[int] = None
        if kv_cache_sharded and ratio > 0 and (self._kv_cache is not None):
            kv_owner_tpb = self._kv_cache.get_seq_size_per_block(
                CSA_KV if ratio == 4 else HCA_KV
            )
            if kv_owner_tpb > 0 and kv_owner_tpb % ratio == 0:
                cmp_owner_block_size = kv_owner_tpb // ratio
        cmp_reader = make_compressed_k_pool_reader(
            cp_ctx=cp_ctx_local,
            kv_cache_sharded=kv_cache_sharded,
            per_req_total_kv_lens=per_req_total_kv_lens,
            block_size=cmp_eb if cmp_eb > 0 else None,
            owner_block_size=cmp_owner_block_size,
            total_kv_len=(
                total_compressed_kv if per_req_total_kv_lens is not None else None
            ),
        )
        use_cp_raw_q_merge = False
        swa_byte_sliced = self._swa_cp_byte_sliced()
        if (
            _use_cp_cache_hit_raw_q_merge()
            and (not swa_byte_sliced)
            and (N > 0)
            and (cp_ctx_local is not None)
            and (cp_ctx_local.cp_size > 1)
            and bool(getattr(cp_ctx_local, "kv_cache_sharded", False))
            and (prefix_lengths is not None)
            and (input_lengths is not None)
        ):
            input_src = (
                cp_ctx_local.input_lengths_global
                if cp_ctx_local.input_lengths_global is not None
                else input_lengths
            )
            if _force_all_cp_raw_q_merge():
                use_cp_raw_q_merge = int(input_src.to(torch.long).sum().item()) > 0
            else:
                prefix_src = (
                    cp_ctx_local.prefix_lengths
                    if cp_ctx_local.prefix_lengths is not None
                    else prefix_lengths
                )
                prefix_len = int(prefix_src.to(torch.long).sum().item())
                input_len = int(input_src.to(torch.long).sum().item())
                use_cp_raw_q_merge = (
                    prefix_len > 0
                    and input_len > 0
                    and prefer_raw_q_merge_attention_conservative(
                        prefix_len=prefix_len,
                        input_len=input_len,
                        compress_ratio=int(self.compress_ratio),
                        include_topk_gather=int(self.compress_ratio) == 4,
                    )
                )
        swa_cache_slot_mapping = _build_suffix_pool_slot_mapping(
            block_table=swa_bt_int32,
            seq_lens=swa_cache_seq_lens,
            gather_lens=swa_cache_gather_lens,
            entries_per_block=swa_eb,
            tokens_per_block_for_block_table=swa_tokens_per_block,
            ring_entries=swa_eb,
        )
        swa_cache_compaction = self._build_swa_cp_byte_compaction(
            swa_cache_slot_mapping,
            full_entries_per_block=swa_eb,
            validation_site="swa.gather_cp_byte.slot_indices",
            negative_mode="skip_any",
            gather_lens=swa_cache_gather_lens,
        )
        return WorkspaceMeta(
            M=M,
            N=N,
            swa_eb=swa_eb,
            cmp_eb=cmp_eb,
            swa_bt_int32=swa_bt_int32,
            cmp_bt_int32=cmp_bt_int32,
            swa_seq_lens=swa_seq_lens,
            cmp_seq_lens=cmp_seq_lens,
            swa_gather_lens=swa_gather_lens,
            swa_cache_seq_lens=swa_cache_seq_lens,
            swa_cache_gather_lens=swa_cache_gather_lens,
            qsl=qsl,
            dense_cmp_topk=dense_cmp_topk,
            new_k_slot_in_flat=new_k_slot_in_flat,
            cmp_reader=cmp_reader,
            use_cp_raw_q_merge=use_cp_raw_q_merge,
            swa_cache_slot_mapping=swa_cache_slot_mapping,
            swa_cache_compaction=swa_cache_compaction,
        )

    def _build_compressor_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        has_prefix: bool,
    ):
        """Run the main compressor's ``prepare_metadata`` with its pool
        context temporarily bound. Returns ``CompressorMeta``. The pool
        context binding (``_set_compressor_pool_context``) reads CSA_KV/
        CSA_STATE for ratio=4 and HCA_KV/HCA_STATE for ratio=128 based
        on ``self.compress_ratio``.

        ``use_varlen`` is required — set by ``_build_shared_prefill_meta``
        (the single env-read point + contract guard for the whole prefill
        stack). UT helpers must pass it explicitly so a missing kwarg
        surfaces as ``TypeError`` instead of silently picking up the
        ambient env.
        """
        from rtp_llm.models_py.modules.dsv41.fp8.compressor import (
            build_prepare_metadata_args,
        )

        cmp_args = build_prepare_metadata_args(
            use_varlen=use_varlen,
            has_prefix=has_prefix,
            device=device,
            sp_int=sp_int,
            seqlen=seqlen,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            seq_start_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
        )
        self._set_compressor_pool_context()
        try:
            return self.compressor.prepare_metadata(**cmp_args)
        finally:
            self._clear_compressor_pool_context()

    def _prefill_common_setup(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> PrefillMeta:
        """Return the per-forward metadata and workspace owned by the caller."""
        return self._prefill_meta_shared

    def _build_swa_prefill_meta_varlen(
        self,
        *,
        seqlen: int,
        device: torch.device,
        any_cont: bool,
        batch_size: int,
        cu_seqlens: torch.Tensor,
        input_lengths: torch.Tensor,
        prefix_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        req_id_per_token: torch.Tensor,
        topk_length_kv_full: Optional[torch.Tensor] = None,
        reuse_write_meta: Optional[SwaPrefillMeta] = None,
        host_swa_table: Optional[torch.Tensor] = None,
        write_only: bool = False,
    ) -> SwaPrefillMeta:
        """Varlen path: B>=1, per-request tensor plumbing.

        Three return points, each constructing a complete ``SwaPrefillMeta``
        with explicit field values (no ``_replace`` chaining):

          1. **Warmup** (pool unbound): only ``topk_length_kv_full`` set
             (SWA-only); all pool / Group-1 / Group-2 fields are ``None``/0.
          2. **CSA/HCA layer** (pool bound, ``compress_ratio != 0``):
             Group-1 write meta only; Group-2 lives on ``WorkspaceMeta``.
          3. **SWA-only layer** (pool bound, ``compress_ratio == 0``):
             Full meta. ``cache_*`` / ``combined_*`` populated on
             continuation; ``None``/0 on all-cold.
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_ops_triton as _swa_ops

        cu_seqlens = _flat_1d(cu_seqlens)
        input_lengths = _flat_1d(input_lengths)
        prefix_lengths = _flat_1d(prefix_lengths)
        position_ids = _flat_1d(position_ids)
        req_id_per_token = _flat_1d(req_id_per_token)
        win = self.window_size
        is_swa_only = self.compress_ratio == 0 and (not write_only)
        num_tokens = seqlen
        if topk_length_kv_full is None:
            sp_per_token = prefix_lengths.to(torch.int32).gather(
                0, req_id_per_token.to(torch.int64)
            )
            local_pos = position_ids.to(torch.int32) - sp_per_token
            topk_length_kv_full = torch.clamp(local_pos + 1, max=win)
        bt = (
            self._block_tables_by_type.get(SWA_KV)
            if self._block_tables_by_type is not None
            else None
        )
        eb = self._swa_entries_per_block()
        if self._kv_cache is None or bt is None or bt.numel() == 0 or (eb <= 0):
            return SwaPrefillMeta(
                slot_mapping=None,
                query_start_loc=None,
                combined_seq_lens=None,
                topk_length_kv_full=topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                cache_slot_mapping=None,
            )
        swa_tokens_per_block = _dsv4_pool_tokens_per_block(
            self._kv_cache, region=SWA_KV
        )
        B = batch_size
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_on_write = (
            cp_ctx is not None
            and cp_ctx.cp_size > 1
            and (cp_ctx.cu_seqlens_global is not None)
            and (cp_ctx.input_lengths_global is not None)
        )
        query_start_loc = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
        combined_seq_lens = (
            prefix_lengths.to(torch.int32) + input_lengths.to(torch.int32)
        ).contiguous()
        if cp_on_write:
            write_B = int(cp_ctx.input_lengths_global.numel())
            write_query_start_loc = _flat_1d(
                cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.int32)
            ).contiguous()
            write_combined_seq_lens = (
                prefix_lengths.to(torch.int32)[:write_B]
                + _flat_1d(cp_ctx.input_lengths_global.to(torch.int32))
            ).contiguous()
            write_num_tokens = cp_ctx.seq_len_full
        else:
            write_B = B
            write_query_start_loc = query_start_loc
            write_combined_seq_lens = combined_seq_lens
            write_num_tokens = num_tokens
        bt_swa = bt[:write_B].to(device=device, dtype=torch.int32).contiguous()
        if reuse_write_meta is not None and reuse_write_meta.slot_mapping is not None:
            slot_mapping = reuse_write_meta.slot_mapping
            slot_compaction = reuse_write_meta.slot_compaction
        else:
            planned = None
            if host_swa_table is not None and win == 128 and self._swa_cp_byte_sliced():
                from ._v41_swa_metadata import try_host_slot_metadata

                planned = try_host_slot_metadata(
                    bt_swa,
                    host_swa_table[:write_B],
                    cp_ctx,
                    entries=eb,
                    span=swa_tokens_per_block,
                    num_blocks=int(self._pool_raw_u8(SWA_KV).shape[0]),
                )
            if planned is not None:
                slot_mapping, slot_compaction = planned
            else:
                slot_mapping = _swa_ops.compute_swa_slot_mapping(
                    block_table=bt_swa,
                    query_start_loc=write_query_start_loc,
                    seq_lens=write_combined_seq_lens,
                    num_tokens=write_num_tokens,
                    pool_entries_per_block=eb,
                    tokens_per_block_for_block_table=swa_tokens_per_block,
                    ring_entries=eb,
                )
                slot_compaction = self._build_swa_cp_byte_compaction(
                    slot_mapping,
                    full_entries_per_block=eb,
                    validation_site="swa.quantize_and_insert_cp_byte.slot_mapping",
                    negative_mode="skip_minus_one",
                )
        if not is_swa_only:
            return SwaPrefillMeta(
                slot_mapping=slot_mapping,
                query_start_loc=query_start_loc,
                combined_seq_lens=combined_seq_lens,
                topk_length_kv_full=topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                slot_compaction=slot_compaction,
            )
        combined_gather_lens = _swa_ops.compute_prefill_gather_lens(
            seq_lens=write_combined_seq_lens,
            query_start_loc=write_query_start_loc,
            num_prefills=write_B,
            num_decodes=0,
            window_size=win,
        )
        host_prefixes = getattr(cp_ctx, "prefix_lengths_host", None)
        host_lengths = getattr(cp_ctx, "input_lengths_global_host", None)
        if (
            hasattr(self, "swa_bounded_replay")
            and cp_on_write
            and (host_prefixes is not None)
            and (host_lengths is not None)
        ):
            combined_gather_len_max = max(
                (n + min(p, win - 1) for p, n in zip(host_prefixes, host_lengths))
            )
        else:
            combined_gather_len_max = int(combined_gather_lens.max().item())
        M = max(combined_gather_len_max, 1)
        if any_cont:
            cache_seq_lens = prefix_lengths.to(device=device, dtype=torch.int32)[
                :write_B
            ].contiguous()
            cache_gather_lens = (
                torch.clamp_max(prefix_lengths, win - 1)
                .to(device=device, dtype=torch.int32)[:write_B]
                .contiguous()
            )
            cache_slot_mapping = None
            planned = None
            if host_swa_table is not None and win == 128 and self._swa_cp_byte_sliced():
                from ._v41_swa_metadata import try_host_slot_metadata

                planned = try_host_slot_metadata(
                    bt_swa,
                    host_swa_table[:write_B],
                    cp_ctx,
                    entries=eb,
                    span=swa_tokens_per_block,
                    num_blocks=int(self._pool_raw_u8(SWA_KV).shape[0]),
                    read=True,
                )
            if planned is not None:
                cache_slot_mapping, cache_compaction = planned
            elif hasattr(self, "swa_bounded_replay") and host_prefixes is not None:
                from ._v41_swa_metadata import try_suffix_slots

                cache_slot_mapping = try_suffix_slots(
                    bt_swa,
                    cache_seq_lens,
                    cache_gather_lens,
                    max_gather=max(
                        (min(p, win - 1) for p in host_prefixes[:write_B]), default=0
                    ),
                    entries=eb,
                    span=swa_tokens_per_block,
                    ring=eb,
                )
            if cache_slot_mapping is None:
                cache_slot_mapping = _build_suffix_pool_slot_mapping(
                    block_table=bt_swa,
                    seq_lens=cache_seq_lens,
                    gather_lens=cache_gather_lens,
                    entries_per_block=eb,
                    tokens_per_block_for_block_table=swa_tokens_per_block,
                    ring_entries=eb,
                )
            if planned is None:
                cache_compaction = self._build_swa_cp_byte_compaction(
                    cache_slot_mapping,
                    full_entries_per_block=eb,
                    validation_site="swa.gather_cp_byte.slot_indices",
                    negative_mode="skip_any",
                    gather_lens=cache_gather_lens,
                )
            if cp_on_write:
                topk_indices_empty = torch.empty(
                    (seqlen, 0), dtype=torch.int32, device=device
                )
                combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices_cp(
                    topk_indices=topk_indices_empty,
                    global_positions=_flat_1d(cp_ctx.global_positions),
                    sp_int=(
                        host_prefixes[0]
                        if host_prefixes
                        else int(prefix_lengths[0].item())
                    ),
                    window_size=win,
                    compress_ratio=1,
                    topk=0,
                    M=M,
                    N=0,
                    req_id_per_token=req_id_per_token,
                    prefix_lengths=prefix_lengths,
                )
                slot_in_flat = _swa_ops.compute_swa_slot_in_flat_from_cu(
                    _flat_1d(
                        cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.int32)
                    ),
                    prefix_lengths.to(device=device)[:write_B],
                    num_tokens=cp_ctx.seq_len_full,
                    M=M,
                    window_size=win,
                )
            else:
                topk_indices_empty = torch.empty(
                    (num_tokens, 0), dtype=torch.int32, device=device
                )
                combined_indices, combined_lens = _swa_ops.combine_topk_swa_indices(
                    topk_indices=topk_indices_empty,
                    query_start_loc=query_start_loc,
                    seq_lens=combined_seq_lens,
                    gather_lens=combined_gather_lens,
                    window_size=win,
                    compress_ratio=1,
                    topk=0,
                    M=M,
                    N=0,
                )
                slot_in_flat = _swa_ops.compute_swa_slot_in_flat(
                    position_ids.to(device=device),
                    req_id_per_token.to(device=device),
                    prefix_lengths.to(device=device),
                    M=M,
                    window_size=win,
                )
            prefix_len_max = 1
        else:
            cache_seq_lens = None
            cache_gather_lens = None
            cache_slot_mapping = None
            cache_compaction = None
            combined_indices = None
            combined_lens = None
            slot_in_flat = None
            prefix_len_max = 0
        return SwaPrefillMeta(
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            combined_seq_lens=combined_seq_lens,
            topk_length_kv_full=topk_length_kv_full,
            combined_gather_lens=combined_gather_lens,
            combined_gather_len_max=combined_gather_len_max,
            M=M,
            cache_seq_lens=cache_seq_lens,
            cache_gather_lens=cache_gather_lens,
            prefix_len_max=prefix_len_max,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            slot_in_flat=slot_in_flat,
            cache_slot_mapping=cache_slot_mapping,
            slot_compaction=slot_compaction,
            cache_compaction=cache_compaction,
        )

    def _prefill_compute_qkv(
        self,
        x: torch.Tensor,
        common: PrefillMeta,
        shared_input_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> PrefillQKV:
        """Q/KV path — RMSNorm + LoRA Q + KV linears + fused RMSNorm-RoPE.

        Internally uses ``[1, T, ...]`` so ``fused_rmsnorm_rope`` sees the
        ``(B, S, …)`` layout it expects. Returned tensors keep the 3D
        shape because downstream pool/compressor helpers rely on it.
        """
        x_3d = x.unsqueeze(0)
        rd = common.rd
        fused_qkv = self._try_fused_qr_kv(x_3d, shared_input_quant)
        if fused_qkv is not None:
            if (
                shared_input_quant is not None
                and x.shape[0] >= 32768
                and (x.shape[1] == 5120)
                and torch.is_inference_mode_enabled()
                and all(
                    (t.is_inference() and t._base is None for t in shared_input_quant)
                )
                and (not torch.cuda.is_current_stream_capturing())
            ):
                from rtp_llm.utils.hot_hook_runtime import runtime as hot_hook_runtime

                if not hot_hook_runtime().enabled:
                    for tensor in shared_input_quant:
                        dispose_tensor(tensor)
            shared_input_quant = None
        elif self._can_reuse_qkv_input_quant():
            if shared_input_quant is None:
                with record_function_range("dsv4.fp8.attn.qkv.shared_input_quant"):
                    x_2d = x_3d.reshape(-1, x_3d.shape[-1])
                    shared_input_quant = self.wq_a.quantize_input(x_2d)
        else:
            shared_input_quant = None

        def compute_qr() -> torch.Tensor:
            with record_function_range("dsv4.fp8.attn.qkv.q_lora_a_norm"):
                if fused_qkv is not None:
                    return fused_qkv[0]
                if shared_input_quant is not None:
                    q_proj = self._lin_from_shared_quant(
                        self.wq_a, shared_input_quant, x_3d.shape
                    )
                else:
                    q_proj = self._lin(self.wq_a, x_3d)
                return self._rmsnorm_weighted(q_proj, self.q_norm)

        def compute_kv() -> torch.Tensor:
            with record_function_range("dsv4.fp8.attn.qkv.kv_proj_rope"):
                if fused_qkv is not None:
                    kv_in = fused_qkv[1]
                elif shared_input_quant is not None:
                    kv_in = self._lin_from_shared_quant(
                        self.wkv, shared_input_quant, x_3d.shape
                    )
                else:
                    kv_in = self._lin(self.wkv, x_3d)
                return fused_rmsnorm_rope(
                    kv_in, self.kv_norm, common.freqs_cis, rd, eps=self.eps
                )

        qr = compute_qr()
        kv = compute_kv()
        if common.cp_on:
            with record_function_range("dsv4.fp8.attn.qkv.cp_gather_varlen"):
                with record_function_range(
                    "dsv4.fp8.attn.swa_kv_full.cp_gather_varlen"
                ):
                    kv_flat = kv.reshape(kv.size(0) * kv.size(1), *kv.shape[2:])
                    kv_full_flat = cp_all_gather_full_varlen(
                        kv_flat,
                        common.cp_ctx,
                        replay_only=cp_swa_replay_starts(common.cp_ctx) is not None,
                        profile_name=f"dsv4.cp.all_gather.L{self.layer_id:02d}.swa_kv_full.varlen",
                    )
                    kv_full = kv_full_flat.unsqueeze(0)
        else:
            kv_full = kv
        return PrefillQKV(qr=qr.squeeze(0), q=None, kv_full=kv_full.squeeze(0))

    def _materialize_prefill_q(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> PrefillQKV:
        """Compute the deferred dense Q (``q_lora_b`` + RMSNorm-RoPE) into the
        forward workspace's Q slice and return ``qkv`` with ``q`` filled.

        Deferred until both compressors have drained their CP gather/restore
        buffers — those alias the same union storage as Q (see
        ``PrefillWorkspace``). Idempotent. Mirrors the ``[1, T, ...]`` layout the
        original ``compute_q`` used so ``fused_rmsnorm_rope`` sees ``(B, S, …)``.
        """
        if qkv.q is not None:
            return qkv
        qr_3d = qkv.qr.unsqueeze(0)
        with record_function_range("dsv4.fp8.attn.qkv.q_lora_b_rope"):
            seqlen = int(qr_3d.shape[1])
            q_out = common.workspace.prefill_q(seqlen)
            q_local_flat = self._lin(
                self.wq_b, qr_3d, out=q_out.view(seqlen, self.n_heads * self.head_dim)
            )
            q_local = q_local_flat.view(1, seqlen, self.n_heads, self.head_dim)
            q_local = fused_rmsnorm_rope(
                q_local, None, common.freqs_cis, common.rd, eps=self.eps, out=q_local
            )
        return qkv._replace(q=q_local.squeeze(0))

    def _prefill_write_swa_fp8_paged(
        self, common: PrefillMeta, kv_full: torch.Tensor
    ) -> None:
        """Single-launch quantize + insert into the FP8 SWA pool, using
        the paged ``slot_mapping`` pre-built in ``common.swa_meta``.

        The paged formula matches the decode-side dequant kernel. For large
        physical blocks, only the SWA ring tail before a physical boundary or
        request end is writable; entries with ``slot=-1`` are skipped. No-op
        on warmup (``swa_meta`` write fields are ``None``).
        """
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_kv_insert_triton as _ins

        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        cp_byte_sliced = self._swa_cp_byte_sliced()
        packed_3d = None if cp_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        raw_u8 = self._pool_raw_u8(SWA_KV) if cp_byte_sliced else None
        if (
            cp_byte_sliced
            and raw_u8 is None
            or (not cp_byte_sliced and packed_3d is None)
        ):
            return
        k_bf16 = kv_full.reshape(-1, self.head_dim)
        if k_bf16.dtype != torch.bfloat16:
            k_bf16 = k_bf16.to(torch.bfloat16)
        with record_function_range("dsv4.fp8.attn.swa.quant_insert"):
            if cp_byte_sliced:
                cp_ctx = common.cp_ctx
                _ins.quantize_and_insert_k_cache_cp_byte_sliced(
                    k_bf16,
                    raw_u8,
                    meta.slot_mapping,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_rank=int(cp_ctx.cp_rank),
                    cp_size=int(cp_ctx.cp_size),
                    compaction=meta.slot_compaction,
                )
            else:
                _ins.quantize_and_insert_k_cache(k_bf16, packed_3d, meta.slot_mapping)

    def _flash_mla_sparse_fwd_chunked_projected(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        freqs_cis: torch.Tensor,
        profile_name: str,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run sparse prefill attention in Q chunks and project immediately.

        ``q`` is the live view of the per-forward ``PrefillWorkspace`` Q
        region. Keeping the chunking here (instead of copying Q into another
        scratch buffer) preserves that storage reuse while bounding the only
        large attention temporary to ``[q_chunk, n_heads, head_dim]``. Sparse
        prefill has no dependency across Q rows, so slicing Q, indices,
        topk-length and RoPE frequencies on the same boundaries is equivalent
        to a full launch.

        The projected result is written directly into the final contiguous
        ``[T, dim]`` tensor. The unprojected full ``[T, H, D]`` output is never
        allocated. A caller may preallocate ``out`` to overlap its allocation
        with preceding GPU work; the same tensor is returned after an in-place
        all-reduce. The caller must keep ``kv`` alive until this method returns.
        """
        s_q = int(q.shape[0])
        from rtp_llm.models_py.modules.dsv41.flash_mla import flash_mla_sparse_fwd

        chunk_rows = min(_FLASH_MLA_SPARSE_Q_CHUNK, s_q)
        single_chunk = chunk_rows == s_q
        if out is None:
            out = torch.empty(s_q, self.dim, dtype=torch.bfloat16, device=q.device)
        for start in range(0, s_q, chunk_rows):
            end = min(start + chunk_rows, s_q)
            if single_chunk:
                q_part = q
                indices_part = indices
                topk_length_part = topk_length
                freqs_cis_part = freqs_cis
                out_part = out
            else:
                q_part = q[start:end]
                indices_part = indices[start:end]
                topk_length_part = topk_length[start:end]
                freqs_cis_part = freqs_cis[start:end]
                out_part = out[start:end, :]
            with record_function_range(profile_name):
                o_part, _, _ = flash_mla_sparse_fwd(
                    q=q_part,
                    kv=kv,
                    indices=indices_part,
                    sm_scale=self.softmax_scale,
                    attn_sink=self.attn_sink,
                    topk_length=topk_length_part,
                )
            with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
                self._prefill_output_proj_into(o_part, freqs_cis_part, out=out_part)
            dispose_tensor(o_part)
        self._prefill_output_all_reduce(out)
        return out

    def _attn_fp8_swa_via_kv_full(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """Chunked sparse_fwd over BF16 ``kv_full`` — no FP8 round-trip.

        Used by:
          * cold prefill (``sp == 0``) — pool capacity (``2 * eb``)
            can't hold the full prefill anyway; attend over BF16 K
            we just computed.
          * warmup forward (``self._kv_cache is None``) — pool not yet
            allocated; needed for the framework's dry-run shape inference.

        Each Q chunk is output-projected into the final ``[T, dim]`` result
        before the next chunk starts. Caller is responsible for pre-writing
        the new K to the FP8 SWA pool (via
        ``_prefill_write_swa_fp8_paged``) for future decode reads — write
        order doesn't matter here since this path doesn't read from the pool.
        """
        q = qkv.q
        meta = common.swa_meta
        ti = common.topk_idxs
        if ti.dim() == 3:
            ti = ti.squeeze(0)
        indices = ti.unsqueeze(1).to(torch.int32)
        out = self._flash_mla_sparse_fwd_chunked_projected(
            q=q,
            kv=qkv.kv_full.unsqueeze(1),
            indices=indices,
            topk_length=meta.topk_length_kv_full,
            freqs_cis=common.freqs_cis,
            profile_name="dsv4.fp8.attn.swa.flash_mla_kv_full",
        )
        dispose_tensor(qkv.kv_full)
        return out

    def _attn_fp8_swa_via_concat(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """Continuation prefill (any req with prefix > 0): prefix-from-cache
        + new-K-bf16 concat. Varlen-aware ``[B, M, D]`` workspace.

        Pipeline (per request b, ``S_b = input_lengths[b]``,
        ``P_b = min(prefix_lengths[b], win-1)``):
          1. ``dequantize_and_gather_k_cache`` reads each request's trailing
             ``P_b`` cached tokens (SWA prefix tail, abs pos ``[sp_b - P_b, sp_b)``)
             into ``workspace[b, :P_b, :]``. Already batched via
             ``cache_seq_lens`` / ``cache_gather_lens`` ``[B]`` tensors.
          2. Vectorized scatter places the freshly computed new K into
             ``workspace[b, P_b:P_b+S_b, :]`` — no per-request Python loop.
          3. Chunked ``flash_mla_sparse_fwd`` over
             ``workspace.view(B*M, 1, D)``; ``combined_indices`` already
             carries ``M*batch_idx + slot`` from
             ``combine_topk_swa_indices``. Each chunk is immediately output-
             projected into the final ``[T, dim]`` result.

        """
        q = qkv.q
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_dequant_triton as _swa_dq

        meta = common.swa_meta
        cp_byte_sliced = self._swa_cp_byte_sliced()
        packed_3d = None if cp_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        raw_u8 = self._pool_raw_u8(SWA_KV) if cp_byte_sliced else None
        D = self.head_dim
        B = common.batch_size
        workspace = torch.empty((B, meta.M, D), dtype=torch.bfloat16, device=q.device)
        if meta.prefix_len_max > 0:
            with record_function_range("dsv4.fp8.attn.swa_concat.gather_prefix"):
                if cp_byte_sliced:
                    cp_ctx = common.cp_ctx
                    _swa_dq.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        out=workspace,
                        k_cache_raw=raw_u8,
                        slot_mapping=meta.cache_slot_mapping,
                        gather_lens=meta.cache_gather_lens,
                        offset=0,
                        full_entries_per_block=self._swa_entries_per_block(),
                        cp_rank=int(cp_ctx.cp_rank),
                        cp_size=int(cp_ctx.cp_size),
                        compaction=meta.cache_compaction,
                    )
                else:
                    _swa_dq.dequantize_and_gather_k_cache_slots(
                        out=workspace,
                        k_cache=packed_3d,
                        slot_mapping=meta.cache_slot_mapping,
                        gather_lens=meta.cache_gather_lens,
                        offset=0,
                    )
        kv_owner_is_ready = (
            qkv.kv_full.dim() == 2
            and qkv.kv_full.shape[-1] == D
            and (qkv.kv_full.dtype == torch.bfloat16)
        )
        kv_source = (
            qkv.kv_full
            if kv_owner_is_ready
            else qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
        )
        with record_function_range("dsv4.fp8.attn.swa_concat.overlay_new_k"):
            workspace.view(B * meta.M, D).index_copy_(0, meta.slot_in_flat, kv_source)
        if not kv_owner_is_ready:
            dispose_tensor(kv_source)
        dispose_tensor(qkv.kv_full)
        return self._flash_mla_sparse_fwd_chunked_projected(
            q=q,
            kv=workspace.view(B * meta.M, 1, D),
            indices=meta.combined_indices.unsqueeze(1),
            topk_length=meta.combined_lens,
            freqs_cis=common.freqs_cis,
            profile_name="dsv4.fp8.attn.swa_concat.flash_mla",
        )

    def _prefill_output_all_reduce(self, out: torch.Tensor) -> None:
        if self.tp_size <= 1:
            return
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        with record_function_range("dsv4.fp8.attn.out.tp_all_reduce"):
            all_reduce(out, Group.TP, inplace=True)

    def _prefill_output_proj(
        self, o: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """Inverse-RoPE + grouped wo_a + wo_b into a fresh ``[T, dim]`` tensor."""
        seqlen = o.shape[-3]
        out = torch.empty(seqlen, self.dim, dtype=torch.bfloat16, device=o.device)
        self._prefill_output_proj_into(o, freqs_cis, out=out)
        return out

    def _prefill_output_proj_into(
        self, o: torch.Tensor, freqs_cis: torch.Tensor, *, out: torch.Tensor
    ) -> None:
        """Inverse-RoPE + grouped wo_a + wo_b into an existing ``[T, dim]`` tensor.

        The fused Triton kernel
        (``fused_inv_rope_fp8_quant``) emits the exact ``(fp8 [M,G,K],
        scale [M,G,K/512])`` layout ``deep_gemm.fp8_einsum`` consumes.
        """
        o_3d = o.view(-1, self.n_heads, self.head_dim)
        seqlen = o_3d.shape[0]
        with record_function_range("dsv4.fp8.attn.out.fused_inv_rope_quant"):
            o_fp8, o_scale = fused_inv_rope_fp8_quant(
                o_3d,
                freqs_cis,
                n_groups=self.n_groups,
                heads_per_group=self.n_heads // self.n_groups,
                nope_dim=self.head_dim - self.rope_head_dim,
                rope_head_dim=self.rope_head_dim,
            )
        with record_function_range("dsv4.fp8.attn.out.wo_a_einsum"):
            o_proj = self._wo_a_einsum_from_fp8(o_fp8, o_scale, 1, seqlen)
        with record_function_range("dsv4.fp8.attn.out.wo_b"):
            wo_b_in = o_proj.flatten(2).reshape(seqlen, -1)
            self.wo_b(wo_b_in, out=out)


class CommitOnlyAttentionFP8(AttentionFP8):
    """Minimal DSpARK attention object used by a prefill commit worker.

    ``forward_commit`` never evaluates a query.  It only runs the per-layer
    ``wkv -> RMSNorm + RoPE`` projection before writing the SWA pool.  The
    regular :class:`AttentionFP8` constructor also materializes Q/O linears,
    compressor/indexer state, and their associated caches; constructing those
    objects for a prefill-only process needlessly loads several GiB of MTP
    weights.  This subclass deliberately initializes only the attributes
    consumed by the shared commit path and inherits the pool/CP helpers from
    ``AttentionFP8`` so there is one implementation of cache geometry.

    It is intentionally not a general attention implementation.  Proposal or
    ordinary prefill/decode calls must use ``AttentionFP8`` (the model selects
    this class only when ``V4Args.commit_only`` is true).
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-06,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        del max_batch_size, index_topk
        if layer_weights is None:
            raise ValueError("commit-only DSpARK attention requires layer weights")
        if int(compress_ratio) != 0:
            raise ValueError(
                f"commit-only DSpARK attention supports SWA layers only, got compress_ratio={compress_ratio}"
            )
        nn.Module.__init__(self)
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.q_lora_rank = int(q_lora_rank)
        self.o_lora_rank = int(o_lora_rank)
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.window_size = int(window_size)
        self.compress_ratio = 0
        self.eps = float(norm_eps)
        self.softmax_scale = self.head_dim ** (-0.5)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        if self.tp_size <= 0:
            raise ValueError(f"invalid attention tp_size={self.tp_size}")
        if int(n_heads) % self.tp_size:
            raise ValueError(
                f"n_heads={n_heads} is not divisible by tp_size={self.tp_size}"
            )
        if int(o_groups) % self.tp_size:
            raise ValueError(
                f"o_groups={o_groups} is not divisible by tp_size={self.tp_size}"
            )
        self.n_heads = int(n_heads) // self.tp_size
        self.n_groups = int(o_groups) // self.tp_size
        from rtp_llm.utils.model_weight import W

        self.wkv = _v4_fp8_linear(
            layer_weights[W.v4_attn_wkv_w], layer_weights[W.v4_attn_wkv_s]
        )
        self.kv_norm = layer_weights[W.v4_attn_kv_norm]
        self.attn_sink = None
        self.wq_a = None
        self.wq_b = None
        self.wo_a_w = None
        self.wo_a_s = None
        self.wo_b = None
        self.q_norm = None
        self.compressor = None
        self.indexer = None
        self._rope_base = float(rope_theta)
        self._rope_o_seq_len = 0
        self._rope_factor = float(rope_factor)
        self._rope_beta_fast = int(beta_fast)
        self._rope_beta_slow = int(beta_slow)
        self._rope_dim = int(rope_head_dim)
        self._rope_max_seq_len = int(max_seq_len)
        self.freqs_cis = precompute_freqs_cis(
            self._rope_dim,
            self._rope_max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            self._rope_factor,
            self._rope_beta_fast,
            self._rope_beta_slow,
        )
        self._fp8_decode_op: Optional[Any] = None
        self._cp_ctx: Optional[CPContext] = None
        self._prefill_meta_shared: Optional["PrefillMeta"] = None
        self._kv_cache: Optional[Any] = None
        self._block_tables_by_type: Optional[Dict[str, torch.Tensor]] = None
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        idx_hd = int(index_head_dim)
        kv_spec = (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
        indexer_kv_spec = (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES)
        self._pool_spec: Dict[str, tuple] = {
            SWA_KV: kv_spec,
            CSA_KV: kv_spec,
            HCA_KV: kv_spec,
            INDEXER_KV: indexer_kv_spec,
            CSA_STATE: (torch.float32, 4 * self.head_dim),
            HCA_STATE: (torch.float32, 2 * self.head_dim),
            INDEXER_STATE: (torch.float32, 4 * idx_hd),
        }


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decode.compute_qkv import DecodeQKV
    from .decode.decode_attn_metadata import DSv4DecodeAttnMetadataFP8
