"""DeepSeek-V4 Compressor — FP8 KV pool path (per-token state pool).

Companion to ``compressor.py`` (BF16 path). Single class for both pool
flavors; ``head_dim`` selects the writer kernel + FP8 KV slot layout:

  * ``head_dim == 512`` (CSA / HCA): per-slot 584B striped layout
    (448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales). Reader:
    ``flash_mla_sparse_fwd`` after dequant.

  * ``head_dim == 128`` (indexer compressor): per-slot 132B grouped
    layout (128 fp8 K + 4-byte fp32 scale). Reader: DeepGEMM
    ``fp8_paged_mqa_logits``.

Post-commit ``e76867719`` ("fix - align state size to 256") the C++
state pools (INDEXER_STATE / CSA_STATE / HCA_STATE) all use
``entries_per_block=256``: every token gets its own slot. We mirror
vLLM's ``DeepseekCompressor`` flow:

  1. ``_save_partial_states_kernel`` writes per-token (kv | score+ape)
     into the framework-allocated state pool.
  2. ``_fused_kv_compress_norm_rope_insert_*_attn`` self-skips
     non-boundary tokens, otherwise gathers the ``(1+overlap)*ratio``
     window from the state pool, does softmax → RMSNorm → RoPE → FP8
     UE8M0 quant → KV-pool slot store.

Public API:
  * ``set_pool_context(kv_view, kv_bt, kv_eb, state_view, state_bt,
    state_eb, state_tokens_per_block, kv_tokens_per_block)`` — shared with
    ``PoolBackedModule``.
  * ``forward(x, start_pos, sequence_lengths=None)`` for prefill.
  * ``forward_decode_vectorized(x, start_pos)`` for batched decode.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.ops.compute_ops import rtp_llm_ops

_CUBLAS_GEMM_BF16_BF16_FP32 = getattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32", None)


def _linear_bf16_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """F.linear(x, weight) with BF16 operands and FP32 accumulation/output."""
    assert x.dtype == torch.bfloat16, f"expected BF16 input, got {x.dtype}"
    assert weight.dtype == torch.bfloat16, f"expected BF16 weight, got {weight.dtype}"
    assert x.is_contiguous(), "expected contiguous input"
    assert weight.is_contiguous(), "expected contiguous weight"
    assert (
        _CUBLAS_GEMM_BF16_BF16_FP32 is not None
    ), "cublas_gemm_bf16_bf16_fp32 op is not built"
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _CUBLAS_GEMM_BF16_BF16_FP32(x_2d, weight)
    return out_2d.reshape(*leading_shape, weight.shape[0])


from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_INDEXER,
    _CP_ROLE_MAIN,
    CPContext,
    CPCudaAsyncGatherHandle,
    cp_all_gather_full_async,
    cp_should_gather,
    cp_wait_gather_full,
    cp_wait_gather_raw,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule

# Process-local cache for the device-side cos_sin tensor derived from a
# given freqs_cis source. DSV4 has ~91 CompressorFP8 instances (main +
# indexer + indexer.compressor per layer), so active instances share one cache
# tensor. Weak values prevent this module global from pinning a previous model's
# large device allocation across unload/reload; each running compressor keeps a
# strong ``self._cos_sin_cache`` reference, so eviction cannot cause UAF.
_SHARED_COS_SIN_CACHE: weakref.WeakValueDictionary[
    Tuple[int, torch.device, Tuple[int, ...], torch.dtype], torch.Tensor
] = weakref.WeakValueDictionary()


@dataclass(frozen=True)
class _CPStateReadSelection:
    block_ids: torch.Tensor
    read_block_table: torch.Tensor
    compacted: bool = False


def _select_cp_state_read_tail_blocks(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    seq_start_per_req: torch.Tensor,
    token_count: int,
    state_tokens_per_block: int,
) -> Optional[_CPStateReadSelection]:
    if (
        token_count <= 1
        or state_tokens_per_block <= 0
        or block_table.dim() != 2
        or int(block_table.shape[1]) <= 0
    ):
        return None

    bt = block_table.to(device=state_cache.device, dtype=torch.long).contiguous()
    B = min(int(bt.shape[0]), int(seq_start_per_req.numel()))
    max_blocks = int(bt.shape[1])
    read_bt = torch.zeros_like(bt)
    if B <= 0:
        return _CPStateReadSelection(
            block_ids=torch.empty((0,), device=bt.device, dtype=torch.long),
            read_block_table=read_bt.to(dtype=block_table.dtype),
        )

    suffix_len = int(token_count) - 1
    block_tokens = int(state_tokens_per_block)
    # Correctness relies on prefix-cache match producing reuse lengths aligned
    # to the state block size. With that invariant, the prefix suffix window
    # [seq_start - suffix_len, seq_start) is fully covered by the tail state
    # block below. If the cache match granularity is changed to allow
    # unaligned seq_start_per_req, this must be changed back to a multi-block
    # selector. Do not add a runtime alignment check here: seq_start_per_req is
    # a CUDA tensor on the hot path, and checking it on host would reintroduce
    # the device-host sync this path is meant to avoid.
    starts = seq_start_per_req[:B].to(device=bt.device, dtype=torch.long).reshape(B)
    last_pos = (starts - 1).clamp_min(0)
    last_block = last_pos // block_tokens
    cols = (last_block % max_blocks).view(B, 1)

    block_ids = bt[:B].gather(1, cols).reshape(-1)
    slot_valid = (starts > 0) & (block_ids > 0)
    block_ids = torch.where(slot_valid, block_ids, torch.zeros_like(block_ids))
    block_ids = block_ids.contiguous()

    slot_ids = torch.arange(1, B + 1, device=bt.device, dtype=torch.long)
    slot_ids = torch.where(slot_valid, slot_ids, torch.zeros_like(slot_ids))
    read_bt[:B].scatter_(1, cols, slot_ids.view(B, 1).to(dtype=read_bt.dtype))
    return _CPStateReadSelection(
        block_ids=block_ids,
        read_block_table=read_bt.to(dtype=block_table.dtype),
    )


def _select_cp_state_read_blocks(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    seq_start_per_req: Optional[torch.Tensor],
    token_count: int,
    state_tokens_per_block: int,
) -> _CPStateReadSelection:
    if seq_start_per_req is not None:
        selection = _select_cp_state_read_tail_blocks(
            state_cache,
            block_table,
            seq_start_per_req=seq_start_per_req,
            token_count=token_count,
            state_tokens_per_block=state_tokens_per_block,
        )
        if selection is not None:
            return selection

    bt = block_table.to(device=state_cache.device, dtype=torch.long).contiguous()
    # Conservative fallback for legacy callers without per-request prefix
    # metadata. The optimized CP prefill path should provide seq_start_per_req
    # and return above.
    flat_ids = bt.reshape(-1)
    valid = flat_ids > 0
    block_ids = flat_ids[valid]
    needed_valid = valid.view_as(bt)

    read_bt = torch.zeros_like(bt)
    if int(block_ids.numel()) > 0:
        inverse = torch.arange(
            int(block_ids.numel()), device=bt.device, dtype=torch.long
        )
        read_bt[needed_valid] = (inverse + 1).to(read_bt.dtype)
    return _CPStateReadSelection(
        block_ids=block_ids,
        read_block_table=read_bt.to(dtype=block_table.dtype),
    )


def _fill_cp_state_read_cache(
    read_cache: torch.Tensor,
    gathered: torch.Tensor,
    cp_size: int,
    num_blocks: int,
    local_eb: int,
    hidden: int,
) -> torch.Tensor:
    read_cache[0].zero_()
    if num_blocks <= 0:
        return read_cache
    dst = read_cache[1:].view(num_blocks, cp_size, local_eb, hidden)
    src = gathered.view(cp_size, num_blocks, local_eb, hidden)
    # One strided copy beats the per-rank loop (cp_size separate kernel
    # launches accumulate to ~cp_size * 50us of launch overhead per layer).
    dst.copy_(src.permute(1, 0, 2, 3))
    return read_cache


# Selections only depend on (block_table, seq_start_per_req, token_count,
# state_tokens_per_block, state_cache.shape[1]); all of these are the same
# across every layer that shares the same CompressorMeta (i.e. all layers of
# the same attention-type within a single prefill). Cache by meta identity so
# layers 2..N reuse layer 1's tail-block selection instead of relaunching
# small tensor ops per layer.
# A WeakKeyDictionary auto-evicts when the per-forward meta is garbage
# collected, so we don't need explicit invalidation.
_CP_STATE_READ_SELECTION_CACHE: (
    "weakref.WeakKeyDictionary[Any, _CPStateReadSelection]"
) = weakref.WeakKeyDictionary()


def _compact_cp_state_read_selection(
    selection: _CPStateReadSelection,
) -> _CPStateReadSelection:
    """Drop dummy block-id 0 entries (cold requests) from tail-block selection."""
    if selection.compacted or int(selection.block_ids.numel()) == 0:
        return selection
    valid = selection.block_ids > 0
    active_slots = torch.nonzero(valid).squeeze(-1)
    if int(active_slots.numel()) == int(selection.block_ids.numel()):
        return _CPStateReadSelection(
            block_ids=selection.block_ids,
            read_block_table=selection.read_block_table,
            compacted=True,
        )
    new_block_ids = selection.block_ids.index_select(0, active_slots).contiguous()
    slot_remap = torch.zeros(
        int(selection.block_ids.numel()) + 1,
        dtype=selection.read_block_table.dtype,
        device=selection.read_block_table.device,
    )
    if int(active_slots.numel()) > 0:
        slot_remap[active_slots + 1] = torch.arange(
            1,
            int(active_slots.numel()) + 1,
            dtype=slot_remap.dtype,
            device=slot_remap.device,
        )
    new_read_bt = (
        slot_remap.index_select(0, selection.read_block_table.reshape(-1).long())
        .view_as(selection.read_block_table)
        .contiguous()
    )
    return _CPStateReadSelection(
        block_ids=new_block_ids,
        read_block_table=new_read_bt,
        compacted=True,
    )


def _build_cp_full_state_read_cache(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    cp_size: int,
    *,
    seq_start_per_req: Optional[torch.Tensor] = None,
    token_count: int = 0,
    state_tokens_per_block: int = 0,
    selection_cache_key: Optional[Any] = None,
    tail_rows: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather CP-sliced fixed-state blocks into a compact full-ring read cache.

    CP prefill stores INDEXER/CSA/HCA state as intra-block slices: each rank
    owns ``local_eb`` rows of the full ``local_eb * cp_size`` state ring.
    Skips dummy block 0 (cold requests) and, when ``tail_rows > 0`` under
    block-aligned prefix hits, broadcasts/gathers only the last ``tail_rows``
    rows of each cached tail block.
    """
    if cp_size <= 1:
        return state_cache, block_table
    if block_table is None or int(block_table.numel()) == 0:
        return state_cache, block_table
    if not state_cache.is_cuda:
        raise RuntimeError("CP-sliced DSV4 state-cache read requires CUDA state pools")

    local_eb = int(state_cache.shape[1])
    hidden = int(state_cache.shape[2])
    selection: Optional[_CPStateReadSelection] = None
    if selection_cache_key is not None:
        selection = _CP_STATE_READ_SELECTION_CACHE.get(selection_cache_key)
    if selection is None:
        selection = _select_cp_state_read_blocks(
            state_cache,
            block_table,
            seq_start_per_req=seq_start_per_req,
            token_count=token_count,
            state_tokens_per_block=state_tokens_per_block,
        )
    if not selection.compacted:
        selection = _compact_cp_state_read_selection(selection)
        if selection_cache_key is not None:
            try:
                _CP_STATE_READ_SELECTION_CACHE[selection_cache_key] = selection
            except TypeError:
                pass
    num_blocks = int(selection.block_ids.numel())
    full_eb = cp_size * local_eb

    read_cache = torch.zeros(
        (num_blocks + 1, full_eb, hidden),
        dtype=state_cache.dtype,
        device=state_cache.device,
    )
    if num_blocks == 0:
        return read_cache, selection.read_block_table

    if 0 < tail_rows <= local_eb and tail_rows < full_eb:
        if torch.distributed.is_initialized():
            from rtp_llm.models_py.distributed import collective_torch

            pg = collective_torch._get_group(Group.TP)
            ranks = torch.distributed.get_process_group_ranks(pg)
            if len(ranks) == cp_size:
                owner_global = ranks[cp_size - 1]
                if torch.distributed.get_rank() == owner_global:
                    tail_buf = (
                        state_cache.index_select(0, selection.block_ids)[
                            :, local_eb - tail_rows :
                        ]
                        .contiguous()
                    )
                else:
                    tail_buf = torch.empty(
                        (num_blocks, tail_rows, hidden),
                        dtype=state_cache.dtype,
                        device=state_cache.device,
                    )
                with record_function_range(
                    "dsv4.cp.broadcast.state_read_tail.sync.launch"
                ):
                    torch.distributed.broadcast(tail_buf, src=owner_global, group=pg)
                read_cache[1:, full_eb - tail_rows :].copy_(tail_buf)
                return read_cache, selection.read_block_table

        local_tail = (
            state_cache.index_select(0, selection.block_ids)[:, local_eb - tail_rows :]
            .reshape(num_blocks * tail_rows, hidden)
            .contiguous()
        )
        with record_function_range("dsv4.cp.all_gather.state_read_tail.sync.launch"):
            gathered = all_gather(local_tail, group=Group.TP)
        read_cache[1:, full_eb - tail_rows :].copy_(
            gathered.view(cp_size, num_blocks, tail_rows, hidden)[cp_size - 1]
        )
        return read_cache, selection.read_block_table

    local_blocks = state_cache.index_select(0, selection.block_ids)
    local_2d = local_blocks.reshape(num_blocks * local_eb, hidden).contiguous()
    with record_function_range("dsv4.cp.all_gather.state_read_cache.sync.launch"):
        gathered = all_gather(local_2d, group=Group.TP)
    _fill_cp_state_read_cache(
        read_cache, gathered, cp_size, num_blocks, local_eb, hidden
    )
    return read_cache, selection.read_block_table


def _cp_sliced_state_read_needed(
    meta: "CompressorMeta", cp_ctx: Optional[CPContext], raw_disabled: bool
) -> bool:
    if raw_disabled:
        if cp_ctx is None or cp_ctx.cp_size <= 1:
            return False
        if not getattr(cp_ctx, "kv_cache_sharded", False):
            return False
        return bool(meta.has_prefix)
    if cp_ctx is None or cp_ctx.cp_size <= 1:
        return False
    if not getattr(cp_ctx, "kv_cache_sharded", False):
        return False
    if not meta.has_prefix or meta.can_skip_state_cache_read:
        return False
    if not meta.any_cached_boundary_global:
        return False
    return True


@dataclass(frozen=True)
class CompressorMeta:
    """Pre-computed per-token launch metadata.

    Built once per (state_block_table, kv_block_table, positions, b_idx)
    tuple — typically by the attention layer just after ``set_pool_context``,
    so the math is amortized across both the host compressor and any nested
    indexer compressor that shares the same positions/b_idx layout.

    Fields are device tensors of length ``N_tok``:
      * ``positions``    : int64 absolute token positions
      * ``b_idx``        : int64 request index per token
      * ``state_slots``  : int64 state-pool slot per token (-1 = skip)
      * ``kv_slots``     : int64 KV-pool slot per token (-1 if non-boundary
                           or unallocated)
      * ``token_to_req`` : int32 alias of ``b_idx`` for the fused KV writer
      * ``is_batched``   : True when this meta uses the varlen/per-request
                           raw path. This includes B==1 CP prefill, where
                           keeping the same path avoids reintroducing a
                           scalar-B special case.
    """

    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    has_prefix: bool
    is_batched: bool = False
    seq_start_per_req: Optional[torch.Tensor] = None
    cu_seq_per_req: Optional[torch.Tensor] = None
    compressed_lens_per_token: Optional[torch.Tensor] = None
    active_state_indices: Optional[torch.Tensor] = None
    raw_boundary_indices: Optional[torch.Tensor] = None
    cached_boundary_indices: Optional[torch.Tensor] = None
    active_boundary_indices: Optional[torch.Tensor] = None
    can_skip_state_cache_read: bool = False
    cp_state_tail_rows: int = 0
    any_cached_boundary_global: bool = True


def _cache_cp_state_read_selection(
    meta: CompressorMeta,
    state_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    *,
    cp_ctx: Optional[CPContext],
    token_count: int,
    state_tokens_per_block: int,
) -> CompressorMeta:
    if (
        state_cache is None
        or block_table is None
        or meta.seq_start_per_req is None
        or not _cp_sliced_state_read_needed(meta, cp_ctx, raw_disabled=False)
    ):
        return meta
    try:
        with record_function_range("dsv4.fp8.compressor.meta.cp_state_read_selection"):
            sel = _select_cp_state_read_blocks(
                state_cache,
                block_table,
                seq_start_per_req=meta.seq_start_per_req,
                token_count=token_count,
                state_tokens_per_block=state_tokens_per_block,
            )
            _CP_STATE_READ_SELECTION_CACHE[meta] = _compact_cp_state_read_selection(
                sel
            )
    except TypeError:
        pass
    return meta


def _resolve_host_prefixes(
    compressor: "CompressorFP8",
    seq_start_per_req: Optional[torch.Tensor],
    prefix_lengths_host: Optional[Tuple[int, ...]] = None,
) -> Optional[Tuple[int, ...]]:
    if prefix_lengths_host is not None:
        return tuple(int(p) for p in prefix_lengths_host)
    cp_ctx = getattr(compressor, "_cp_ctx", None)
    if cp_ctx is not None:
        cp_host = getattr(cp_ctx, "prefix_lengths_host", None)
        if cp_host is not None:
            return tuple(int(p) for p in cp_host)
        if getattr(cp_ctx, "prefix_lengths", None) is None:
            return (int(getattr(cp_ctx, "prefix_length", 0)),)
    if seq_start_per_req is not None and seq_start_per_req.device.type == "cpu":
        return tuple(int(v) for v in seq_start_per_req.tolist())
    return None


def _build_host_compacted_slot_indices(
    positions: torch.Tensor,
    *,
    host_input_lengths: Tuple[int, ...],
    host_prefixes: Tuple[int, ...],
    has_prefix: bool,
    can_skip_state_cache_read: bool,
    window_size: int,
    compress_ratio: int,
    cp_size: int,
    cp_rank: int,
    kv_owner_tokens_per_block: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
]:
    """Compute compacted boundary indices purely on host (zero GPU nonzero sync).

    Returns ``active_state_indices=None`` so ``run_save_partial_states`` scans
    ``state_slots`` directly on-device without a host sync.
    """
    device = positions.device
    B = len(host_input_lengths)
    if (
        B == 1
        and cp_size <= 1
        and (can_skip_state_cache_read or not has_prefix or host_prefixes[0] <= 0)
    ):
        L0 = int(host_input_lengths[0])
        P0 = int(host_prefixes[0])
        off0 = (compress_ratio - 1 - (P0 % compress_ratio)) % compress_ratio
        if off0 >= L0:
            empty = positions.new_empty((0,), dtype=torch.int32)
            return None, empty, empty, empty, False
        count = (L0 - 1 - off0) // compress_ratio + 1
        raw_t = off0 + compress_ratio * torch.arange(
            count, dtype=torch.int32, device=device
        )
        return None, raw_t, positions.new_empty((0,), dtype=torch.int32), raw_t, False

    raw_list: list[int] = []
    cached_list: list[int] = []
    active_list: list[int] = []
    any_global = False
    cu_q = 0
    for req_b in range(B):
        L_b = int(host_input_lengths[req_b])
        P_b = int(host_prefixes[req_b])
        if L_b > 0:
            off = (compress_ratio - 1 - (P_b % compress_ratio)) % compress_ratio
            while off < L_b:
                pos = P_b + off
                needs_cache = (
                    (not can_skip_state_cache_read)
                    and has_prefix
                    and P_b > 0
                    and (pos - window_size + 1) < P_b
                )
                if needs_cache:
                    any_global = True
                owned = (
                    True
                    if cp_size <= 1
                    else ((pos // kv_owner_tokens_per_block) % cp_size) == cp_rank
                )
                if owned:
                    tok_idx = cu_q + off
                    active_list.append(tok_idx)
                    if needs_cache:
                        cached_list.append(tok_idx)
                    else:
                        raw_list.append(tok_idx)
                off += compress_ratio
        cu_q += L_b

    empty = positions.new_empty((0,), dtype=torch.int32)
    if not active_list:
        return None, empty, empty, empty, any_global
    if not cached_list:
        raw_t = torch.tensor(raw_list, dtype=torch.int32, device=device)
        return None, raw_t, empty, raw_t, any_global
    if not raw_list:
        cac_t = torch.tensor(cached_list, dtype=torch.int32, device=device)
        return None, empty, cac_t, cac_t, any_global
    raw_t = torch.tensor(raw_list, dtype=torch.int32, device=device)
    cac_t = torch.tensor(cached_list, dtype=torch.int32, device=device)
    act_t = torch.tensor(active_list, dtype=torch.int32, device=device)
    return None, raw_t, cac_t, act_t, any_global


def _build_compacted_slot_indices(
    positions: torch.Tensor,
    b_idx: torch.Tensor,
    state_slots: torch.Tensor,
    kv_slots: torch.Tensor,
    seq_start_per_req: torch.Tensor,
    *,
    has_prefix: bool,
    can_skip_state_cache_read: bool,
    window_size: int,
    compress_ratio: int,
    host_input_lengths: Optional[Tuple[int, ...]] = None,
    host_prefixes: Optional[Tuple[int, ...]] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    kv_owner_tokens_per_block: int = 0,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
]:
    """Compact active state & boundary indices once per step outside CUDA graph capture."""
    N = int(positions.shape[0])
    if N <= 0:
        return None, None, None, None, False
    if positions.is_cuda and torch.cuda.is_current_stream_capturing():
        return (
            None,
            None,
            None,
            None,
            bool(has_prefix and not can_skip_state_cache_read),
        )

    if (
        host_input_lengths is not None
        and host_prefixes is not None
        and len(host_input_lengths) > 0
        and len(host_input_lengths) == len(host_prefixes)
        and len(host_input_lengths) == int(seq_start_per_req.numel())
        and sum(int(x) for x in host_input_lengths) == N
        and (cp_size <= 1 or kv_owner_tokens_per_block > 0)
    ):
        return _build_host_compacted_slot_indices(
            positions,
            host_input_lengths=host_input_lengths,
            host_prefixes=host_prefixes,
            has_prefix=has_prefix,
            can_skip_state_cache_read=can_skip_state_cache_read,
            window_size=window_size,
            compress_ratio=compress_ratio,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_owner_tokens_per_block=kv_owner_tokens_per_block,
        )

    active_state_indices: Optional[torch.Tensor] = None
    if N > 32:
        active_state_indices = (
            torch.nonzero(state_slots >= 0).squeeze(-1).to(torch.int32).contiguous()
        )

    active_kv_mask = kv_slots >= 0
    active_boundary_indices = (
        torch.nonzero(active_kv_mask).squeeze(-1).to(torch.int32).contiguous()
    )
    if can_skip_state_cache_read or not has_prefix:
        return (
            active_state_indices,
            active_boundary_indices,
            positions.new_empty((0,), dtype=torch.int32),
            active_boundary_indices,
            False,
        )

    # Derive collective participation from rank-invariant global coordinates
    # rather than rank-local active_boundary_indices (which can be empty on
    # non-owning CP ranks under sharded KV page-RR ownership).
    starts = seq_start_per_req.to(
        device=positions.device, dtype=torch.long
    ).reshape(-1)
    if int(starts.numel()) == 1:
        all_starts = starts.expand_as(positions)
    else:
        all_starts = starts.index_select(0, b_idx.long())
    global_need_cache = (
        (positions >= 0)
        & (((positions + 1) % int(compress_ratio)) == 0)
        & (all_starts > 0)
        & ((positions - window_size + 1) < all_starts)
    )
    if int(active_boundary_indices.numel()) == 0:
        any_global = bool(global_need_cache.any().item())
        return (
            active_state_indices,
            active_boundary_indices,
            positions.new_empty((0,), dtype=torch.int32),
            active_boundary_indices,
            any_global,
        )

    active_idx_long = active_boundary_indices.long()
    need_cache = global_need_cache.index_select(0, active_idx_long)
    n_cached_val, any_global_val = torch.stack(
        [
            need_cache.sum().to(torch.int64),
            global_need_cache.any().to(torch.int64),
        ]
    ).tolist()
    n_cached = int(n_cached_val)
    any_global = bool(any_global_val)
    if n_cached == 0:
        return (
            active_state_indices,
            active_boundary_indices,
            positions.new_empty((0,), dtype=torch.int32),
            active_boundary_indices,
            any_global,
        )
    if n_cached == int(active_boundary_indices.numel()):
        return (
            active_state_indices,
            positions.new_empty((0,), dtype=torch.int32),
            active_boundary_indices,
            active_boundary_indices,
            any_global,
        )
    order = torch.argsort(need_cache.to(torch.uint8), descending=True)
    sorted_active = active_boundary_indices.index_select(0, order).contiguous()
    return (
        active_state_indices,
        sorted_active[n_cached:],
        sorted_active[:n_cached],
        active_boundary_indices,
        any_global,
    )


@dataclass
class _CompressorPending:
    """In-flight state between :meth:`CompressorFP8.start_prefill` and
    :meth:`CompressorFP8.finish_prefill`.

    Captured at ``start_prefill`` time so the orchestrator (attention
    overlap path) can interleave other work on the default stream while
    the CP all-gather drains on ``cp_gather_stream``. Holds:

    * ``fused_flat`` — rank-local ``[T_local, 2*out_dim]`` fused KV/gate
      projection. When CP is on, this is the source of the NCCL gather
      (``fused_gather_handle``) and ``finish_prefill`` overwrites the
      local reference with the gathered full-seq tensor.
    * ``fused_gather_handle`` — ``CPCudaAsyncGatherHandle`` (or sync handle)
      when CP is active; ``None`` when CP is off (single-rank prefill).
    * ``sp / bsz / seqlen`` — pre-resolved scalars mirroring what
      ``forward`` derives from ``start_pos`` / input shape, so
      ``finish_prefill`` does not need to re-inspect ``x``.
    * ``meta`` — caller-provided ``CompressorMeta`` (CP requires it; non-CP
      may pass ``None`` and let ``finish_prefill`` rebuild positions/b_idx
      from ``sp``/``bsz``/``seqlen``, matching ``forward``'s fallback).
    * ``out_dim`` — ``(1 + overlap) * head_dim``. Captured at start_prefill
      time so finish_prefill stays pure (no self peek).
    * ``restored_buf`` — optional full-sequence destination used by the
      non-prefix CP restore path. It is kept on the pending object so the
      restored fused tensor's storage stays owned through ``finish_prefill``.

    The overlap orchestrator may call ``wait_prefill_gather`` before
    ``finish_prefill`` to fence NCCL without making the compressed-pool write
    visible early. The baseline ``forward`` path is unchanged.
    """

    fused_flat: torch.Tensor
    fused_gather_handle: Optional[Any]
    sp: int
    bsz: int
    seqlen: int
    meta: Optional[CompressorMeta]
    out_dim: int
    profile_label: Optional[str] = None
    restored_buf: Optional[torch.Tensor] = None
    unpad_restore: Optional[torch.Tensor] = None


class _CompressorNorm(nn.Module):
    """RMSNorm weight holder — bf16 (vLLM kernel reads bf16 weight)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class CompressorFP8(PoolBackedModule):
    """FP8 KV cache compressor — vLLM-aligned per-token state pool path.

    Compress ratio: CSA uses ratio=4 (overlap=True), HCA uses ratio=128
    (overlap=False). The indexer compressor follows the host attention
    layer's ratio (typically 4, overlap=True).
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        *,
        cp_role: str,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        compressor_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``compressor_weights`` is a 4-key dict ``{"ape", "wkv", "wgate",
        "norm"}`` extracted by the caller from ``layer_weights[W.v4_*compressor_*]``."""
        super().__init__()
        assert head_dim in (KV_HEAD_DIM, INDEXER_HEAD_DIM), (
            f"CompressorFP8 supports head_dim in {{{KV_HEAD_DIM}, "
            f"{INDEXER_HEAD_DIM}}}; got {head_dim}"
        )
        assert compressor_weights is not None, (
            "CompressorFP8 requires compressor_weights — meta-tensor / "
            "stand-alone construction is not supported (use the BF16 path)."
        )
        self.dim = dim
        self.head_dim = head_dim
        # Which per-forward workspace CP buffer pair this compressor's gather
        # draws from — passed EXPLICITLY by the caller (main CSA/HCA vs nested
        # indexer) rather than inferred from ``head_dim``, so the semantics are
        # not hidden in a dimension constant. The two roles need distinct buffers
        # so both can be in flight within one layer under the overlap
        # orchestrator. Validated, never silently defaulted.
        assert cp_role in (_CP_ROLE_MAIN, _CP_ROLE_INDEXER), (
            f"CompressorFP8 cp_role must be one of "
            f"{{{_CP_ROLE_MAIN!r}, {_CP_ROLE_INDEXER!r}}}; got {cp_role!r}"
        )
        self._cp_role = cp_role
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = norm_eps
        coff = 1 + self.overlap
        self.coff = coff
        # Per-slot bytes of the FP8 KV pool entry (selects layout/kernel).
        self._pool_entry_bytes = (
            KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
        )

        # ape stays FP32. wkv/wgate are stored as BF16, while the fused
        # projection below uses BF16 operands with FP32 accumulation/output.
        # Register ape as a non-trainable Parameter so .to(device) follows the
        # module.
        self.ape = nn.Parameter(
            compressor_weights["ape"].float().contiguous(), requires_grad=False
        )
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        with torch.no_grad():
            self.wkv.weight = nn.Parameter(
                compressor_weights["wkv"].to(torch.bfloat16), requires_grad=False
            )
            self.wgate.weight = nn.Parameter(
                compressor_weights["wgate"].to(torch.bfloat16), requires_grad=False
            )
        self.norm = _CompressorNorm(head_dim)
        self.norm.weight = nn.Parameter(
            compressor_weights["norm"].to(torch.bfloat16),
            requires_grad=False,
        )

        # Fuse wkv + wgate into one bf16 weight matrix; saves one
        # GEMM launch per compressor decode call (~92/step at bs16).
        self._wkv_wgate_fused: Optional[torch.Tensor] = None
        self._fuse_wkv_wgate(coff)

        # Legacy attribute kept for attention.py's cmp_T fallback (line 1583).
        self._kv_cache_t: int = 0

        # Built from freqs_cis during model loading. Runtime only reads it.
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cos_sin_cache: Optional[torch.Tensor] = None
        self._state_tokens_per_block: int = 0
        self._cp_ctx: Optional[CPContext] = None
        self._cp_gather_stream: Optional[Any] = None
        self._kv_cache_sharded: bool = False
        self._profile_label: Optional[str] = None

    def _cp_profile_name(self, profile_label: Optional[str]) -> str:
        label = profile_label or self._profile_label
        if not label:
            role = "indexer" if self._cp_role == _CP_ROLE_INDEXER else "attn"
            label = f"{role}.ratio{self.compress_ratio}.hd{self.head_dim}"
        return f"dsv4.cp.all_gather.{label}.kv_score"

    def _fuse_wkv_wgate(self, coff: int) -> None:
        """Concat wkv + wgate along out-dim into one fused bf16 weight,
        then re-point ``wkv.weight`` / ``wgate.weight`` to views of the
        fused storage (zero memory overhead)."""
        with torch.no_grad():
            fused = torch.cat(
                [self.wkv.weight.data, self.wgate.weight.data], dim=0
            ).contiguous()
            self._wkv_wgate_fused = fused
            out_dim = coff * self.head_dim
            self.wkv.weight = nn.Parameter(fused[:out_dim], requires_grad=False)
            self.wgate.weight = nn.Parameter(fused[out_dim:], requires_grad=False)

    # ----------------------------------------------------------------------
    # Compatibility shims (kept to match the BF16 ``Compressor`` API surface
    # so callers don't need to special-case the FP8 class).
    # ----------------------------------------------------------------------
    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        """Stores ``_kv_cache_t`` only as informational metadata so legacy
        readers (e.g. ``attention.py:1583`` ``cmp_T`` fallback) keep working.
        The FP8 path does NOT allocate any per-step bf16 cache from this."""
        self._kv_cache_t = int(kv_cache_t)

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx
        # When ``kv_cache_sharded=True`` flows in via the CPContext, paged KV
        # pools use page-RR ownership and fixed STATE pools use intra-block
        # slices. Both writers therefore need CP-aware slot mappings.
        self._kv_cache_sharded = bool(
            cp_ctx is not None
            and getattr(cp_ctx, "kv_cache_sharded", False)
            and cp_ctx.cp_size > 1
        )

    # ----------------------------------------------------------------------
    # Metadata preparation (call once per forward, OFF the hot path)
    # ----------------------------------------------------------------------
    def prepare_metadata(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
        has_prefix: bool,
        is_batched: bool = False,
        seq_start_per_req: Optional[torch.Tensor] = None,
        cu_seq_per_req: Optional[torch.Tensor] = None,
        prefix_lengths_host: Optional[Tuple[int, ...]] = None,
        host_input_lengths: Optional[Tuple[int, ...]] = None,
    ) -> CompressorMeta:
        """Compute slot mappings + token_to_req from current pool context.

        Pure function of ``(positions, b_idx, self._state_block_table,
        self._kv_block_table, self.compress_ratio, self._state_eb,
        self._kv_eb)`` — safe to call once per attention forward and reuse
        across the host compressor and any nested indexer compressor that
        shares the same positions/b_idx (when their pool context is bound).
        """
        assert (
            positions.dim() == 1
        ), f"positions must be flat [N], got {positions.shape}"
        assert b_idx.dim() == 1, f"b_idx must be flat [N], got {b_idx.shape}"
        assert (
            positions.numel() == b_idx.numel()
        ), f"positions/b_idx length mismatch: {positions.numel()} vs {b_idx.numel()}"

        # Warmup: pool context unbound — return None-slotted meta so
        # callers (compressor.forward, _forward_prefill_compressed) can
        # short-circuit without crashing.
        if self._state_block_table is None or self._state_eb <= 0:
            return CompressorMeta(
                positions=positions,
                b_idx=b_idx,
                state_slots=None,
                kv_slots=None,
                token_to_req=b_idx.to(torch.int32),
                has_prefix=has_prefix,
                is_batched=is_batched,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        assert seq_start_per_req is not None and cu_seq_per_req is not None, (
            "seq_start_per_req and cu_seq_per_req are required for ring write mask; "
            "caller must supply per-request metadata"
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import _fused_compressor_meta_triton

        if not _fused_compressor_meta_triton._TRITON_AVAILABLE:
            raise RuntimeError(
                "DSV4 FP8 compressor requires fused Triton metadata preparation"
            )
        pool_rows = 0
        if self._kv_pool_view is not None:
            pool_rows = int(self._kv_pool_view.numel() // self._kv_pool_view.shape[-1])
        cp_ctx = self._cp_ctx if self._kv_cache_sharded else None
        cp_size = int(cp_ctx.cp_size) if cp_ctx is not None else 1
        cp_rank = int(cp_ctx.cp_rank) if cp_ctx is not None else 0
        (
            state_slots,
            kv_slots,
            token_to_req,
        ) = _fused_compressor_meta_triton.fused_compressor_slot_mapping(
            positions,
            b_idx,
            self._state_block_table,
            self._state_eb,
            self._kv_block_table,
            self._kv_eb,
            self.compress_ratio,
            seq_start_per_req,
            cu_seq_per_req,
            self._state_tokens_per_block,
            pool_rows=pool_rows,
            kv_tokens_per_block=self._kv_tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_owner_tokens_per_block=int(
                getattr(self, "_kv_owner_tokens_per_block", self._kv_tokens_per_block)
            ),
        )
        overlap = bool(getattr(self, "overlap", False))
        host_prefixes = _resolve_host_prefixes(
            self, seq_start_per_req, prefix_lengths_host
        )
        can_skip_state_cache_read = (not has_prefix) or (
            not overlap
            and host_prefixes is not None
            and len(host_prefixes) > 0
            and all((p % self.compress_ratio) == 0 for p in host_prefixes)
        )
        full_state_eb = self._state_eb * max(1, cp_size)
        cp_state_tail_rows = (
            self.compress_ratio
            if (
                overlap
                and has_prefix
                and host_prefixes is not None
                and len(host_prefixes) > 0
                and self._state_tokens_per_block > self.compress_ratio
                and full_state_eb > self.compress_ratio
                and all(
                    (p % self._state_tokens_per_block) == 0
                    and (p % full_state_eb) == 0
                    for p in host_prefixes
                )
            )
            else 0
        )
        host_inputs = host_input_lengths
        if host_inputs is None and cp_ctx is not None:
            cp_in = getattr(cp_ctx, "input_lengths_global_host", None)
            if cp_in is not None:
                host_inputs = tuple(int(v) for v in cp_in)
        window_size = (1 + int(overlap)) * self.compress_ratio
        (
            active_state_indices,
            raw_boundary_indices,
            cached_boundary_indices,
            active_boundary_indices,
            any_cached_boundary_global,
        ) = _build_compacted_slot_indices(
            positions,
            b_idx,
            state_slots,
            kv_slots,
            seq_start_per_req,
            has_prefix=has_prefix,
            can_skip_state_cache_read=can_skip_state_cache_read,
            window_size=window_size,
            compress_ratio=self.compress_ratio,
            host_input_lengths=host_inputs,
            host_prefixes=host_prefixes,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_owner_tokens_per_block=int(
                getattr(self, "_kv_owner_tokens_per_block", self._kv_tokens_per_block)
            ),
        )
        meta = CompressorMeta(
            positions=positions,
            b_idx=b_idx,
            state_slots=state_slots,
            kv_slots=kv_slots,
            token_to_req=token_to_req,
            has_prefix=has_prefix,
            is_batched=is_batched,
            seq_start_per_req=seq_start_per_req,
            cu_seq_per_req=cu_seq_per_req,
            active_state_indices=active_state_indices,
            raw_boundary_indices=raw_boundary_indices,
            cached_boundary_indices=cached_boundary_indices,
            active_boundary_indices=active_boundary_indices,
            can_skip_state_cache_read=can_skip_state_cache_read,
            cp_state_tail_rows=cp_state_tail_rows,
            any_cached_boundary_global=any_cached_boundary_global,
        )
        return _cache_cp_state_read_selection(
            meta,
            getattr(self, "_state_pool_3d", None),
            self._state_block_table,
            cp_ctx=self._cp_ctx,
            token_count=window_size,
            state_tokens_per_block=self._state_tokens_per_block,
        )

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def init_rope_cache(self, freqs_cis: torch.Tensor) -> None:
        """Bind RoPE tables and build the shared kernel cache at model load."""
        self.freqs_cis = freqs_cis
        device = freqs_cis.device
        key = (
            id(freqs_cis),
            device,
            tuple(int(v) for v in freqs_cis.shape),
            freqs_cis.dtype,
        )
        shared = _SHARED_COS_SIN_CACHE.get(key)
        if shared is None:
            shared, _ = build_cos_sin_cache(freqs_cis)
            _SHARED_COS_SIN_CACHE[key] = shared
        self._cos_sin_cache = shared

    def _compute_state_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
        seq_end_per_req: torch.Tensor,  # [B] int64
    ) -> torch.Tensor:
        """state_slot[t] = state_block_table[b, (pos//tpb)%max_blocks] * eb + pos%eb.

        State pools are SWA-type ring tables. Block-table indexing uses
        ``_state_tokens_per_block`` (physical block size) with modulo
        wrapping; in-block ring offset uses ``_state_eb``.
        Returns -1 where the resolved block_id is negative (unallocated).

        Ring write mask: only the last R positions before each block boundary
        (or sequence end) actually write. Earlier positions whose ring entries
        would be overwritten by later tokens in the same block are masked to -1.
        """
        bt = self._state_block_table
        eb = self._state_eb
        tpb = self._state_tokens_per_block
        assert bt is not None and eb > 0, "state pool context unbound"
        if self._kv_cache_sharded and self._cp_ctx is not None:
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_state_slot_mapping,
            )

            return cp_state_slot_mapping(
                positions,
                bt,
                b_idx,
                eb,
                tpb,
                self._cp_ctx.cp_size,
                self._cp_ctx.cp_rank,
                seq_end_per_req=seq_end_per_req,
            )
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        if max_blocks <= 0:
            return torch.full_like(positions, -1)
        block_in_seq = (positions // tpb) % max_blocks
        in_block = positions % eb
        block_id = bt_long[b_idx, block_in_seq]
        valid = block_id > 0
        block_end = (positions // tpb + 1) * tpb
        seq_end = seq_end_per_req[b_idx]
        effective_end = torch.minimum(block_end, seq_end)
        valid = valid & ((positions + eb) >= effective_end)
        slot = block_id * eb + in_block
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _compute_kv_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> torch.Tensor:
        """KV-pool slot for each token. -1 unless (pos+1) % ratio == 0
        (i.e. boundary token that produces a compressed entry).

        Block addressing follows the framework convention for FULL paged
        pools: the block_table is indexed in raw-token space using
        ``_kv_tokens_per_block``. The KV pool's per-block entry count is
        ``kv_eb = kernel_tokens_per_block / ratio``, so the in-block offset
        is the compressed-entry offset within that raw-token block.

          block_in_seq = pos // TOKENS_PER_BLOCK              # token -> block
          in_block     = (pos % TOKENS_PER_BLOCK) // ratio    # compressed offset
          slot         = block_id * kv_eb + in_block

        Also masks out any slot that would land past the pool's row count; a
        malformed block_table can otherwise produce a slot above
        ``pool_view.shape[0]`` and silently corrupt an unrelated pool entry.
        """
        bt = self._kv_block_table
        kv_eb = self._kv_eb
        ratio = self.compress_ratio
        tokens_per_block = self._kv_tokens_per_block
        if bt is None or kv_eb <= 0 or tokens_per_block <= 0:
            return torch.full_like(positions, -1)
        if self._kv_cache_sharded and self._cp_ctx is not None:
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_kv_slot_mapping,
            )

            slot = cp_kv_slot_mapping(
                positions,
                bt,
                b_idx,
                tokens_per_block,
                kv_eb,
                ratio,
                self._cp_ctx.cp_size,
                self._cp_ctx.cp_rank,
                owner_tokens_per_block=self._kv_owner_tokens_per_block,
            )
            if self._kv_pool_view is not None:
                pool_rows = int(
                    self._kv_pool_view.numel() // self._kv_pool_view.shape[-1]
                )
                slot = torch.where(slot < pool_rows, slot, torch.full_like(slot, -1))
            return slot
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        if max_blocks <= 0:
            return torch.full_like(positions, -1)

        boundary = ((positions + 1) % ratio) == 0
        block_in_seq = positions // tokens_per_block
        in_block = (positions % tokens_per_block) // ratio
        in_capacity = block_in_seq < max_blocks
        safe_block_in_seq = block_in_seq.clamp(min=0, max=max_blocks - 1)
        block_id = bt_long[b_idx, safe_block_in_seq]
        slot = block_id * kv_eb + in_block
        valid = boundary & in_capacity & (block_id >= 0)
        if self._kv_pool_view is not None:
            pool_rows = int(self._kv_pool_view.numel() // self._kv_pool_view.shape[-1])
            valid = valid & (slot < pool_rows)
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32 (or padded rank-major with unpad_restore)
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32 (or padded rank-major with unpad_restore)
        meta: CompressorMeta,
        seq_start: Optional[int] = None,
        *,
        unpad_restore: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the two vLLM kernels (state write + boundary KV write).

        ``seq_start`` is the absolute position of ``kv_flat[0]`` for
        sequentially-laid-out batches (prefill: ``sp_int``). When provided
        the fused kernel reads any overlap-window position with
        ``flat_idx = pos - seq_start in [0, N)`` directly from
        ``kv_flat / score_flat`` instead of reading back through the state
        pool, where current-launch writes are still in flight.

        Pass ``None`` to disable the raw path (decode: ``kv_flat`` is
        indexed by ``req_idx``, not by absolute position offset).

        When ``unpad_restore`` is provided on the CP raw prefill path, both
        kernels gather directly from rank-major ``kv_flat / score_flat`` via
        ``unpad_restore`` so ``cp_restore`` does not need to materialize a full
        restored buffer.

        All slot-mapping math is consumed from ``meta`` — this method only
        does kernel launches. Designed to stay branch-light so it composes
        cleanly with CUDA graph capture.
        """
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._state_block_table is None
            or self._kv_block_table is None
        ):
            # Warmup / unbound: nothing to write.
            return

        N = int(meta.positions.shape[0])
        if N == 0:
            return

        cos_sin_cache = self._cos_sin_cache
        assert cos_sin_cache is not None

        with record_function_range("dsv4.fp8.compressor.launch.save_partial_states"):
            run_save_partial_states(
                kv_flat,
                score_flat,
                self.ape,
                meta.positions,
                self._state_pool_3d,
                meta.state_slots,
                compress_ratio=self.compress_ratio,
                unpad_restore=unpad_restore,
                active_indices=meta.active_state_indices,
            )

        use_varlen_raw = (
            meta.is_batched
            and meta.seq_start_per_req is not None
            and meta.cu_seq_per_req is not None
        )
        raw_disabled = (seq_start is None) and not use_varlen_raw
        eff_seq_start = 0 if (raw_disabled or use_varlen_raw) else seq_start
        eff_seq_start_per_req = meta.seq_start_per_req if use_varlen_raw else None
        eff_cu_seq_per_req = meta.cu_seq_per_req if use_varlen_raw else None

        split_boundary_write = (
            not raw_disabled
            and meta.raw_boundary_indices is not None
            and meta.cached_boundary_indices is not None
        )

        if split_boundary_write:
            if int(meta.raw_boundary_indices.numel()) > 0:
                with record_function_range(
                    "dsv4.fp8.compressor.launch.compress_kv_write"
                ):
                    run_fused_compress_kv_write(
                        self._state_pool_3d,
                        meta.token_to_req,
                        meta.positions,
                        meta.state_slots,
                        self._state_block_table,
                        self.norm.weight,
                        self.norm_eps,
                        cos_sin_cache,
                        self._kv_pool_view,
                        meta.kv_slots,
                        kv_flat,
                        score_flat,
                        self.ape,
                        eff_seq_start,
                        disable_raw_path=False,
                        head_dim=self.head_dim,
                        rope_head_dim=self.rope_head_dim,
                        compress_ratio=self.compress_ratio,
                        overlap=self.overlap,
                        seq_start_per_req=eff_seq_start_per_req,
                        cu_seq_per_req=eff_cu_seq_per_req,
                        state_tokens_per_block=self._state_tokens_per_block,
                        may_use_cache=False,
                        unpad_restore=unpad_restore,
                        active_token_indices=meta.raw_boundary_indices,
                    )
            state_cache_for_read = self._state_pool_3d
            state_block_table_for_read = self._state_block_table
            if _cp_sliced_state_read_needed(meta, self._cp_ctx, raw_disabled=False):
                with record_function_range(
                    "dsv4.fp8.compressor.launch.cp_gather_state_read_cache"
                ):
                    state_cache_for_read, state_block_table_for_read = (
                        _build_cp_full_state_read_cache(
                            self._state_pool_3d,
                            self._state_block_table,
                            int(self._cp_ctx.cp_size),
                            seq_start_per_req=eff_seq_start_per_req,
                            token_count=(1 + int(self.overlap))
                            * self.compress_ratio,
                            state_tokens_per_block=self._state_tokens_per_block,
                            selection_cache_key=meta,
                            tail_rows=meta.cp_state_tail_rows,
                        )
                    )
            if int(meta.cached_boundary_indices.numel()) > 0:
                with record_function_range(
                    "dsv4.fp8.compressor.launch.compress_kv_write"
                ):
                    run_fused_compress_kv_write(
                        state_cache_for_read,
                        meta.token_to_req,
                        meta.positions,
                        meta.state_slots,
                        state_block_table_for_read,
                        self.norm.weight,
                        self.norm_eps,
                        cos_sin_cache,
                        self._kv_pool_view,
                        meta.kv_slots,
                        kv_flat,
                        score_flat,
                        self.ape,
                        eff_seq_start,
                        disable_raw_path=False,
                        head_dim=self.head_dim,
                        rope_head_dim=self.rope_head_dim,
                        compress_ratio=self.compress_ratio,
                        overlap=self.overlap,
                        seq_start_per_req=eff_seq_start_per_req,
                        cu_seq_per_req=eff_cu_seq_per_req,
                        state_tokens_per_block=self._state_tokens_per_block,
                        may_use_cache=True,
                        unpad_restore=unpad_restore,
                        active_token_indices=meta.cached_boundary_indices,
                    )
            return

        may_use_cache = bool(
            raw_disabled or (meta.has_prefix and not meta.can_skip_state_cache_read)
        )
        state_cache_for_read = self._state_pool_3d
        state_block_table_for_read = self._state_block_table
        if _cp_sliced_state_read_needed(meta, self._cp_ctx, raw_disabled):
            with record_function_range(
                "dsv4.fp8.compressor.launch.cp_gather_state_read_cache"
            ):
                state_cache_for_read, state_block_table_for_read = (
                    _build_cp_full_state_read_cache(
                        self._state_pool_3d,
                        self._state_block_table,
                        int(self._cp_ctx.cp_size),
                        seq_start_per_req=eff_seq_start_per_req,
                        token_count=(1 + int(self.overlap)) * self.compress_ratio,
                        state_tokens_per_block=self._state_tokens_per_block,
                        selection_cache_key=meta,
                        tail_rows=(
                            0 if raw_disabled else meta.cp_state_tail_rows
                        ),
                    )
                )

        with record_function_range("dsv4.fp8.compressor.launch.compress_kv_write"):
            run_fused_compress_kv_write(
                state_cache_for_read,
                meta.token_to_req,
                meta.positions,
                meta.state_slots,
                state_block_table_for_read,
                self.norm.weight,
                self.norm_eps,
                cos_sin_cache,
                self._kv_pool_view,
                meta.kv_slots,
                kv_flat,
                score_flat,
                self.ape,
                eff_seq_start,
                disable_raw_path=raw_disabled,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                seq_start_per_req=eff_seq_start_per_req,
                cu_seq_per_req=eff_cu_seq_per_req,
                state_tokens_per_block=self._state_tokens_per_block,
                may_use_cache=may_use_cache,
                unpad_restore=unpad_restore,
                active_token_indices=meta.active_boundary_indices,
            )

    # ----------------------------------------------------------------------
    # Overlap orchestration: split-phase prefill (start / finish).
    #
    # The baseline :meth:`forward` below is a strict superset of these two
    # split-phase methods (start + finish performed sequentially). The
    # overlap orchestrator in attention.py uses the split-phase variant so
    # it can interleave the CP all-gather (queued on a side stream by
    # ``start_prefill``) with default-stream compute (SWA pool write,
    # indexer compute_q) before waiting on the gather inside
    # ``finish_prefill``.
    #
    # Contract:
    #   * Every ``start_prefill`` call must be paired with exactly one
    #     ``finish_prefill`` call on the returned handle.
    #   * On warmup (no pool bound) ``start_prefill`` returns ``None`` and
    #     ``finish_prefill(None)`` is a no-op — callers can chain
    #     unconditionally without warmup branching.
    #   * ``cp_gather_stream`` is the caller-owned CP gather stream. Sharing
    #     one stream across multiple compressor calls in the same layer
    #     (e.g. CSA: nested indexer + main) is required for FIFO ordering
    #     of NCCL collectives within the ProcessGroup.
    # ----------------------------------------------------------------------
    def start_prefill(
        self,
        x: torch.Tensor,
        start_pos,
        *,
        meta: Optional[CompressorMeta] = None,
        cp_gather_stream: Optional[Any] = None,
        profile_label: Optional[str] = None,
        # REQUIRED, never Optional — by design. ``workspace`` is the single
        # authoritative per-forward prefill scratch (allocated once in
        # ``prefill/forward.py::forward_layers`` and threaded through the meta
        # broadcast). It is dereferenced only on the CP gather branch and is
        # unused on the non-CP path, but it is ALWAYS passed so there is exactly
        # one buffer-lifecycle contract and no hidden ``None``/fallback-allocate
        # branch a future edit could trip over. Do NOT relax this to Optional.
        workspace: "PrefillWorkspace",
    ) -> Optional[_CompressorPending]:
        """Begin a prefill compressor launch without waiting on the CP gather.

        Steps performed eagerly on the **default** stream:
          * shape / sp resolve (mirrors ``forward``);
          * warmup early return → ``None`` (no pool bound by framework);
          * fused KV/gate projection (``_linear_bf16_bf16_fp32``).

        Steps deferred to :meth:`finish_prefill`:
          * waiting the CP all-gather (queued here on
            ``cp_gather_stream`` when CP is active);
          * splitting fused → kv_flat + score_flat;
          * the ``_launch`` writer kernel.
        """
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        out_dim = (1 + self.overlap) * self.head_dim
        with record_function_range("dsv4.fp8.compressor.prefill.fused_linear"):
            fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
            N = bsz * seqlen
            fused_flat = fused_out.reshape(N, -1)

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        fused_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            assert meta is not None, (
                "CompressorFP8 CP start_prefill requires hoisted "
                "CompressorMeta; non-CP path may pass meta=None"
            )
            N_full = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N_full, (
                f"CP compressor meta/token length mismatch: "
                f"meta={meta.positions.numel()} seq_len_full={N_full}"
            )
            # The non-prefix restore destination is the per-forward workspace's
            # role buffer (``cp_restore_{main,idx}``), selected in the gather
            # impl by ``cp_role`` — no fresh per-layer alloc here (that churn is
            # exactly what the workspace buffers eliminate under overlap).
            gather_stream = cp_gather_stream
            if gather_stream is None and fused_flat.is_cuda:
                gather_stream = torch.cuda.Stream(device=fused_flat.device)
            profile_name = self._cp_profile_name(profile_label)
            gather_range = "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            if profile_label:
                gather_range = (
                    f"dsv4.fp8.compressor.prefill.{profile_label}.cp_gather_kv_score"
                )
            with record_function_range(gather_range):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    profile_name=profile_name,
                    workspace=workspace,
                    cp_role=self._cp_role,
                )

        return _CompressorPending(
            fused_flat=fused_flat,
            fused_gather_handle=fused_gather_handle,
            sp=sp,
            bsz=bsz,
            seqlen=seqlen,
            meta=meta,
            out_dim=out_dim,
            profile_label=profile_label,
            restored_buf=None,
        )

    def _wait_cp_gather(
        self, handle: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fence CP gather; skip cp_restore when compressor kernels read raw."""
        if (
            type(self)._launch is CompressorFP8._launch
            and isinstance(handle, CPCudaAsyncGatherHandle)
            and not handle.cp_ctx.unpad_restore_is_prefix
            and handle.cp_ctx.unpad_restore.device == handle.gathered.device
        ):
            return cp_wait_gather_raw(handle), handle.cp_ctx.unpad_restore
        return cp_wait_gather_full(handle), None

    def _call_launch(
        self,
        kv_flat: torch.Tensor,
        score_flat: torch.Tensor,
        meta: CompressorMeta,
        seq_start: Optional[int],
        unpad_restore: Optional[torch.Tensor],
    ) -> None:
        if unpad_restore is not None:
            self._launch(
                kv_flat,
                score_flat,
                meta,
                seq_start=seq_start,
                unpad_restore=unpad_restore,
            )
        else:
            self._launch(kv_flat, score_flat, meta, seq_start=seq_start)

    def wait_prefill_gather(self, pending: Optional[_CompressorPending]) -> None:
        """Fence the split prefill CP gather without writing the FP8 pool.

        CSA overlap uses this to ensure the main compressor's NCCL gather has
        completed before indexer score/topk, while preserving the baseline
        order where the main CSA pool write happens after indexer topk.
        """
        if pending is None or pending.fused_gather_handle is None:
            return
        wait_range = "dsv4.fp8.compressor.prefill.cp_wait_kv_score"
        if pending.profile_label:
            wait_range = (
                f"dsv4.fp8.compressor.prefill.{pending.profile_label}.cp_wait_kv_score"
            )
        with record_function_range(wait_range):
            pending.fused_flat, pending.unpad_restore = self._wait_cp_gather(
                pending.fused_gather_handle
            )
            pending.fused_gather_handle = None

    def finish_prefill(self, pending: Optional[_CompressorPending]) -> None:
        """Drain a :meth:`start_prefill` and write the FP8 KV pool.

        Mirrors the second half of :meth:`forward` (wait → split → launch).
        ``pending=None`` is a no-op (warmup early-return path from
        ``start_prefill``).
        """
        if pending is None:
            return  # warmup, mirrors forward()'s early return
        self.wait_prefill_gather(pending)
        fused_flat = pending.fused_flat
        out_dim = pending.out_dim
        meta = pending.meta

        with record_function_range("dsv4.fp8.compressor.prefill.split_kv_score"):
            assert fused_flat.dim() == 2, (
                f"CompressorFP8 prefill expects flat fused projection, got "
                f"{tuple(fused_flat.shape)}"
            )
            assert fused_flat.size(1) == 2 * out_dim, (
                f"CompressorFP8 fused hidden mismatch: got {fused_flat.size(1)}, "
                f"expected {2 * out_dim}"
            )
            kv_flat = fused_flat[:, :out_dim]
            score_flat = fused_flat[:, out_dim:]

        if meta is None:
            # Non-CP fallback: rebuild positions/b_idx from sp/bsz/seqlen.
            device = fused_flat.device
            with record_function_range("dsv4.fp8.compressor.prefill.build_meta"):
                positions, b_idx = _build_prefill_positions(
                    pending.sp, pending.bsz, pending.seqlen, device
                )
                seq_starts = torch.full(
                    (pending.bsz,), pending.sp, dtype=torch.long, device=device
                )
                cu_seqs = (
                    torch.arange(pending.bsz + 1, dtype=torch.long, device=device)
                    * pending.seqlen
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=pending.sp > 0,
                    seq_start_per_req=seq_starts,
                    cu_seq_per_req=cu_seqs,
                )

        seq_start = None if meta.is_batched else pending.sp
        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._call_launch(
                kv_flat, score_flat, meta, seq_start, pending.unpad_restore
            )

    # ----------------------------------------------------------------------
    # Forward (prefill)
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
        meta: Optional[CompressorMeta] = None,
        *,
        # REQUIRED, never Optional — see ``start_prefill``: one authoritative
        # per-forward buffer, always passed (unused on non-CP), no fallback.
        workspace: "PrefillWorkspace",
    ) -> Optional[torch.Tensor]:
        """Prefill entry. ``bsz==1`` (FIFO scheduler).

        Returns ``None`` — downstream readers gather compressed K from the
        FP8 KV pool directly.

        ``meta`` lets the caller (typically ``attention.py``) hoist the
        slot-mapping compute out of the per-layer hot path and amortize it
        across the host compressor and any nested indexer compressor that
        share the same positions/b_idx. When ``None`` the compressor falls
        back to the in-body compute path (warmup / standalone / UT).
        """
        del sequence_lengths  # not needed: positions derived from start_pos+arange
        # Phase-3a: accept either flat ``[T_total, dim]`` (vLLM-native /
        # batched prefill) or legacy ``[B, S, dim]``. The compressor kernels
        # consume token-major 2D tensors, so the prefill path flattens once
        # immediately after the fused projection and never reintroduces B.
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        # Warmup forward (no pool bound by framework): no-op.
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        with record_function_range("dsv4.fp8.compressor.prefill.fused_linear"):
            fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
            N = bsz * seqlen
            fused_flat = fused_out.reshape(N, -1)

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        fused_gather_handle = None
        unpad_restore: Optional[torch.Tensor] = None
        if cp_gather:
            assert cp_ctx is not None
            gather_stream = (
                torch.cuda.Stream(device=fused_flat.device)
                if fused_flat.is_cuda
                else None
            )
            # Gather the fused ``[kv | score]`` projection once. This removes
            # one NCCL launch and keeps the CP path token-major 2D; the split
            # below is a strided view and the Triton wrappers pass row strides
            # explicitly.
            with record_function_range(
                "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            ):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    profile_name=self._cp_profile_name(None),
                    workspace=workspace,
                    cp_role=self._cp_role,
                )
            assert meta is not None, (
                "CompressorFP8 CP prefill requires full-sequence metadata from "
                "rtp_llm.models_py.modules.dsv4.fp8.prefill_meta; rebuilding it "
                "inside compressor.forward is intentionally disabled."
            )
            N = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N, (
                f"CP compressor meta/token length mismatch: meta={meta.positions.numel()} "
                f"seq_len_full={N}"
            )
            assert fused_gather_handle is not None
            with record_function_range("dsv4.fp8.compressor.prefill.cp_wait_kv_score"):
                fused_flat, unpad_restore = self._wait_cp_gather(fused_gather_handle)

        with record_function_range("dsv4.fp8.compressor.prefill.split_kv_score"):
            assert fused_flat.dim() == 2, (
                f"CompressorFP8 prefill expects flat fused projection, got "
                f"{tuple(fused_flat.shape)}"
            )
            assert fused_flat.size(1) == 2 * out_dim, (
                f"CompressorFP8 fused hidden mismatch: got {fused_flat.size(1)}, "
                f"expected {2 * out_dim}"
            )
            kv_flat = fused_flat[:, :out_dim]
            score_flat = fused_flat[:, out_dim:]
        if meta is None:
            with record_function_range("dsv4.fp8.compressor.prefill.build_meta"):
                positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=sp > 0,
                    seq_start_per_req=torch.tensor(
                        [sp], dtype=torch.long, device=device
                    ),
                    cu_seq_per_req=torch.tensor(
                        [0, seqlen], dtype=torch.long, device=device
                    ),
                )
        # Varlen prefill carries per-request raw arrays, so ``_launch`` routes
        # through the kernel's BATCHED branch even when CP has only one
        # request. Legacy scalar metadata keeps the old ``seq_start`` path.
        seq_start = None if meta.is_batched else sp
        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._call_launch(kv_flat, score_flat, meta, seq_start, unpad_restore)
        return None

    # ----------------------------------------------------------------------
    # Forward (decode, vectorized over B)
    # ----------------------------------------------------------------------
    def forward_decode_vectorized(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        meta: Optional[CompressorMeta] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Batched decode entry. ``position_ids`` enables q_len > 1 verify."""
        bsz, q_len = int(x.size(0)), int(x.size(1))
        T = bsz * q_len
        if position_ids is None:
            assert q_len == 1, "decode q_len > 1 requires flat position_ids"
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        # ``kv`` and ``score`` are last-dim slices of ``fused_out``. Keep
        # them as strided row views instead of materialising two BF16 copy
        # kernels; the Triton writer consumes row stride explicitly.
        kv_flat = kv.view(T, -1)
        score_flat = score.view(T, -1)
        if meta is None:
            if position_ids is None:
                positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                cu_seq_per_req = torch.arange(
                    0, bsz + 1, device=device, dtype=torch.int64
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=True,
                    seq_start_per_req=positions,
                    cu_seq_per_req=cu_seq_per_req,
                )
            else:
                positions = (
                    position_ids.to(device=device, dtype=torch.long)
                    .reshape(T)
                    .contiguous()
                )
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                b_idx = b_idx.repeat_interleave(q_len).contiguous()
                position_ids_2d = positions.view(bsz, q_len)
                cu_seq_per_req = torch.arange(
                    0,
                    (bsz + 1) * q_len,
                    q_len,
                    device=device,
                    dtype=torch.long,
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=True,
                    is_batched=q_len > 1,
                    seq_start_per_req=position_ids_2d[:, 0].to(torch.long).contiguous(),
                    cu_seq_per_req=cu_seq_per_req,
                )
        self._launch(kv_flat, score_flat, meta)
        return None


# ---------------------------------------------------------------------------
# Free helpers — exposed so the attention layer can build positions/b_idx
# once per forward and feed them through ``prepare_metadata`` / ``forward``.
# ---------------------------------------------------------------------------
def _build_prefill_positions(
    sp: int, bsz: int, seqlen: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``positions = sp + arange(seqlen)`` flat ``[seqlen]``; ``b_idx``
    all-zeros ``[seqlen]``.

    Single-request helper only — batched prefill feeds
    ``(position_ids, req_id_per_token)`` straight into
    ``compressor.prepare_metadata`` so this builder must not be reached
    from a batched call site.
    """
    assert bsz == 1, (
        f"_build_prefill_positions is the legacy B==1 helper; got bsz={bsz}. "
        "Varlen callers must use position_ids / req_id_per_token directly."
    )
    positions = torch.arange(sp, sp + seqlen, device=device, dtype=torch.long)
    b_idx = torch.zeros(seqlen, device=device, dtype=torch.long)
    return positions, b_idx


def build_prefill_metadata(
    compressor: "CompressorFP8", sp: int, bsz: int, seqlen: int, device: torch.device
) -> CompressorMeta:
    """Convenience: build positions/b_idx + ``CompressorMeta`` in one call."""
    positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
    return compressor.prepare_metadata(
        positions,
        b_idx,
        has_prefix=sp > 0,
        seq_start_per_req=torch.tensor([sp], dtype=torch.long, device=device),
        cu_seq_per_req=torch.tensor([0, seqlen], dtype=torch.long, device=device),
    )


def build_prepare_metadata_args(
    *,
    use_varlen: bool,
    has_prefix: bool,
    device: torch.device,
    sp_int: int,
    seqlen: int,
    position_ids: Optional[torch.Tensor] = None,
    req_id_per_token: Optional[torch.Tensor] = None,
    seq_start_per_req: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return the kwargs dict for ``CompressorFP8.prepare_metadata``,
    branching on ``use_varlen``. Single source of truth for the three
    call sites that used to inline this dispatch:

      * ``Attention._build_compressor_meta`` (HCA + standalone CSA path)
      * ``Attention._build_csa_prefill_meta`` (CSA inline that shares the
        pool bind with ``IndexerFP8.prepare``)
      * ``IndexerFP8.prepare`` nested compressor hoist

    Under varlen we pass the upper-layer-derived per-request tensors
    straight through; the legacy B==1 path collapses to the same
    ``(arange(sp, sp+T), zeros)`` pair ``_build_prefill_positions`` produces
    so bisecting between the two stays bit-equal.

    ``seq_start_per_req`` accepts either ``sp_per_req`` (Attention) or
    ``prefix_lengths`` (Indexer); both are forwarded as int64 per-request
    metadata for the compressor varlen raw path.
    """
    if use_varlen:
        assert (
            position_ids is not None
            and req_id_per_token is not None
            and seq_start_per_req is not None
            and cu_seqlens is not None
        ), "varlen dispatch requires position_ids/req_id_per_token/seq_start_per_req/cu_seqlens"
        return dict(
            positions=position_ids.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
            b_idx=req_id_per_token.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
            has_prefix=has_prefix,
            is_batched=True,
            seq_start_per_req=seq_start_per_req.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
            cu_seq_per_req=cu_seqlens.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
        )
    positions, b_idx = _build_prefill_positions(sp_int, 1, seqlen, device)
    return dict(
        positions=positions,
        b_idx=b_idx,
        has_prefix=has_prefix,
        is_batched=False,
        seq_start_per_req=None,
        cu_seq_per_req=None,
    )


def build_decode_metadata(
    compressor: "CompressorFP8", start_pos: torch.Tensor, bsz: int
) -> CompressorMeta:
    device = start_pos.device
    positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz).contiguous()
    b_idx = torch.arange(bsz, device=device, dtype=torch.long)
    cu_seq_per_req = torch.arange(0, bsz + 1, device=device, dtype=torch.long)
    return compressor.prepare_metadata(
        positions,
        b_idx,
        has_prefix=True,
        seq_start_per_req=positions,
        cu_seq_per_req=cu_seq_per_req,
    )
