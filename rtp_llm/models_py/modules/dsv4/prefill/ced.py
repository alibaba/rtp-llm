"""Bounded, forward-local CED transport for the V4.1 isolated generation probe.

Integration contract (all four TP ranks call each operation in the same order)::

    plan = CEDTail.create(v4, cp_ctx, attn_inputs,
                          prepare_hidden_fn=prepare_hidden_fn,
                          cache_store_active=write_cache_store_impl is not None)
    # After the COMPLETE L20 block, before L21:
    h, input_ids = plan.compact(v4, h, input_ids, shared)
    # Main installs plan.context, positions, plan.cu_seqlens and fresh prefill
    # metadata; shared["ced_swa_start"] = plan.swa_start disables prefix reads.
    # Run L21..39, then head reduction and norm on the compact rows.
    h = plan.restore_rows(h)
    plan.restore_aux(v4)
    # Main restores plan.original_context and removes its marker in finally.

The caller owns the default-off flag, generation-only eligibility, vision/loss
exclusion, and disabling ALL persistent prefix reuse (device/memory/remote).
Cold input alone does not make publication of incomplete historical SWA safe.
Cache-store metadata may be present on a PREFILL-role local request. Only the
isolated local-generation caller may allow it; the native writer still checks
each request's PD flag. No active PD transfer is supported by this experiment.
It neither installs attention metadata nor changes KV pools or global[20].

The 3072-row L20-output tail leaves 659 exact L39-output rows for window128,
covering the final hidden and aux/ring tails. Early halo outputs may differ and
must not be exposed as full-prompt outputs. Prefill draft commit is rowwise; it
does not add three attention windows. Every allocation here belongs to this
forward; no module-global cache, persistent scratch, or new JIT kernel is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.cp import CPContext

_CP_SIZE = 4
_TAIL_TOKENS = 3072
_LOCAL_ROWS = _TAIL_TOKENS // _CP_SIZE
_MIN_TOKENS = 32768
_DTYPES = (torch.bfloat16, torch.float32, torch.int32, torch.int64)
_DERIVED_KEYS = (
    "prefill_chunk_meta",
    "prefill_index_plan",
    "prefill_candidate_mask",
    "candidate_mask",
    "prefill_sparse_plans",
    "prefill_score_bounds",
    "prefill_meta_common",
)


@dataclass(frozen=True)
class _TailCPInfo:
    """Internal raw metadata; never replace the engine's PyContextParallelParams."""

    prefill_cp_padding_lengths: torch.Tensor
    prefill_cp_chunk_lengths: torch.Tensor
    prefill_shuffle_indices: torch.Tensor
    prefill_qkv_restore_indice: torch.Tensor
    prefill_qkv_padding_mask: torch.Tensor
    prefill_actual_input_lengths_cpu: torch.Tensor
    prefill_prefix_lengths_cpu: torch.Tensor
    prefill_mm_spans: torch.Tensor


def _tensor(value, shape, device=None, dtypes=None) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(shape)
        and (device is None or value.device == device)
        and (dtypes is None or value.dtype in dtypes)
    )


def _agree(values, device, group) -> bool:
    """Bounded control exchange: all ranks either proceed or reject together."""
    local = torch.tensor([values], dtype=torch.int64, device=device)
    received = torch.empty((_CP_SIZE, len(values)), dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(received, local, group=group)
    rows = received.cpu().tolist()
    return rows[0][0] == 1 and all(row == rows[0] for row in rows[1:])


def _supported_model(v4) -> bool:
    args = getattr(v4, "args", None)
    layers = getattr(v4, "layers", ())
    captures = tuple(getattr(v4, "capture_aux_hidden_layer_ids", ()))
    if (
        args is None
        or getattr(args, "v41_config", None) is None
        or not getattr(v4, "fp8_kv_cache", False)
        or getattr(args, "n_layers", 0) != 40
        or len(layers) != 40
        or getattr(args, "ep_size", 0) != _CP_SIZE
        or getattr(args, "window_size", 0) != 128
        or getattr(args, "dim", 0) != 5120
        or getattr(args, "hc_mult", 0) != 4
        or getattr(args, "n_hash_layers", -1) != 0
        or len(set(captures)) != len(captures)
        or any(layer not in (37, 38, 39) for layer in captures)
        # Ordinary MTP has a different full-hidden handoff contract.
        or (not captures and getattr(v4, "_mtp_hidden_buffer", None) is not None)
    ):
        return False
    for index in range(20, 40):
        layer = layers[index]
        attn = getattr(layer, "attn", None)
        if (
            attn is None
            or getattr(attn, "layer_id", -1) != index
            or getattr(attn, "kv_source_layer_id", None) != 20
            or getattr(attn, "index_source_layer_id", None)
            != 20 + (index - 20) // 4 * 4
            or getattr(attn, "compress_ratio", 0) != 1
            or getattr(attn, "window_size", 0) != 128
            or bool(getattr(attn, "is_kv_source", False)) != (index == 20)
            or bool(getattr(attn, "is_index_source", False)) != (index % 4 == 0)
            or getattr(layer, "engram", None) is not None
        ):
            return False
    return True


def _metadata_shapes(cp, attn, device) -> bool:
    integer = (torch.int32, torch.int64)
    rows, n, padded = cp.chunk_length, cp.seq_len_full, cp.padded_seq_len
    info = cp.cp_info
    return all(
        _tensor(value, shape, expected_device, dtype)
        for value, shape, expected_device, dtype in (
            (cp.global_positions, (rows,), device, integer),
            (cp.relative_positions, (rows,), device, integer),
            (cp.local_is_real, (rows,), device, (torch.bool,)),
            (cp.req_id_per_token, (rows,), device, integer),
            (cp.unpad_restore, (n,), device, integer),
            (cp.prefix_lengths, (1,), device, integer),
            (cp.input_lengths_global, (1,), device, integer),
            (cp.cu_seqlens_global, (2,), device, integer),
            (getattr(attn, "input_lengths", None), (1,), None, integer),
            (getattr(attn, "prefix_lengths", None), (1,), None, integer),
            (getattr(attn, "sequence_lengths", None), (0,), None, integer),
            (
                getattr(info, "prefill_qkv_restore_indice", None),
                (padded,),
                None,
                integer,
            ),
            (getattr(info, "prefill_qkv_padding_mask", None), (padded,), None, integer),
        )
    )


def _original_tail_is_rank_zero(cp, attn, first: int) -> bool:
    """Check the actual inverse map, not just the assumed zigzag arithmetic."""
    n, padded = cp.seq_len_full, cp.padded_seq_len
    start = n - _TAIL_TOKENS
    device = cp.global_positions.device
    expected = torch.arange(first, first + _TAIL_TOKENS, device=device)
    info = cp.cp_info
    checks = [
        (cp.unpad_restore[start:n] == expected).all(),
        (info.prefill_qkv_restore_indice[start:n].to(device) == expected).all(),
        (info.prefill_qkv_padding_mask[start:n].to(device) == 1).all(),
        (info.prefill_qkv_padding_mask[n:padded].to(device) == 0).all(),
        (cp.prefix_lengths == 0).all(),
        (cp.input_lengths_global == n).all(),
        (cp.cu_seqlens_global == torch.tensor([0, n], device=device)).all(),
        (attn.prefix_lengths.to(device) == 0).all(),
        (attn.input_lengths.to(device) == cp.chunk_length).all(),
    ]
    if cp.cp_rank == 0:
        selected = slice(first, first + _TAIL_TOKENS)
        positions = torch.arange(start, n, device=device)
        checks.extend(
            (
                (cp.global_positions[selected] == positions).all(),
                (cp.relative_positions[selected] == positions).all(),
                cp.local_is_real[selected].all(),
                (cp.req_id_per_token[selected] == 0).all(),
                (~cp.local_is_real[first + _TAIL_TOKENS :]).all(),
            )
        )
    return bool(torch.stack(checks).all().item())


def _tail_context(original: CPContext) -> CPContext:
    """Explicitly replace every CP field; no original restore/position cache survives."""
    device = original.global_positions.device
    rank = original.cp_rank
    start = original.seq_len_full - _TAIL_TOKENS
    half = _LOCAL_ROWS // 2
    offset = torch.arange(half, dtype=torch.int64, device=device)
    relative = torch.cat(
        (rank * half + offset, _TAIL_TOKENS - (rank + 1) * half + offset)
    )
    canonical = torch.arange(_TAIL_TOKENS, dtype=torch.int64, device=device)
    pair = canonical // half
    owner = torch.where(pair < _CP_SIZE, pair, 2 * _CP_SIZE - 1 - pair)
    local = canonical % half + torch.where(pair < _CP_SIZE, 0, half)
    restore = (owner * _LOCAL_ROWS + local).contiguous()
    lengths = torch.tensor([_TAIL_TOKENS], dtype=torch.int32, device=device)
    prefixes = torch.tensor([start], dtype=torch.int64, device=device)
    info = _TailCPInfo(
        prefill_cp_padding_lengths=torch.zeros(1, dtype=torch.int32, device=device),
        prefill_cp_chunk_lengths=torch.tensor(
            [_LOCAL_ROWS], dtype=torch.int32, device=device
        ),
        prefill_shuffle_indices=relative.to(torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32),
        prefill_qkv_padding_mask=torch.ones(
            _TAIL_TOKENS, dtype=torch.int32, device=device
        ),
        prefill_actual_input_lengths_cpu=torch.tensor(
            [_TAIL_TOKENS], dtype=torch.int32
        ),
        prefill_prefix_lengths_cpu=torch.tensor([start], dtype=torch.int32),
        prefill_mm_spans=torch.empty((0, 3), dtype=torch.int32, device=device),
    )
    # dataclasses.replace resets CPContext's init=False position cache to None.
    return replace(
        original,
        cp_size=_CP_SIZE,
        cp_rank=rank,
        chunk_length=_LOCAL_ROWS,
        padded_seq_len=_TAIL_TOKENS,
        seq_len_full=_TAIL_TOKENS,
        relative_positions=relative,
        prefix_length=start,
        global_positions=(relative + start).contiguous(),
        local_is_real=torch.ones(_LOCAL_ROWS, dtype=torch.bool, device=device),
        unpad_restore=restore,
        seq_len_total=original.seq_len_full,
        cp_info=info,
        req_id_per_token=torch.zeros(_LOCAL_ROWS, dtype=torch.int32, device=device),
        prefix_lengths=prefixes,
        input_lengths_global=lengths,
        cu_seqlens_global=torch.tensor(
            [0, _TAIL_TOKENS], dtype=torch.int32, device=device
        ),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=(_LOCAL_ROWS,),
        kv_cache_sharded=original.kv_cache_sharded,
        input_lengths_global_host=(_TAIL_TOKENS,),
        prefix_lengths_host=(start,),
    )


@dataclass
class CEDTail:
    """One forward's transport plan. Do not store this object on a model/cache."""

    original_context: CPContext
    context: CPContext
    cu_seqlens: torch.Tensor
    _source_tail_start: int = field(repr=False)
    _group: Any = field(repr=False)
    _source_global_rank: int = field(repr=False)
    _capture_ids: Tuple[int, ...] = field(repr=False)
    _compacted: bool = field(default=False, init=False, repr=False)
    _aux_restored: bool = field(default=False, init=False, repr=False)

    @property
    def original_rows(self) -> int:
        return self.original_context.chunk_length

    @property
    def tail_tokens(self) -> int:
        return _TAIL_TOKENS

    @property
    def swa_start(self) -> int:
        return self.context.prefix_length

    @classmethod
    @torch.inference_mode()
    def create(
        cls,
        v4,
        cp_ctx: Optional[CPContext],
        attn_inputs,
        *,
        prepare_hidden_fn=None,
        cache_store_active: bool = False,
        allow_local_cache_store: bool = False,
    ) -> Optional[CEDTail]:
        """Return None on unsupported input, before any model state is changed.

        Main must call on every TP rank after its rank-consistent enable gate.
        This is eager CUDA CP4 only. Local metadata/model checks are voted on
        before any payload broadcast, so one rejected rank rejects the plan.
        """
        if (
            not isinstance(cp_ctx, CPContext)
            or not isinstance(cp_ctx.global_positions, torch.Tensor)
            or not cp_ctx.global_positions.is_cuda
            or not torch.distributed.is_initialized()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        group = collective_torch._get_group(Group.TP)
        if torch.distributed.get_world_size(group) != _CP_SIZE:
            return None
        rank = torch.distributed.get_rank(group)
        device = cp_ctx.global_positions.device
        n, rows, padded = (
            cp_ctx.seq_len_full,
            cp_ctx.chunk_length,
            cp_ctx.padded_seq_len,
        )
        first = rows - (padded - n) - _TAIL_TOKENS
        captures = tuple(getattr(v4, "capture_aux_hidden_layer_ids", ()))
        eligible = (
            prepare_hidden_fn is None
            and (
                allow_local_cache_store
                or (
                    not cache_store_active
                    and getattr(attn_inputs, "cache_store_inputs", None) is None
                )
            )
            and bool(getattr(attn_inputs, "is_prefill", False))
            and not bool(getattr(attn_inputs, "is_target_verify", False))
            and not bool(getattr(attn_inputs, "is_cuda_graph", False))
            and cp_ctx.cp_size == _CP_SIZE
            and cp_ctx.cp_rank == rank
            and cp_ctx.kv_cache_sharded
            and _MIN_TOKENS <= n <= 1048576
            and padded == ((n + 7) // 8) * 8
            and rows * _CP_SIZE == padded
            and rows % 2 == 0
            and first >= rows // 2
            and cp_ctx.prefix_length == 0
            and cp_ctx.seq_len_total == n
            and cp_ctx.input_lengths_global_host == (n,)
            and cp_ctx.prefix_lengths_host == (0,)
            and cp_ctx.chunk_lengths_per_req == (rows,)
            and _supported_model(v4)
            and _metadata_shapes(cp_ctx, attn_inputs, device)
        )
        # Encode order as well as membership: aux segments follow capture order.
        capture_signature = (
            sum((int(layer) + 1) << (8 * i) for i, layer in enumerate(captures))
            if eligible
            else 0
        )
        if not _agree(
            [int(eligible), n, rows, padded, capture_signature], device, group
        ):
            return None
        valid = _original_tail_is_rank_zero(cp_ctx, attn_inputs, first)
        if not _agree([int(valid)], device, group):
            return None
        return cls(
            original_context=cp_ctx,
            context=_tail_context(cp_ctx),
            cu_seqlens=torch.tensor([0, _LOCAL_ROWS], dtype=torch.int32, device=device),
            _source_tail_start=first,
            _group=group,
            # broadcast's src is a WORLD rank, not a TP-local rank.
            _source_global_rank=torch.distributed.get_global_rank(group, 0),
            _capture_ids=captures,
        )

    def _require_all(self, valid: bool, *signature: int) -> None:
        if not _agree(
            [int(valid), *signature], self.context.global_positions.device, self._group
        ):
            raise ValueError("CED tail state/geometry disagrees across TP ranks")

    def _broadcast_tail(self, value: torch.Tensor) -> torch.Tensor:
        shape = (_TAIL_TOKENS, *value.shape[1:])
        if self.context.cp_rank == 0:
            # Own the bounded slice; never retain the full prompt's storage.
            full_tail = value.narrow(0, self._source_tail_start, _TAIL_TOKENS).clone()
        else:
            full_tail = torch.empty(shape, dtype=value.dtype, device=value.device)
        collective_torch.broadcast(full_tail, self._source_global_rank, Group.TP)
        return full_tail.index_select(0, self.context.relative_positions).contiguous()

    @torch.inference_mode()
    def compact(self, v4, hidden: torch.Tensor, input_ids: torch.Tensor, shared):
        """After L20: transport HC, delayed mix, IDs and L20 query selections.

        Mutates only the existing L20 pre_mix_out and shared query state. Main
        must install the new context/meta and its SWA-read override before L21.
        No old hidden or pre-mix reference is saved on this plan.
        """
        device = self.context.global_positions.device
        pre_mix = v4.layers[20].ffn_hc.pre_mix_out
        topk = shared.get("topk", {}).get(20)
        candidates = shared.get("candidates")
        sources = shared.get("global", {}).get(20)
        source_ready = (
            isinstance(sources, (list, tuple))
            and len(sources) == 1
            and isinstance(sources[0], (list, tuple))
            and len(sources[0]) == 2
            and _tensor(
                sources[0][0], (self.original_context.seq_len_full, 512), device
            )
            and hasattr(sources[0][1], "__len__")
            and len(sources[0][1]) == self.original_context.seq_len_full
        )
        valid = (
            not self._compacted
            and source_ready
            and shared is v4.layers[20].attn._shared_attention
            and _tensor(
                hidden, (self.original_rows, 4, 5120), device, (torch.bfloat16,)
            )
            and _tensor(pre_mix, (self.original_rows, 4), device, (torch.float32,))
            and _tensor(
                input_ids, (self.original_rows,), device, (torch.int32, torch.int64)
            )
            and _tensor(topk, (self.original_rows, 512), device, (torch.int32,))
            and _tensor(candidates, (self.original_rows, 2048), device, (torch.int32,))
        )
        ids_dtype = _DTYPES.index(input_ids.dtype) if input_ids.dtype in _DTYPES else -1
        self._require_all(valid, ids_dtype)
        compact_hidden = self._broadcast_tail(hidden)
        compact_ids = self._broadcast_tail(input_ids)
        compact_mix = self._broadcast_tail(pre_mix)
        compact_topk = self._broadcast_tail(topk)
        compact_candidates = self._broadcast_tail(candidates)
        v4.layers[20].ffn_hc.pre_mix_out = compact_mix
        shared["topk"] = {20: compact_topk}
        shared["candidates"] = compact_candidates
        # The small BF16 indexer head-weight GEMM changes rounding when cuBLAS
        # changes its M-dependent algorithm. Preserve its original row geometry;
        # attention, Q projection and MoE still execute only the compact tail.
        shared["ced_indexer_layout"] = (
            self.original_rows,
            self.context.relative_positions + self._source_tail_start,
        )
        for key in _DERIVED_KEYS:
            shared.pop(key, None)
        self._compacted = True
        return compact_hidden, compact_ids

    def _gather_rows(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Restore only the bounded canonical tail, on its original owner."""
        device = self.context.global_positions.device
        valid = (
            self._compacted
            and tensor.ndim == 2
            and tensor.shape[0] == _LOCAL_ROWS
            and tensor.shape[1] > 0
            and tensor.device == device
            and tensor.dtype in _DTYPES
        )
        width = tensor.shape[1] if tensor.ndim == 2 else -1
        dtype_id = _DTYPES.index(tensor.dtype) if tensor.dtype in _DTYPES else -1
        self._require_all(valid, width, dtype_id)
        local = tensor.contiguous()
        gathered = torch.empty((_TAIL_TOKENS, width), dtype=tensor.dtype, device=device)
        # Use an ordinary ephemeral allocation, not the persistent symmetric
        # memory fast path in collective_torch.all_gather.
        torch.distributed.all_gather_into_tensor(gathered, local, group=self._group)
        if self.context.cp_rank == 0:
            return gathered.index_select(0, self.context.unpad_restore)
        return None

    @torch.inference_mode()
    def restore_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collect a compact [768, width] matrix; zero-fill the original layout.

        Only original rank0 owns any of the selected 3072 tokens. Transport is
        bounded by the compact rows, regardless of the original prompt length.
        Use this AFTER the compact final HC reduction/norm, not before them.
        The returned full-row output is the caller's required output, not scratch.
        """
        canonical = self._gather_rows(tensor)
        output = tensor.new_zeros((self.original_rows, tensor.shape[1]))
        if canonical is not None:
            output.narrow(0, self._source_tail_start, _TAIL_TOKENS).copy_(canonical)
        return output

    @torch.inference_mode()
    def restore_aux(self, v4) -> None:
        """Restore aux in the EXISTING fixed-address buffer and fix its row count.

        Call once after all selected target layers have captured compact aux,
        before C++ asks get_mtp_target_hidden_states(-1). Source/destination may
        alias: gather the bounded tail before zeroing the original buffer. No
        full-prompt aux scratch or allocation retained across forwards is needed.
        """
        if not self._capture_ids:
            return
        buffer = v4._mtp_hidden_buffer
        width = 5120 * len(self._capture_ids)
        valid = (
            self._compacted
            and not self._aux_restored
            and tuple(v4.capture_aux_hidden_layer_ids) == self._capture_ids
            and isinstance(buffer, torch.Tensor)
            and buffer.ndim == 2
            and buffer.shape[0] >= self.original_rows
            and buffer.shape[1] == width
            and buffer.device == self.context.global_positions.device
            and buffer.dtype == torch.bfloat16
        )
        self._require_all(valid)
        canonical = self._gather_rows(buffer[:_LOCAL_ROWS])
        buffer[: self.original_rows].zero_()
        if canonical is not None:
            buffer.narrow(0, self._source_tail_start, _TAIL_TOKENS).copy_(canonical)
        v4._note_aux_hidden_rows(self.original_rows, is_cuda_graph=False)
        self._aux_restored = True


__all__ = ["CEDTail"]
