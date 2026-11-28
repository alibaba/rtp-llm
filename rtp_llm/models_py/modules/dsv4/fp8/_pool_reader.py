"""Per-iteration compressed-K pool reader for DSV4 prefill (Stage 5b-3).

Two implementations behind a single ``fill(out=workspace, ...)`` interface so
attention/indexer call sites stay agnostic of CP storage layout:

* ``LocalPoolReader`` — single-rank / non-sharded. Forwards directly to
  ``dequantize_and_gather_k_cache``.
* ``CPShardedPoolReader`` — Stage 5b. Each CP rank physically owns
  ``1/cp_size`` of the request's blocks (RR by ``logical_block_idx %
  cp_size``). The reader follows the branch
  ``feat/support_distributed_cp_kvcache_rebase`` per-iteration pattern
  (mirrors ``flashmla_sparse_cp_impl.py``):

    1. Each rank gathers its OWNED prefix slice from the local pool into a
       packed FP8 ``[total_local_kv, 584]`` uint8 buffer (one Triton call per
       fill, NOT per-request).
    2. ONE NCCL ``all_gather`` produces
       ``[cp_size * total_local_kv, 584]``.
    3. Pre-built ``restore_indices`` (from
       ``cp.build_kv_allgather_restore_indices``) reorder the gathered
       buffer into request-concatenated logical token order.
    4. The restored packed slots are dequantized locally to BF16 and scattered
       into ``out[r, offset:offset+seq_lens[r], :]`` per request (the
       workspace contract used by ``_attn_via_workspace`` / indexer).

Compared to a per-request gather loop this collapses ``B`` collectives into
one and avoids ``B`` Triton kernel launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_actual_owned_kv_len,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_len,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    cp_direct_flat_pack_enabled,
    dequantize_and_gather_k_cache,
    dequantize_packed_k_cache_flat,
    gather_k_cache_packed,
    try_gather_k_cache_packed_to_flat,
    try_restore_dequantize_scatter_packed_k_cache_flat,
)


if _TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize=[
            "total",
            "batch_size",
            "D",
            "padded_stride0",
            "padded_stride1",
            "out_stride0",
        ]
    )
    def _pack_padded_to_flat_kernel(
        padded_ptr,
        lens_ptr,
        out_ptr,
        total,
        batch_size,
        D,
        padded_stride0,
        padded_stride1,
        out_stride0,
        TILE_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BATCH_BLOCK: tl.constexpr,
    ):
        t_offs = tl.program_id(0).to(tl.int64) * TILE_M + tl.arange(0, TILE_M)
        t_valid = t_offs < total

        batch_offsets = tl.arange(0, BATCH_BLOCK)
        batch_mask = batch_offsets < batch_size
        lens = tl.load(lens_ptr + batch_offsets, mask=batch_mask, other=0).to(tl.int64)
        ends = tl.cumsum(lens, axis=0)
        starts = ends - lens

        batch_idx = tl.sum((t_offs[:, None] >= ends[None, :]).to(tl.int32), axis=1)
        eq_mask = batch_idx[:, None] == batch_offsets[None, :]
        req_start = tl.sum(tl.where(eq_mask, starts[None, :], 0), axis=1)
        s_idx = t_offs - req_start

        d_cols = tl.arange(0, BLOCK_D)
        tile_mask = t_valid[:, None] & (d_cols[None, :] < D)
        src_ptrs = (
            padded_ptr
            + batch_idx.to(tl.int64)[:, None] * padded_stride0.to(tl.int64)
            + s_idx[:, None] * padded_stride1.to(tl.int64)
            + d_cols[None, :]
        )
        vals = tl.load(src_ptrs, mask=tile_mask, other=0)
        tl.store(
            out_ptr + t_offs[:, None] * out_stride0.to(tl.int64) + d_cols[None, :],
            vals,
            mask=tile_mask,
        )

    @triton.jit(
        do_not_specialize=[
            "total",
            "batch_size",
            "D",
            "offset",
            "src_stride0",
            "out_stride0",
            "out_stride1",
        ]
    )
    def _scatter_flat_to_workspace_kernel(
        restored_ptr,
        out_ptr,
        seq_lens_ptr,
        cu_dst_offsets_ptr,
        total,
        batch_size,
        D,
        offset,
        src_stride0,
        out_stride0,
        out_stride1,
        TILE_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BATCH_BLOCK: tl.constexpr,
        HAS_CU_DST_OFFSETS: tl.constexpr,
    ):
        t_offs = tl.program_id(0).to(tl.int64) * TILE_M + tl.arange(0, TILE_M)
        t_valid = t_offs < total

        batch_offsets = tl.arange(0, BATCH_BLOCK)
        batch_mask = batch_offsets < batch_size
        seq_lens = tl.load(seq_lens_ptr + batch_offsets, mask=batch_mask, other=0).to(
            tl.int64
        )
        ends = tl.cumsum(seq_lens, axis=0)
        starts = ends - seq_lens

        batch_idx = tl.sum((t_offs[:, None] >= ends[None, :]).to(tl.int32), axis=1)
        valid_batch = t_valid & (batch_idx >= 0) & (batch_idx < batch_size)
        safe_batch = tl.where(valid_batch, batch_idx, 0).to(tl.int64)
        eq_mask = batch_idx[:, None] == batch_offsets[None, :]
        seq_start = tl.sum(tl.where(eq_mask, starts[None, :], 0), axis=1)
        in_req = t_offs - seq_start

        if HAS_CU_DST_OFFSETS:
            dst_base = tl.load(
                cu_dst_offsets_ptr + safe_batch, mask=valid_batch, other=0
            ).to(tl.int64)
            out_row_off = (
                dst_base + offset.to(tl.int64) + in_req
            ) * out_stride0.to(tl.int64)
        else:
            out_row_off = (
                safe_batch * out_stride0.to(tl.int64)
                + (offset.to(tl.int64) + in_req) * out_stride1.to(tl.int64)
            )

        d_cols = tl.arange(0, BLOCK_D)
        tile_mask = valid_batch[:, None] & (d_cols[None, :] < D)
        vals = tl.load(
            restored_ptr + t_offs[:, None] * src_stride0.to(tl.int64) + d_cols[None, :],
            mask=tile_mask,
            other=0,
        )
        tl.store(out_ptr + out_row_off[:, None] + d_cols[None, :], vals, mask=tile_mask)


class CompressedKPoolReader:
    """Materialize per-request compressed K from the paged pool into the
    workspace slice ``out[r, offset:offset+seq_lens[r], :]``.

    Implementations match the contract of ``dequantize_and_gather_k_cache``
    so attention / indexer call sites are agnostic to the underlying
    storage layout (full local pool vs CP-sharded across ranks).
    """

    def fill(
        self,
        *,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: Optional[torch.Tensor],
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
        cu_dst_offsets: Optional[torch.Tensor] = None,
        max_tokens_hint: int = 0,
    ) -> None:
        raise NotImplementedError


class LocalPoolReader(CompressedKPoolReader):
    def fill(
        self,
        *,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: Optional[torch.Tensor],
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
        cu_dst_offsets: Optional[torch.Tensor] = None,
        max_tokens_hint: int = 0,
    ) -> None:
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
            cu_dst_offsets=cu_dst_offsets,
            max_tokens_hint=max_tokens_hint,
        )


@dataclass
class CPShardConfig:
    """Per-prefill-iteration CP shard metadata.

    Fields:
      * ``cp_ctx`` — CP context for the all_gather collective.
      * ``per_req_total_kv_lens`` — ``[B]`` int64 — per-request KV length to
        gather (== prefix that lives RR-sharded across ranks).
      * ``restore_indices`` — ``[total_logical_kv]`` int64. Pre-built by
        ``cp.build_kv_allgather_restore_indices(per_req_total_kv_lens,
        cp_size, block_size, device)``.
      * ``block_size`` — pool block entries used by ``gather_k_cache_packed``.
      * ``owner_block_size`` — logical owner block entries used by CP restore.
      * ``total_local_kv`` — sum of per-rank local lengths (==
        ``ceil(T_r/(owner_block_size*cp_size)) * owner_block_size`` per req,
        summed).
    """

    cp_ctx: CPContext
    per_req_total_kv_lens: torch.Tensor
    restore_indices: torch.Tensor
    block_size: int
    total_local_kv: int
    owner_block_size: int = 0
    seq_lens_total: Optional[int] = None
    local_seq_lens_padded: Optional[torch.Tensor] = None
    local_seq_lens_actual: Optional[torch.Tensor] = None
    max_local_seq_len_padded: int = 0
    has_local_seq_len_actual: Optional[bool] = None

    def __post_init__(self) -> None:
        owner_block_size = int(self.owner_block_size or self.block_size)
        self.owner_block_size = owner_block_size
        if self.seq_lens_total is None:
            # Legacy/test callers may not carry the independently computed host
            # total. The restore shape is still a safe host-side fallback.
            self.seq_lens_total = int(self.restore_indices.numel())
        else:
            self.seq_lens_total = int(self.seq_lens_total)
        if self.seq_lens_total < 0:
            raise ValueError(
                f"seq_lens_total must be non-negative, got {self.seq_lens_total}"
            )
        if int(self.restore_indices.numel()) != self.seq_lens_total:
            raise ValueError(
                f"restore_indices rows ({self.restore_indices.numel()}) != host "
                f"seq_lens total ({self.seq_lens_total})"
            )

        device = self.per_req_total_kv_lens.device
        if self.local_seq_lens_padded is None:
            self.local_seq_lens_padded = cp_padded_local_kv_lens(
                self.per_req_total_kv_lens,
                self.cp_ctx.cp_size,
                owner_block_size,
            ).to(device=device, dtype=torch.int32)
        else:
            self.local_seq_lens_padded = self.local_seq_lens_padded.to(
                device=device, dtype=torch.int32
            )

        if self.local_seq_lens_actual is None:
            self.local_seq_lens_actual = cp_actual_owned_kv_lens(
                self.per_req_total_kv_lens,
                self.cp_ctx.cp_size,
                owner_block_size,
                self.cp_ctx.cp_rank,
            ).to(device=device, dtype=torch.int32)
        else:
            self.local_seq_lens_actual = self.local_seq_lens_actual.to(
                device=device, dtype=torch.int32
            )

        if (
            self.max_local_seq_len_padded == 0
            and self.has_local_seq_len_actual is None
            and self.local_seq_lens_padded.numel() > 1
            and self.local_seq_lens_padded.is_cuda
            and self.local_seq_lens_actual.numel() > 0
            and self.local_seq_lens_actual.is_cuda
        ):
            max_pad_v, max_act_v = (
                int(v)
                for v in torch.stack(
                    [
                        self.local_seq_lens_padded.max().to(torch.int64),
                        self.local_seq_lens_actual.max().to(torch.int64),
                    ]
                ).tolist()
            )
            self.max_local_seq_len_padded = max_pad_v
            self.has_local_seq_len_actual = max_act_v > 0
        else:
            if self.max_local_seq_len_padded == 0 and self.local_seq_lens_padded.numel():
                if self.local_seq_lens_padded.numel() == 1:
                    self.max_local_seq_len_padded = int(self.total_local_kv)
                else:
                    self.max_local_seq_len_padded = int(
                        self.local_seq_lens_padded.max().item()
                    )
            if (
                self.has_local_seq_len_actual is None
                and self.local_seq_lens_actual.numel()
            ):
                self.has_local_seq_len_actual = (
                    int(self.local_seq_lens_actual.max().item()) > 0
                )


@dataclass
class CPShardedPoolReadHandle:
    local_flat: torch.Tensor
    gathered: torch.Tensor
    work: Any
    completion_event: torch.cuda.Event
    stream: torch.cuda.Stream
    producer_stream: torch.cuda.Stream
    out: torch.Tensor
    seq_lens: torch.Tensor
    offset: int
    cu_dst_offsets: Optional[torch.Tensor] = None
    done_event: Optional[torch.cuda.Event] = None
    work_waited: bool = False


class CPShardedPoolReader(CompressedKPoolReader):
    """Per-iteration: gather owned packed FP8 → all_gather → restore → dequant."""

    def __init__(self, shard_config: CPShardConfig) -> None:
        self.cfg = shard_config

    def fill(
        self,
        *,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: Optional[torch.Tensor],
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
        cu_dst_offsets: Optional[torch.Tensor] = None,
        max_tokens_hint: int = 0,
    ) -> None:
        del max_tokens_hint
        self._validate_call(block_size=block_size, gather_lens=gather_lens)
        device = out.device

        local_flat = self._build_local_flat(
            k_cache=k_cache,
            block_table=block_table,
            block_size=block_size,
            device=device,
        )

        # Step 3: all_gather across cp ranks → [cp_size * total_local_kv, 584].
        # Use raw rank-major all_gather; cp_all_gather_full_async asserts
        # T_local == cp_ctx.chunk_length (prefill-token space), but
        # local_flat lives in KV-pool-entry space (block-aligned).
        # INVARIANT: the compressor writer (cp_wait_gather_full + _launch)
        # always lands its pool writes on the current stream, so NCCL
        # stream-ordering alone suffices for visibility. No CPU-blocking
        # synchronize() here — at ~60 layers × per-fill it would serialize
        # the prefill pipeline. If the writer ever moves to a side stream,
        # add ``current_stream.wait_event(writer_event)`` instead.
        with record_function_range("dsv4.cp.all_gather.pool_reader.gather_cmp.launch"):
            gathered = all_gather(local_flat, group=Group.TP)
        # gathered shape is rank-major: [cp_size * total_local_kv, 584].

        # Step 4: restore to request-concatenated logical order.
        self._restore_dequant_scatter(
            gathered, out, seq_lens, offset, cu_dst_offsets=cu_dst_offsets
        )

    def start_fill_async(
        self,
        *,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: Optional[torch.Tensor],
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
        stream: torch.cuda.Stream,
        cu_dst_offsets: Optional[torch.Tensor] = None,
        max_tokens_hint: int = 0,
    ) -> Optional[CPShardedPoolReadHandle]:
        """Launch only the CP all-gather stage on ``stream``.

        The local pack stays on the caller's current stream.  The communication
        stream waits for it, and ``prepare_fill_async`` later queues restore /
        dequant / scatter on the shared post stream.
        """
        del max_tokens_hint
        self._validate_call(block_size=block_size, gather_lens=gather_lens)
        if (
            stream is None
            or not out.is_cuda
            or not torch.distributed.is_initialized()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None

        cfg = self.cfg
        device = out.device
        local_flat = self._build_local_flat(
            k_cache=k_cache,
            block_table=block_table,
            block_size=block_size,
            device=device,
        )

        from rtp_llm.models_py.distributed import collective_torch

        process_group = collective_torch._get_group(Group.TP)
        world_size = torch.distributed.get_world_size(process_group)
        if world_size != cfg.cp_ctx.cp_size:
            raise RuntimeError(
                f"CP pool reader world_size({world_size}) != cp_size({cfg.cp_ctx.cp_size})"
            )

        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            gathered = torch.empty(
                (world_size * int(local_flat.shape[0]), ENTRY_BYTES),
                dtype=local_flat.dtype,
                device=device,
            )
            with record_function_range(
                "dsv4.cp.all_gather.pool_reader.gather_cmp.launch"
            ):
                work = torch.distributed.all_gather_into_tensor(
                    gathered,
                    local_flat,
                    group=process_group,
                    async_op=True,
                )
            try:
                completion_event = torch.cuda.Event()
                completion_event.record(stream)
            except Exception:
                # Drain the in-flight NCCL Work before propagating; the
                # caller never sees the pending handle, so nothing else
                # would wait it.
                work.wait()
                current_stream.wait_stream(stream)
                raise

        return CPShardedPoolReadHandle(
            local_flat=local_flat,
            gathered=gathered,
            work=work,
            completion_event=completion_event,
            stream=stream,
            producer_stream=current_stream,
            out=out,
            seq_lens=seq_lens,
            offset=offset,
            cu_dst_offsets=cu_dst_offsets,
        )

    def prepare_fill_async(
        self,
        handle: CPShardedPoolReadHandle,
        *,
        stream: torch.cuda.Stream,
        cu_dst_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        if handle.done_event is not None:
            return
        if stream is None:
            raise ValueError("prepare_fill_async requires an explicit stream")
        cu_dst_offsets_eff = (
            cu_dst_offsets if cu_dst_offsets is not None else handle.cu_dst_offsets
        )
        current_stream = torch.cuda.current_stream(handle.out.device)
        # ``out`` is the caller's workspace.  Queue postprocess after any
        # current-stream writes to that workspace; otherwise disjoint-looking
        # range updates can still race through a future caller change.
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            stream.wait_event(handle.completion_event)
            # NCCL Work.wait() is stream-affine: call it while the consumer
            # stream is current so restore/scatter below is fenced correctly.
            with record_function_range(
                "dsv4.cp.all_gather.pool_reader.gather_cmp.wait_work"
            ):
                self._wait_fill_work_once(handle)
            try:
                self._restore_dequant_scatter(
                    handle.gathered,
                    handle.out,
                    handle.seq_lens,
                    handle.offset,
                    cu_dst_offsets=cu_dst_offsets_eff,
                )
            finally:
                try:
                    handle.done_event = torch.cuda.Event()
                    handle.done_event.record(stream)
                except Exception:
                    stream.synchronize()
                    raise

    def wait_fill_async(self, handle: CPShardedPoolReadHandle) -> None:
        done_event = handle.done_event
        current_stream = torch.cuda.current_stream(handle.out.device)
        terminal_event = done_event or handle.completion_event
        current_stream.wait_event(terminal_event)
        if handle.producer_stream != current_stream:
            handle.producer_stream.wait_event(terminal_event)
        if done_event is not None:
            handle.stream.wait_event(done_event)
        self._wait_fill_work_once(handle)

    def discard_fill_async(self, handle: CPShardedPoolReadHandle) -> None:
        self.wait_fill_async(handle)

    @staticmethod
    def _wait_fill_work_once(handle: CPShardedPoolReadHandle) -> None:
        if not handle.work_waited:
            handle.work.wait()
            handle.work_waited = True

    def _validate_call(
        self,
        *,
        block_size: int,
        gather_lens: Optional[torch.Tensor],
    ) -> None:
        if self.cfg.block_size != block_size:
            raise ValueError(
                f"block_size mismatch: cfg={self.cfg.block_size} call={block_size}"
            )
        if gather_lens is not None:
            raise NotImplementedError(
                "CPShardedPoolReader does not support gather_lens yet; "
                "pass full per-request seq_lens or add suffix-aware restore/scatter."
            )

    def _build_local_flat(
        self,
        *,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        cfg = self.cfg
        # Step 1: gather owned slice into packed [total_local_kv, 584] uint8.
        # Two distinct per-rank lengths matter here:
        #   * padded — what each rank's local buffer is sized for; uniform
        #     across ranks so all_gather has matching shapes.
        #   * actual — how many KV entries this rank's pool ACTUALLY holds
        #     valid data for. The compressor's CP-sharded write path only
        #     populates owned logical blocks (logical_idx % cp_size ==
        #     cp_rank), AND the last owned block may be partial. Reading
        #     past `actual` lands on uninitialized FP8 bytes that decode to
        #     NaN and propagate through all_gather → restore → workspace.
        local_seq_lens_padded = cfg.local_seq_lens_padded
        local_seq_lens_actual = cfg.local_seq_lens_actual
        assert local_seq_lens_padded is not None
        assert local_seq_lens_actual is not None
        if local_seq_lens_padded.device != device:
            local_seq_lens_padded = local_seq_lens_padded.to(device=device)
        if local_seq_lens_actual.device != device:
            local_seq_lens_actual = local_seq_lens_actual.to(device=device)
        if cp_direct_flat_pack_enabled():
            local_flat = torch.empty(
                (cfg.total_local_kv, ENTRY_BYTES),
                dtype=torch.uint8,
                device=device,
            )
            if try_gather_k_cache_packed_to_flat(
                local_flat,
                k_cache,
                block_table,
                local_seq_lens_padded,
                local_seq_lens_actual,
                block_size=block_size,
                has_actual_tokens=bool(cfg.has_local_seq_len_actual),
            ):
                return local_flat

        # Use the rank's existing block_table directly — Stage 5a guarantees
        # block_table[r, :] holds this rank's local entries in compact form.
        # zeros() (not empty()) so the [actual, padded) tail of each request
        # is a deterministic 0 and contributes nothing to the gather.
        local_packed = torch.zeros(
            (
                int(local_seq_lens_padded.shape[0]),
                cfg.max_local_seq_len_padded,
                ENTRY_BYTES,
            ),
            dtype=torch.uint8,
            device=device,
        )
        if local_packed.shape[1] > 0 and bool(cfg.has_local_seq_len_actual):
            gather_k_cache_packed(
                out=local_packed,
                k_cache=k_cache,
                seq_lens=local_seq_lens_actual,
                gather_lens=None,
                block_table=block_table,
                block_size=block_size,
                offset=0,
            )
        # Step 2: pack to flat [total_local_kv, 584] (drop per-row padding).
        return _pack_padded_to_flat(
            local_packed,
            local_seq_lens_padded,
            cfg.total_local_kv,
            ENTRY_BYTES,
        )

    def _restore_dequant_scatter(
        self,
        gathered: torch.Tensor,
        out: torch.Tensor,
        seq_lens: torch.Tensor,
        offset: int,
        *,
        cu_dst_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        cfg = self.cfg
        D = out.shape[-1]
        if int(seq_lens.numel()) != int(cfg.per_req_total_kv_lens.numel()):
            raise ValueError(
                f"restore seq_lens batch ({seq_lens.numel()}) != configured batch "
                f"({cfg.per_req_total_kv_lens.numel()})"
            )
        with record_function_range("dsv4.cp.all_gather.pool_reader.gather_cmp.restore"):
            fused_kwargs = {"seq_lens_total": int(cfg.seq_lens_total)}
            if cu_dst_offsets is not None:
                fused_kwargs["cu_dst_offsets"] = cu_dst_offsets
            if try_restore_dequantize_scatter_packed_k_cache_flat(
                out,
                gathered,
                cfg.restore_indices,
                seq_lens,
                offset,
                **fused_kwargs,
            ):
                return

            restored_packed = gathered[cfg.restore_indices].contiguous()

            # Step 5: dequant locally after communication. This keeps NCCL payload
            # at 584B/token instead of 1024B/token BF16 [512].
            restored = torch.empty(
                (int(restored_packed.shape[0]), D),
                dtype=out.dtype,
                device=out.device,
            )
            dequantize_packed_k_cache_flat(restored, restored_packed)

            # Step 6: scatter restored into workspace rows.
            if cu_dst_offsets is not None:
                _scatter_flat_to_workspace(
                    restored, out, seq_lens, offset, cu_dst_offsets=cu_dst_offsets
                )
            else:
                _scatter_flat_to_workspace(restored, out, seq_lens, offset)


def _pack_padded_to_flat(
    padded: torch.Tensor, lens: torch.Tensor, total: int, D: int
) -> torch.Tensor:
    """Drop per-row padding → flat [total, D]."""
    if total == 0:
        return torch.empty((0, D), dtype=padded.dtype, device=padded.device)
    device = padded.device
    B = int(lens.shape[0])
    if B == 1:
        return padded[0, :total, :].contiguous()
    if (
        _TRITON_AVAILABLE
        and padded.is_cuda
        and padded.dim() == 3
        and int(padded.stride(2)) == 1
        and B <= 256
        and 0 < D <= 2048
    ):
        lens_d = lens if (lens.device == device and lens.is_contiguous()) else lens.to(device=device).contiguous()
        out = torch.empty((total, D), dtype=padded.dtype, device=device)
        tile_m = 8
        _pack_padded_to_flat_kernel[(triton.cdiv(total, tile_m),)](
            padded,
            lens_d,
            out,
            total=total,
            batch_size=B,
            D=D,
            padded_stride0=int(padded.stride(0)),
            padded_stride1=int(padded.stride(1)),
            out_stride0=int(out.stride(0)),
            TILE_M=tile_m,
            BLOCK_D=triton.next_power_of_2(D),
            BATCH_BLOCK=triton.next_power_of_2(B),
            num_warps=4,
        )
        return out

    lens_l = lens.to(device=device, dtype=torch.int64)
    b_idx = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        lens_l,
        output_size=total,
    )
    cu_starts = torch.zeros(B, device=device, dtype=torch.int64)
    if B > 1:
        cu_starts[1:] = torch.cumsum(lens_l[:-1], dim=0)
    s_idx = torch.arange(
        total, device=device, dtype=torch.int64
    ) - cu_starts.index_select(0, b_idx)
    return padded[b_idx, s_idx]


def _scatter_flat_to_workspace(
    restored: torch.Tensor,
    out: torch.Tensor,
    seq_lens: torch.Tensor,
    offset: int,
    cu_dst_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Copy restored[r-th-slice] → workspace rows."""
    total = int(restored.shape[0])
    if total == 0:
        return
    device = restored.device
    B = int(seq_lens.shape[0])
    if cu_dst_offsets is None and B == 1:
        out[0, offset : offset + total, :].copy_(restored)
        return

    D = int(restored.shape[1]) if restored.dim() == 2 else 0
    has_cu = cu_dst_offsets is not None
    if has_cu:
        assert out.dim() == 2 and int(cu_dst_offsets.numel()) >= B
    if (
        _TRITON_AVAILABLE
        and restored.is_cuda
        and out.is_cuda
        and out.device == device
        and restored.dtype == out.dtype
        and restored.dim() == 2
        and int(restored.stride(1)) == 1
        and B > 0
        and B <= 256
        and 0 < D <= 2048
        and ((has_cu and int(out.stride(1)) == 1) or (not has_cu and out.dim() == 3 and int(out.stride(2)) == 1))
    ):
        seq_lens_d = (
            seq_lens
            if (seq_lens.device == device and seq_lens.is_contiguous())
            else seq_lens.to(device=device).contiguous()
        )
        cu_dst_d = (
            (
                cu_dst_offsets[:B]
                if (cu_dst_offsets.device == device and cu_dst_offsets.is_contiguous())
                else cu_dst_offsets[:B].to(device=device).contiguous()
            )
            if has_cu
            else seq_lens_d
        )
        tile_m = 8
        _scatter_flat_to_workspace_kernel[(triton.cdiv(total, tile_m),)](
            restored,
            out,
            seq_lens_d,
            cu_dst_d,
            total=total,
            batch_size=B,
            D=D,
            offset=int(offset),
            src_stride0=int(restored.stride(0)),
            out_stride0=int(out.stride(0)),
            out_stride1=int(out.stride(1)) if not has_cu else 0,
            TILE_M=tile_m,
            BLOCK_D=triton.next_power_of_2(D),
            BATCH_BLOCK=triton.next_power_of_2(B),
            HAS_CU_DST_OFFSETS=has_cu,
            num_warps=4,
        )
        return

    seq_lens_l = seq_lens.to(device=device, dtype=torch.int64)
    if not has_cu:
        b_idx = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int64),
            seq_lens_l,
            output_size=total,
        )
        cu_starts = torch.zeros(B, device=device, dtype=torch.int64)
        if B > 1:
            cu_starts[1:] = torch.cumsum(seq_lens_l[:-1], dim=0)
        col_idx = (
            torch.arange(total, device=device, dtype=torch.int64)
            - cu_starts.index_select(0, b_idx)
            + offset
        )
        out.index_put_((b_idx, col_idx), restored, accumulate=False)
        return

    b_idx = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        seq_lens_l,
        output_size=total,
    )
    cu_starts = torch.zeros(B, device=device, dtype=torch.int64)
    if B > 1:
        cu_starts[1:] = torch.cumsum(seq_lens_l[:-1], dim=0)
    dst_bases = (
        cu_dst_offsets[:B].to(device=device, dtype=torch.int64).index_select(0, b_idx)
    )
    row_idx = (
        dst_bases
        + offset
        + torch.arange(total, device=device, dtype=torch.int64)
        - cu_starts.index_select(0, b_idx)
    )
    out.index_copy_(0, row_idx, restored)


def make_compressed_k_pool_reader(
    cp_ctx: Optional[CPContext],
    kv_cache_sharded: bool,
    per_req_total_kv_lens: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    owner_block_size: Optional[int] = None,
    total_kv_len: Optional[int] = None,
) -> CompressedKPoolReader:
    """Pick the right reader for this prefill iteration.

    Returns ``LocalPoolReader`` unless CP sharding is on with cp_size>1 and
    at least one request has non-zero KV to gather. In the sharded case,
    builds ``CPShardConfig`` with ``restore_indices`` pre-computed via
    ``cp.build_kv_allgather_restore_indices``.
    """
    if not kv_cache_sharded:
        return LocalPoolReader()
    if cp_ctx is None or cp_ctx.cp_size <= 1:
        return LocalPoolReader()
    if per_req_total_kv_lens is None or block_size is None:
        raise ValueError(
            "kv_cache_sharded + cp_size>1 requires per_req_total_kv_lens and "
            "block_size; got None. Pass torch.zeros(B, dtype=int64) only when "
            "this iteration has no compressed-K rows to restore."
        )
    if total_kv_len is not None and int(total_kv_len) <= 0:
        return LocalPoolReader()

    owner_bs = int(owner_block_size or block_size)
    if owner_bs <= 0:
        raise ValueError(f"owner_block_size must be positive, got {owner_bs}")

    from rtp_llm.models_py.modules.dsv4.cp import build_kv_allgather_restore_indices

    device = per_req_total_kv_lens.device
    batch_size = int(per_req_total_kv_lens.numel())
    if batch_size == 0:
        return LocalPoolReader()

    if batch_size == 1 and total_kv_len is not None:
        total_kv_host = int(total_kv_len)
        total_local = cp_padded_local_kv_len(
            total_kv_host, cp_ctx.cp_size, owner_bs
        )
        max_local = total_local
        actual_local = cp_actual_owned_kv_len(
            total_kv_host,
            cp_ctx.cp_size,
            owner_bs,
            cp_ctx.cp_rank,
        )
        has_actual = actual_local > 0
        local_lens = torch.full(
            (1,), total_local, dtype=torch.int32, device=device
        )
        local_actual_lens = torch.full(
            (1,), actual_local, dtype=torch.int32, device=device
        )
    elif device.type == "cpu":
        per_req_cpu = per_req_total_kv_lens.to(dtype=torch.int64)
        total_kv_host = (
            int(total_kv_len)
            if total_kv_len is not None
            else int(per_req_cpu.sum().item())
        )
        if total_kv_host <= 0:
            return LocalPoolReader()
        local_lens = cp_padded_local_kv_lens(
            per_req_cpu, cp_ctx.cp_size, owner_bs
        ).to(dtype=torch.int32)
        local_actual_lens = cp_actual_owned_kv_lens(
            per_req_cpu, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
        ).to(dtype=torch.int32)
        total_local = int(local_lens.sum().item())
        max_local = int(local_lens.max().item())
        has_actual = int(local_actual_lens.max().item()) > 0
    else:
        local_lens = cp_padded_local_kv_lens(
            per_req_total_kv_lens, cp_ctx.cp_size, owner_bs
        ).to(device=device, dtype=torch.int32)
        local_actual_lens = cp_actual_owned_kv_lens(
            per_req_total_kv_lens, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
        ).to(device=device, dtype=torch.int32)
        if total_kv_len is None:
            total_kv_host, total_local, max_local, max_actual = (
                int(v)
                for v in torch.stack(
                    [
                        per_req_total_kv_lens.sum().to(torch.int64),
                        local_lens.sum().to(torch.int64),
                        local_lens.max().to(torch.int64),
                        local_actual_lens.max().to(torch.int64),
                    ]
                ).tolist()
            )
            if total_kv_host <= 0:
                return LocalPoolReader()
        else:
            total_kv_host = int(total_kv_len)
            total_local, max_local, max_actual = (
                int(v)
                for v in torch.stack(
                    [
                        local_lens.sum().to(torch.int64),
                        local_lens.max().to(torch.int64),
                        local_actual_lens.max().to(torch.int64),
                    ]
                ).tolist()
            )
        has_actual = max_actual > 0

    restore = build_kv_allgather_restore_indices(
        per_req_total_kv_lens,
        cp_ctx.cp_size,
        owner_bs,
        device,
        total_kv_len=total_kv_host,
        total_local_kv=total_local,
    )
    return CPShardedPoolReader(
        CPShardConfig(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req_total_kv_lens,
            restore_indices=restore,
            block_size=block_size,
            total_local_kv=total_local,
            owner_block_size=owner_bs,
            seq_lens_total=total_kv_host,
            local_seq_lens_padded=local_lens,
            local_seq_lens_actual=local_actual_lens,
            max_local_seq_len_padded=max_local,
            has_local_seq_len_actual=has_actual,
        )
    )
