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

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_actual_owned_kv_lens,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    dequantize_and_gather_k_cache,
    dequantize_packed_k_cache_flat,
    gather_k_cache_packed,
)


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
    ) -> None:
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
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
    local_seq_lens_padded: Optional[torch.Tensor] = None
    local_seq_lens_actual: Optional[torch.Tensor] = None
    max_local_seq_len_padded: int = 0
    has_local_seq_len_actual: Optional[bool] = None

    def __post_init__(self) -> None:
        owner_block_size = int(self.owner_block_size or self.block_size)
        self.owner_block_size = owner_block_size

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

        if self.max_local_seq_len_padded == 0 and self.local_seq_lens_padded.numel():
            if self.local_seq_lens_padded.numel() == 1:
                self.max_local_seq_len_padded = int(self.total_local_kv)
            else:
                self.max_local_seq_len_padded = int(
                    self.local_seq_lens_padded.max().item()
                )
        if self.has_local_seq_len_actual is None and self.local_seq_lens_actual.numel():
            self.has_local_seq_len_actual = (
                int(self.local_seq_lens_actual.max().item()) > 0
            )


@dataclass
class CPShardedPoolReadHandle:
    gathered: torch.Tensor
    work: Any
    completion_event: torch.cuda.Event
    stream: torch.cuda.Stream
    out: torch.Tensor
    seq_lens: torch.Tensor
    offset: int
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
    ) -> None:
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
        self._restore_dequant_scatter(gathered, out, seq_lens, offset)

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
    ) -> Optional[CPShardedPoolReadHandle]:
        """Launch only the CP all-gather stage on ``stream``.

        The local pack stays on the caller's current stream.  The communication
        stream waits for it, and ``prepare_fill_async`` later queues restore /
        dequant / scatter on the shared post stream.
        """
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

        from rtp_llm.models_py.distributed import collective_torch, pynccl_cp

        process_group = collective_torch._get_group(Group.TP)
        world_size = torch.distributed.get_world_size(process_group)
        if world_size != cfg.cp_ctx.cp_size:
            raise RuntimeError(
                f"CP pool reader world_size({world_size}) != cp_size({cfg.cp_ctx.cp_size})"
            )

        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        local_flat.record_stream(stream)
        # CP all-gather backend is selected inside the dispatcher. gather_cmp is
        # intentionally not a symmetric-memory role because its async consumer
        # lifetime is separate from the workspace-backed compressor gathers.
        # Under DSV4_CP_SYMM it falls back to ordinary pynccl. ``work`` is None
        # on pynccl (stream-ordered, fenced via completion_event).
        gather_rows = world_size * int(local_flat.shape[0])
        gathered, work, completion_event = pynccl_cp.cp_all_gather(
            local_flat,
            lambda r, c, d: torch.empty((r, c), dtype=d, device=device),
            role="gather_cmp",
            rows=gather_rows,
            cols=ENTRY_BYTES,
            process_group=process_group,
            gather_stream=stream,
            profile_name="dsv4.cp.all_gather.pool_reader.gather_cmp",
            symm_variable=True,
        )

        return CPShardedPoolReadHandle(
            gathered=gathered,
            work=work,
            completion_event=completion_event,
            stream=stream,
            out=out,
            seq_lens=seq_lens,
            offset=offset,
        )

    def prepare_fill_async(
        self,
        handle: CPShardedPoolReadHandle,
        *,
        stream: torch.cuda.Stream,
    ) -> None:
        if handle.done_event is not None:
            return
        if stream is None:
            raise ValueError("prepare_fill_async requires an explicit stream")
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
            handle.gathered.record_stream(stream)
            handle.out.record_stream(stream)
            try:
                self._restore_dequant_scatter(
                    handle.gathered,
                    handle.out,
                    handle.seq_lens,
                    handle.offset,
                )
            finally:
                handle.done_event = torch.cuda.Event()
                handle.done_event.record(stream)

    def wait_fill_async(self, handle: CPShardedPoolReadHandle) -> None:
        done_event = handle.done_event
        current_stream = torch.cuda.current_stream(handle.out.device)
        if done_event is not None:
            current_stream.wait_event(done_event)
        else:
            current_stream.wait_event(handle.completion_event)
        self._wait_fill_work_once(handle)

    def discard_fill_async(self, handle: CPShardedPoolReadHandle) -> None:
        self.wait_fill_async(handle)

    @staticmethod
    def _wait_fill_work_once(handle: CPShardedPoolReadHandle) -> None:
        if not handle.work_waited:
            if handle.work is not None:  # None on the pynccl path (stream-ordered)
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
    ) -> None:
        cfg = self.cfg
        D = out.shape[-1]
        with record_function_range("dsv4.cp.all_gather.pool_reader.gather_cmp.restore"):
            restored_packed = gathered[cfg.restore_indices].contiguous()

            # Step 5: dequant locally after communication. This keeps NCCL payload
            # at 584B/token instead of 1024B/token BF16 [512].
            restored = torch.empty(
                (int(restored_packed.shape[0]), D),
                dtype=out.dtype,
                device=out.device,
            )
            dequantize_packed_k_cache_flat(restored, restored_packed)

            # Step 6: scatter restored into out[r, offset:offset+T_r, :].
            _scatter_flat_to_workspace(restored, out, seq_lens, offset)


def _pack_padded_to_flat(
    padded: torch.Tensor, lens: torch.Tensor, total: int, D: int
) -> torch.Tensor:
    """Drop per-row padding → flat [total, D].

    Vectorized: build per-token ``(b_idx, s_idx)`` via repeat_interleave +
    arange − cu_starts, then advanced-index in one shot. Replaces a
    per-request Python loop with B .item() syncs (~B per layer per
    reader).
    """
    if total == 0:
        return torch.empty((0, D), dtype=padded.dtype, device=padded.device)
    device = padded.device
    lens_l = lens.to(device=device, dtype=torch.int64)
    B = int(lens_l.shape[0])
    if B == 1:
        return padded[0, :total, :].contiguous()
    b_idx = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64), lens_l
    )
    cu_starts = torch.zeros(B, device=device, dtype=torch.int64)
    if B > 1:
        cu_starts[1:] = torch.cumsum(lens_l[:-1], dim=0)
    s_idx = torch.arange(
        total, device=device, dtype=torch.int64
    ) - cu_starts.index_select(0, b_idx)
    return padded[b_idx, s_idx]


def _scatter_flat_to_workspace(
    restored: torch.Tensor, out: torch.Tensor, seq_lens: torch.Tensor, offset: int
) -> None:
    """Copy restored[r-th-slice] → out[r, offset:offset+T_r, :].

    Vectorized: build per-token ``(b_idx, col_idx)`` and use
    ``out.index_put_`` for a single GPU op. Replaces a per-request loop
    with B .item() syncs.
    """
    total = int(restored.shape[0])
    if total == 0:
        return
    device = restored.device
    seq_lens_l = seq_lens.to(device=device, dtype=torch.int64)
    B = int(seq_lens_l.shape[0])
    if B == 1:
        out[0, offset : offset + total, :].copy_(restored)
        return
    b_idx = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64), seq_lens_l
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


def make_compressed_k_pool_reader(
    cp_ctx: Optional[CPContext],
    kv_cache_sharded: bool,
    per_req_total_kv_lens: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    owner_block_size: Optional[int] = None,
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
    if not torch.any(per_req_total_kv_lens > 0):
        return LocalPoolReader()
    # Lazy import to avoid circular: cp.py imports nothing from here.
    from rtp_llm.models_py.modules.dsv4.cp import build_kv_allgather_restore_indices

    device = per_req_total_kv_lens.device
    owner_bs = int(owner_block_size or block_size)
    if owner_bs <= 0:
        raise ValueError(f"owner_block_size must be positive, got {owner_bs}")
    restore = build_kv_allgather_restore_indices(
        per_req_total_kv_lens, cp_ctx.cp_size, owner_bs, device
    )
    local_lens = cp_padded_local_kv_lens(
        per_req_total_kv_lens, cp_ctx.cp_size, owner_bs
    )
    local_actual_lens = cp_actual_owned_kv_lens(
        per_req_total_kv_lens, cp_ctx.cp_size, owner_bs, cp_ctx.cp_rank
    )
    # Host scalars are needed to size the per-layer flat gather buffers. Keep
    # these syncs in metadata construction; the read/restore hot path below must
    # stay kernel-only aside from the single NCCL wait.
    total_local = int(local_lens.sum().item())
    if not local_lens.numel():
        max_local = 0
    elif local_lens.numel() == 1:
        max_local = total_local
    else:
        max_local = int(local_lens.max().item())
    has_actual = (
        int(local_actual_lens.max().item()) > 0 if local_actual_lens.numel() else False
    )
    return CPShardedPoolReader(
        CPShardConfig(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req_total_kv_lens,
            restore_indices=restore,
            block_size=block_size,
            total_local_kv=total_local,
            owner_block_size=owner_bs,
            local_seq_lens_padded=local_lens,
            local_seq_lens_actual=local_actual_lens,
            max_local_seq_len_padded=max_local,
            has_local_seq_len_actual=has_actual,
        )
    )
