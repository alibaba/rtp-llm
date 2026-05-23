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
from typing import Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
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
      * ``block_size`` — pool's tokens-per-block.
      * ``total_local_kv`` — sum of per-rank local lengths (==
        ``ceil(T_r/(block_size*cp_size)) * block_size`` per req, summed).
    """

    cp_ctx: CPContext
    per_req_total_kv_lens: torch.Tensor
    restore_indices: torch.Tensor
    block_size: int
    total_local_kv: int


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
        cfg = self.cfg
        if cfg.block_size != block_size:
            raise ValueError(
                f"block_size mismatch: cfg={cfg.block_size} call={block_size}"
            )
        if gather_lens is not None:
            raise NotImplementedError(
                "CPShardedPoolReader does not support gather_lens yet; "
                "pass full per-request seq_lens or add suffix-aware restore/scatter."
            )
        cp_ctx = cfg.cp_ctx
        device = out.device
        D = out.shape[-1]

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
        local_seq_lens_padded = cp_padded_local_kv_lens(
            cfg.per_req_total_kv_lens, cp_ctx.cp_size, block_size
        ).to(device=device, dtype=torch.int32)
        local_seq_lens_actual = cp_actual_owned_kv_lens(
            cfg.per_req_total_kv_lens, cp_ctx.cp_size, block_size, cp_ctx.cp_rank
        ).to(device=device, dtype=torch.int32)
        # Use the rank's existing block_table directly — Stage 5a guarantees
        # block_table[r, :] holds this rank's local entries in compact form.
        # zeros() (not empty()) so the [actual, padded) tail of each request
        # is a deterministic 0 and contributes nothing to the gather.
        local_packed = torch.zeros(
            (
                int(local_seq_lens_padded.shape[0]),
                (
                    int(local_seq_lens_padded.max().item())
                    if local_seq_lens_padded.numel()
                    else 0
                ),
                ENTRY_BYTES,
            ),
            dtype=torch.uint8,
            device=device,
        )
        if local_packed.shape[1] > 0 and int(local_seq_lens_actual.max().item()) > 0:
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
        local_flat = _pack_padded_to_flat(
            local_packed, local_seq_lens_padded, cfg.total_local_kv, ENTRY_BYTES
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
        gathered = all_gather(local_flat, group=Group.TP)
        # gathered shape is rank-major: [cp_size * total_local_kv, 584].

        # Step 4: restore to request-concatenated logical order.
        restored_packed = gathered[cfg.restore_indices].contiguous()  # [sum(T_r), 584]

        # Step 5: dequant locally after communication. This keeps NCCL payload
        # at 584B/token instead of 1024B/token BF16 [512].
        restored = torch.empty(
            (int(restored_packed.shape[0]), D), dtype=out.dtype, device=device
        )
        dequantize_packed_k_cache_flat(restored, restored_packed)

        # Step 6: scatter restored into out[r, offset:offset+T_r, :].
        _scatter_flat_to_workspace(restored, out, seq_lens, offset)


_compute_local_seq_lens = cp_padded_local_kv_lens
_compute_local_owned_kv_lens = cp_actual_owned_kv_lens


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
            "block_size; got None. Pass torch.zeros(B, dtype=int64) for a "
            "guaranteed-cold prefill iteration."
        )
    if not torch.any(per_req_total_kv_lens > 0):
        return LocalPoolReader()
    # Lazy import to avoid circular: cp.py imports nothing from here.
    from rtp_llm.models_py.modules.dsv4.cp import build_kv_allgather_restore_indices

    device = per_req_total_kv_lens.device
    restore = build_kv_allgather_restore_indices(
        per_req_total_kv_lens, cp_ctx.cp_size, block_size, device
    )
    local_lens = cp_padded_local_kv_lens(
        per_req_total_kv_lens, cp_ctx.cp_size, block_size
    )
    total_local = int(local_lens.sum().item())
    return CPShardedPoolReader(
        CPShardConfig(
            cp_ctx=cp_ctx,
            per_req_total_kv_lens=per_req_total_kv_lens,
            restore_indices=restore,
            block_size=block_size,
            total_local_kv=total_local,
        )
    )
