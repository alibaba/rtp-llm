"""Byte-preserving staging for DSV4 compressed KV in mapped host memory.

This is the transfer primitive, not a cache allocator or replacement policy.
The caller owns source-page lifetimes and orders compressor writes before
gather, then orders attention after gather. No host tensor values are read on
the decode path, so an existing instance can be captured in a CUDA graph.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_ROPE_STRIDE,
    SCALE_BYTES_PER_TOKEN,
)


@triton.jit
def _gather_model1_kv(
    source,
    selected,
    destination,
    remapped,
    NUM_SLOTS: tl.constexpr,
    SOURCE_SLOTS: tl.constexpr,
    SOURCE_ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_ENTRIES: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    byte = tl.arange(0, BLOCK)
    for position in range(tl.program_id(0), NUM_SLOTS, tl.num_programs(0)):
        source_slot = tl.load(selected + position).to(tl.int64)
        valid = (source_slot >= 0) & (source_slot < SOURCE_SLOTS)
        # MODEL1 stores all payloads first, then all scales within each block.
        source_offset = (source_slot // SOURCE_ENTRIES) * SOURCE_STRIDE + tl.where(
            byte < DATA_BYTES,
            (source_slot % SOURCE_ENTRIES) * DATA_BYTES + byte,
            SOURCE_ENTRIES * DATA_BYTES
            + (source_slot % SOURCE_ENTRIES) * SCALE_BYTES
            + byte
            - DATA_BYTES,
        )
        dest_slot = position.to(tl.int64)
        dest_offset = (dest_slot // DEST_ENTRIES) * DEST_STRIDE + tl.where(
            byte < DATA_BYTES,
            (dest_slot % DEST_ENTRIES) * DATA_BYTES + byte,
            DEST_ENTRIES * DATA_BYTES
            + (dest_slot % DEST_ENTRIES) * SCALE_BYTES
            + byte
            - DATA_BYTES,
        )
        payload = tl.load(
            source + source_offset,
            mask=valid & (byte < DATA_BYTES + SCALE_BYTES),
            other=0,
        )
        tl.store(destination + dest_offset, payload, byte < DATA_BYTES + SCALE_BYTES)
        tl.store(remapped + position, tl.where(valid, position, -1))


class CsaKvStaging:
    """Fixed-address compressed-KV staging, partitioned by batch row.

    ``source`` is the packed MODEL1 pool ``[blocks, entries, 584]``, with
    optional block-stride padding. It may live on the target GPU or in pinned
    CPU memory. ``gather`` takes translated global source slots ``[B, topk]``;
    negative or out-of-pool slots become masked attention entries. Duplicate
    selections retain their order and multiplicity.

    Every call refreshes the selected bytes; this class deliberately has no
    cross-step hit state that could survive request cancellation or page reuse.
    Allocate one instance per simultaneously live attention working set outside
    graph capture. A caller using a prefetch stream must provide its own event
    dependencies and retain this object until that stream has completed.
    """

    def __init__(
        self,
        source: torch.Tensor,
        *,
        max_batch_size: int,
        topk: int,
        device: torch.device | str = "cuda",
        max_ctas: int = 64,
    ) -> None:
        if (
            source.dtype != torch.uint8
            or source.ndim != 3
            or source.shape[-1] != ENTRY_BYTES
            or source.shape[0] <= 0
            or source.shape[1] <= 0
            or source.stride(2) != 1
            or source.stride(1) != ENTRY_BYTES
            or source.stride(0) < source.shape[1] * ENTRY_BYTES
        ):
            raise ValueError(
                "source must be a packed MODEL1 [blocks, entries, 584] pool"
            )
        if source.device.type == "cpu":
            if not source.is_pinned():
                raise ValueError("CPU source must be pinned for CUDA access")
        elif source.device.type != "cuda":
            raise ValueError("source must be on CUDA or pinned CPU memory")
        if max_batch_size <= 0 or topk <= 0 or max_ctas <= 0:
            raise ValueError("max_batch_size, topk and max_ctas must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("staging requires a CUDA destination")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if source.is_cuda and source.device != device:
            raise ValueError("CUDA source and staging must be on the same device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("allocate CsaKvStaging before CUDA graph capture")
            # FlashMLA's TMA path requires a 576-byte-aligned block stride.
            stride = (
                triton.cdiv(topk * ENTRY_BYTES, NOPE_ROPE_STRIDE) * NOPE_ROPE_STRIDE
            )
            self._storage = torch.zeros(
                (max_batch_size, stride), dtype=torch.uint8, device=device
            )
            self.pool = self._storage.as_strided(
                (max_batch_size, topk, ENTRY_BYTES), (stride, ENTRY_BYTES, 1)
            )
            self.indices = torch.full(
                (max_batch_size, 1, topk), -1, dtype=torch.int32, device=device
            )
        self.source = source
        self.max_batch_size = max_batch_size
        self.topk = topk
        self.max_ctas = max_ctas

    def gather(self, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fixed GPU pool and remapped ``[B, 1, topk]`` indices.

        Returned tensors are reused by the next call. Source contents may
        change between replays, but the source allocation must remain alive
        and its writes must have completed before this operation starts.
        """
        if (
            selected.ndim != 2
            or selected.shape[1] != self.topk
            or not 0 < selected.shape[0] <= self.max_batch_size
            or selected.dtype not in (torch.int32, torch.int64)
            or selected.device != self.pool.device
            or not selected.is_contiguous()
        ):
            raise ValueError("selected must be contiguous CUDA int32/int64 [B, topk]")
        count = selected.numel()
        with torch.cuda.device(self.pool.device):
            _gather_model1_kv[(min(self.max_ctas, count),)](
                self.source,
                selected,
                self.pool,
                self.indices,
                NUM_SLOTS=count,
                SOURCE_SLOTS=self.source.shape[0] * self.source.shape[1],
                SOURCE_ENTRIES=self.source.shape[1],
                SOURCE_STRIDE=self.source.stride(0),
                DEST_ENTRIES=self.topk,
                DEST_STRIDE=self.pool.stride(0),
                DATA_BYTES=NOPE_ROPE_STRIDE,
                SCALE_BYTES=SCALE_BYTES_PER_TOKEN,
                BLOCK=triton.next_power_of_2(ENTRY_BYTES),
                num_warps=4,
            )
        return self.pool, self.indices[: selected.shape[0]]


@triton.jit
def _check_model1_selection(
    Source,
    Selected,
    Pool,
    Slots,
    Errors,
    N: tl.constexpr,
    SOURCE_SLOTS: tl.constexpr,
    DEST_SLOTS: tl.constexpr,
    ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    ASSERT: tl.constexpr,
):
    byte = tl.arange(0, 1024)
    for index in range(tl.program_id(0), N, tl.num_programs(0)):
        source = tl.load(Selected + index).to(tl.int64)
        dest = tl.load(Slots + index).to(tl.int64)
        valid = (source >= 0) & (source < SOURCE_SLOTS)
        mapped = (dest >= 0) & (dest < DEST_SLOTS)
        source_offset = source // ENTRIES * SOURCE_STRIDE + tl.where(
            byte < 576,
            source % ENTRIES * 576 + byte,
            ENTRIES * 576 + source % ENTRIES * 8 + byte - 576,
        )
        dest_offset = dest // ENTRIES * DEST_STRIDE + tl.where(
            byte < 576,
            dest % ENTRIES * 576 + byte,
            ENTRIES * 576 + dest % ENTRIES * 8 + byte - 576,
        )
        mask = valid & mapped & (byte < 584)
        expected = tl.load(Source + source_offset, mask, other=0)
        actual = tl.load(Pool + dest_offset, mask, other=0)
        mismatch = tl.sum(((expected != actual) & mask).to(tl.int32), 0)
        mismatch += (valid & ~mapped).to(tl.int32)
        mismatch += (~valid & (dest != -1)).to(tl.int32)
        if mismatch != 0:
            tl.atomic_add(Errors, mismatch)
        if ASSERT:
            tl.device_assert(mismatch == 0, "CSA GPU/CPU selected KV bytes differ")


class CsaByteValidator:
    """Opt-in graph-safe diagnostic; excluded from performance runs."""

    def __init__(self, device):
        self.errors = torch.zeros(1, dtype=torch.int64, device=device)

    def check(self, source, selected, pool, slots, *, assertions=True):
        _check_model1_selection[(min(256, selected.numel()),)](
            source,
            selected,
            pool,
            slots,
            self.errors,
            N=selected.numel(),
            SOURCE_SLOTS=source.shape[0] * source.shape[1],
            DEST_SLOTS=pool.shape[0] * pool.shape[1],
            ENTRIES=source.shape[1],
            SOURCE_STRIDE=source.stride(0),
            DEST_STRIDE=pool.stride(0),
            ASSERT=assertions,
            debug=True,
            num_warps=4,
        )
