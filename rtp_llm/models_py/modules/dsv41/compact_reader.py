"""Experimental native V4.1 compact KV reader.

Set DSV41_NATIVE_COMPACT_READER=1 explicitly. The attention kernel reads at most
128 SWA rows and 512 selected global rows per query, in 16-row tiles. There is no
history-sized score or decoded-KV allocation. It is a correctness candidate,
with no performance-selection or release-acceptance claim.

Byte layouts follow the frozen HF revision
2bc89ac599031fa673cab993f1df02fc4a98c673: post-RoPE SWA FP8 group32/UE8M0,
global FP4 group16/E4M3, and index FP4 group32/UE8M0. Each row stores payload
followed by its scales; FP4 packs the even channel in the low nibble.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from rtp_llm.models_py.modules.dsv41._compact_reader_triton import (
    compact_attention_kernel,
    gather_compact_kernel,
)
from rtp_llm.models_py.modules.dsv41.cache_layout import ENCODINGS, CacheRegion

MAX_GATHER_BYTES = 64 * 1024 * 1024
_FORMAT_IDS = {CacheRegion.SWA: 0, CacheRegion.GLOBAL: 1, CacheRegion.INDEX_K: 2}


def is_supported(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 10


def _require_enabled(tensor: torch.Tensor) -> None:
    if os.environ.get("DSV41_NATIVE_COMPACT_READER", "0") != "1":
        raise RuntimeError(
            "native compact reader requires DSV41_NATIVE_COMPACT_READER=1"
        )
    if not is_supported(tensor):
        raise RuntimeError("native compact reader requires a Blackwell CUDA device")


def _integer_tensor(tensor: torch.Tensor, shape: Tuple[int, ...], device) -> None:
    if (
        tensor.device != device
        or tensor.dtype not in (torch.int32, torch.int64)
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "metadata must be contiguous CUDA integer tensors of the declared shape"
        )


def _output_tensor(
    tensor: Optional[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype, device
) -> torch.Tensor:
    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "output buffers must match shape, dtype, device and contiguity"
        )
    return tensor


def _separate_outputs(outputs, inputs):
    def bounds(tensor):
        first = tensor.data_ptr()
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        return first, first + span * tensor.element_size()

    retained = [tensor for tensor in inputs if tensor is not None and tensor.numel()]
    for output in outputs:
        if not output.numel():
            continue
        first, last = bounds(output)
        for tensor in retained:
            begin, end = bounds(tensor)
            if first < end and begin < last:
                raise ValueError("output buffers must not alias inputs or each other")
        retained.append(output)


@dataclass(frozen=True)
class CompactPages:
    data: torch.Tensor
    region: CacheRegion
    entries_per_page: int

    def validate(self, device) -> None:
        row_bytes = ENCODINGS[self.region].entry_bytes
        if (
            self.data.device != device
            or self.data.dtype != torch.uint8
            or self.data.ndim != 2
            or self.data.shape[0] < 1
            or self.entries_per_page <= 0
            or self.data.shape[1] < self.entries_per_page * row_bytes
            or self.data.stride(1) != 1
            or self.data.stride(0) < self.data.shape[1]
            or self.data.stride(0) % 512
        ):
            raise ValueError("invalid compact pool geometry or physical page stride")


@dataclass(frozen=True)
class SwaBinding:
    pages: CompactPages
    page_ids: torch.Tensor
    valid_starts: torch.Tensor
    valid_ends: torch.Tensor

    def validate(self, device) -> int:
        if not isinstance(self.pages, CompactPages):
            raise TypeError("native SWA binding requires row-interleaved compact pages")
        self.pages.validate(device)
        if self.pages.region != CacheRegion.SWA or self.pages.entries_per_page < 128:
            raise ValueError(
                "SWA binding requires complete 528-byte rings of at least 128 rows"
            )
        requests = self.page_ids.numel()
        for tensor in (self.page_ids, self.valid_starts, self.valid_ends):
            _integer_tensor(tensor, (requests,), device)
        return requests


@dataclass(frozen=True)
class GlobalBinding:
    pages: CompactPages
    page_table: torch.Tensor
    compress_ratio: int

    def validate(self, requests: int, device) -> None:
        if not isinstance(self.pages, CompactPages):
            raise TypeError(
                "native global binding requires row-interleaved compact pages"
            )
        self.pages.validate(device)
        if self.pages.region != CacheRegion.GLOBAL or self.compress_ratio not in (1, 2):
            raise ValueError(
                "global attention needs 288-byte pages and ratio1 or ratio2"
            )
        if self.page_table.ndim != 2 or self.page_table.shape[0] != requests:
            raise ValueError("global page table must contain one row per request")
        _integer_tensor(self.page_table, tuple(self.page_table.shape), device)


@dataclass(frozen=True)
class ReaderResult:
    output: torch.Tensor
    status: torch.Tensor
    lse: Optional[torch.Tensor] = None

    def check(self) -> None:
        """Check after completion, outside graph capture and before consuming output."""
        with torch.cuda.device(self.status.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "reader status must be checked after graph execution"
                )
        errors = self.status.detach().cpu()
        if torch.any(errors != 0):
            codes = sorted(set(errors.reshape(-1).tolist()) - {0})
            raise RuntimeError(
                f"compact reader rejected metadata: status={codes}; 1=missing KV, 2=invalid metadata"
            )


def gather_compact(
    pages: CompactPages,
    page_table: torch.Tensor,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    visible_lengths: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    status: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> ReaderResult:
    """Decode a bounded list of logical KV rows; padding/future positions become zero.

    Page zero and negative IDs are unmapped. Missing pages for visible selected
    rows set status=1. A caller must check status after the GPU completes.
    The table is read inside the kernel on every invocation or graph replay.
    """
    if not isinstance(pages, CompactPages):
        raise TypeError("native gather requires row-interleaved compact pages")
    _require_enabled(pages.data)
    pages.validate(pages.data.device)
    if positions.ndim != 2 or page_table.ndim != 2:
        raise ValueError("positions and page table must be matrices")
    rows, slots = positions.shape
    requests = page_table.shape[0]
    device = pages.data.device
    for tensor, shape in (
        (page_table, tuple(page_table.shape)),
        (request_ids, (rows,)),
        (positions, (rows, slots)),
        (visible_lengths, (rows,)),
    ):
        _integer_tensor(tensor, shape, device)
    if slots > 512:
        raise ValueError("gather requires query tiling with at most 512 selected rows")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("compact dequant output must be BF16 or diagnostic FP32")
    dim = ENCODINGS[pages.region].head_dim
    item_bytes = 2 if output_dtype == torch.bfloat16 else 4
    if rows * slots * dim * item_bytes > MAX_GATHER_BYTES:
        raise ValueError("gather exceeds 64 MiB workspace; tile the query rows")
    output = _output_tensor(output, (rows, slots, dim), output_dtype, device)
    status = _output_tensor(status, (rows, slots), torch.int32, device)
    _separate_outputs(
        (output, status),
        (pages.data, page_table, request_ids, positions, visible_lengths),
    )
    if rows and slots:
        gather_compact_kernel[(rows, slots)](
            pages.data,
            page_table,
            request_ids,
            positions,
            visible_lengths,
            output,
            status,
            NUM_REQUESTS=requests,
            NUM_PAGES=pages.data.shape[0],
            TABLE_WIDTH=page_table.shape[1],
            ROWS_PER_PAGE=pages.entries_per_page,
            PAGE_STRIDE=pages.data.stride(0),
            SLOT_COUNT=slots,
            HEAD_DIM=dim,
            ROW_BYTES=ENCODINGS[pages.region].entry_bytes,
            FORMAT=_FORMAT_IDS[pages.region],
            num_warps=4,
        )
    return ReaderResult(output, status)


def compact_attention(
    query: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    replay_floors: torch.Tensor,
    swa: SwaBinding,
    sinks: torch.Tensor,
    *,
    global_kv: Optional[GlobalBinding] = None,
    global_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    status: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> ReaderResult:
    """Attend post-RoPE Q to compact SWA/global KV with one shared sink softmax.

    Input Q is [rows, heads, 512] BF16. Global indices contain a strictly sorted
    valid prefix followed by -1 padding; causality is enforced per query even if
    future physical KV rows exist. A position of -1 denotes a padding query.
    SWA page IDs refer to each request's complete active ring. CP byte shards must
    be gathered before this local reader; no collective is hidden in this API.
    The returned attention output precedes the model's inverse RoPE transform.

    Stable caller-owned output/LSE/status tensors allow graph capture. Request,
    page, range, replay-floor and candidate tensor contents are all read on GPU
    during replay, so refreshed page IDs do not require recapture.
    """
    if not isinstance(swa, SwaBinding) or (
        global_kv is not None and not isinstance(global_kv, GlobalBinding)
    ):
        raise TypeError("native attention requires row-interleaved compact bindings")
    _require_enabled(query)
    if (
        query.ndim != 3
        or query.shape[-1] != 512
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
    ):
        raise ValueError("main Q must be contiguous BF16 [rows, heads, 512]")
    rows, heads, _ = query.shape
    if heads <= 0 or output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("invalid head count or output dtype")
    device = query.device
    requests = swa.validate(device)
    for tensor in (request_ids, query_positions, replay_floors):
        _integer_tensor(tensor, (rows,), device)
    if (
        sinks.device != device
        or sinks.dtype != torch.float32
        or tuple(sinks.shape) != (heads,)
        or not sinks.is_contiguous()
    ):
        raise ValueError("attention sinks must be contiguous FP32 [heads]")
    if global_kv is None:
        if global_indices is not None:
            raise ValueError("global indices require a global KV binding")
        global_pool = swa.pages.data
        global_table = swa.page_ids
        indices = query_positions
        ratio, capacity, global_entries, global_width = 0, 0, 1, 0
    else:
        global_kv.validate(requests, device)
        if (
            global_indices is None
            or global_indices.ndim != 2
            or global_indices.shape[0] != rows
            or global_indices.shape[1] > 512
        ):
            raise ValueError("global top-k indices must have shape [rows, at most 512]")
        _integer_tensor(global_indices, tuple(global_indices.shape), device)
        global_pool = global_kv.pages.data
        global_table = global_kv.page_table
        indices = global_indices
        ratio = global_kv.compress_ratio
        capacity = global_indices.shape[1]
        global_entries = global_kv.pages.entries_per_page
        global_width = global_table.shape[1]
    output = _output_tensor(output, tuple(query.shape), output_dtype, device)
    lse = _output_tensor(lse, (rows, heads), torch.float32, device)
    status = _output_tensor(status, (rows, heads), torch.int32, device)
    _separate_outputs(
        (output, lse, status),
        (
            query,
            request_ids,
            query_positions,
            replay_floors,
            swa.pages.data,
            swa.page_ids,
            swa.valid_starts,
            swa.valid_ends,
            sinks,
            global_pool,
            global_table,
            indices,
        ),
    )
    if rows:
        compact_attention_kernel[(rows, heads)](
            query,
            request_ids,
            query_positions,
            replay_floors,
            swa.pages.data,
            swa.page_ids,
            swa.valid_starts,
            swa.valid_ends,
            global_pool,
            global_table,
            indices,
            sinks,
            output,
            lse,
            status,
            HEADS=heads,
            NUM_REQUESTS=requests,
            SWA_NUM_PAGES=swa.pages.data.shape[0],
            SWA_ENTRIES=swa.pages.entries_per_page,
            SWA_PAGE_STRIDE=swa.pages.data.stride(0),
            GLOBAL_NUM_PAGES=global_pool.shape[0],
            GLOBAL_ENTRIES=global_entries,
            GLOBAL_PAGE_STRIDE=global_pool.stride(0),
            GLOBAL_TABLE_WIDTH=global_width,
            GLOBAL_CAPACITY=capacity,
            COMPRESS_RATIO=ratio,
            SCALE=1.0 / math.sqrt(512),
            BLOCK_N=16,
            num_warps=8,
        )
    return ReaderResult(output, status, lse)
