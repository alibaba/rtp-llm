"""Local V4.1 sparse reader using FlashMLA 07a1089857b63e74e3133630c02b083b75e8d4b2.

PlanarPages makes the upstream page ABI explicit. Converting a compact pool
only copies bytes; callers must refresh converted pages after their writers
finish. No persistent second cache, CP collective, PD conversion or fused
norm/RoPE path is hidden in this reader.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import ENCODINGS, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    ReaderResult,
    SwaBinding,
    _integer_tensor,
    _output_tensor,
)
from rtp_llm.models_py.modules.dsv41.native_aot import native_identity

FLASHMLA_REVISION = "07a1089857b63e74e3133630c02b083b75e8d4b2"


def is_supported(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) in (
        (10, 0),
        (10, 3),
    )


def _require_enabled(tensor):
    if os.environ.get("DSV41_FLASHMLA", "0") != "1":
        raise RuntimeError("V4.1 FlashMLA requires DSV41_FLASHMLA=1")
    if not is_supported(tensor):
        raise RuntimeError("V4.1 FlashMLA requires SM100 or SM103")
    native_identity("flash-mla")


@dataclass(frozen=True)
class PlanarPages:
    data: torch.Tensor
    region: CacheRegion
    entries_per_page: int

    def validate(self, device):
        if self.region not in (CacheRegion.SWA, CacheRegion.GLOBAL):
            raise ValueError("FlashMLA only accepts V4.1 SWA and global cache pages")
        CompactPages(self.data, self.region, self.entries_per_page).validate(device)
        if self.data.shape[0] * self.entries_per_page > torch.iinfo(torch.int32).max:
            raise ValueError("FlashMLA physical indices must fit in int32")

    def kernel_view(self):
        row_bytes = ENCODINGS[self.region].entry_bytes
        return self.data[:, : self.entries_per_page * row_bytes].view(
            self.data.shape[0], self.entries_per_page, 1, row_bytes
        )


def to_planar(pages: CompactPages, *, out: PlanarPages | None = None) -> PlanarPages:
    """Copy interleaved rows to upstream page planes without decoding any byte."""
    _require_enabled(pages.data)
    pages.validate(pages.data.device)
    if not isinstance(pages, CompactPages) or pages.region not in (
        CacheRegion.SWA,
        CacheRegion.GLOBAL,
    ):
        raise ValueError("planar conversion needs explicit compact SWA/global pages")
    encoding = ENCODINGS[pages.region]
    row_bytes = encoding.entry_bytes
    payload = 512 if pages.region == CacheRegion.SWA else 256
    scales = row_bytes - payload
    n, entries = pages.data.shape[0], pages.entries_per_page
    if out is None:
        stride = (entries * row_bytes + 511) // 512 * 512
        out = PlanarPages(
            torch.empty((n, stride), dtype=torch.uint8, device=pages.data.device),
            pages.region,
            entries,
        )
    out.validate(pages.data.device)
    if (
        out.region != pages.region
        or out.entries_per_page != entries
        or out.data.shape[0] != n
        or out.data.untyped_storage().data_ptr()
        == pages.data.untyped_storage().data_ptr()
    ):
        raise ValueError(
            "planar destination must have matching geometry and separate storage"
        )
    source = pages.data[:, : entries * row_bytes].view(n, entries, row_bytes)
    out.data[:, : entries * payload].view(n, entries, payload).copy_(
        source[:, :, :payload]
    )
    out.data[:, entries * payload : entries * row_bytes].view(n, entries, scales).copy_(
        source[:, :, payload:]
    )
    return out


def copy_to_compact(pages: PlanarPages, out: CompactPages) -> None:
    """Reverse the byte permutation for explicit copy/offload integration probes."""
    _require_enabled(pages.data)
    pages.validate(pages.data.device)
    out.validate(pages.data.device)
    if (
        not isinstance(pages, PlanarPages)
        or not isinstance(out, CompactPages)
        or out.region != pages.region
        or out.entries_per_page != pages.entries_per_page
        or out.data.shape[0] != pages.data.shape[0]
        or out.data.untyped_storage().data_ptr()
        == pages.data.untyped_storage().data_ptr()
    ):
        raise ValueError(
            "compact destination must match planar geometry without aliasing"
        )
    n, entries = pages.data.shape[0], pages.entries_per_page
    row_bytes = ENCODINGS[pages.region].entry_bytes
    payload = 512 if pages.region == CacheRegion.SWA else 256
    target = out.data[:, : entries * row_bytes].view(n, entries, row_bytes)
    target[:, :, :payload].copy_(
        pages.data[:, : entries * payload].view(n, entries, payload)
    )
    target[:, :, payload:].copy_(
        pages.data[:, entries * payload : entries * row_bytes].view(
            n, entries, row_bytes - payload
        )
    )


@dataclass(frozen=True)
class PlanarSwaBinding:
    pages: PlanarPages
    page_ids: torch.Tensor
    valid_starts: torch.Tensor
    valid_ends: torch.Tensor

    @classmethod
    def from_compact(cls, binding: SwaBinding, *, out=None):
        return cls(
            to_planar(binding.pages, out=out),
            binding.page_ids,
            binding.valid_starts,
            binding.valid_ends,
        )

    def validate(self, device):
        if not isinstance(self.pages, PlanarPages):
            raise TypeError("FlashMLA SWA binding must contain explicit planar pages")
        self.pages.validate(device)
        if self.pages.region != CacheRegion.SWA or self.pages.entries_per_page < 128:
            raise ValueError("FlashMLA main cache needs complete V4.1 FP8 SWA rings")
        requests = self.page_ids.numel()
        for tensor in (self.page_ids, self.valid_starts, self.valid_ends):
            _integer_tensor(tensor, (requests,), device)
        return requests


@dataclass(frozen=True)
class PlanarGlobalBinding:
    pages: PlanarPages
    page_table: torch.Tensor
    compress_ratio: int

    @classmethod
    def from_compact(cls, binding: GlobalBinding, *, out=None):
        return cls(
            to_planar(binding.pages, out=out),
            binding.page_table,
            binding.compress_ratio,
        )

    def validate(self, requests, device):
        if not isinstance(self.pages, PlanarPages):
            raise TypeError(
                "FlashMLA global binding must contain explicit planar pages"
            )
        self.pages.validate(device)
        if self.pages.region != CacheRegion.GLOBAL or self.compress_ratio not in (1, 2):
            raise ValueError("FlashMLA extra cache needs V4.1 global FP4 with ratio1/2")
        if self.page_table.ndim != 2 or self.page_table.shape[0] != requests:
            raise ValueError("global page table must contain one row per request")
        _integer_tensor(self.page_table, tuple(self.page_table.shape), device)


@dataclass(frozen=True)
class FlashMLAIndices:
    main: torch.Tensor
    main_lengths: torch.Tensor
    extra: torch.Tensor | None
    extra_lengths: torch.Tensor | None
    status: torch.Tensor
    query_valid: torch.Tensor


def build_indices(
    request_ids,
    query_positions,
    replay_floors,
    swa,
    *,
    global_kv=None,
    global_indices=None,
):
    """Map logical rows to safe physical indices on GPU, including during replay."""
    device = request_ids.device
    requests = swa.validate(device)
    rows = request_ids.numel()
    for tensor in (request_ids, query_positions, replay_floors):
        _integer_tensor(tensor, (rows,), device)
    request, position, floor = (
        tensor.to(torch.int64)
        for tensor in (request_ids, query_positions, replay_floors)
    )
    active = position >= 0
    query_valid = (
        active
        & (position < 1048576)
        & (request >= 0)
        & (request < requests)
        & (floor >= 0)
        & (floor <= position)
    )
    status = ((position < -1) | (active & ~query_valid)).to(torch.int32) * 2
    selected_request = request.clamp(0, max(requests - 1, 0))

    def select(tensor):
        if requests == 0:
            return tensor.new_zeros((rows,))
        return tensor[selected_request].to(torch.int64)

    page, start, end = (
        select(tensor) for tensor in (swa.page_ids, swa.valid_starts, swa.valid_ends)
    )
    begin = torch.maximum(
        torch.maximum(position - 127, floor), torch.zeros_like(position)
    )
    tokens = begin[:, None] + torch.arange(128, device=device)
    expected = query_valid[:, None] & (tokens <= position[:, None])
    ring_valid = (
        (start >= 0) & (end >= start) & (end - start <= swa.pages.entries_per_page)
    )
    usable = (
        expected
        & ring_valid[:, None]
        & (page[:, None] > 0)
        & (page[:, None] < swa.pages.data.shape[0])
        & (tokens >= start[:, None])
        & (tokens < end[:, None])
    )
    status |= (expected & ~usable).any(dim=1).to(torch.int32)
    main = (
        torch.where(
            usable,
            page[:, None] * swa.pages.entries_per_page
            + tokens % swa.pages.entries_per_page,
            -1,
        )
        .to(torch.int32)
        .unsqueeze(1)
    )
    main_lengths = expected.sum(dim=1, dtype=torch.int32)
    extra, extra_lengths = None, None
    if global_kv is None:
        if global_indices is not None:
            raise ValueError("global indices require an explicit planar global binding")
    else:
        global_kv.validate(requests, device)
        if (
            global_indices is None
            or global_indices.ndim != 2
            or global_indices.shape[0] != rows
            or global_indices.shape[1] > 512
        ):
            raise ValueError("FlashMLA global indices must be [rows, at most 512]")
        _integer_tensor(global_indices, tuple(global_indices.shape), device)
        capacity = global_indices.shape[1]
        width = max(64, (capacity + 63) // 64 * 64)
        logical = torch.full((rows, width), -1, device=device, dtype=torch.int64)
        logical[:, :capacity].copy_(global_indices)
        malformed = (logical < -1).any(dim=1)
        malformed |= (
            (logical[:, 1:] >= 0)
            & ((logical[:, :-1] < 0) | (logical[:, 1:] <= logical[:, :-1]))
        ).any(dim=1)
        status |= (query_valid & malformed).to(torch.int32) * 2
        visible = (position + 1) // global_kv.compress_ratio
        expected = query_valid[:, None] & (logical >= 0) & (logical < visible[:, None])
        block = logical.clamp_min(0) // global_kv.pages.entries_per_page
        table_width = global_kv.page_table.shape[1]
        if requests and table_width:
            physical = global_kv.page_table[
                selected_request[:, None], block.clamp(0, table_width - 1)
            ].to(torch.int64)
        else:
            physical = torch.zeros_like(block)
        usable = (
            expected
            & (block < table_width)
            & (physical > 0)
            & (physical < global_kv.pages.data.shape[0])
        )
        status |= (expected & ~usable).any(dim=1).to(torch.int32)
        extra = (
            torch.where(
                usable,
                physical * global_kv.pages.entries_per_page
                + logical.clamp_min(0) % global_kv.pages.entries_per_page,
                -1,
            )
            .to(torch.int32)
            .unsqueeze(1)
        )
        extra_lengths = expected.sum(dim=1, dtype=torch.int32)
    return FlashMLAIndices(
        main, main_lengths, extra, extra_lengths, status, query_valid
    )


def flashmla_attention(
    query,
    request_ids,
    query_positions,
    replay_floors,
    swa: PlanarSwaBinding,
    sinks,
    *,
    global_kv: PlanarGlobalBinding | None = None,
    global_indices=None,
    output=None,
    lse=None,
    status=None,
) -> ReaderResult:
    """Return the compact-reader output/LSE contract before inverse RoPE.

    Each query becomes an independent native batch row, so speculative rows
    have their own lengths. Every Python invocation creates fresh scheduling
    metadata; any required scheduler launch is therefore part of capture.
    Native outputs/workspaces still use PyTorch's Graph private pool, whose
    lifetime belongs to the caller's retained CUDAGraph object.
    """
    _require_enabled(query)
    if (
        query.ndim != 3
        or query.shape[1] not in (64, 128)
        or query.shape[2] != 512
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
    ):
        raise ValueError("FlashMLA Q must be contiguous BF16 [rows,64/128,512]")
    rows, heads, _ = query.shape
    device = query.device
    if (
        sinks.shape != (heads,)
        or sinks.dtype != torch.float32
        or sinks.device != device
        or not sinks.is_contiguous()
    ):
        raise ValueError("FlashMLA attention sinks must be contiguous FP32 [heads]")
    if not isinstance(swa, PlanarSwaBinding) or (
        global_kv is not None and not isinstance(global_kv, PlanarGlobalBinding)
    ):
        raise TypeError("FlashMLA requires explicit planar cache bindings")
    if request_ids.device != device or request_ids.shape != (rows,):
        raise ValueError("FlashMLA query and request rows must share shape/device")
    metadata = build_indices(
        request_ids,
        query_positions,
        replay_floors,
        swa,
        global_kv=global_kv,
        global_indices=global_indices,
    )
    output = _output_tensor(output, tuple(query.shape), torch.bfloat16, device)
    lse = _output_tensor(lse, (rows, heads), torch.float32, device)
    status = _output_tensor(status, (rows, heads), torch.int32, device)
    if rows:
        from flash_mla.flash_mla_interface import (
            FlashMLASchedMeta,
            flash_mla_with_kvcache,
        )

        native_output, native_lse = flash_mla_with_kvcache(
            query.view(rows, 1, heads, 512),
            swa.pages.kernel_view(),
            None,
            None,
            512,
            FlashMLASchedMeta(),
            softmax_scale=1.0 / math.sqrt(512),
            causal=False,
            is_fp8_kvcache=True,
            indices=metadata.main,
            attn_sink=sinks,
            extra_k_cache=None if global_kv is None else global_kv.pages.kernel_view(),
            extra_indices_in_kvcache=metadata.extra,
            topk_length=metadata.main_lengths,
            extra_topk_length=metadata.extra_lengths,
        )
        active = metadata.query_valid[:, None]
        output.copy_(torch.where(active[:, :, None], native_output.view_as(output), 0))
        have_kv = (metadata.main >= 0).any(dim=(1, 2))
        if metadata.extra is not None:
            have_kv |= (metadata.extra >= 0).any(dim=(1, 2))
        # Upstream LSE excludes the sink and uses +inf for no-KV queries.
        attention_lse = torch.where(
            have_kv[:, None], native_lse.reshape(rows, heads), -torch.inf
        )
        combined_lse = torch.logaddexp(attention_lse, sinks[None, :])
        lse.copy_(torch.where(active, combined_lse, -torch.inf))
        status.copy_(metadata.status[:, None].expand(-1, heads))
    return ReaderResult(output, status, lse)
