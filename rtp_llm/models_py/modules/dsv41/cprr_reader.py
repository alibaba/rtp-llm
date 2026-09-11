"""Local restoration of received CP8 pages for the V4.1 compact readers.

Communication is explicit caller work: inputs already contain all eight ranks
in CP rank order. SWA rank-local page IDs may differ between ranks. Global and
index pages retain their rank-major receive storage and only remap page IDs.
These functions do not implement a collective, PD transport or a second cache.
The receive buffers may contain a bounded working set with remapped local IDs;
they are not required to contain the complete history of each rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    CacheIdentity,
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    MAX_GATHER_BYTES,
    CompactPages,
    ReaderResult,
    _integer_tensor,
    _output_tensor,
    is_supported,
)


def _page(layout, slot):
    if layout.cp_size != 8:
        raise ValueError("CPRR restoration requires the declared CP8 layout")
    for page in layout.pages:
        if page.slot == slot:
            return page
    raise ValueError("CPRR source slot is not owned by the declared layout")


def _received(received, page):
    if not is_supported(received):
        raise RuntimeError("CPRR reader restoration requires Blackwell CUDA")
    if (
        received.dtype != torch.uint8
        or received.ndim != 3
        or received.shape[0] != page.cp_size
        or received.shape[1] < 1
        or received.shape[2] != page.prefill_shard_bytes
        or not received.is_contiguous()
    ):
        raise ValueError("received pages must match all eight rank-local strides")
    if received.numel() > MAX_GATHER_BYTES:
        raise ValueError("CPRR receive workspace exceeds 64 MiB; tile selected pages")


def _separate(output, *inputs):
    if output.numel() and any(
        value.numel()
        and output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        for value in inputs
    ):
        raise ValueError("CPRR output buffers must not alias source storage")


@dataclass(frozen=True)
class CprrSwaRestore:
    pages: CompactPages
    status: torch.Tensor

    def check(self):
        ReaderResult(self.pages.data, self.status).check()


def restore_cprr_swa(
    layout: CacheLayout,
    layer: int,
    received: torch.Tensor,
    rank_page_ids: torch.Tensor,
    *,
    output: CompactPages | None = None,
    status: torch.Tensor | None = None,
) -> CprrSwaRestore:
    """Restore one complete ring per request, including its alignment padding.

    received is [8, rank-local pages, shard bytes]; rank_page_ids is [8,
    requests]. Output request i occupies page i+1; page zero is unmapped.
    A missing byte slice sets status=1, invalid IDs set status=2. Either error
    zeros the entire affected ring. Check status before consuming restored KV.
    Caller-owned output/status buffers support capture and changing IDs on replay.
    """
    page = _page(layout, RegionSlot(CacheRegion.SWA, layer))
    _received(received, page)
    device = received.device
    if rank_page_ids.ndim != 2 or rank_page_ids.shape[0] != layout.cp_size:
        raise ValueError("SWA page IDs must contain all eight CP ranks")
    requests = rank_page_ids.shape[1]
    _integer_tensor(rank_page_ids, (layout.cp_size, requests), device)
    shape = (requests + 1, page.page_stride_bytes)
    if (requests + 1) * page.page_stride_bytes > MAX_GATHER_BYTES:
        raise ValueError("restored SWA workspace exceeds 64 MiB; tile requests")
    if output is None:
        output = CompactPages(
            torch.empty(shape, dtype=torch.uint8, device=device),
            CacheRegion.SWA,
            page.entries,
        )
    output.validate(device)
    if (
        output.region != CacheRegion.SWA
        or output.entries_per_page != page.entries
        or tuple(output.data.shape) != shape
        or not output.data.is_contiguous()
    ):
        raise ValueError("SWA destination must match the complete declared page stride")
    status = _output_tensor(status, (requests,), torch.int32, device)
    _separate(output.data, received, rank_page_ids, status)
    _separate(status, received, rank_page_ids)
    valid = (rank_page_ids > 0) & (rank_page_ids < received.shape[1])
    malformed = (rank_page_ids < -1) | (rank_page_ids >= received.shape[1])
    status.copy_(
        torch.where(malformed.any(dim=0), 2, (~valid).any(dim=0).to(torch.int32))
    )
    rank = torch.arange(layout.cp_size, device=device)[:, None]
    slices = received[rank, rank_page_ids.long().clamp(0, received.shape[1] - 1)]
    restored = output.data[1:].view(requests, layout.cp_size, page.prefill_shard_bytes)
    restored.copy_(
        torch.where((status == 0)[None, :, None], slices, 0).permute(1, 0, 2)
    )
    output.data[0].zero_()
    return CprrSwaRestore(output, status)


@dataclass(frozen=True)
class CprrPagedRestore:
    pages: CompactPages
    page_table: torch.Tensor
    status: torch.Tensor

    def check(self):
        ReaderResult(self.pages.data, self.status).check()


def bind_cprr_paged(
    layout: CacheLayout,
    slot: RegionSlot,
    received: torch.Tensor,
    rank_page_tables: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
) -> CprrPagedRestore:
    """Map CPRR pages without copying or decoding their payload.

    Tables are [8, requests, virtual blocks], using each rank's received local
    page IDs. Logical block b belongs to rank b%8 and virtual block b//8,
    exactly as RegionPage.paged_location. Missing IDs 0/-1 remain unmapped;
    malformed IDs set status=2 without producing an out-of-bounds reader index.
    The returned pages alias received, which must remain alive and unchanged
    until its last consumer completes. New contents or IDs need no recapture.
    """
    page = _page(layout, slot)
    if slot.region == CacheRegion.SWA:
        raise ValueError("SWA pages require complete byte-slice restoration")
    _received(received, page)
    if rank_page_tables.ndim != 3 or rank_page_tables.shape[0] != layout.cp_size:
        raise ValueError("paged tables must contain all eight CP ranks")
    device = received.device
    _integer_tensor(rank_page_tables, tuple(rank_page_tables.shape), device)
    _, requests, virtual_blocks = rank_page_tables.shape
    if virtual_blocks * layout.reuse_unit > 1048576:
        raise ValueError("CPRR page table exceeds the model context capacity")
    shape = (requests, virtual_blocks * layout.cp_size)
    page_table = _output_tensor(page_table, shape, torch.int32, device)
    status = _output_tensor(status, shape, torch.int32, device)
    _separate(page_table, received, rank_page_tables, status)
    _separate(status, received, rank_page_tables)
    local_pages = received.shape[1]
    if layout.cp_size * local_pages > torch.iinfo(torch.int32).max:
        raise ValueError("restored physical page IDs must fit int32")
    local = rank_page_tables.permute(1, 2, 0).reshape(shape).long()
    rank = torch.arange(shape[1], device=device) % layout.cp_size
    valid = (local > 0) & (local < local_pages)
    page_table.copy_(torch.where(valid, rank[None, :] * local_pages + local, 0))
    status.copy_(((local < -1) | (local >= local_pages)).to(torch.int32) * 2)
    pages = CompactPages(
        received.view(layout.cp_size * local_pages, page.page_stride_bytes),
        slot.region,
        page.entries,
    )
    return CprrPagedRestore(pages, page_table, status)


@dataclass(frozen=True)
class CprrReadIdentity:
    request_id: str
    cache: CacheIdentity
    epoch: int
    chunk_start: int
    chunk_end: int

    def __post_init__(self):
        if (
            not self.request_id
            or type(self.epoch) is not int
            or self.epoch < 0
            or not 0 <= self.chunk_start <= self.chunk_end <= 1048576
        ):
            raise ValueError("CPRR read requires request, epoch and chunk identity")


class CprrReaderLease:
    """Keep a received source alive from its ready event to its last consumer.

    Lifecycle methods run outside capture, on the streams doing the work. A
    consumer may enqueue a captured reader graph between acquire and complete.
    Graph objects retaining these buffers must not replay after release. This
    lease does not hide synchronization or communication inside reader kernels.
    """

    def __init__(self, identity, layout, slot, resources, consumers):
        _page(layout, slot)
        if identity.cache.layout_fingerprint != layout.fingerprint:
            raise ValueError("CPRR source and cache layout identities disagree")
        self.identity, self.slot = identity, slot
        self.resources = tuple(resources)
        self.consumers = frozenset(consumers)
        if not self.resources or not self.consumers:
            raise ValueError(
                "CPRR lease needs backing resources and explicit consumers"
            )
        self.device = self.resources[0].device
        if any(not is_supported(t) or t.device != self.device for t in self.resources):
            raise ValueError("CPRR lease resources must share one Blackwell device")
        for layer in self.consumers:
            source = layer_sources(layer)
            owner = {
                CacheRegion.SWA: layer,
                CacheRegion.GLOBAL: source.global_owner,
                CacheRegion.INDEX_K: (
                    source.index_k_owner if source.scores_queries else None
                ),
            }[slot.region]
            if owner != slot.owner_layer:
                raise ValueError(
                    "consumer does not read the declared V4.1 source owner"
                )
        self.active, self.completed = {}, {}
        self.released = False
        self._outside_capture()
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.device))

    def _outside_capture(self):
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("manage CPRR lease lifetime outside Graph capture")

    def _validate(self, identity):
        self._outside_capture()
        if self.released or identity != self.identity:
            raise ValueError("CPRR lease has stale or released source identity")

    def acquire(self, layer, identity):
        self._validate(identity)
        if (
            layer not in self.consumers
            or layer in self.active
            or layer in self.completed
        ):
            raise ValueError("CPRR consumer is unregistered or already acquired")
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.ready)
        for tensor in self.resources:
            tensor.record_stream(stream)
        self.active[layer] = stream

    def complete(self, layer, identity):
        self._validate(identity)
        stream = torch.cuda.current_stream(self.device)
        if layer not in self.active or stream != self.active[layer]:
            raise ValueError("complete the acquired CPRR consumer on its own stream")
        event = torch.cuda.Event()
        event.record(stream)
        self.completed[layer] = event
        del self.active[layer]

    def release(self, identity):
        self._validate(identity)
        if self.active or self.completed.keys() != self.consumers:
            raise ValueError("CPRR source still has unfinished consumers")
        stream = torch.cuda.current_stream(self.device)
        for event in self.completed.values():
            stream.wait_event(event)
        for tensor in self.resources:
            tensor.record_stream(stream)
        self.released = True
