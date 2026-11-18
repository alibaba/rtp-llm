"""Forward-local page-planar FP4 Indexer readback for CP Prefill.

Pages use the existing native 256-entry ABI: 256x64 payload bytes followed
by 256x4 scale bytes. The public 4D view is the native paged-provider ABI,
not an assertion that the packed bytes are token-interleaved.
"""

from dataclasses import dataclass

import torch


PAGE = 256
WIDTH = 68


@dataclass(frozen=True)
class IndexerPageLayout:
    prefix_rows: int
    request_counts: tuple[int, ...]
    tail_page_bases: tuple[int, ...]
    request_table_host: tuple[tuple[int, ...], ...]
    pages: int


def page_layout(counts, prefix_rows):
    counts = tuple(counts)
    if (type(prefix_rows) is not int or prefix_rows <= 0 or prefix_rows % PAGE
            or len(counts) < 2 or any(type(count) is not int or count < prefix_rows for count in counts)):
        raise ValueError("paged Indexer readback requires a page-aligned shared prefix")
    prefix_pages = prefix_rows // PAGE
    offsets, tables, total = [], [], prefix_pages
    maximum = max((count + PAGE - 1) // PAGE for count in counts)
    for count in counts:
        tails = (count - prefix_rows + PAGE - 1) // PAGE
        offsets.append(total)
        table = tuple(range(prefix_pages)) + tuple(range(total, total + tails))
        # Page zero is a valid common-prefix page. Logical lengths exclude
        # these ragged-table padding columns before a score can consume them.
        tables.append(table + (0,) * (maximum - len(table)))
        total += tails
    return IndexerPageLayout(prefix_rows, counts, tuple(offsets), tuple(tables), total)


@dataclass(frozen=True)
class IndexerReadbackPages:
    storage: torch.Tensor
    pages: torch.Tensor
    layout: IndexerPageLayout

    @classmethod
    def create(cls, storage, counts, prefix_rows):
        layout = page_layout(counts, prefix_rows)
        if not (isinstance(storage, torch.Tensor) and storage.dtype == torch.uint8
                and storage.ndim == 2 and storage.shape == (layout.pages, PAGE * WIDTH)
                and storage.is_contiguous()):
            raise ValueError("Indexer readback storage must contain complete planar256 pages")
        with torch.inference_mode(False):
            pages = storage.view(layout.pages, PAGE, 1, WIDTH)
        return cls(storage, pages, layout)


@dataclass(frozen=True)
class SharedIndexerBank:
    pages: torch.Tensor
    prefix_rows: int
    request_counts: tuple[int, ...]
    tail_page_bases: tuple[int, ...]
    request_table: torch.Tensor
    request_table_host: tuple[tuple[int, ...], ...]
    cache: object
    source: int
    epoch: int
    owners: tuple

    @classmethod
    def from_readback(cls, readback, cache, source, epoch, request_table, *, owners=()):
        if type(readback) is not IndexerReadbackPages:
            raise ValueError("shared Indexer bank requires a native readback receipt")
        layout = readback.layout
        if not (isinstance(request_table, torch.Tensor) and request_table.dtype == torch.int32
                and request_table.device == readback.pages.device and request_table.is_contiguous()
                and request_table.shape == (len(layout.request_counts), len(layout.request_table_host[0]))
                and type(source) is int and type(epoch) is int):
            raise ValueError("shared Indexer request table differs from its host layout")
        return cls(readback.pages, layout.prefix_rows, layout.request_counts, layout.tail_page_bases,
                   request_table, layout.request_table_host, cache, source, epoch,
                   (readback.storage, readback.pages, request_table, *owners))

    def valid(self):
        validate = getattr(self.cache, "_valid_shared_indexer", None)
        return callable(validate) and validate(self)

    def materialize(self, request, count, *, split=False):
        """Independent snapshot bytes for legacy consumers and verification.

        Cache readers validate the current receipt first. Verification may
        inspect the immutable readback before it is published as current.
        """
        if (type(request) is not int or not 0 <= request < len(self.request_counts)
                or type(count) is not int or not 0 <= count <= self.request_counts[request]):
            raise ValueError("Indexer request range is outside its readback bank")
        logical = torch.arange(count, device=self.pages.device, dtype=torch.int64)
        physical = self.request_table[request, logical // PAGE].long()
        entry = logical % PAGE
        storage = self.pages.view(self.pages.shape[0], PAGE * WIDTH)
        payload = storage[:, :PAGE * 64].reshape(-1, PAGE, 64)[physical, entry].contiguous()
        scales = storage[:, PAGE * 64:].reshape(-1, PAGE, 4)[physical, entry].contiguous()
        if split:
            return payload.view(torch.int8), scales.view(torch.int32).reshape(-1)
        return torch.cat((payload, scales), dim=1)


@dataclass(frozen=True)
class SharedIndexerSlice:
    bank: SharedIndexerBank
    request: int


def materialize_indexer(value, count, *, split, stats):
    if type(value) is not SharedIndexerSlice:
        raise TypeError("expected a shared Indexer slice")
    if not value.bank.valid():
        raise RuntimeError("stale CP shared Indexer readback")
    stats["materializations"] += 1
    stats["materialized_rows"] += count
    stats["materialized_bytes"] += count * WIDTH
    return value.bank.materialize(value.request, count, split=split)
