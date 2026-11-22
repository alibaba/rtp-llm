"""Decode and Verify Graph attention over the native packed KV pools."""

from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value as optimization_env_value
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import (
    enabled,
    load_packed_backend,
    packed_pool_view,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_decode import packed_decode

GRAPH_PACKED_ATTENTION_ENV = "RTP_V41_PPU_GRAPH_PACKED_ATTENTION"
VERIFY_PACKED_ATTENTION_ENV = "RTP_V41_PPU_VERIFY_PACKED_ATTENTION"
GRAPH_VISIBLE_INDEXER_ENV = "RTP_V41_PPU_GRAPH_VISIBLE_INDEXER"
GRAPH_SHARED_MLA_METADATA_ENV = "RTP_V41_PPU_GRAPH_SHARED_MLA_METADATA"
VERIFY_DIRECT_JOURNAL_ENV = "RTP_V41_PPU_VERIFY_DIRECT_JOURNAL"
_PACKED_WIDTHS = {"swa_kv": 528, "global_kv_1": 288, "global_kv_2": 288}


def _can_fuse_tail_copies(pages, payloads):
    if (optimization_env_value("RTP_V41_PPU_DECODE_LAUNCH_REUSE", "0") != "1"
            or len(pages) not in (1, 2) or len(pages) != len(payloads)):
        return False
    tensors = [*pages, *payloads]
    if not all(
        type(t) is torch.Tensor and t.is_cuda and t.dtype == torch.uint8
        and t.ndim == 2 and 0 < t.numel() <= torch.iinfo(torch.int32).max
        and t.shape[1] in (288, 528) and t.is_contiguous()
        and t.device == pages[0].device for t in tensors
    ) or any(page.shape != payload.shape for page, payload in zip(pages, payloads)):
        return False
    # Each destination and source must occupy a distinct byte interval. Keep
    # ordered copy_ semantics for overlapping or custom/strided adapters.
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel()) for t in tensors)
    return all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


@lru_cache(maxsize=1)
def _tail_copy_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def backup_and_stage(page0, payload0, backup0, page1, payload1, backup1,
                         SIZE0: tl.constexpr, SIZE1: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        if tl.program_id(0) == 0:
            old = tl.load(page0 + offset, offset < SIZE0, other=0)
            new = tl.load(payload0 + offset, offset < SIZE0, other=0)
            tl.store(backup0 + offset, old, offset < SIZE0)
            tl.store(page0 + offset, new, offset < SIZE0)
        elif SIZE1:
            old = tl.load(page1 + offset, offset < SIZE1, other=0)
            new = tl.load(payload1 + offset, offset < SIZE1, other=0)
            tl.store(backup1 + offset, old, offset < SIZE1)
            tl.store(page1 + offset, new, offset < SIZE1)

    @triton.jit
    def restore_tail(page0, backup0, page1, backup1,
                     SIZE0: tl.constexpr, SIZE1: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        if tl.program_id(0) == 0:
            old = tl.load(backup0 + offset, offset < SIZE0, other=0)
            tl.store(page0 + offset, old, offset < SIZE0)
        elif SIZE1:
            old = tl.load(backup1 + offset, offset < SIZE1, other=0)
            tl.store(page1 + offset, old, offset < SIZE1)

    @triton.jit
    def stage_tail(page0, payload0, page1, payload1,
                   SIZE0: tl.constexpr, SIZE1: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        if tl.program_id(0) == 0:
            tl.store(page0 + offset, tl.load(payload0 + offset, offset < SIZE0, other=0), offset < SIZE0)
        elif SIZE1:
            tl.store(page1 + offset, tl.load(payload1 + offset, offset < SIZE1, other=0), offset < SIZE1)

    @triton.jit
    def copy_reserved_pages(pages, backups, sizes, RESTORE: tl.constexpr, BLOCK: tl.constexpr):
        page_id = tl.program_id(0)
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        size = tl.load(sizes + page_id)
        page = tl.load(pages + page_id).to(tl.pointer_type(tl.uint8))
        backup = tl.load(backups + page_id).to(tl.pointer_type(tl.uint8))
        if RESTORE:
            tl.store(page + offset, tl.load(backup + offset, offset < size, other=0), offset < size)
        else:
            tl.store(backup + offset, tl.load(page + offset, offset < size, other=0), offset < size)

    @triton.jit
    def exchange_reserved_pages(pages, backups, payloads, sizes, RESTORE: tl.constexpr, BLOCK: tl.constexpr):
        page_id = tl.program_id(0)
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = offset < tl.load(sizes + page_id)
        page = tl.load(pages + page_id).to(tl.pointer_type(tl.uint8))
        backup = tl.load(backups + page_id).to(tl.pointer_type(tl.uint8))
        payload = tl.load(payloads + page_id).to(tl.pointer_type(tl.uint8))
        old = tl.load(page + offset, live, 0)
        if RESTORE:
            tl.store(payload + offset, old, live)
            tl.store(page + offset, tl.load(backup + offset, live, 0), live)
        else:
            tl.store(backup + offset, old, live)
            tl.store(page + offset, tl.load(payload + offset, live, 0), live)

    return triton, backup_and_stage, restore_tail, stage_tail, copy_reserved_pages, exchange_reserved_pages


@contextmanager
def _fused_tail_copies(pages, payloads):
    triton, stage, restore, _, _, _ = _tail_copy_kernels()
    backups = [torch.empty_like(page) for page in pages]
    paired = len(pages) == 2
    page1, payload1, backup1 = (pages[1], payloads[1], backups[1]) if paired else (None, None, None)
    size0, size1 = pages[0].numel(), pages[1].numel() if paired else 0
    grid = (len(pages), triton.cdiv(max(size0, size1), 1024))
    # Compilation/enqueue failure occurs before entering the transaction, so
    # uninitialized backups are never restored. The one launch writes every
    # backup and staged destination before the same stream reaches the consumer.
    stage[grid](pages[0], payloads[0], backups[0], page1, payload1, backup1,
                size0, size1, 1024, num_warps=4)
    try:
        yield
    finally:
        restore[grid](pages[0], backups[0], page1, backup1, size0, size1, 1024, num_warps=4)


@lru_cache(maxsize=1)
def _packed_metadata_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def packed_slots_and_lengths(
        positions, valid, selected, table, slots, lengths, write_slots,
        verify_requests, verify_starts, verify_prefixes, verify_generations,
        POSITION_STRIDE: tl.constexpr, VALID_STRIDE: tl.constexpr,
        SELECTED_ROW_STRIDE: tl.constexpr, SELECTED_COL_STRIDE: tl.constexpr,
        TABLE_ROW_STRIDE: tl.constexpr, TABLE_COL_STRIDE: tl.constexpr,
        TABLE_COLS: tl.constexpr, POOL_BLOCKS: tl.constexpr,
        SPAN: tl.constexpr, ENTRIES: tl.constexpr, WIDTH: tl.constexpr,
        RATIO: tl.constexpr, GLOBAL: tl.constexpr, ROWS: tl.constexpr,
        VERIFY_LAG: tl.constexpr, BLOCK: tl.constexpr,
        STAGE_PAGE0=None, STAGE_PAYLOAD0=None, STAGE_PAGE1=None, STAGE_PAYLOAD1=None,
        STAGE_WIDTH0: tl.constexpr = 0, STAGE_WIDTH1: tl.constexpr = 0,
    ):
        row = tl.program_id(0)
        segment = tl.program_id(1)
        column = segment * BLOCK + tl.arange(0, BLOCK)
        active = tl.load(valid + row * VALID_STRIDE)
        end = tl.load(positions + row * POSITION_STRIDE).to(tl.int64) + 1
        if GLOBAL:
            # Match torch floor division even for negative positions.
            visible = (end - tl.where(end < 0, RATIO - 1, 0)) // RATIO
            key = tl.load(
                selected + row * SELECTED_ROW_STRIDE + column * SELECTED_COL_STRIDE,
                column < WIDTH, other=-1,
            ).to(tl.int64)
            raw = key * RATIO
            live = active & (key >= 0) & (key < visible)
            offset = (raw % SPAN) // RATIO
        else:
            visible = end
            raw = tl.maximum(end - WIDTH, 0) + column
            live = active & (column < visible)
            offset = raw % ENTRIES
        page = raw // SPAN
        mapped = live & (column < WIDTH) & (raw >= 0) & (page < TABLE_COLS)
        block = tl.load(
            table + row * TABLE_ROW_STRIDE + page * TABLE_COL_STRIDE,
            mapped, other=0,
        ).to(tl.int64)
        mapped = mapped & (block > 0) & (block < POOL_BLOCKS) & (offset < ENTRIES)
        slot = tl.where(mapped, block * ENTRIES + offset, -1)
        if verify_generations is not None:
            request = tl.load(verify_requests + row)
            start = tl.load(verify_starts + request)
            prefix = tl.load(verify_prefixes + request)
            relative = raw + VERIFY_LAG - prefix
            candidate = tl.minimum(tl.maximum(start + relative, 0), ROWS - 1)
            tentative = (
                mapped & (relative >= 0)
                & tl.load(valid + candidate * VALID_STRIDE)
                & (tl.load(verify_requests + candidate) == request)
                & (tl.load(verify_generations + candidate) == raw)
            )
            # Candidate rows occupy offsets [0, ROWS) of reserved block 0.
            slot = tl.where(tentative, candidate, slot)
        tl.store(
            slots + row * WIDTH + column,
            slot,
            column < WIDTH,
        )
        if segment == 0:
            # FlashMLA skips sentinel slots inside the positional prefix.
            # Missing pages and rejected TopK IDs do not shorten that prefix.
            if lengths is not None:
                tl.store(lengths + row, tl.where(active, tl.minimum(visible, WIDTH), 0))
            if write_slots is not None:
                position = end - 1
                write_page = position // SPAN
                writable = active & (position >= 0) & (write_page < TABLE_COLS)
                write_block = tl.load(
                    table + row * TABLE_ROW_STRIDE + write_page * TABLE_COL_STRIDE,
                    writable, other=0,
                ).to(tl.int64)
                writable = writable & (write_block > 0) & (write_block < POOL_BLOCKS)
                tl.store(
                    write_slots + row,
                    tl.where(writable, write_block * ENTRIES + position % ENTRIES, -1),
                )
        if STAGE_WIDTH0:
            value0 = tl.load(STAGE_PAYLOAD0 + row * STAGE_WIDTH0 + column, column < STAGE_WIDTH0, other=0)
            tl.store(STAGE_PAGE0 + row * STAGE_WIDTH0 + column, value0, column < STAGE_WIDTH0)
        if STAGE_WIDTH1:
            value1 = tl.load(STAGE_PAYLOAD1 + row * STAGE_WIDTH1 + column, column < STAGE_WIDTH1, other=0)
            tl.store(STAGE_PAGE1 + row * STAGE_WIDTH1 + column, value1, column < STAGE_WIDTH1)

    return triton, packed_slots_and_lengths


class GraphPackedAttention:
    """Bind the provider and zero-copy pool views before bucket capture."""

    @staticmethod
    def supports_verify(cache, rows):
        return all(
            rows <= cache.entries[cache._table_tag(owner, tag)]
            for owner, tag in cache.geometry.pool_keys() if tag in _PACKED_WIDTHS
        )

    def __init__(self, cache, geometry, *, journal=None):
        backend, _ = load_packed_backend()
        self.native = backend.flash_mla_with_kvcache
        self.metadata_factory = backend.get_mla_metadata
        self.share_mla_metadata = enabled(GRAPH_SHARED_MLA_METADATA_ENV, default=True)
        self.cache = cache
        self.journal = journal
        g = geometry
        if g.window != 128 or g.head_dim != 512 or not 0 < g.n_heads <= 128:
            raise ValueError("unsupported PPU packed Decode Graph geometry")
        if journal is not None and not self.supports_verify(cache, journal.rows):
            raise ValueError("packed Verify Graph exceeds the reserved page capacity")
        self.offsets = torch.arange(g.window, dtype=torch.int64, device=cache.device)
        self.views = {}
        self.swa_domains = {}
        self._tail_transaction_active = False
        self._tail_transaction = None
        self._direct_journal = None
        for owner, tag in cache.geometry.pool_keys():
            if tag not in _PACKED_WIDTHS:
                continue
            tensor = cache.pool(owner, tag)
            domain = cache._table_tag(owner, tag)
            entries = cache.entries[domain]
            self.views[owner, tag] = packed_pool_view(
                SimpleNamespace(tensor=tensor, entries=entries), _PACKED_WIDTHS[tag]
            )
            if tag == "swa_kv":
                self.swa_domains[owner] = (domain, tensor.shape[0])
        if (journal is not None
            and optimization_env_value("RTP_V41_PPU_DECODE_LAUNCH_REUSE", "0") == "1"
            and optimization_env_value("RTP_V41_PPU_VERIFY_BATCHED_JOURNAL", "0") == "1"):
            pages = [view[0, :journal.rows, 0] for view in self.views.values()]
            payloads = [journal.payloads[key] for key in self.views]
            tensors = [*pages, *payloads]
            spans = sorted((t.data_ptr(), t.data_ptr() + t.numel()) for t in tensors)
            if (
                pages and all(_can_fuse_tail_copies([p], [v]) for p, v in zip(pages, payloads))
                and all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
            ):
                backups = [torch.empty_like(page) for page in pages]
                self._tail_transaction = SimpleNamespace(
                    pages=pages, backups=backups,
                    page_ptrs=torch.tensor([p.data_ptr() for p in pages], dtype=torch.int64, device=cache.device),
                    backup_ptrs=torch.tensor([p.data_ptr() for p in backups], dtype=torch.int64, device=cache.device),
                    sizes=torch.tensor([p.numel() for p in pages], dtype=torch.int32, device=cache.device),
                    max_size=max(p.numel() for p in pages),
                )
                if optimization_env_value(VERIFY_DIRECT_JOURNAL_ENV, "0") == "1":
                    from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal

                    # The publication descriptor proves that the full
                    # native pool ranges do not alias (not just block zero).
                    if (
                        type(journal) is VerifyGraphJournal
                        and journal._batched_publication is not None
                    ):
                        private = journal.payloads
                        self._direct_journal = SimpleNamespace(
                            private=private,
                            mapped={**private, **dict(zip(self.views, pages))},
                            payload_ptrs=torch.tensor(
                                [p.data_ptr() for p in payloads], dtype=torch.int64, device=cache.device,
                            ),
                        )

    @contextmanager
    def verify_tail_transaction(self):
        """Borrow block 0 for one serialized Verify, restoring even on failure."""
        transaction = getattr(self, "_tail_transaction", None)
        if transaction is None:
            yield
            return
        if self._tail_transaction_active:
            raise RuntimeError("packed Verify journal transaction is already active")
        triton, _, _, _, copy, exchange = _tail_copy_kernels()
        grid = (len(transaction.pages), triton.cdiv(transaction.max_size, 1024))
        direct = getattr(self, "_direct_journal", None)
        if direct is None:
            copy[grid](transaction.page_ptrs, transaction.backup_ptrs, transaction.sizes, False, 1024)
        else:
            if self.journal.payloads is not direct.private:
                raise RuntimeError("Verify journal payload ownership changed")
            exchange[grid](transaction.page_ptrs, transaction.backup_ptrs,
                           direct.payload_ptrs, transaction.sizes, False, 1024)
            self.journal.payloads = direct.mapped
        self._tail_transaction_active = True
        try:
            yield
        finally:
            try:
                if direct is None:
                    copy[grid](transaction.page_ptrs, transaction.backup_ptrs, transaction.sizes, True, 1024)
                else:
                    self.journal.payloads = direct.private
                    exchange[grid](transaction.page_ptrs, transaction.backup_ptrs,
                                   direct.payload_ptrs, transaction.sizes, True, 1024)
            finally:
                self._tail_transaction_active = False

    def _native_indices(
        self, owner, tag, positions, valid, width, *, selected=None, ratio=1,
        write_slots=None, shared=None, stage=None, verify_generations=None,
    ):
        cache = self.cache
        if (
            type(cache) is not NativeGraphCache
            or not positions.is_cuda
            or positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or valid.dtype != torch.bool
            or valid.shape != positions.shape
            or valid.device != positions.device
            or width <= 0
            or not positions.numel()
        ):
            return None
        rows = positions.numel()
        if selected is not None and (
            ratio not in (1, 2)
            or cache.geometry.ratio(owner) != ratio
            or selected.shape != (rows, width)
            or selected.dtype not in (torch.int32, torch.int64)
            or selected.device != positions.device
        ):
            return None
        domain = cache._table_tag(owner, tag)
        journal = self.journal
        table = cache.tables[domain] if journal is None else journal.row_tables[domain]
        if (
            table.device != positions.device
            or table.dtype not in (torch.int32, torch.int64)
            or table.ndim != 2
            or table.shape[0] < rows
            or not table.shape[1]
        ):
            return None
        slots = torch.empty((rows, 1, width), dtype=torch.int32, device=positions.device)
        # This native kernel's prefix is a function of the request positions,
        # valid mask, compression ratio and width. Physical mappings, missing
        # pages and selected IDs can differ between layers without changing it.
        prefix = ("packed_prefix", id(positions), id(valid), selected is not None, ratio, width)
        lengths = shared.get(prefix) if shared is not None else None
        length_output = None
        if lengths is None:
            lengths = length_output = torch.empty(rows, dtype=torch.int32, device=positions.device)
            if shared is not None:
                shared[prefix] = lengths
        triton, kernel = _packed_metadata_kernel()
        block = 1024 if stage is not None else 128
        options = {}
        if stage is not None:
            pages, payloads = stage
            if not self._tail_transaction_active or not _can_fuse_tail_copies(pages, payloads):
                raise RuntimeError("fused Verify staging requires an active disjoint journal transaction")
            options = dict(STAGE_PAGE0=pages[0], STAGE_PAYLOAD0=payloads[0],
                           STAGE_WIDTH0=pages[0].shape[1])
            if len(pages) == 2:
                options.update(STAGE_PAGE1=pages[1], STAGE_PAYLOAD1=payloads[1],
                               STAGE_WIDTH1=pages[1].shape[1])
        kernel[(rows, triton.cdiv(width, block))](
            positions, valid, selected, table, slots, length_output, write_slots,
            None if journal is None else journal.requests,
            None if journal is None else journal.starts,
            None if journal is None else journal.prefixes,
            (None if journal is None else journal.generations[owner, tag])
            if verify_generations is None else verify_generations,
            positions.stride(0), valid.stride(0),
            0 if selected is None else selected.stride(0),
            0 if selected is None else selected.stride(1),
            table.stride(0), table.stride(1), table.shape[1],
            cache.pool(owner, tag).shape[0], cache.spans[domain], cache.entries[domain],
            width, ratio, selected is not None, rows, int(selected is not None and ratio == 2),
            block, num_warps=4, **options,
        )
        return slots, lengths

    def _swa_metadata(self, layer_id, positions, valid, shared, *, stage=None, verify_slots=None):
        journal = self.journal
        key = (
            "packed_swa", id(positions), id(valid), self.swa_domains[layer_id],
            layer_id if journal is not None and verify_slots is None else None,
            id(verify_slots) if verify_slots is not None else None,
        )
        if key not in shared or stage is not None:
            width = self.offsets.numel()
            writes = None if journal is not None else torch.empty_like(positions, dtype=torch.int64)
            indices = self._native_indices(
                layer_id, "swa_kv", positions, valid, width, write_slots=writes, shared=shared,
                stage=stage, verify_generations=verify_slots,
            )
            if indices is None:
                if journal is not None:
                    raise ValueError("invalid paged SWA metadata for packed Verify")
                end = positions + 1
                lengths = torch.where(valid, end.clamp_max(width), 0).int()
                raw = (end - width).clamp_min(0)[:, None] + self.offsets[None]
                raw = torch.where(self.offsets[None] < lengths[:, None], raw, -1)
                slots = self.cache.slots(layer_id, "swa_kv", raw).int()
                indices = slots[:, None], lengths
                writes = self.cache.slots(
                    layer_id, "swa_kv", torch.where(valid, positions, -1)
                )
            shared[key] = (*indices, writes)
        return shared[key]

    def _swa_indices(self, layer_id, positions, valid, shared, *, stage=None, verify_slots=None):
        slots, lengths, _ = self._swa_metadata(
            layer_id, positions, valid, shared, stage=stage, verify_slots=verify_slots,
        )
        return slots, lengths

    def swa_write_slots(self, layer_id, positions, valid, shared):
        return self._swa_metadata(layer_id, positions, valid, shared)[2]

    def _global_indices(self, source, index_source, tag, ratio, selected, positions, valid, shared):
        key = ("packed_global", id(positions), id(valid), source, index_source)
        if key not in shared:
            indices = self._native_indices(
                source, tag, positions, valid, selected.shape[-1],
                selected=selected, ratio=ratio, shared=shared,
            )
            if indices is None:
                if self.journal is not None:
                    raise ValueError("invalid paged global KV metadata for packed Verify")
                visible = (positions + 1) // ratio
                lengths = torch.where(valid, visible.clamp_max(selected.shape[-1]), 0).int()
                live = valid[:, None] & (selected >= 0) & (selected < visible[:, None])
                raw = torch.where(live, selected.long() * ratio, -1)
                slots = self.cache.slots(source, tag, raw).int()
                indices = slots[:, None], lengths
            shared[key] = indices
        return shared[key]

    @contextmanager
    def _stage_verify_tail(self, keys, *, already_staged=False):
        if self.journal is None:
            yield
            return
        if getattr(self, "_direct_journal", None) is not None and self._tail_transaction_active:
            yield
            return
        # The framework never allocates block 0 to a live request. The
        # synchronous Verify stream borrows it only across this MLA call,
        # then restores its previous bytes for other Graph modes.
        pages = [self.views[key][0, :self.journal.rows, 0] for key in keys]
        payloads = [self.journal.payloads[key] for key in keys]
        if getattr(self, "_tail_transaction_active", False):
            if not _can_fuse_tail_copies(pages, payloads):
                raise RuntimeError("packed Verify journal storage changed inside transaction")
            if already_staged:
                yield
                return
            triton, _, _, stage, _, _ = _tail_copy_kernels()
            size0 = pages[0].numel()
            size1 = pages[1].numel() if len(pages) == 2 else 0
            stage[(len(pages), triton.cdiv(max(size0, size1), 1024))](
                pages[0], payloads[0], pages[1] if size1 else None,
                payloads[1] if size1 else None, size0, size1, 1024,
            )
            yield
            return
        if _can_fuse_tail_copies(pages, payloads):
            with _fused_tail_copies(pages, payloads):
                yield
            return
        backups = [page.clone() for page in pages]
        try:
            for key, page in zip(keys, pages):
                page.copy_(self.journal.payloads[key])
            yield
        finally:
            for page, backup in zip(pages, backups):
                page.copy_(backup)

    def forward(self, attn, q, selected, metadata, shared, *, mask_output=True, share_scheduler=False,
                mega=None, mega_phase="decode", verify_slots=None):
        positions, valid = metadata.positions, metadata.valid_mask
        keys = [(attn.layer_id, "swa_kv")]
        global_view = global_indices = global_lengths = None
        if selected is not None:
            source, ratio = attn.kv_source_layer_id, attn.compress_ratio
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            keys.append((source, tag))
            global_view = self.views[source, tag]
        from rtp_llm.models_py.modules.dsv41.ppu.decode_fusions import enabled as fusion_enabled

        stage = None
        active = getattr(self, "_tail_transaction_active", False)
        direct = active and getattr(self, "_direct_journal", None) is not None
        if fusion_enabled("JOURNAL") and active and not direct:
            pages = [self.views[key][0, :self.journal.rows, 0] for key in keys]
            payloads = [self.journal.payloads[key] for key in keys]
            stage = pages, payloads
        if stage is None:
            swa_indices, swa_lengths = self._swa_indices(
                attn.layer_id, positions, valid, shared, **({"verify_slots": verify_slots}
                if direct and verify_slots is not None else {}),
            )
        else:
            swa_indices, swa_lengths = self._swa_indices(attn.layer_id, positions, valid, shared, stage=stage)
        if selected is not None:
            global_indices, global_lengths = self._global_indices(
                source, attn.index_source_layer_id, tag, ratio, selected,
                positions, valid, shared,
            )
        # The provider masks -1 slots inside each live prefix, including
        # unmapped warmup pages. Both prefixes keep their original key order.
        # Share only while building a graph. A fresh per-forward
        # dictionary makes the first layer of each group capture the provider
        # scheduler, whose device work then runs on every replay.
        scheduler_cache = None
        if (
            (share_scheduler or self.journal is not None)
            and self.share_mla_metadata
            and type(self.cache) is NativeGraphCache
            and q.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            scheduler_cache = shared
        plan = SimpleNamespace(
            native=self.native,
            metadata_factory=self.metadata_factory,
            swa_cache=self.views[attn.layer_id, "swa_kv"],
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            global_cache=global_view,
            global_indices=global_indices,
            global_lengths=global_lengths,
            valid_mask=valid,
        )
        if mega is not None:
            if mega.decode_reason(q, plan, positions, phase=mega_phase) is not None:
                return None
            from rtp_llm.models_py.modules.dsv41.ppu.mega_attention import PPUMegaAttention

            options = {"shared": scheduler_cache} if type(mega) is PPUMegaAttention else {}
            with self._stage_verify_tail(keys, **({"already_staged": True} if stage is not None else {})):
                return mega.decode(
                    q, plan, positions, packed_q=True,
                    mask_output=mask_output, phase=mega_phase, **options,
                )
        with self._stage_verify_tail(keys, **({"already_staged": True} if stage is not None else {})):
            return packed_decode(
                q, plan, softmax_scale=attn.geometry.sm_scale(), attn_sink=attn.attn_sink,
                mask_output=mask_output, scheduler_cache=scheduler_cache,
            )
