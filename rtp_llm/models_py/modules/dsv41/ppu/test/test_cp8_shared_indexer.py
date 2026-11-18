"""CP8 Indexer page ownership, readback bytes and cache lifetime contracts."""

from datetime import timedelta
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_shared_global as shared_global
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_shared_indexer as shared_indexer
from rtp_llm.models_py.modules.dsv41.ppu import prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache, _source_tensor_key


def _rows(first, count, request, device="cpu"):
    positions = torch.arange(first, first + count, device=device)[:, None]
    columns = torch.arange(68, device=device)[None]
    return ((positions * 13 + columns * 3 + request * 37) % 251).to(torch.uint8)


def _readback(counts=(529, 513, 256, 769), prefix=256, device="cpu"):
    layout = shared_indexer.page_layout(counts, prefix)
    storage = torch.zeros((layout.pages, 256 * 68), dtype=torch.uint8, device=device)

    def store(page, rows):
        count = rows.shape[0]
        storage[page, :count * 64].copy_(rows[:, :64].reshape(-1))
        storage[page, 256 * 64:256 * 64 + count * 4].copy_(rows[:, 64:].reshape(-1))

    for page in range(prefix // 256):
        store(page, _rows(page * 256, 256, 0, device))
    expected = []
    for request, count in enumerate(counts):
        tail = _rows(prefix, count - prefix, request + 1, device)
        for first in range(0, tail.shape[0], 256):
            store(layout.tail_page_bases[request] + first // 256, tail[first:first + 256])
        expected.append(torch.cat((_rows(0, prefix, 0, device), tail)))
    return shared_indexer.IndexerReadbackPages.create(storage, counts, prefix), tuple(expected)


def _stats():
    return dict(enabled=False, enable_calls=0, banks=0, read_hits=0, pages=0,
                prefix_rows=0, tail_rows=0, restore_rows=0, page_padding_rows=0,
                logical_rows=0, prefix_copy_bytes_avoided=0, materializations=0,
                materialized_rows=0, materialized_bytes=0, invalidations=0)


def _cache(readback):
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache._is_prefill, cache.sharded, cache.device = True, True, readback.pages.device
    cache.cp = dict(size=8, rank=0)
    cache.lengths = readback.layout.request_counts
    cache.geometry = NS(ratio=lambda source: 1)
    cache._source_reads, cache._local_source_reads, cache._fresh_swa = {}, {}, {}
    cache._cp_shared_global_reads, cache._cp_shared_indexer_reads = {}, {}
    cache._cp_shared_global_epoch = 7
    cache._cp_shared_indexer_consumer_ready = False
    cache._cp_shared_indexer_stats = _stats()
    cache._cp_shared_global_stats = dict(enabled=False, banks=0, logical_rows=0, packed_rows=0,
                                        prefix_copy_bytes_avoided=0, legacy_materializations=0,
                                        invalidations=0)
    cache._cp_shared_prefix_signature = lambda key: (key[1] + "-proof", (readback.storage,))
    cache._local_source_cache_eligible = lambda: False
    cache.prefill_plans = {}
    return cache


def _publish(cache, readback, source=20, *, owners=()):
    with torch.inference_mode(False):
        table = torch.tensor(readback.layout.request_table_host, dtype=torch.int32, device=readback.pages.device)
    bank = shared_indexer.SharedIndexerBank.from_readback(
        readback, cache, source, cache._cp_shared_global_epoch, table, owners=owners,
    )
    signature = cache._cp_shared_prefix_signature((source, "indexer_kv"))[0]
    cache._cp_shared_indexer_reads[source] = bank, signature, cache._shared_indexer_metadata(bank)
    for request, count in enumerate(bank.request_counts):
        cache._source_reads[source, request] = (
            torch.zeros((count, 288), dtype=torch.uint8, device=bank.pages.device),
            shared_indexer.SharedIndexerSlice(bank, request),
        )
    return bank


def _padding_is_zero(bank):
    raw = bank.pages.view(-1, 256 * 68)
    for request, count in enumerate(bank.request_counts):
        tail = count - bank.prefix_rows
        live = tail % 256
        if live:
            page = bank.tail_page_bases[request] + tail // 256
            if bool(raw[page, live * 64:256 * 64].any()) or bool(raw[page, 256 * 64 + live * 4:].any()):
                return False
    return True


class CP8SharedIndexerContracts(unittest.TestCase):
    def test_page_layout_has_only_shared_prefix_and_private_whole_tail_pages(self):
        layout = shared_indexer.page_layout((256, 257, 511, 512, 513), 256)
        self.assertEqual(layout.tail_page_bases, (1, 1, 2, 3, 4))
        self.assertEqual(layout.pages, 6)
        self.assertEqual(layout.request_table_host,
                         ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 5)))
        private = []
        for count, table in zip(layout.request_counts, layout.request_table_host):
            used = table[1:(count + 255) // 256]
            self.assertTrue(all(0 < page < layout.pages for page in used))
            private.extend(used)
        self.assertEqual(len(private), len(set(private)))
        for counts, prefix in (((512, 512), 0), ((512, 512), 255), ((512, 512), 257),
                               ((255, 512), 256), ((512,), 256), ((True, 512), 256)):
            with self.subTest(counts=counts, prefix=prefix), self.assertRaises(ValueError):
                shared_indexer.page_layout(counts, prefix)

    def test_native_planar_pages_match_legacy_packed_and_split_bytes_exactly(self):
        readback, expected = _readback()
        cache = _cache(readback)
        bank = _publish(cache, readback)
        self.assertEqual(tuple(bank.pages.shape), (8, 256, 1, 68))
        self.assertTrue(bank.valid())
        self.assertTrue(_padding_is_zero(bank))
        self.assertTrue(any(owner is readback.storage for owner in bank.owners))
        for request, count in enumerate(bank.request_counts):
            for length in (0, 255, 256, min(257, count), count):
                with self.subTest(request=request, count=length):
                    packed = bank.materialize(request, length)
                    payload, scales = bank.materialize(request, length, split=True)
                    self.assertTrue(torch.equal(packed, expected[request][:length]))
                    self.assertTrue(torch.equal(payload.view(torch.uint8), packed[:, :64]))
                    self.assertTrue(torch.equal(scales.view(torch.uint8).reshape(-1, 4), packed[:, 64:]))
                    self.assertEqual((payload.dtype, scales.dtype), (torch.int8, torch.int32))
                    self.assertFalse(torch._C._is_alias_of(packed, bank.pages))
                    self.assertFalse(torch._C._is_alias_of(payload, bank.pages))
                    self.assertFalse(torch._C._is_alias_of(scales, bank.pages))
        bank.materialize(0, 256).zero_()
        self.assertTrue(torch.equal(bank.materialize(1, 256), expected[1][:256]))

    def test_restore_plan_preserves_packets_and_pads_only_the_last_private_stripe(self):
        for counts in ((529, 529), (256, 257, 513, 529)):
            with self.subTest(counts=counts):
                groups = tuple(prefill_readback._stripe_groups(counts, 64, 3, 4, indexer_pages=True))
                pages = shared_indexer.page_layout(counts, 256)
                visited = []
                for size, rows, group in groups:
                    self.assertEqual(size, len(group))
                    self.assertEqual(rows, max(record[4] for record in group))
                    for request, column, count, destination, padded in group:
                        if padded != count:
                            self.assertGreaterEqual(column * 64, 256)
                            self.assertEqual(column * 64 + count, counts[request])
                            self.assertEqual((destination + padded) % 256, 0)
                        visited.extend(range(destination, destination + padded))
                self.assertEqual(sorted(visited), list(range(pages.pages * 256)))
                self.assertEqual(len(visited), len(set(visited)))
                if len(set(counts)) == 1:
                    legacy = [record[:3] for _, _, group in prefill_readback._stripe_groups(counts, 64, 3, 4)
                              for record in group]
                    actual = [record[:3] for _, _, group in groups for record in group]
                    self.assertEqual(actual, legacy)
        no_share = [record for _, _, group in prefill_readback._stripe_groups((529, 513), 64, 3)
                    for record in group]
        self.assertEqual(sum(record[2] for record in no_share), 1042)
        for stripes in (0, 3):
            with self.assertRaises(ValueError):
                tuple(prefill_readback._stripe_groups((529, 529), 64, 3, stripes, indexer_pages=True))

    def test_capability_is_explicit_and_reads_never_initiate_collectives(self):
        readback, _ = _readback((529, 529))
        cache = _cache(readback)
        bank = _publish(cache, readback)
        key = 20, "indexer_kv"
        cache._cp_shared_global_eligible = lambda key: True
        cache.shared_prefix_stripes = {key: 1}
        cache.placements = {key: NS(size=8, physical_span=32, ratio=1)}
        self.assertFalse(cache._cp_shared_indexer_eligible(key))
        with patch("torch.distributed.all_reduce", side_effect=AssertionError("no implicit collective")), \
                patch("torch.distributed.all_gather_into_tensor", side_effect=AssertionError("no implicit collective")):
            cache.enable_shared_indexer()
            self.assertTrue(cache._cp_shared_indexer_eligible(key))
            self.assertIs(cache.read_shared_indexer(20), bank)
            self.assertIsNone(cache.read_shared_indexer(99))
            self.assertTrue(bank.valid())
        self.assertEqual(cache._cp_shared_indexer_stats["enable_calls"], 1)
        self.assertEqual(cache._cp_shared_indexer_stats["read_hits"], 1)
        cache.placements[key].physical_span = 8
        self.assertFalse(cache._cp_shared_indexer_eligible(key))
        cache.placements[key].physical_span = 32
        cache._cp_shared_global_eligible = lambda key: False
        self.assertFalse(cache._cp_shared_indexer_eligible(key))

    def test_current_receipt_checks_epoch_source_and_page_table_metadata(self):
        for change in ("epoch", "proof", "pages", "table", "shape", "source"):
            with self.subTest(change=change):
                readback, _ = _readback((529, 529))
                cache = _cache(readback)
                bank = _publish(cache, readback)
                if change == "epoch":
                    cache._cp_shared_global_epoch += 1
                elif change == "proof":
                    cache._cp_shared_prefix_signature = lambda key: ("new-pool-or-table", ())
                elif change == "pages":
                    bank.pages.add_(1)
                elif change == "table":
                    bank.request_table.add_(1)
                elif change == "shape":
                    bank.pages.transpose_(1, 2)
                else:
                    cache._cp_shared_indexer_reads[20] = (object(), "indexer_kv-proof", ())
                self.assertFalse(bank.valid())
                with self.assertRaisesRegex(RuntimeError, "stale CP"):
                    shared_indexer.materialize_indexer(shared_indexer.SharedIndexerSlice(bank, 0), 529,
                                                       split=False, stats=cache._cp_shared_indexer_stats)
        readback, _ = _readback((529, 529))
        cache = _cache(readback)
        bank = shared_indexer.SharedIndexerBank.from_readback(
            readback, cache, 20, 7, torch.tensor(readback.layout.request_table_host, dtype=torch.int32),
        )
        self.assertFalse(bank.valid())  # A layout is not a cache-issued current receipt.

    def test_source_preparation_publishes_independent_global_and_indexer_proofs(self):
        readback, _ = _readback((529, 529))
        cache = _cache(readback)
        global_bank = shared_global.SharedGlobalBank.create(torch.zeros((802, 288), dtype=torch.uint8),
                                                           256, (529, 529))
        keys = ((20, "global_kv_1"), (20, "indexer_kv"))
        cache.pools = dict.fromkeys(keys, object())
        cache.placements = dict.fromkeys(keys, object())
        cache.shared_prefix_stripes = dict.fromkeys(keys, 1)
        cache._cp_shared_global_eligible = lambda key: True
        cache._cp_shared_indexer_eligible = lambda key: True
        options = []

        def gather(pool, placement, counts, **kwargs):
            options.append(kwargs)
            return readback if kwargs["planar"] else global_bank

        def upload(values, device, dtype):
            return torch.tensor(values, device=device, dtype=dtype)

        with patch.object(prefill_readback, "gather_pool", side_effect=gather), \
                patch("rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload", side_effect=upload) as uploaded:
            cache._prepare_source_batched(20, verify=False)
            first = cache.read_shared_indexer(20)
            self.assertIs(cache.read_shared_global(20), global_bank)
            self.assertEqual(cache._cp_shared_indexer_reads[20][1], "indexer_kv-proof")
            self.assertTrue(first.valid())
            self.assertEqual([option["planar"] for option in options], [False, True])
            self.assertTrue(options[0]["compact_shared_prefix"])
            self.assertTrue(options[1]["page_planar_indexer"])
            first.request_table.add_(1)
            self.assertFalse(first.valid())
            cache._prepare_source_batched(20, verify=False)
            second = cache.read_shared_indexer(20)
            self.assertTrue(second.valid())
            self.assertFalse(first.valid())
            self.assertTrue(torch.equal(second.request_table,
                                        torch.tensor(readback.layout.request_table_host, dtype=torch.int32)))
            self.assertEqual(uploaded.call_count, 2)
        stats = cache._cp_shared_indexer_stats
        self.assertEqual(stats["banks"], 2)
        self.assertEqual(stats["restore_rows"], 2 * 5 * 256)
        self.assertEqual(stats["tail_rows"], 2 * 2 * 273)
        self.assertEqual(stats["page_padding_rows"], 2 * 478)
        self.assertEqual(stats["prefix_copy_bytes_avoided"], 2 * 256 * 68)
        self.assertEqual(stats["materializations"], 0)

    def test_legacy_read_source_and_split_fp4_remain_functional_and_counted(self):
        readback, expected = _readback((529, 529))
        cache = _cache(readback)
        bank = _publish(cache, readback)
        _, packed = cache.read_source(20, 1, global_entries=529)
        payload, scales = cache.read_indexer_fp4(20, 1, global_entries=257)
        self.assertTrue(torch.equal(packed, expected[1]))
        self.assertTrue(torch.equal(payload.view(torch.uint8), expected[1][:257, :64]))
        self.assertTrue(torch.equal(scales.view(torch.uint8).reshape(-1, 4), expected[1][:257, 64:]))
        self.assertEqual(cache._cp_shared_indexer_stats["materializations"], 2)
        self.assertEqual(cache._cp_shared_indexer_stats["materialized_rows"], 786)
        self.assertEqual(cache._cp_shared_indexer_stats["materialized_bytes"], 786 * 68)
        packed.zero_()
        self.assertTrue(bank.valid())
        self.assertTrue(torch.equal(bank.materialize(0, 529), expected[0]))

    def test_full_pool_writes_invalidate_all_aliases_before_store(self):
        readback, expected = _readback((529, 529))
        cache = _cache(readback)
        first, second = _publish(cache, readback), _publish(cache, readback, 21)
        cache.placements = {}

        def stored(*args, **kwargs):
            self.assertEqual(cache._cp_shared_indexer_reads, {})
            self.assertEqual(cache._source_reads, {})
            self.assertEqual(cache._cp_shared_global_epoch, 8)
            self.assertFalse(first.valid())
            self.assertFalse(second.valid())

        with patch.object(NativeCacheBridge, "_copy_rows", side_effect=stored):
            cache._copy_rows((21, "indexer_kv"), 0, torch.arange(2), torch.empty((2, 68), dtype=torch.uint8))
        self.assertTrue(torch.equal(first.materialize(0, 529), expected[0]))
        self.assertEqual(cache._cp_shared_indexer_stats["invalidations"], 1)
        cache = _cache(readback)
        bank = _publish(cache, readback)
        cache.cp8_row_prep_enabled = lambda: False
        self.assertFalse(cache._try_store_prefill_range((20, "indexer_kv"), 0, 0, expected[0]))
        self.assertFalse(bank.valid())
        self.assertEqual(cache._source_reads, {})

    def test_native_pool_table_and_handle_proof_and_release_own_the_snapshot_lifetime(self):
        readback, expected = _readback((529, 529))
        cache = _cache(readback)
        del cache._cp_shared_prefix_signature
        host = torch.tensor([[1], [1]], dtype=torch.int32)
        table, native = host.clone(), torch.zeros((3, 256 * 68), dtype=torch.uint8)
        handle = NS(kv_cache_base=native)
        pool = NS(framework_tag="indexer", tensor=native, table=table, handle=handle, entries=256, span=256)
        cache.pools = {(20, "indexer_kv"): pool}
        cache.placements = {(20, "indexer_kv"): NS(table=table, size=8, rank=0, physical_span=256, pages=1)}
        cache.inputs = {"indexer": NS(kv_cache_block_id=host, kv_cache_block_id_device=table)}
        signature, owners = cache._cp_shared_prefix_signature((20, "indexer_kv"))
        bank = _publish(cache, readback, owners=owners)
        native.add_(1)  # Explicit native writes invalidate; fresh-data versions are not an alias proof.
        self.assertEqual(cache._cp_shared_prefix_signature((20, "indexer_kv"))[0], signature)
        self.assertTrue(bank.valid())
        host.add_(1)
        self.assertFalse(bank.valid())
        bank = _publish(cache, readback, owners=owners)
        pool.handle = NS(kv_cache_base=native)
        self.assertFalse(bank.valid())
        bank = _publish(cache, readback, owners=owners)
        self.assertFalse(cache._check_local_source_reads())
        self.assertTrue(bank.valid())
        cache._fresh_swa[20, 0] = object()
        cache.release_swa(20)
        self.assertIsNone(cache.read_shared_indexer(20))
        self.assertFalse(bank.valid())
        self.assertEqual(cache._fresh_swa, {})
        self.assertTrue(torch.equal(bank.materialize(1, 529), expected[1]))
        self.assertTrue(any(owner is readback.storage for owner in bank.owners))
        self.assertTrue(any(owner is handle for owner in bank.owners))


def _world8_worker(rank, store):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                            timeout=timedelta(seconds=180))
    try:
        device = torch.device("cuda", rank)
        owner_rows, page_rows, entries, pages, prefix = 64, 32, 64, 2, 512
        page_bytes = entries * 68 + 16
        block_bytes = pages * page_bytes
        for ratio, tail in ((1, 257), (2, 513)):
            counts = (prefix + tail,) * 8
            columns = (counts[0] + 8 * owner_rows - 1) // (8 * owner_rows)
            block_count = 2 + 8 * (columns - 1)
            tensor = torch.full((block_count * pages, page_bytes), 231, dtype=torch.uint8, device=device)
            host_table = [[1] + [2 + request * (columns - 1) + column for column in range(columns - 1)]
                          for request in range(8)]

            def fill(block, first, count, request):
                for page in range(pages):
                    position = first + rank * owner_rows + page * page_rows
                    live = max(0, min(page_rows, count - position))
                    data = _rows(position, live, request, device)
                    tensor[block * pages + page, :live * 64] = data[:, :64].reshape(-1)
                    tensor[block * pages + page, entries * 64:entries * 64 + live * 4] = data[:, 64:].reshape(-1)

            fill(1, 0, prefix, 0)
            for request in range(8):
                for column in range(1, columns):
                    fill(host_table[request][column], column * 8 * owner_rows, counts[request], request + 1)
            table = torch.tensor(host_table, dtype=torch.int32, device=device)
            pool = NS(span=page_rows * ratio, entries=entries, tensor=tensor, table=table,
                      framework_tag="indexer", handle=NS(kv_cache_base=tensor))
            placement = NS(size=8, rank=rank, ratio=ratio, physical_span=owner_rows * ratio, pages=pages, table=table)
            packets = []

            def gather(local):
                packets.append(tuple(local.shape))
                output = torch.empty((8 * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=device)
                dist.all_gather_into_tensor(output, local, group=dist.group.WORLD)
                return output

            options = dict(width=68, planar=True, shared_prefix_stripes=1, receive=NS(gather=gather))
            # Multiple groups exercise exact packet order and restored tail
            # pages across gather boundaries; all three paths use this budget.
            with patch.object(prefill_readback, "_SCRATCH_BYTES", 9 * block_bytes * 3):
                legacy = prefill_readback.gather_pool(pool, placement, counts, **options)
                legacy_packets = packets[:]
                packets.clear()
                split = prefill_readback.gather_pool(pool, placement, counts, split_indexer=True, **options)
                assert packets == legacy_packets
                packets.clear()
                with torch.inference_mode():
                    readback = prefill_readback.gather_pool(pool, placement, counts, split_indexer=True,
                                                           page_planar_indexer=True, **options)
                assert packets == legacy_packets and len(packets) > 1
            assert not readback.pages.is_inference()
            cache = _cache(readback)
            del cache._cp_shared_prefix_signature
            cache.lengths = tuple(count * ratio for count in counts)
            cache.geometry = NS(ratio=lambda source: ratio)
            cache.cp = dict(size=8, rank=rank)
            cache.pools, cache.placements = {(20, "indexer_kv"): pool}, {(20, "indexer_kv"): placement}
            host = torch.tensor(host_table, dtype=torch.int32)
            cache.inputs = {"indexer": NS(kv_cache_block_id=host, kv_cache_block_id_device=table)}
            with patch("rtp_llm.models_py.distributed.collective_torch._get_group", return_value=dist.group.WORLD):
                if rank == 7:
                    host[-1, 0] = 2
                cache.bind_shared_source_prefix((prefix * ratio,) * 8)
                assert cache.shared_prefix_stripes[(20, "indexer_kv")] == 0
                host.copy_(torch.tensor(host_table, dtype=torch.int32))
                cache.bind_shared_source_prefix((prefix * ratio,) * 8)
                assert cache.shared_prefix_stripes[(20, "indexer_kv")] == 1
            _, owners = cache._cp_shared_prefix_signature((20, "indexer_kv"))
            bank = _publish(cache, readback, owners=owners)
            assert bank.valid() and _padding_is_zero(bank)
            assert bank.pages.shape[0] == prefix // 256 + 8 * ((tail + 255) // 256)
            assert bank.request_table.tolist() == [list(row) for row in bank.request_table_host]
            assert cache.read_shared_indexer(20) is bank
            packet_count = len(packets)
            for request in range(8):
                expected = torch.cat((_rows(0, prefix, 0, device), _rows(prefix, tail, request + 1, device)))
                assert torch.equal(legacy[request], expected)
                assert torch.equal(split[request][0].view(torch.uint8), expected[:, :64])
                assert torch.equal(split[request][1].view(torch.uint8).reshape(-1, 4), expected[:, 64:])
                assert torch.equal(bank.materialize(request, counts[request]), expected)
                _, compatibility = cache.read_source(20, request, global_entries=counts[request])
                assert torch.equal(compatibility, expected)
            assert len(packets) == packet_count  # Reads/materializations add no collective.
            assert cache._cp_shared_indexer_stats["materializations"] == 8
            cache._invalidate_source_readbacks()
            tensor.fill_(17)
            assert not bank.valid()
            assert torch.equal(bank.materialize(7, counts[7]),
                               torch.cat((_rows(0, prefix, 0, device), _rows(prefix, tail, 8, device))))
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-PPU lease required")
class CP8SharedIndexerWorld8(unittest.TestCase):
    def test_actual_eight_rank_planar_readback_matches_legacy_and_pads_private_pages(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory(prefix="cp8-shared-indexer-") as directory:
            torch.multiprocessing.spawn(_world8_worker, args=(str(Path(directory) / "store"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
