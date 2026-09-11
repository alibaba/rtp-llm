"""CP8-format restore/reader probes on one real GPU, without a CP collective."""

import hashlib
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_compact_reader import _attention_oracle, _fixture, _ints

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    CompactPages,
    GlobalBinding,
    SwaBinding,
    compact_attention,
    gather_compact,
)
from rtp_llm.models_py.modules.dsv41.cprr_reader import (
    CprrReadIdentity,
    CprrReaderLease,
    bind_cprr_paged,
    restore_cprr_swa,
)


def page_for(layout, region, layer):
    return next(p for p in layout.pages if p.slot == RegionSlot(region, layer))


def swa_fixture(layout):
    page = page_for(layout, CacheRegion.SWA, 21)
    compact, decoded = _fixture(CacheRegion.SWA, pages=9, entries=page.entries)
    complete = torch.full(
        (9, page.page_stride_bytes), 173, dtype=torch.uint8, device="cuda"
    )
    complete[:, : compact.data.shape[1]].copy_(compact.data)
    received = torch.full(
        (8, 9, page.prefill_shard_bytes), 255, dtype=torch.uint8, device="cuda"
    )
    ids = _ints(
        [[1 + (rank + request) % 8 for request in range(8)] for rank in range(8)]
    )
    for rank in range(8):
        first, last = page.swa_byte_slice(rank)
        for request in range(8):
            received[rank, 1 + (rank + request) % 8].copy_(
                complete[request + 1, first:last]
            )
    return received, ids, complete, decoded


def paged_fixture(layout, owner, region=CacheRegion.GLOBAL):
    page = page_for(layout, region, owner)
    received = torch.zeros((8, 3, page.page_stride_bytes), dtype=torch.uint8)
    decoded = torch.zeros((24, page.entries, page.encoding.head_dim))
    for rank in range(8):
        received[rank, 0].fill_(255)
        for local in (1, 2):
            rows = received[
                rank, local, : page.entries * page.encoding.entry_bytes
            ].view(page.entries, page.encoding.entry_bytes)
            rows[:, : page.encoding.payload_bytes].fill_(0x22)
            if region == CacheRegion.GLOBAL:
                value = (rank + 1) * local
                scale = (
                    torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8).item()
                )
            else:
                scale = 125 + rank + local
                value = 2 ** (scale - 127)
            rows[:, page.encoding.payload_bytes :].fill_(scale)
            decoded[rank * 3 + local].fill_(value)
            received[rank, local, page.entries * page.encoding.entry_bytes :].fill_(
                rank + local
            )
    tables = _ints(
        [
            [
                [1 + (rank + request + block) % 2 for block in range(2)]
                for request in range(8)
            ]
            for rank in range(8)
        ]
    )
    return received.cuda(), tables, decoded


def independent_table(tables, local_pages):
    source = tables.cpu().tolist()
    requests, virtual = tables.shape[1:]
    return _ints(
        [
            [
                (
                    (block % 8) * local_pages + source[block % 8][request][block // 8]
                    if source[block % 8][request][block // 8] > 0
                    else 0
                )
                for block in range(8 * virtual)
            ]
            for request in range(requests)
        ]
    )


class CprrReaderGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            os.geteuid() == 0
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] != 10
            or not torch.version.cuda.startswith("13.")
        ):
            raise RuntimeError(
                "required CPRR reader probes need non-root CUDA13 Blackwell"
            )
        cls.backend = os.environ["DSV41_CPRR_READER_BACKEND"]
        if cls.backend not in ("native", "flashmla"):
            raise RuntimeError(
                "select the explicit native or FlashMLA reader candidate"
            )
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / "cprr_reader_components.json").write_text(
                json.dumps(
                    {
                        "scope": "one-GPU CP8-format component; collective/PD/W05 pending",
                        "backend": cls.backend,
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "torch": str(torch.__version__),
                        "observations": cls.records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_device=False)

        def digest(tensor):
            return hashlib.sha256(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()

        self.records.append(
            {
                "test": self.id(),
                "shape": list(actual.shape),
                "actual_sha256": digest(actual),
                "expected_sha256": digest(expected),
            }
        )

    def test_swa_restore_all_rank_local_ids_and_padding_bytes(self):
        for block in (128, 256):
            layout = CacheLayout(token_block_size=block)
            received, ids, complete, _ = swa_fixture(layout)
            result = restore_cprr_swa(layout, 21, received, ids)
            result.check()
            self.exact(result.pages.data[1:], complete[1:])
            self.exact(result.pages.data[0], torch.zeros_like(complete[0]))
            self.assertEqual(result.pages.entries_per_page, 136)

    def test_swa_restore_rejects_planar_destination_before_writing(self):
        from rtp_llm.models_py.modules.dsv41.flashmla import PlanarPages

        layout = CacheLayout()
        received, ids, complete, _ = swa_fixture(layout)
        before = complete.clone()
        with self.assertRaisesRegex(TypeError, "row-interleaved"):
            restore_cprr_swa(
                layout,
                21,
                received,
                ids,
                output=PlanarPages(complete, CacheRegion.SWA, layout.swa_entries),
            )
        self.exact(complete, before)

    def test_paged_sources_map_all_rank_owners_without_requantization(self):
        for block in (128, 256):
            for owner in (2, 8, 14, 20):
                for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
                    layout = CacheLayout(token_block_size=block)
                    received, tables, decoded = paged_fixture(layout, owner, region)
                    result = bind_cprr_paged(
                        layout, RegionSlot(region, owner), received, tables
                    )
                    result.check()
                    expected_table = independent_table(tables, 3)
                    self.exact(result.page_table, expected_table)
                    self.assertEqual(result.pages.data.data_ptr(), received.data_ptr())
                    entries = result.pages.entries_per_page
                    positions = _ints([[block * entries for block in range(16)]])
                    actual = gather_compact(
                        result.pages,
                        result.page_table,
                        _ints([0]),
                        positions,
                        _ints([16 * entries]),
                        output_dtype=torch.float32,
                    )
                    actual.check()
                    expected = decoded[expected_table[0].cpu().long(), 0][None]
                    self.exact(actual.output, expected)

    def test_invalid_ranks_pages_strides_and_context_capacity(self):
        layout = CacheLayout()
        received, ids, complete, _ = swa_fixture(layout)
        for data, mapping in (
            (received[:7], ids),
            (received, ids[:7]),
            (received[:, :, :-1], ids),
        ):
            with self.assertRaisesRegex(ValueError, "eight"):
                restore_cprr_swa(layout, 21, data, mapping)
        with self.assertRaisesRegex(ValueError, "CP8"):
            restore_cprr_swa(CacheLayout(cp_size=1), 21, received, ids)
        with self.assertRaisesRegex(ValueError, "alias"):
            restore_cprr_swa(
                layout,
                21,
                received,
                ids,
                output=CompactPages(
                    received.view_as(complete),
                    CacheRegion.SWA,
                    layout.swa_entries,
                ),
            )
        bad = ids.clone()
        bad[0, 0], bad[1, 1], bad[2, 2], bad[3, 3] = -1, -2, 9, 0
        result = restore_cprr_swa(layout, 21, received, bad)
        self.exact(result.status, _ints([1, 2, 2, 1, 0, 0, 0, 0]))
        self.exact(result.pages.data[1:5], torch.zeros_like(result.pages.data[1:5]))
        with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
            result.check()
        received, tables, _ = paged_fixture(layout, 20)
        for value in (-2, 3):
            bad = tables.clone()
            bad[7, 0, 1] = value
            result = bind_cprr_paged(
                layout, RegionSlot(CacheRegion.GLOBAL, 20), received, bad
            )
            self.assertEqual(result.page_table[0, 15].item(), 0)
            self.assertEqual(result.status[0, 15].item(), 2)
            with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
                result.check()
        with self.assertRaisesRegex(ValueError, "source slot"):
            bind_cprr_paged(
                layout, RegionSlot(CacheRegion.GLOBAL, 24), received, tables
            )
        with self.assertRaisesRegex(ValueError, "context capacity"):
            bind_cprr_paged(
                layout,
                RegionSlot(CacheRegion.GLOBAL, 20),
                received,
                torch.zeros((8, 1, 1025), dtype=torch.int32, device="cuda"),
            )

    def test_missing_global_pages_stay_unmapped_for_visible_reader_checks(self):
        layout = CacheLayout()
        received, tables, _ = paged_fixture(layout, 20)
        tables[7, 0, 0] = -1
        result = bind_cprr_paged(
            layout, RegionSlot(CacheRegion.GLOBAL, 20), received, tables
        )
        result.check()
        position = 7 * result.pages.entries_per_page
        for visible, expected_status in ((position, 0), (position + 1, 1)):
            read = gather_compact(
                result.pages,
                result.page_table,
                _ints([0]),
                _ints([[position]]),
                _ints([visible]),
            )
            self.assertEqual(read.status.item(), expected_status)
            if expected_status:
                with self.assertRaisesRegex(RuntimeError, "missing KV"):
                    read.check()
            else:
                read.check()

    def test_empty_local_requests_preserve_zero_row_contract(self):
        layout = CacheLayout()
        received, ids, _, _ = swa_fixture(layout)
        result = restore_cprr_swa(layout, 21, received, ids[:, :0].contiguous())
        result.check()
        self.assertEqual(result.pages.data.shape[0], 1)
        self.assertEqual(result.status.numel(), 0)
        received, tables, _ = paged_fixture(layout, 20)
        result = bind_cprr_paged(
            layout,
            RegionSlot(CacheRegion.GLOBAL, 20),
            received,
            tables[:, :0].contiguous(),
        )
        result.check()
        self.assertEqual(tuple(result.page_table.shape), (0, 16))

    def reader(self, query, request, position, floor, swa, sinks, global_kv, indices):
        if self.backend == "native":
            return compact_attention(
                query,
                request,
                position,
                floor,
                swa,
                sinks,
                global_kv=global_kv,
                global_indices=indices,
                output_dtype=torch.float32,
            )
        from rtp_llm.models_py.modules.dsv41.flashmla import (
            PlanarGlobalBinding,
            PlanarSwaBinding,
            flashmla_attention,
        )

        return flashmla_attention(
            query,
            request,
            position,
            floor,
            PlanarSwaBinding.from_compact(swa),
            sinks,
            global_kv=PlanarGlobalBinding.from_compact(global_kv),
            global_indices=indices,
        )

    def test_restored_reader_matches_independent_source_pages(self):
        for block in (128, 256):
            for owner in (2, 20):
                layout = CacheLayout(token_block_size=block)
                shards, ids, complete, swa_values = swa_fixture(layout)
                received, tables, global_values = paged_fixture(layout, owner)
                restored = restore_cprr_swa(layout, 21, shards, ids)
                global_result = bind_cprr_paged(
                    layout, RegionSlot(CacheRegion.GLOBAL, owner), received, tables
                )
                restored.check()
                global_result.check()
                positions = (
                    torch.arange(8, dtype=torch.int32, device="cuda") + 8 * block
                )
                swa = SwaBinding(
                    restored.pages,
                    _ints(list(range(1, 9))),
                    positions - 135,
                    positions + 1,
                )
                global_kv = GlobalBinding(
                    global_result.pages,
                    global_result.page_table,
                    layer_sources(owner).ratio,
                )
                entries = global_result.pages.entries_per_page
                indices = _ints(
                    [[i * entries for i in range(8)] + [8 * entries + 32]] * 8
                )
                query = (
                    torch.randn(
                        (8, 64, 512),
                        device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(41),
                    ).bfloat16()
                    * 0.03125
                )
                requests, sinks = _ints(list(range(8))), torch.linspace(
                    -2, 2, 64, device="cuda"
                )
                for floor in (torch.zeros_like(positions), positions - 3):
                    actual = self.reader(
                        query,
                        requests,
                        positions,
                        floor,
                        swa,
                        sinks,
                        global_kv,
                        indices,
                    )
                    actual.check()
                    direct_swa = replace(
                        swa,
                        pages=CompactPages(
                            complete, CacheRegion.SWA, layout.swa_entries
                        ),
                    )
                    direct_global = replace(
                        global_kv, page_table=independent_table(tables, 3)
                    )
                    direct = self.reader(
                        query,
                        requests,
                        positions,
                        floor,
                        direct_swa,
                        sinks,
                        direct_global,
                        indices,
                    )
                    direct.check()
                    self.exact(actual.output, direct.output)
                    self.exact(actual.lse, direct.lse)
                    if self.backend == "native":
                        expected, expected_lse = _attention_oracle(
                            query,
                            requests,
                            positions,
                            floor,
                            direct_swa,
                            swa_values,
                            sinks,
                            direct_global,
                            global_values,
                            indices,
                        )
                        torch.testing.assert_close(
                            actual.output.cpu(), expected, rtol=2e-5, atol=2e-5
                        )
                        torch.testing.assert_close(
                            actual.lse.cpu(), expected_lse, rtol=2e-5, atol=2e-5
                        )
                invalid = self.reader(
                    query,
                    requests,
                    positions,
                    floor,
                    replace(swa, valid_ends=positions),
                    sinks,
                    global_kv,
                    indices,
                )
                with self.assertRaisesRegex(RuntimeError, "rejected metadata"):
                    invalid.check()

    def test_graph_reuses_backing_and_refreshes_rank_ids_pages_and_lengths(self):
        layout = CacheLayout()
        shards, ids, complete, _ = swa_fixture(layout)
        received, tables, _ = paged_fixture(layout, 20)
        restored = restore_cprr_swa(layout, 21, shards, ids)
        paged = bind_cprr_paged(
            layout, RegionSlot(CacheRegion.GLOBAL, 20), received, tables
        )
        positions = _ints(list(range(1024, 1032)))
        swa = SwaBinding(
            restored.pages, _ints(list(range(1, 9))), positions - 135, positions + 1
        )
        global_kv = GlobalBinding(paged.pages, paged.page_table, 1)
        requests, floor = _ints(list(range(8))), positions - 3
        indices = _ints([[i * 128 for i in range(8)]] * 8)
        query = torch.zeros((8, 64, 512), dtype=torch.bfloat16, device="cuda")
        sinks = torch.zeros(64, device="cuda")

        def operation():
            restore_cprr_swa(
                layout, 21, shards, ids, output=restored.pages, status=restored.status
            )
            bind_cprr_paged(
                layout,
                RegionSlot(CacheRegion.GLOBAL, 20),
                received,
                tables,
                page_table=paged.page_table,
                status=paged.status,
            )
            return self.reader(
                query, requests, positions, floor, swa, sinks, global_kv, indices
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            operation()
            operation()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = operation()
        pointers = (
            restored.pages.data.data_ptr(),
            paged.pages.data.data_ptr(),
            paged.page_table.data_ptr(),
            captured.output.data_ptr(),
        )
        for step in range(3):
            ids.copy_(ids.roll(1, dims=1))
            tables.copy_(3 - tables)
            complete[1:].copy_(complete[1:].roll(1, dims=0))
            received[:, 1:].copy_(received[:, 1:].flip(1))
            positions.add_(1)
            swa.valid_starts.add_(1)
            swa.valid_ends.add_(1)
            floor.add_(1)
            indices[:, -1].fill_(-1 if step % 2 else 7 * 128)
            query.add_(0.001953125)
            graph.replay()
            torch.cuda.synchronize()
            restored.check()
            paged.check()
            captured.check()
            self.assertEqual(
                pointers,
                (
                    restored.pages.data.data_ptr(),
                    paged.pages.data.data_ptr(),
                    paged.page_table.data_ptr(),
                    captured.output.data_ptr(),
                ),
            )
            self.exact(restored.pages.data[1:], complete[1:])
            self.exact(paged.page_table, independent_table(tables, 3))
            direct = self.reader(
                query,
                requests,
                positions,
                floor,
                replace(
                    swa,
                    pages=CompactPages(complete, CacheRegion.SWA, layout.swa_entries),
                ),
                sinks,
                replace(global_kv, page_table=independent_table(tables, 3)),
                indices,
            )
            direct.check()
            self.exact(captured.output, direct.output)
            self.exact(captured.lse, direct.lse)
        self.records.append(
            {"test": self.id(), "captures": 1, "replays": 3, "same_backing": True}
        )

    def test_cross_stream_ready_last_consumer_and_stale_source_identity(self):
        layout = CacheLayout()
        cache_identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity(
            "component", layout
        )
        identity = CprrReadIdentity("request-a", cache_identity, 4, 0, 128)
        source = torch.empty((1024,), dtype=torch.uint8, device="cuda")
        producer, left, right = (torch.cuda.Stream() for _ in range(3))
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            source.fill_(17)
            lease = CprrReaderLease(
                identity, layout, RegionSlot(CacheRegion.GLOBAL, 20), [source], [20, 21]
            )
        for stale in (
            replace(identity, request_id="request-b"),
            replace(identity, epoch=5),
            replace(identity, chunk_start=1),
            replace(
                identity,
                cache=ReplayConfig(ReplayMode.FULL).cache_identity("component", layout),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "stale"):
                lease.acquire(20, stale)
        with torch.cuda.stream(left):
            lease.acquire(20, identity)
            first = source.clone()
        with self.assertRaisesRegex(ValueError, "own stream"):
            lease.complete(20, identity)
        with torch.cuda.stream(left):
            lease.complete(20, identity)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            lease.release(identity)
        with torch.cuda.stream(right):
            lease.acquire(21, identity)
            second = source.clone()
            lease.complete(21, identity)
        lease.release(identity)
        source.zero_()
        torch.cuda.synchronize()
        self.exact(first, torch.full_like(first, 17))
        self.exact(second, torch.full_like(second, 17))
        with self.assertRaisesRegex(ValueError, "released"):
            lease.acquire(21, identity)
        with self.assertRaisesRegex(ValueError, "source owner"):
            CprrReaderLease(
                identity, layout, RegionSlot(CacheRegion.INDEX_K, 20), [source], [21]
            )


if __name__ == "__main__":
    unittest.main()
