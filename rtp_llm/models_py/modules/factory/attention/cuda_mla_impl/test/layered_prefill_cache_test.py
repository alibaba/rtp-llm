"""Byte-exact CUDA coverage of contiguous staging, writeback and block reuse."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
    ContiguousPrefillCache,
    chunk_runs,
    contiguous_runs,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.prefill_cache_diagnostics import (
    PrefillCacheDiagnostics,
)
from rtp_llm.ops.compute_ops import LayerKVCache, copy_paged_kv_cache_async


class LayeredPrefillCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("layered prefill tests require an allocated CUDA GPU")

    def make_cache(self, dtype):
        width = 584 if dtype == torch.uint8 else 512
        backing = []
        for layer in range(8):
            tensor = torch.empty(
                (16, 64, width),
                dtype=dtype,
                device="cuda" if layer == 0 else "cpu",
                pin_memory=layer != 0,
            )
            tensor.fill_(layer + 1)
            backing.append(tensor)
        return ContiguousPrefillCache(backing, [0, 0, 0, 0, 1, 1, 1, 1], 8 * 64, 1, 3)

    def make_fmha(self, sharded):
        table = torch.tensor([[7, 2, 9, -1], [7, 3, 0, 0]], dtype=torch.int32)
        slots = torch.tensor([7 * 64 + 60, 2 * 64 + 3, 3 * 64 + 8, -1], device="cuda")
        op = SimpleNamespace(
            block_table=table.cuda(),
            kv_cache_sharded=sharded,
            sharded_slot_mapping=slots,
            write_cache_store_impl=None,
        )
        return SimpleNamespace(
            attn_inputs=SimpleNamespace(
                is_prefill=True,
                cache_store_inputs=None,
                kv_cache_kernel_block_id_host=table,
                kv_cache_block_id_host=table,
            ),
            fmha_impl=op,
            rope_params=SimpleNamespace(slot_mapping=slots),
            write_cache_store_impl=None,
        )

    def test_coalescing_preserves_gaps_and_destinations(self):
        self.assertEqual(
            contiguous_runs([(2, 0), (3, 1), (5, 2), (6, 4)]),
            [(2, 0, 2), (5, 2, 1), (6, 4, 1)],
        )

    def test_indexer_diagnostics_namespace_requests_and_exclude_padding(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": directory}
        ):
            root = Path(directory)
            (root / "prefill_indexer_capture.json").write_text(
                json.dumps({"id": "test", "min_batch": 2})
            )
            diagnostics = PrefillCacheDiagnostics(0)
            op = SimpleNamespace(
                attn_inputs=SimpleNamespace(
                    input_lengths=torch.tensor([2, 1]),
                    prefix_lengths=torch.tensor([2, 2]),
                ),
                precomputed_req_ids=torch.tensor([0, 0, 1]),
            )
            diagnostics.begin(SimpleNamespace(fmha_impl=op))
            diagnostics.indexer(0, torch.tensor([[0, 1, -1], [0, 2, 99], [0, 1, 3]]))
            diagnostics.finish(None, False)
            row = json.loads((root / "prefill_forwards_rank0.jsonl").read_text())
            self.assertEqual(row["batch"], 2)
            self.assertEqual(row["exec_tokens"], 3)
            self.assertEqual(row["kv_tokens"], 7)
            self.assertEqual(row["indexer"][0]["selection_count"], 6)
            self.assertEqual(row["indexer"][0]["unique_request_tokens"], 5)
            diagnostics.begin(SimpleNamespace(fmha_impl=op))
            self.assertIsNone(diagnostics.capture)
            diagnostics.finish(None, False)

    def test_fragmented_prefetch_overlaps_owner_compute(self):
        # Fragmentation must not turn prefetch into hundreds of Python tensor
        # copies which finish before the owner kernel can even be submitted.
        backing = [
            torch.full(
                (1024, 64, 656),
                layer + 1,
                dtype=torch.uint8,
                device="cuda" if layer == 0 else "cpu",
                pin_memory=layer != 0,
            )
            for layer in range(4)
        ]
        cache = ContiguousPrefillCache(backing, [0, 0, 0, 0], 512 * 64, 1, 3)
        fmha = self.make_fmha(True)
        table = torch.arange(0, 1024, 2, dtype=torch.int32).reshape(1, -1)
        fmha.attn_inputs.kv_cache_kernel_block_id_host = table
        fmha.fmha_impl.block_table = table.cuda()
        fmha.fmha_impl.sharded_slot_mapping = torch.tensor([0], device="cuda")
        # Resolve the binding before measuring the first-forward submissions.
        copy_paged_kv_cache_async(backing[1], cache.staging[0], [])
        torch.cuda.synchronize()
        reference = torch.cuda.Event(enable_timing=True)
        copy_begin = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        owner_begin = torch.cuda.Event(enable_timing=True)
        owner_end = torch.cuda.Event(enable_timing=True)
        compute = torch.cuda.current_stream()
        original_copy = cache._copy_runs
        calls = 0

        def submit(source, dest, runs):
            nonlocal calls
            if calls == 0:
                copy_begin.record()
            original_copy(source, dest, runs)
            if calls == 0:
                with torch.cuda.stream(compute):
                    owner_begin.record()
                    torch.cuda._sleep(10_000_000)
                    owner_end.record()
            calls += 1

        reference.record()
        with patch.object(cache, "_copy_runs", submit):
            cache.begin_forward(fmha)
        copy_end.record(cache.prefetch_stream)
        cache.end_forward()
        torch.cuda.synchronize()
        overlap_ms = min(
            reference.elapsed_time(copy_end), reference.elapsed_time(owner_end)
        ) - max(reference.elapsed_time(copy_begin), reference.elapsed_time(owner_begin))
        self.assertGreater(overlap_ms, 0, "H2D must overlap the owner kernel")
        for layer in range(1, 4):
            torch.testing.assert_close(
                cache.staging[layer - 1].cpu(),
                backing[layer][::2],
                rtol=0,
                atol=0,
            )
        print(f"fragmented H2D/owner overlap: {overlap_ms:.3f} ms", flush=True)

    def test_native_copy_validates_all_runs_before_submission(self):
        cache = self.make_cache(torch.uint8)
        for runs, message in (
            ([(0, 0, 1), (16, 1, 1)], "bounds"),
            ([(0, 0, 2), (2, 1, 1)], "disjoint"),
            ([(0, 0, 0)], "bounds"),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                copy_paged_kv_cache_async(cache.backing[1], cache.staging[0], runs)
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            copy_paged_kv_cache_async(
                torch.empty_like(cache.backing[1], pin_memory=False),
                cache.staging[0],
                [(0, 0, 1)],
            )

    def test_decoder_joins_prefill_and_decode_prefetch_before_moe(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer

        for enabled in (False, True):
            for kind in ("pinned", "layered"):
                for nested in (False, True):
                    with self.subTest(enabled=enabled, kind=kind, nested=nested):
                        calls = []
                        hidden = torch.empty((4, 8))
                        residual = torch.empty_like(hidden)
                        layer = object.__new__(GenericMoeDecoderLayer)
                        torch.nn.Module.__init__(layer)
                        # Offset one also covers a dense owner whose first
                        # follower is the group's first Mega MoE layer.
                        layer.layer_idx = 3
                        layer._drain_pinned_mla_before_moe = enabled
                        layer._fuse_input_norm_quant = False
                        layer.input_layernorm = lambda *_: (hidden, residual)
                        layer.self_attn = Mock(
                            side_effect=lambda *a, **kw: calls.append("attention")
                            or hidden
                        )
                        layer._fwd_mlp_or_moe = Mock(
                            side_effect=lambda *a: calls.append("moe")
                            or (hidden, residual)
                        )
                        working = SimpleNamespace(
                            wait_prefetch_complete=lambda: calls.append("ready")
                        )
                        fmha = (
                            SimpleNamespace(pinned_mla_groups={3: (working, 1)})
                            if kind == "pinned"
                            else SimpleNamespace(
                                pinned_mla_groups={}, layered_prefill_cache=working
                            )
                        )
                        if nested:
                            fmha = SimpleNamespace(fmha_impl=fmha)
                        layer.forward(hidden, residual, fmha)
                        self.assertEqual(
                            calls,
                            ["attention"] + (["ready"] if enabled else []) + ["moe"],
                        )

    def test_partial_pages_slot_reuse_and_recycled_ids(self):
        for dtype in (torch.uint8, torch.bfloat16):
            for sharded in (False, True):
                with self.subTest(dtype=dtype, sharded=sharded):
                    cache = self.make_cache(dtype)
                    fmha = self.make_fmha(sharded)
                    original_table = fmha.fmha_impl.block_table
                    original_slots = fmha.rope_params.slot_mapping.clone()
                    expected = [tensor.cpu().clone() for tensor in cache.backing]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    for iteration in range(3):
                        # Recycle a backing ID without any persistent staging
                        # invalidation; the next forward must see the new bytes.
                        if iteration:
                            cache.backing[1][7].fill_(31 + iteration)
                            expected[1][7].fill_(31 + iteration)
                        snapshots = []
                        with torch.cuda.stream(stream):
                            cache.begin_forward(fmha)
                            for layer in range(8):
                                cache.before_layer(layer)
                                if layer == 0:
                                    # Keep owner compute busy while copy engines
                                    # fetch the subsequent host layers.
                                    torch.cuda._sleep(100_000)
                                    cache.prefetch_owner(layer)
                                    continue
                                backing_view = LayerKVCache()
                                backing_view.kv_cache_base = cache.backing[layer]
                                with cache.attention(
                                    layer, fmha, backing_view
                                ) as target:
                                    pages = fmha.fmha_impl.block_table
                                    snapshots.append(
                                        (
                                            layer,
                                            target.kv_cache_base[pages[0, 0]].clone(),
                                        )
                                    )
                                    slots = (
                                        fmha.fmha_impl.sharded_slot_mapping
                                        if sharded
                                        else fmha.rope_params.slot_mapping
                                    )
                                    value = 70 + layer + iteration
                                    target.kv_cache_base.flatten(0, 1)[
                                        slots[:3], :8
                                    ] = value
                                    torch.cuda._sleep(100_000)
                                    for page, row in ((7, 60), (2, 3), (3, 8)):
                                        expected[layer][page, row, :8] = value
                            cache.end_forward()
                        # Output readiness on the compute stream implies D2H
                        # readiness too, so the CPU can safely recycle pages.
                        stream.synchronize()
                        for layer, tensor in enumerate(cache.backing):
                            torch.testing.assert_close(
                                tensor.cpu(), expected[layer], rtol=0, atol=0
                            )
                        for layer, snapshot in snapshots:
                            # Unwritten historical rows survive partial append
                            # and the next layer's reuse of this staging slot.
                            torch.testing.assert_close(
                                snapshot[:59].cpu(),
                                expected[layer][7, :59],
                                rtol=0,
                                atol=0,
                            )
                        self.assertIs(fmha.fmha_impl.block_table, original_table)
                        torch.testing.assert_close(
                            fmha.rope_params.slot_mapping,
                            original_slots,
                            rtol=0,
                            atol=0,
                        )

    def test_exception_restores_metadata_and_drains_transfers(self):
        cache = self.make_cache(torch.uint8)
        fmha = self.make_fmha(True)
        original_table = fmha.fmha_impl.block_table
        original_slots = fmha.fmha_impl.sharded_slot_mapping
        cache.begin_forward(fmha)
        cache.before_layer(0)
        cache.prefetch_owner(0)
        cache.before_layer(1)
        view = LayerKVCache()
        view.kv_cache_base = cache.backing[1]
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with cache.attention(1, fmha, view):
                raise RuntimeError("injected attention failure")
        cache.end_forward(failed=True)
        self.assertFalse(cache.active)
        self.assertIs(fmha.fmha_impl.block_table, original_table)
        self.assertIs(fmha.fmha_impl.sharded_slot_mapping, original_slots)
        # A subsequent request can reuse the same buffers after the failure.
        cache.begin_forward(fmha)
        cache.before_layer(0)
        cache.prefetch_owner(0)
        cache.end_forward()
        torch.cuda.synchronize()

    def test_store_publishes_backing_with_writeback_readiness(self):
        cache = self.make_cache(torch.uint8)
        fmha = self.make_fmha(True)
        view = LayerKVCache()
        view.kv_cache_base = cache.backing[1]
        published = []

        def store(backing):
            event = torch.cuda.Event()
            event.record()
            published.append((backing, torch.cuda.current_stream().cuda_stream, event))

        fmha.attn_inputs.cache_store_inputs = object()
        fmha.write_cache_store_impl = store
        fmha.fmha_impl.write_cache_store_impl = store
        cache.begin_forward(fmha)
        cache.before_layer(0)
        cache.prefetch_owner(0)
        cache.before_layer(1)
        with cache.attention(1, fmha, view) as target:
            self.assertIsNone(fmha.write_cache_store_impl)
            self.assertIsNone(fmha.fmha_impl.write_cache_store_impl)
            torch.cuda._sleep(100_000)
            target.kv_cache_base.flatten(0, 1)[
                fmha.fmha_impl.sharded_slot_mapping[0], :8
            ] = 99
        self.assertEqual(len(published), 1)
        backing, stream, ready = published[0]
        self.assertEqual(backing.kv_cache_base.data_ptr(), cache.backing[1].data_ptr())
        self.assertEqual(stream, cache.writeback_stream.cuda_stream)
        ready.synchronize()
        self.assertTrue(torch.all(backing.kv_cache_base[7, 60, :8] == 99))
        self.assertIs(fmha.write_cache_store_impl, store)
        self.assertIs(fmha.fmha_impl.write_cache_store_impl, store)
        cache.end_forward()
        torch.cuda.synchronize()

    def make_contiguous_cache(self, prefix=4, layers=12, slots=3):
        backing = [
            torch.full(
                (16, 64, 584),
                layer + 1,
                dtype=torch.uint8,
                device="cuda" if layer < prefix else "cpu",
                pin_memory=layer >= prefix,
            )
            for layer in range(layers)
        ]
        return ContiguousPrefillCache(
            backing, [i // 4 for i in range(layers)], 8 * 64, prefix, slots
        )

    def test_chunks_preserve_sparse_pages_and_bound_submissions(self):
        runs = [(2, 0, 3), (8, 3, 5), (20, 8, 1)]
        chunks = chunk_runs(runs, 2)
        self.assertTrue(all(count <= 2 for _, count in chunks))
        expected = [(s + i, d + i) for s, d, n in runs for i in range(n)]
        actual = [
            (s + i, d + i) for chunk, _ in chunks for s, d, n in chunk for i in range(n)
        ]
        self.assertEqual(actual, expected)

    def test_contiguous_ring_writeback_and_recycled_blocks(self):
        for sharded in (False, True):
            cache = self.make_contiguous_cache()
            fmha = self.make_fmha(sharded)
            expected = [tensor.cpu().clone() for tensor in cache.backing]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            for iteration in range(2):
                if iteration:
                    cache.backing[4][7].fill_(43)
                    expected[4][7].fill_(43)
                snapshots = []
                with torch.cuda.stream(stream):
                    cache.begin_forward(fmha)
                    for layer in range(12):
                        cache.before_layer(layer)
                        view = LayerKVCache()
                        view.kv_cache_base = cache.backing[layer]
                        torch.cuda._sleep(100_000)
                        with cache.attention(layer, fmha, view) as target:
                            if layer < 4:
                                continue
                            snapshots.append(
                                (
                                    layer,
                                    target.kv_cache_base[
                                        fmha.fmha_impl.block_table[0, 0]
                                    ].clone(),
                                )
                            )
                            slots = (
                                fmha.fmha_impl.sharded_slot_mapping
                                if sharded
                                else fmha.rope_params.slot_mapping
                            )
                            target.kv_cache_base.flatten(0, 1)[slots[:3], :8] = (
                                70 + layer
                            )
                            for page, row in ((7, 60), (2, 3), (3, 8)):
                                expected[layer][page, row, :8] = 70 + layer
                        cache.wait_prefetch_complete()
                        torch.cuda._sleep(100_000)  # Synthetic Mega MoE window.
                    cache.end_forward()
                stream.synchronize()
                for layer, actual in enumerate(cache.backing):
                    torch.testing.assert_close(
                        actual.cpu(), expected[layer], rtol=0, atol=0
                    )
                for layer, snapshot in snapshots:
                    torch.testing.assert_close(
                        snapshot[:59].cpu(), expected[layer][7, :59], rtol=0, atol=0
                    )
                self.assertEqual(
                    cache.copy_stats["h2d_bytes"], 8 * 5 * cache.page_bytes
                )
                self.assertEqual(
                    cache.copy_stats["d2h_bytes"], 8 * 3 * cache.page_bytes
                )
                self.assertGreater(cache.copy_stats["layers_submitted_ahead"], 0)
                self.assertEqual(cache.copy_stats["demand_h2d_bytes"], 0)

    def test_contiguous_weighted_windows_fence_only_submitted_copies(self):
        with patch.dict(os.environ, {"RTP_LLM_DSA_PREFILL_COPY_SCHEDULE": "window"}):
            cache = self.make_contiguous_cache(prefix=1, layers=17, slots=4)
        fmha = self.make_fmha(True)
        original_copy = cache._copy_runs
        intervals, moe, indexer, h2d = [], [], [], []

        def slow_copy(source, dest, runs):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            torch.cuda._sleep(200_000)
            original_copy(source, dest, runs)
            end.record()
            intervals.append((begin, end))
            if not source.is_cuda:
                h2d.append((begin, end))

        cache._copy_runs = slow_copy
        cache.begin_forward(fmha)
        reference = torch.cuda.Event(enable_timing=True)
        reference.record()
        for layer in range(17):
            cache.before_layer(layer)
            view = LayerKVCache()
            view.kv_cache_base = cache.backing[layer]
            torch.cuda._sleep(100_000)  # Projection submitted before H2D.
            begin_indexer, end_indexer = (
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            )
            cache.prefetch_before_indexer(layer)
            begin_indexer.record()
            torch.cuda._sleep(10_000_000)
            end_indexer.record()
            indexer.append((begin_indexer, end_indexer))
            with cache.attention(layer, fmha, view):
                torch.cuda._sleep(1_000_000)
            cache.wait_prefetch_complete()
            if layer == 0:
                # A fence must not force all four lookahead slots to complete.
                self.assertTrue(any(cache.jobs.values()))
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            torch.cuda._sleep(1_000_000)
            end.record()
            moe.append((begin, end))
        cache.end_forward()
        torch.cuda.synchronize()
        for begin, end in intervals:
            copy_interval = (reference.elapsed_time(begin), reference.elapsed_time(end))
            for first, last in moe:
                overlap = min(copy_interval[1], reference.elapsed_time(last)) - max(
                    copy_interval[0], reference.elapsed_time(first)
                )
                self.assertLessEqual(overlap, 0, "KV H2D/D2H must not overlap Mega MoE")
        overlap = sum(
            max(
                0,
                min(
                    reference.elapsed_time(copy_end),
                    reference.elapsed_time(compute_end),
                )
                - max(
                    reference.elapsed_time(copy_begin),
                    reference.elapsed_time(compute_begin),
                ),
            )
            for copy_begin, copy_end in h2d
            for compute_begin, compute_end in indexer
        )
        self.assertGreater(overlap, 0, "independent H2D must overlap Indexer compute")
        windows = cache.copy_stats["copy_windows"]
        self.assertGreater(windows[0]["h2d_bytes"], windows[1]["h2d_bytes"])
        self.assertEqual(cache.copy_stats["demand_h2d_bytes"], 0)

    def test_contiguous_whole_layers_are_fifo_and_do_not_fence_moe(self):
        cache = self.make_contiguous_cache(prefix=4, layers=12, slots=4)
        fmha = self.make_fmha(True)
        original_copy = cache._copy_runs
        copies, writes, moe, snapshots = [], {}, [], []
        by_ptr = {t.data_ptr(): layer for layer, t in enumerate(cache.backing)}

        def slow_copy(source, dest, runs):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            torch.cuda._sleep(2_000_000)
            original_copy(source, dest, runs)
            end.record()
            if source.is_cuda:
                writes[by_ptr[dest.data_ptr()]] = end
            else:
                copies.append((by_ptr[source.data_ptr()], begin, end))

        cache._copy_runs = slow_copy
        reference = torch.cuda.Event(enable_timing=True)
        reference.record()
        cache.begin_forward(fmha)
        # All available slots must be queued before any compute-layer hook.
        self.assertEqual([layer for layer, _, _ in copies], list(range(4, 8)))
        self.assertEqual(cache.copy_stats["initial_h2d_calls"], 4)
        for layer in range(12):
            cache.before_layer(layer)
            before = len(copies)
            cache.prefetch_before_indexer(layer)
            view = LayerKVCache()
            view.kv_cache_base = cache.backing[layer]
            with cache.attention(layer, fmha, view) as target:
                if layer >= 4:
                    # Keep the host free to submit subsequent work. Checking
                    # GPU values here would synchronize and hide reuse races.
                    snapshots.append((layer, target.kv_cache_base[:5].clone()))
            if 4 <= layer < 8:
                # Refill at attention release, before the next layer hook.
                self.assertEqual(len(copies), before + 1)
                self.assertEqual(copies[-1][0], layer + 4)
            else:
                self.assertEqual(len(copies), before)
            cache.wait_prefetch_complete()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            torch.cuda._sleep(2_000_000)
            end.record()
            moe.append((begin, end))
        cache.end_forward()
        torch.cuda.synchronize()
        for layer, snapshot in snapshots:
            self.assertTrue(torch.all(snapshot.cpu() == layer + 1))
        self.assertEqual([layer for layer, _, _ in copies], list(range(4, 12)))
        for (_, _, previous), (_, following, _) in zip(copies, copies[1:]):
            self.assertLessEqual(
                reference.elapsed_time(previous), reference.elapsed_time(following)
            )
        for layer, begin, _ in copies:
            if layer >= 8:
                self.assertLessEqual(
                    reference.elapsed_time(writes[layer - 4]),
                    reference.elapsed_time(begin),
                    "slot reuse must wait for the previous layer's writeback",
                )
        overlap = sum(
            max(
                0,
                min(reference.elapsed_time(last), reference.elapsed_time(b))
                - max(reference.elapsed_time(first), reference.elapsed_time(a)),
            )
            for _, first, last in copies
            for a, b in moe
        )
        self.assertGreater(overlap, 0, "the layer schedule must allow copy/MoE overlap")
        self.assertEqual(cache.copy_stats["h2d_calls"], 8)
        self.assertEqual(cache.copy_stats["copy_schedule"], "layer")

    def test_contiguous_skips_only_fully_overwritten_pages(self):
        for full_only in (False, True):
            cache = self.make_contiguous_cache(prefix=1, layers=5, slots=2)
            fmha = self.make_fmha(True)
            table = torch.tensor([[7]] if full_only else [[7, 2]], dtype=torch.int32)
            slots = torch.arange(7 * 64, 8 * 64, device="cuda")
            if not full_only:
                # Duplicates must not incorrectly mark page 2 as overwritten.
                slots = torch.cat((slots, torch.full((80,), 2 * 64, device="cuda")))
            fmha.attn_inputs.kv_cache_kernel_block_id_host = table
            fmha.fmha_impl.block_table = table.cuda()
            fmha.fmha_impl.sharded_slot_mapping = slots
            expected = [tensor.cpu().clone() for tensor in cache.backing]
            cache.begin_forward(fmha)
            self.assertEqual(cache.skipped_pages, 1)
            self.assertEqual(sum(n for _, _, n in cache.fetch_runs), int(not full_only))
            for layer in range(5):
                cache.before_layer(layer)
                cache.prefetch_before_indexer(layer)
                calls_before = cache.copy_stats["h2d_calls"]
                view = LayerKVCache()
                view.kv_cache_base = cache.backing[layer]
                with cache.attention(layer, fmha, view) as target:
                    if layer:
                        target.kv_cache_base.flatten(0, 1)[
                            fmha.fmha_impl.sharded_slot_mapping
                        ] = (90 + layer)
                        expected[layer][7].fill_(90 + layer)
                        if not full_only:
                            expected[layer][2, 0].fill_(90 + layer)
                    # Attention itself never performs a demand fetch. Exiting
                    # the context may immediately prefetch a future layer.
                    self.assertEqual(cache.copy_stats["h2d_calls"], calls_before)
                cache.wait_prefetch_complete()
            cache.end_forward()
            torch.cuda.synchronize()
            self.assertEqual(
                cache.copy_stats["h2d_bytes"], 4 * cache.page_bytes * int(not full_only)
            )
            for layer, tensor in enumerate(cache.backing):
                torch.testing.assert_close(
                    tensor.cpu(), expected[layer], rtol=0, atol=0
                )

    def test_contiguous_timing_is_opt_in_and_never_synchronizes(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(
            os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": root, "WORLD_RANK": "0"}
        ):
            Path(root, "prefill_copy_timing.enable").touch()
            cache = self.make_contiguous_cache(prefix=1, layers=2, slots=1)
            fmha = self.make_fmha(True)
            cache.begin_forward(fmha)
            for layer in range(2):
                cache.before_layer(layer)
                view = LayerKVCache()
                view.kv_cache_base = cache.backing[layer]
                with cache.attention(layer, fmha, view):
                    torch.cuda._sleep(100_000)
                cache.wait_prefetch_complete()
            cache.end_forward()
            torch.cuda.synchronize()
            cache.flush_timings()
            rows = json.loads(
                Path(root, "prefill_copy_timing_rank0.jsonl").read_text()
            )["layers"]
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[1]["non_moe_window_ms"], 0)
            self.assertGreaterEqual(rows[1]["attention_kv_wait_ms"], 0)
            Path(root, "prefill_copy_timing.enable").unlink()
            cache.begin_forward(fmha)
            self.assertFalse(cache.capture_timing)
            cache.end_forward()
            torch.cuda.synchronize()

    def test_contiguous_failure_restores_and_drains(self):
        cache = self.make_contiguous_cache(prefix=1)
        fmha = self.make_fmha(True)
        original = fmha.fmha_impl.block_table
        cache.begin_forward(fmha)
        cache.before_layer(0)
        cache.prefetch_owner(0)
        cache.before_layer(1)
        view = LayerKVCache()
        view.kv_cache_base = cache.backing[1]
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with cache.attention(1, fmha, view):
                raise RuntimeError("injected")
        cache.end_forward(failed=True)
        self.assertIs(fmha.fmha_impl.block_table, original)
        self.assertFalse(cache.active)
        cache.begin_forward(fmha)
        cache.before_layer(0)
        cache.prefetch_owner(0)
        cache.end_forward(failed=True)

    def test_rejects_overflow_and_out_of_range_before_transfer(self):
        cache = self.make_cache(torch.uint8)
        fmha = self.make_fmha(True)
        table = torch.arange(9, dtype=torch.int32).reshape(1, -1)
        fmha.attn_inputs.kv_cache_kernel_block_id_host = table
        fmha.fmha_impl.block_table = table.cuda()
        with self.assertRaisesRegex(ValueError, "capacity"):
            cache.begin_forward(fmha)
        self.assertFalse(cache.active)
        table[0, 0] = 16
        cache.capacity = 16
        with self.assertRaisesRegex(ValueError, "logical cache capacity"):
            cache.begin_forward(fmha)


if __name__ == "__main__":
    unittest.main()
