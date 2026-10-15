"""Interleaved KV reuse, invalidation, and CUDA foreach ordering."""

import importlib.util
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "kv_workspace", Path(__file__).resolve().parents[1] / "_v41_prefill_kv_workspace.py"
)
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def reference(globals_by_req, swa):
    return torch.cat([t for (g, _), sw in zip(globals_by_req, swa) for t in (g, sw)])


def inputs(batch=3, device="cuda"):
    globals_by_req = [
        (
            torch.full(
                ([16384, 32768, 65536][i % 3], 512),
                i + 1,
                dtype=torch.bfloat16,
                device=device,
            ),
            None,
        )
        for i in range(batch)
    ]
    swa = [
        torch.full(
            ([255, 639, 2175, 4223][i % 4], 512),
            -i - 1,
            dtype=torch.bfloat16,
            device=device,
        )
        for i in range(batch)
    ]
    return globals_by_req, swa


class CPUFallbackTest(unittest.TestCase):
    def test_ragged_descriptor_is_lazy_and_supports_padded_row_starts(self):
        flat = torch.arange(70).view(14, 5)
        rows = workspace.SwaRows(flat, (0, 3, 2), row_starts=(0, 4, 10))
        self.assertIsNone(rows._views)
        self.assertEqual(len(rows), 3)
        self.assertEqual([t.shape[0] for t in rows], [0, 3, 2])
        self.assertTrue(torch.equal(rows[1], flat[4:7]))
        self.assertIs(rows[1], rows[1])
        with self.assertRaises(ValueError):
            workspace.SwaRows(flat, (3, 2))
        with self.assertRaises(ValueError):
            workspace.SwaRows(flat, (3,), row_starts=(13,))

    def test_ragged_descriptor_cpu_fallback(self):
        globals_by_req = [(torch.ones(3, 4), None), (torch.ones(5, 4) * 2, None)]
        source = torch.arange(12).reshape(3, 4).float()
        rows = workspace.SwaRows(source, (2, 1))
        result = workspace.combine_kv({}, globals_by_req, rows)
        self.assertTrue(torch.equal(result, reference(globals_by_req, rows)))
        self.assertIsInstance(workspace.split_swa_rows(source, (2, 1)), list)
        self.assertIsNone(workspace.prepare_global_workspace(rows, (3, 5)))

    def test_cpu_never_retains_batch_workspace(self):
        shared = {}
        globals_by_req = [(torch.ones(3, 4), None), (torch.ones(5, 4) * 2, None)]
        swa = [torch.zeros(2, 4), torch.zeros(1, 4)]
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        self.assertNotIn("prefill_kv_workspace", shared)

    def test_single_request_reuse_still_works(self):
        shared = {}
        globals_by_req = [(torch.ones(3, 4), None)]
        swa = [torch.zeros(2, 4)]
        first = workspace.combine_kv(shared, globals_by_req, swa)
        swa[0].fill_(7)
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
        self.assertTrue(torch.equal(first, reference(globals_by_req, swa)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CUDAWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "swa_refresh",
            Path(__file__).resolve().parents[1] / "_v41_swa_refresh_triton.py",
        )
        cls.refresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.refresh)
        cls.refresh.warmup(torch.device("cuda"))
        patcher = patch.object(workspace, "_refresh_backend", return_value=cls.refresh)
        patcher.start()
        cls.addClassCleanup(patcher.stop)

    def prepared(self, rows, counts):
        plan = workspace.prepare_global_workspace(rows, counts)
        self.assertIsNotNone(plan)
        gs = [
            (
                plan.output.as_strided(
                    (count, rows.flat.shape[1]),
                    plan.output.stride(),
                    start * rows.flat.shape[1],
                ),
                None,
            )
            for count, start in zip(counts, plan.global_starts)
        ]
        for g, _ in gs:
            g.view(torch.int16).random_(-32768, 32767)
        return plan, plan.publish(gs)

    def test_direct_producer_ragged_bits_no_cat_or_request_views(self):
        for counts, sizes, starts in (
            ((0, 1, 257), (31, 0, 129), (0, 64, 128)),
            ((8191,), (257,), None),
            ((17, 1021, 8193, 7, 0), (0, 1, 127, 255, 33), None),
        ):
            total = (
                sum(sizes)
                if starts is None
                else max(start + size for start, size in zip(starts, sizes))
            )
            flat = torch.empty((total, 516), device="cuda", dtype=torch.bfloat16)[
                :, :512
            ]
            flat.view(torch.int16).random_(-32768, 32767)
            rows = workspace.SwaRows(flat, sizes, starts)
            plan, gs = self.prepared(rows, counts)
            expected = reference(gs, rows)
            rows._views = None
            source_ref = weakref.ref(flat)
            with patch.object(torch, "cat", side_effect=AssertionError("direct cat")):
                output = workspace.combine_kv({}, gs, rows)
            self.assertIs(output, plan.output)
            self.assertIsNone(rows._views)
            self.assertTrue(
                torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            )
            del rows, flat
            self.assertIsNone(source_ref())

    def test_direct_producer_geometry_change_cache_clear_and_request_count(self):
        flat = torch.zeros((256 + 127 + 31, 512), device="cuda", dtype=torch.bfloat16)
        rows = workspace.SwaRows(flat, (256, 127, 31))
        plan, gs = self.prepared(rows, (8192, 16381, 8193))
        with self.assertRaises(ValueError):
            plan.publish(gs[:-1])
        shared = {}
        first = workspace.combine_kv(shared, gs, rows)
        # CED clears shared caches and shortens every SWA range. The producer's
        # plan must not be selected just because the global aliases are equal.
        shared.clear()
        shorter = workspace.SwaRows(flat, (128, 127, 31), (0, 256, 383))
        current = workspace.combine_kv(shared, gs, shorter)
        self.assertIsNot(current, first)
        self.assertTrue(
            torch.equal(
                current.view(torch.uint8), reference(gs, shorter).view(torch.uint8)
            )
        )
        flat.fill_(97)
        self.assertIs(workspace.combine_kv(shared, gs, shorter), current)
        self.assertTrue(
            torch.equal(
                current.view(torch.uint8), reference(gs, shorter).view(torch.uint8)
            )
        )

    def test_direct_producer_nondefault_stream_graph_and_lifetime(self):
        flat = torch.zeros((129 + 33, 512), device="cuda", dtype=torch.bfloat16)
        rows = workspace.SwaRows(flat, (129, 33, 0))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan, gs = self.prepared(rows, (3, 257, 1021))
            output = workspace.combine_kv({}, gs, rows)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            workspace.combine_kv({}, gs, rows)
        for value in (17, -13):
            flat.fill_(value)
            graph.replay()
            self.assertTrue(
                torch.equal(
                    output.view(torch.uint8), reference(gs, rows).view(torch.uint8)
                )
            )
        plan_ref, output_ref = weakref.ref(plan), weakref.ref(output)
        del plan, output, graph
        self.assertIsNotNone(plan_ref())
        del gs
        self.assertIsNone(plan_ref())
        self.assertIsNone(output_ref())

    def test_direct_producer_unsupported_unwarmed_and_invalid_ranges(self):
        flat = torch.ones((3, 512), device="cuda", dtype=torch.bfloat16)
        rows = workspace.SwaRows(flat, (1, 2))
        for counts in ((), (1,), (-1, 2), (1.5, 2)):
            self.assertIsNone(workspace.prepare_global_workspace(rows, counts))
        with patch.object(self.refresh, "_READY", {}):
            self.assertIsNone(workspace.prepare_global_workspace(rows, (2, 3)))
        bad = torch.ones(3 * 512 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
            3, 512
        )
        self.assertIsNone(
            workspace.prepare_global_workspace(workspace.SwaRows(bad, (1, 2)), (2, 3))
        )

    def test_ragged_slab_bitwise_refresh_retention_and_padded_storage(self):
        shared = {}
        gs = [
            (
                torch.full(
                    (8192 + i * 257, 512), i + 1, device="cuda", dtype=torch.bfloat16
                ),
                None,
            )
            for i in range(3)
        ]
        for sizes, starts in (((0, 127, 259), None), ((31, 0, 129), (0, 256, 512))):
            total = (
                sum(sizes)
                if starts is None
                else max(s + n for s, n in zip(starts, sizes))
            )
            flat = torch.empty(
                (total, 1024 if starts else 512), device="cuda", dtype=torch.bfloat16
            )[:, :512]
            rows = workspace.split_swa_rows(flat, sizes, row_starts=starts)
            output = workspace.combine_kv(shared, gs, rows)
            old_source = weakref.ref(flat)
            del rows, flat
            self.assertIsNone(old_source())
            flat = torch.empty(
                (total, 1024 if starts else 512), device="cuda", dtype=torch.bfloat16
            )[:, :512]
            flat.view(torch.int16).random_(-32768, 32767)
            rows = workspace.split_swa_rows(flat, sizes, row_starts=starts)
            with patch.object(
                torch, "_foreach_copy_", side_effect=AssertionError("ragged foreach")
            ):
                self.assertIs(workspace.combine_kv(shared, gs, rows), output)
            self.assertIsNone(rows._views)
            self.assertTrue(
                torch.equal(
                    output.view(torch.uint8), reference(gs, rows).view(torch.uint8)
                )
            )

    def test_ragged_slab_reorder_source_and_request_domain_invalidate(self):
        gs, _ = inputs(3)
        shared = {}
        flat = torch.ones((64 + 127 + 256, 512), device="cuda", dtype=torch.bfloat16)
        rows = workspace.SwaRows(flat, (64, 127, 256))
        output = workspace.combine_kv(shared, gs, rows)
        for mutation in ("source", "reorder", "layout"):
            old = weakref.ref(output)
            del output
            if mutation == "source":
                gs[1] = (gs[1][0].clone(), None)
            elif mutation == "reorder":
                gs.reverse()
            else:
                rows = workspace.SwaRows(flat, (127, 256, 64))
            output = workspace.combine_kv(shared, gs, rows)
            self.assertIsNone(old())
            self.assertTrue(torch.equal(output, reference(gs, rows)))
        old = weakref.ref(output)
        del output
        shared.pop("prefill_kv_workspace")
        self.assertIsNone(old())

    def test_ragged_slab_graph_updates_and_stream_order(self):
        gs, _ = inputs(3)
        flat = torch.zeros((129 + 33 + 0, 512), device="cuda", dtype=torch.bfloat16)
        rows = workspace.SwaRows(flat, (129, 33, 0))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        shared = {}
        with torch.cuda.stream(stream):
            output = workspace.combine_kv(shared, gs, rows)
            workspace.combine_kv(shared, gs, rows)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            workspace.combine_kv(shared, gs, rows)
        for value in (37, -18):
            flat.fill_(value)
            graph.replay()
            self.assertTrue(torch.equal(output, reference(gs, rows)))

    def test_ragged_copy_runtime_width_stride_empty_rows_and_skew(self):
        # The startup kernel must handle different widths, noncontiguous rows,
        # empty requests and a long final tail without compiling a new variant.
        for width in (4, 68, 512):
            sizes = (1, 0, 1, 8191)
            total = sum(sizes)
            source = torch.empty(
                (total, width + 4), device="cuda", dtype=torch.bfloat16
            )[:, :width]
            source.view(torch.int16).random_(-32768, 32767)
            output = torch.zeros(
                (total + 4 * 3, width), device="cuda", dtype=torch.bfloat16
            )
            expected = output.clone()
            layout, src, dst, tiles = [], 0, 0, 0
            for size in sizes:
                dst += 3
                tiles += (
                    size * (width // 4) + self.refresh._BLOCK - 1
                ) // self.refresh._BLOCK
                layout.append((src, dst, size, tiles))
                expected[dst : dst + size].copy_(source[src : src + size])
                src += size
                dst += size
            metadata = torch.tensor(layout, dtype=torch.int64, device="cuda")
            with patch.object(
                self.refresh._copy_swa_rows,
                "warmup",
                side_effect=AssertionError("request compiled a new SWA kernel"),
            ):
                self.refresh.copy_rows(source, output, metadata, tiles)
            self.assertTrue(
                torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_small_mixed_batch_negative_cache_and_readmission(self):
        shared = {}
        globals_by_req = [
            (
                torch.full(
                    ([127, 511, 1021][i % 3], 512),
                    i + 1,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                None,
            )
            for i in range(32)
        ]
        swa = [
            torch.zeros(
                ([0, 13, 67, 129][i % 4], 512), device="cuda", dtype=torch.bfloat16
            )
            for i in range(32)
        ]
        with patch.object(
            torch, "_foreach_copy_", side_effect=AssertionError("small batch foreach")
        ):
            for change in ("first", "layout", "order", "source"):
                if change == "layout":
                    swa[0] = torch.ones((19, 1024), device="cuda")[:, ::2]
                elif change == "order":
                    globals_by_req.reverse()
                    swa.reverse()
                elif change == "source":
                    globals_by_req[0] = (globals_by_req[0][0].clone(), None)
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
                self.assertIsNone(shared["prefill_kv_workspace"][1])
                self.assertIsNone(shared["prefill_kv_workspace"][2])
                ref = weakref.ref(output)
                del output
                self.assertIsNone(ref())
        # A new, profitable source group must not inherit the cached rejection.
        globals_by_req, swa = inputs(32)
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertIs(shared["prefill_kv_workspace"][2], output)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)

    def test_raw_bf16_bits_and_empty_tail(self):
        shared = {}
        globals_by_req, swa = inputs()
        swa[0] = swa[0][:0]
        output = workspace.combine_kv(shared, globals_by_req, swa)
        # Include signed zero, NaN payloads, infinities and arbitrary mantissas.
        for sw in swa:
            sw.view(torch.int16).random_(-32768, 32767)
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )

    def test_misaligned_contiguous_source_falls_back(self):
        shared = {}
        globals_by_req, swa = inputs()
        workspace.combine_kv(shared, globals_by_req, swa)
        base = torch.ones(swa[0].numel() + 1, device="cuda", dtype=torch.bfloat16)
        swa[0] = base[1:].view_as(swa[0])
        self.assertTrue(swa[0].is_contiguous())
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertNotIn("prefill_kv_workspace", shared)
        self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))

    def test_mixed_batch32_exact_bytes_and_no_swa_retention(self):
        shared = {}
        globals_by_req, swa = inputs(32)
        output = workspace.combine_kv(shared, globals_by_req, swa)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )
        previous = [weakref.ref(t) for t in swa]
        swa = [torch.full_like(t, i + 97) for i, t in enumerate(swa)]
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), output)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8),
                reference(globals_by_req, swa).view(torch.uint8),
            )
        )
        for i, (g, _) in enumerate(globals_by_req):
            self.assertTrue(torch.all(g == i + 1).item())

    def test_source_replacement_reordering_shape_change_and_clear(self):
        shared = {}
        globals_by_req, swa = inputs()
        output = workspace.combine_kv(shared, globals_by_req, swa)
        for mutation in ("source", "order", "shape"):
            with self.subTest(mutation=mutation):
                old = weakref.ref(output)
                del output
                if mutation == "source":
                    old_source = weakref.ref(globals_by_req[1][0])
                    globals_by_req[1] = (
                        torch.full_like(globals_by_req[1][0], 53),
                        None,
                    )
                elif mutation == "order":
                    globals_by_req.reverse()
                    swa.reverse()
                else:
                    swa[0] = swa[0][:-1]
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertIsNone(old())
                if mutation == "source":
                    self.assertIsNone(old_source())
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))
        old = weakref.ref(output)
        del output
        shared.pop("prefill_kv_workspace")
        self.assertIsNone(old())

    def test_unsupported_layout_dtype_and_small_global_drop_cache(self):
        for reason in ("strided", "dtype", "small", "copy_volume"):
            with self.subTest(reason=reason):
                shared = {}
                globals_by_req, swa = inputs()
                output = workspace.combine_kv(shared, globals_by_req, swa)
                old = weakref.ref(output)
                del output
                if reason == "strided":
                    swa[0] = torch.zeros(
                        (255, 1024), device="cuda", dtype=torch.bfloat16
                    )[:, ::2]
                elif reason == "dtype":
                    swa[0] = swa[0].float()
                elif reason == "small":
                    globals_by_req = [(g[:1], None) for g, _ in globals_by_req]
                else:
                    swa = [torch.zeros_like(g) for g, _ in globals_by_req]
                    swa[0] = torch.cat((swa[0], swa[0]))
                output = workspace.combine_kv(shared, globals_by_req, swa)
                self.assertIsNone(old())
                if reason == "small":
                    self.assertIsNone(shared["prefill_kv_workspace"][2])
                    for sw in swa:
                        sw.fill_(18)
                    output = workspace.combine_kv(shared, globals_by_req, swa)
                else:
                    self.assertNotIn("prefill_kv_workspace", shared)
                self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))

    def test_batch_single_batch_transition(self):
        shared = {}
        globals_by_req, swa = inputs()
        workspace.combine_kv(shared, globals_by_req, swa)
        for gs, ss in ((globals_by_req[:1], [swa[0][:128]]), (globals_by_req, swa)):
            output = workspace.combine_kv(shared, gs, ss)
            self.assertTrue(torch.equal(output, reference(gs, ss)))

    def test_nondefault_stream_order_and_consumer_snapshot(self):
        globals_by_req, swa = inputs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        shared = {}
        with torch.cuda.stream(stream):
            first = workspace.combine_kv(shared, globals_by_req, swa)
            for sw in swa:
                sw.fill_(91)
            self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
            snapshot = first.clone()
            expected = reference(globals_by_req, swa)
            for sw in swa:
                sw.fill_(17)
            self.assertIs(workspace.combine_kv(shared, globals_by_req, swa), first)
            expected_updated = reference(globals_by_req, swa)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(torch.equal(snapshot, expected))
        self.assertTrue(torch.equal(first, expected_updated))

    def test_graph_replay_updates_same_output(self):
        globals_by_req, swa = inputs()
        shared = {}
        output = workspace.combine_kv(shared, globals_by_req, swa)
        workspace.combine_kv(shared, globals_by_req, swa)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.combine_kv(shared, globals_by_req, swa)
        for value in (39, -13):
            for sw in swa:
                sw.fill_(value)
            graph.replay()
            self.assertTrue(torch.equal(output, reference(globals_by_req, swa)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
