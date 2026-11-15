import gc
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class PLEPrefetchTest(unittest.TestCase):
    def test_fp8_cpu_table_prefetch_and_graph_use_bf16_selected_rows(self):
        shards = [
            (torch.arange(5 * 160).reshape(5, 160) % 32 + i).to(torch.float8_e4m3fn)
            for i in range(2)
        ]
        module = Qwen4ExpNGramEmbedding(
            shards,
            torch.tensor([8, 8], dtype=torch.int64, device="cuda"),
            torch.tensor([0, 8], dtype=torch.int64, device="cuda"),
            torch.tensor([3, 5], dtype=torch.int64, device="cuda"),
            ngram_size=2,
            eos_token_id=0,
            shard_indices=[1, 3],
            total_shards=4,
            distributed_reduce=True,
            weight_scale=torch.tensor([0.25]),
        )
        self.assertTrue(module.supports_prefetch)
        self.assertTrue(all(t.dtype == torch.float8_e4m3fn for t in module.shards))
        ids = torch.tensor([5, 16, -1, 0], dtype=torch.int64, device="cuda")
        expected = module._gather_torch(ids.cpu()).cuda()
        handle = module.start_prefetch(ids)
        rows = module.consume_prefetch(handle)
        self.assertEqual(rows.dtype, torch.bfloat16)
        torch.testing.assert_close(rows, expected, atol=0, rtol=0)
        module.finish_prefetch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module._gather_local(ids).clone()
        ids.copy_(torch.tensor([19, 9, 7, 2], dtype=torch.int64, device="cuda"))
        graph.replay()
        torch.testing.assert_close(
            output, module._gather_torch(ids.cpu()).cuda(), atol=0, rtol=0
        )

    def _module(self, dtype=torch.bfloat16):
        shards = [
            (torch.arange(5 * 160).reshape(5, 160) + i * 1000).to(dtype)
            for i in range(2)
        ]
        module = Qwen4ExpNGramEmbedding(
            shards,
            torch.tensor([8, 8], dtype=torch.int64, device="cuda"),
            torch.tensor([0, 8], dtype=torch.int64, device="cuda"),
            torch.tensor([3, 5], dtype=torch.int64, device="cuda"),
            ngram_size=2,
            eos_token_id=0,
            shard_indices=[1, 3],
            total_shards=4,
            distributed_reduce=True,
        )
        self.assertTrue(module.supports_prefetch)
        self.assertTrue(
            all(t.device.type == "cpu" and t.is_pinned() for t in module.shards)
        )
        return module

    def test_legacy_event_api_supports_eager_prefetch_and_inline_graph(self):
        original_event = torch.cuda.Event

        def legacy_event(enable_timing=False, blocking=False, interprocess=False):
            return original_event(enable_timing, blocking, interprocess)

        ids = torch.tensor([5, 16, -1, 0], dtype=torch.int64, device="cuda")
        with patch.object(torch.cuda, "Event", legacy_event):
            module = self._module()
            handle = module.start_prefetch(ids)
            eager = module.consume_prefetch(handle).clone()
            module.finish_prefetch()
        torch.testing.assert_close(
            eager, module._gather_torch(ids.cpu()).cuda(), rtol=0, atol=0
        )
        module._gather_local(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = module._gather_local(ids).clone()
        ids.copy_(torch.tensor([19, 9, 7, 2], dtype=torch.int64, device="cuda"))
        graph.replay()
        torch.testing.assert_close(
            actual, module._gather_torch(ids.cpu()).cuda(), rtol=0, atol=0
        )
        graph.reset()
        module.close()

    def test_lookup_and_reused_slots_preserve_local_remote_and_invalid_rows(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                module = self._module(dtype)
                originals = [t.clone() for t in module.shards]
                for length in (1, 7, 17, 127, 3):
                    ids = (
                        torch.arange(length, device="cuda", dtype=torch.int64) % 22 - 1
                    )
                    expected = module._gather_torch(ids.cpu()).to("cuda")
                    handle = module.start_prefetch(ids)
                    actual = module.consume_prefetch(handle).clone()
                    module.finish_prefetch()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(len(module._prefetcher._eager_slots), 1)
                for actual, expected in zip(module.shards, originals):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_eager_reuse_waits_for_consumers_on_a_different_stream(self):
        module = self._module()
        first = torch.tensor([5, 6, 16], device="cuda", dtype=torch.int64)
        second = torch.tensor([19, 9, 7], device="cuda", dtype=torch.int64)
        first_expected = module._gather_torch(first.cpu()).to("cuda")
        second_expected = module._gather_torch(second.cpu()).to("cuda")
        consumer = torch.cuda.Stream()
        handle = module.start_prefetch(first)
        with torch.cuda.stream(consumer):
            rows = module.consume_prefetch(handle)
            torch.cuda._sleep(2_000_000)
            first_output = rows.clone()
            module.finish_prefetch()
        handle = module.start_prefetch(second)
        second_output = module.consume_prefetch(handle).clone()
        module.finish_prefetch()
        torch.cuda.synchronize()
        torch.testing.assert_close(first_output, first_expected, atol=0, rtol=0)
        torch.testing.assert_close(second_output, second_expected, atol=0, rtol=0)

    def test_close_drains_consumers_and_releases_lookup_buffers(self):
        module = self._module()
        ids = torch.tensor([5, 6, 16], device="cuda", dtype=torch.int64)
        expected = module._gather_torch(ids.cpu()).to("cuda")
        consumer = torch.cuda.Stream()
        handle = module.start_prefetch(ids)
        with torch.cuda.stream(consumer):
            rows = module.consume_prefetch(handle)
            torch.cuda._sleep(2_000_000)
            output = rows.clone()
            module.finish_prefetch()
        prefetcher = module._prefetcher
        module.close()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        self.assertEqual(prefetcher.retained_eager_bytes, 0)
        self.assertEqual(module.shards, [])
        self.assertFalse(module.supports_prefetch)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            module.gather(ids)
        module.close()

    def test_active_slot_and_foreign_handle_are_rejected(self):
        module = self._module()
        ids = torch.tensor([5, 19], device="cuda", dtype=torch.int64)
        handle = module.start_prefetch(ids)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            module.start_prefetch(ids)
        other = self._module()
        with self.assertRaisesRegex(RuntimeError, "different embedding"):
            other.consume_prefetch(handle)
        module.consume_prefetch(handle)
        module.finish_prefetch()

    def test_abandoned_lookup_survives_slot_growth_and_allocator_churn(self):
        module = self._module()
        first = torch.tensor([5, 6, 16], device="cuda", dtype=torch.int64)
        larger = torch.arange(127, device="cuda", dtype=torch.int64) % 20
        module._gather_local(first)
        module._gather_local(larger)
        torch.cuda.synchronize()
        with torch.cuda.stream(module._prefetcher.stream):
            torch.cuda._sleep(50_000_000)
        handle = module.start_prefetch(first)
        self.assertFalse(handle.ready.query())
        module.finish_prefetch()
        del handle
        handle = module.start_prefetch(larger)
        guards = [
            torch.full((4, 160), 17.0, device="cuda", dtype=torch.bfloat16)
            for _ in range(32)
        ]
        actual = module.consume_prefetch(handle).clone()
        module.finish_prefetch()
        torch.cuda.synchronize()
        expected = module._gather_torch(larger.cpu()).to("cuda")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for tensor in guards:
            torch.testing.assert_close(
                tensor, torch.full_like(tensor, 17.0), atol=0, rtol=0
            )

    def test_module_release_drains_unconsumed_uva_reads(self):
        module = self._module()
        ids = torch.tensor([5, 16], device="cuda", dtype=torch.int64)
        module._gather_local(ids)
        torch.cuda.synchronize()
        with torch.cuda.stream(module._prefetcher.stream):
            torch.cuda._sleep(50_000_000)
        handle = module.start_prefetch(ids)
        ready = handle.ready
        self.assertFalse(ready.query())
        del handle
        del module
        gc.collect()
        self.assertTrue(ready.query())

    def test_memory_budget_falls_back_without_extra_allocation(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_prefetch import PLEPrefetcher

        module = self._module()
        module._prefetcher = PLEPrefetcher(
            module.shards,
            module.shard_indices,
            module.total_shards,
            module._cpu_uva_pointers,
            max_eager_bytes=640,
        )
        small = torch.tensor([5, 16], device="cuda", dtype=torch.int64)
        handle = module.start_prefetch(small)
        self.assertIsNotNone(handle)
        self.assertIsNone(
            module.start_prefetch(torch.arange(17, device="cuda"), slot=1)
        )
        module.consume_prefetch(handle)
        module.finish_prefetch()
        torch.cuda.synchronize()
        handle = module.start_prefetch(small, slot=1)
        self.assertIsNotNone(handle)
        module.consume_prefetch(handle)
        module.finish_prefetch()
        self.assertLessEqual(
            sum(value[0].nbytes for value in module._prefetcher._eager_slots.values()),
            640,
        )

    def test_prefetch_preserves_embedding_module_hooks(self):
        module = self._module()
        ids = torch.tensor([5, 16], device="cuda", dtype=torch.int64)
        for register in (
            module.register_forward_pre_hook,
            module.register_forward_hook,
        ):
            handle = register(lambda *args: None)
            try:
                self.assertFalse(module.supports_prefetch)
                self.assertIsNone(module.start_prefetch(ids))
            finally:
                handle.remove()
        self.assertTrue(module.supports_prefetch)

    def test_prefetch_consumption_during_capture_is_rejected(self):
        module = self._module()
        ids = torch.tensor([5, 19], device="cuda", dtype=torch.int64)
        handle = module.start_prefetch(ids)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "eager execution"):
                module.consume_prefetch(handle)
        module.finish_prefetch()
        module.close()

    def test_native_graph_prefetch_replays_fork_gather_join_with_fixed_banks(self):
        module = self._module()
        graphs = []
        try:
            # Separate banks permit alternating buckets without replacing any
            # address already referenced by a captured graph.
            for count in (2, 4):
                ids = torch.arange(count, device="cuda", dtype=torch.int64) + 5
                module.consume_prefetch(module.start_graph_prefetch(ids))
                module.finish_graph_warmup()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    handle = module.start_graph_prefetch(ids)
                    output = module.consume_prefetch(handle).clone()
                graphs.append((graph, ids, output, handle.bank.rows.data_ptr()))
            retained = module.resource_usage()["graph_retained_bytes"]
            self.assertEqual(retained, (2 + 4) * (8 + 160 * 2))
            self.assertEqual(module.resource_usage()["eager_retained_bytes"], 0)
            self.assertFalse(module._prefetcher._active_slots)
            for step in range(4):
                for graph, ids, output, pointer in reversed(graphs):
                    ids.copy_(
                        (torch.arange(ids.numel(), device="cuda") * 7 + step) % 22 - 1
                    )
                    graph.replay()
                    torch.testing.assert_close(
                        output, module._gather_torch(ids.cpu()).cuda(), atol=0, rtol=0
                    )
                    bank = module._prefetcher._graph_banks[tuple(ids.shape)]
                    self.assertEqual(bank.rows.data_ptr(), pointer)
            self.assertEqual(module.resource_usage()["graph_retained_bytes"], retained)
        finally:
            for graph, *_ in graphs:
                graph.reset()
            module.close()

    def test_graph_bank_budget_owner_shape_stream_and_capture_teardown_guards(self):
        module = self._module()
        ids = torch.tensor([5, 16], device="cuda", dtype=torch.int64)
        prefetcher = module._prefetcher
        prefetcher._max_eager_bytes = 2 * (8 + 160 * 2)
        self.assertIsNone(
            module.start_graph_prefetch(
                torch.zeros(3, device="cuda", dtype=torch.int64)
            )
        )
        handle = module.start_graph_prefetch(ids)
        module.consume_prefetch(handle)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "finish warmup"):
                module.start_graph_prefetch(ids)
        module.finish_graph_warmup()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module.start_graph_prefetch(ids)
            module.consume_prefetch(captured)
        self.assertIsNone(module.start_prefetch(ids))
        other = self._module()
        try:
            with self.assertRaisesRegex(RuntimeError, "different embedding"):
                other.consume_prefetch(handle)
            foreign_stream = torch.cuda.Stream()
            with torch.cuda.stream(foreign_stream):
                with self.assertRaisesRegex(RuntimeError, "producer stream"):
                    module.consume_prefetch(handle)
                with patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "producer stream"):
                        module.start_graph_prefetch(ids)
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "warmed before capture"):
                    module.start_graph_prefetch(ids.view(1, 2))
                with self.assertRaisesRegex(RuntimeError, "capture ends"):
                    module.close()
            self.assertIs(module._prefetcher, prefetcher)
            self.assertTrue(module.shards)
            graph.reset()
            retained = prefetcher.retained_bytes
            with patch.object(
                torch.cuda, "synchronize", side_effect=RuntimeError("drain failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "drain failed"):
                    module.close()
            self.assertEqual(prefetcher.retained_bytes, retained)
            self.assertFalse(prefetcher._closed)
            module.clear_graph_banks()
            self.assertEqual(prefetcher.retained_graph_bytes, 0)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                module.consume_prefetch(handle)
            module.consume_prefetch(module.start_graph_prefetch(ids))
            module.finish_graph_warmup()
        finally:
            graph.reset()
            other.close()
            module.close()


if __name__ == "__main__":
    unittest.main()
