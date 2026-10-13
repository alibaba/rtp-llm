import gc
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import Qwen4ExpNGramEmbedding


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class PLEPrefetchTest(unittest.TestCase):
    def test_fp8_cpu_table_prefetch_and_graph_use_bf16_selected_rows(self):
        shards = [
            (torch.arange(5 * 160).reshape(5, 160) % 32 + i).to(torch.float8_e4m3fn)
            for i in range(2)
        ]
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN4_PLE_CPU_UVA": "1",
                "RTP_LLM_QWEN4_PLE_CPU_PREFETCH": "1",
            },
        ):
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
        module._graph_gather_required = True
        module.prepare_cpu_graph_rows(ids, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module._gather_local(ids).clone()
        ids.copy_(torch.tensor([19, 9, 7, 2], dtype=torch.int64, device="cuda"))
        module.prepare_cpu_graph_rows(ids, 1)
        graph.replay()
        torch.testing.assert_close(
            output, module._gather_torch(ids.cpu()).cuda(), atol=0, rtol=0
        )

    def _module(self, dtype=torch.bfloat16):
        shards = [
            (torch.arange(5 * 160).reshape(5, 160) + i * 1000).to(dtype)
            for i in range(2)
        ]
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN4_PLE_CPU_UVA": "1",
                "RTP_LLM_QWEN4_PLE_CPU_PREFETCH": "1",
            },
        ):
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
        from rtp_llm.models_py.modules.qwen4_exp.ple_prefetch import PLEPrefetcher

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

    def test_external_wait_graph_replay_observes_every_new_lookup(self):
        module = self._module()
        ids = torch.tensor([5, 0, 15, 19], device="cuda", dtype=torch.int64)
        module.prepare_cpu_graph_rows(ids, 123)
        module._prepared_cpu_graph_rows((4, 160), ids.device).clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module._prepared_cpu_graph_rows((4, 160), ids.device).clone()
        address = module._cpu_graph_rows.data_ptr()
        for index in range(20):
            with torch.cuda.stream(module._prefetcher.stream):
                torch.cuda._sleep(200_000)
            new_ids = torch.tensor(
                [5 + index % 5, 16 + index % 4, 0, 20],
                device="cuda",
                dtype=torch.int64,
            )
            ids.copy_(new_ids)
            module.prepare_cpu_graph_rows(ids, 123)
            graph.replay()
            expected = module._gather_torch(new_ids.cpu()).to("cuda")
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            self.assertEqual(module._cpu_graph_rows.data_ptr(), address)
        module.prepare_cpu_graph_rows(ids, 456)
        self.assertNotEqual(module._cpu_graph_rows.data_ptr(), address)


if __name__ == "__main__":
    unittest.main()
