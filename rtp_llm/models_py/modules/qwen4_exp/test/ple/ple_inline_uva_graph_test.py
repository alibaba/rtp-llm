import gc
import unittest
import weakref

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class PLEInlineUVAGraphTest(unittest.TestCase):
    def make_module(self, *, inline=True, partial=False, scale=0.25):
        indices = list(range(0, 128, 2)) if partial else list(range(128))
        shards = [
            ((torch.arange(67 * 160).reshape(67, 160) + i) % 33 - 16).to(
                torch.float8_e4m3fn
            )
            for i in indices
        ]
        return Qwen4ExpNGramEmbedding(
            shards,
            torch.full((32,), 257, device="cuda", dtype=torch.int64),
            torch.arange(32, device="cuda", dtype=torch.int64) * 257,
            torch.tensor([3, 5, 7, 11, 13], device="cuda", dtype=torch.int64),
            ngram_size=5,
            eos_token_id=0,
            shard_indices=indices,
            total_shards=128,
            distributed_reduce=partial,
            weight_scale=torch.tensor([scale]),
            inline_cpu_uva_graph=inline,
        )

    def test_dynamic_tokens_context_reorder_eos_and_graph_keys(self):
        torch.manual_seed(20261019)
        for scale in (0.25, 1.75):
            module = self.make_module(scale=scale)
            owners = [weakref.ref(x) for x in module.shards]
            pool = torch.randint(0, 1000, (43, 4), device="cuda")
            captures = []
            for batch, width in ((1, 1), (3, 4), (4, 4), (7, 4), (8, 4)):
                ids = torch.randint(0, 1000, (batch, width), device="cuda")
                blocks = torch.arange(batch, device="cuda", dtype=torch.int64) + 1

                def call():
                    history = torch.cat((pool.index_select(0, blocks), ids), dim=1)
                    with module.lookup_context(True):
                        return module(history, width)

                for _ in range(3):
                    call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call()
                captures.append((graph, ids, blocks, width, output))
            gc.collect()
            for step in range(6):
                pool.copy_(torch.randint(0, 1000, pool.shape, device="cuda"))
                pool[step, step % 4] = 0
                snapshot = pool.clone()
                for graph, ids, blocks, width, output in reversed(captures):
                    ids.copy_(torch.randint(0, 1000, ids.shape, device="cuda"))
                    ids[:, step % width] = 0
                    blocks.copy_((blocks + 7).remainder(pool.shape[0]))
                    history = torch.cat((pool.index_select(0, blocks), ids), dim=1)
                    hashed = module.hashed_ids(history, width)
                    expected = module._gather_torch(hashed.cpu()).cuda().flatten(-2)
                    graph.replay()
                    self.assertTrue(torch.equal(output, expected))
                    self.assertTrue(torch.equal(pool, snapshot))
            self.assertEqual(module._cpu_graph_buffers, {})
            self.assertIsNone(module._cpu_graph_rows)
            self.assertEqual(module.resource_usage()["graph_host_bytes"], 0)
            self.assertTrue(all(x() is not None and x().is_pinned() for x in owners))
            for graph, *_ in captures:
                graph.reset()
            captures.clear()
            module.close()
            gc.collect()
            self.assertTrue(all(x() is None for x in owners))

    def test_partial_shards_invalid_ids_and_global_scale(self):
        module = self.make_module(partial=True, scale=1.75)
        ids = torch.tensor([-1, 0, 67, 134, 8575, 8576], device="cuda")
        for _ in range(3):
            module._gather_local(ids)
        graph = torch.cuda.CUDAGraph()
        with module.lookup_context(True), torch.cuda.graph(graph):
            output = module._gather_local(ids)
        for values in ([-1, 0, 67, 134, 8575, 8576], [8576, 201, 200, 133, 66, -7]):
            ids.copy_(torch.tensor(values, device="cuda"))
            graph.replay()
            self.assertTrue(torch.equal(output, module._gather_torch(ids.cpu()).cuda()))
        graph.reset()
        module.close()

    def test_default_staged_and_inline_consumers_match(self):
        staged = self.make_module(inline=False)
        inline = self.make_module()
        history = torch.randint(0, 1000, (4, 8), device="cuda")
        graphs = []
        for module in (staged, inline):
            if not module.inline_cpu_uva_graph:
                module.prepare_cpu_graph_rows(module.hashed_ids(history, 4), 123)
            with module.lookup_context(True):
                for _ in range(3):
                    module(history, 4)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = module(history, 4)
            graphs.append((graph, output))
        for _ in range(4):
            history.copy_(torch.randint(0, 1000, history.shape, device="cuda"))
            staged.prepare_cpu_graph_rows(staged.hashed_ids(history, 4), 123)
            for graph, _ in graphs:
                graph.replay()
            self.assertTrue(torch.equal(graphs[0][1], graphs[1][1]))
        self.assertEqual(staged.resource_usage()["graph_host_bytes"], 0)
        self.assertGreater(staged.resource_usage()["graph_device_bytes"], 0)
        for graph, _ in graphs:
            graph.reset()
        staged.close()
        inline.close()

    def test_capture_geometry_and_closed_module_rejected(self):
        module = self.make_module()
        with module.lookup_context(True):
            with self.assertRaisesRegex(RuntimeError, "history geometry"):
                module(torch.ones((1, 5), device="cuda", dtype=torch.int64), 4)
        module.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            module.gather(torch.zeros((1,), device="cuda", dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
