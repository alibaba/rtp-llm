"""Real graph replay after production MoE output-cache capacity growth."""

import gc
import unittest
import weakref
from contextlib import ExitStack
from unittest import mock

import torch
from moe_cache_test_support import moe_layer


class FinalOutCacheCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is a GPU-required target: absence is failure, not a silent skip.
        if not torch.cuda.is_available():
            raise RuntimeError("FinalOutCacheCudaTest requires its leased CUDA GPU")

    def setUp(self):
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)
        self._patches.enter_context(
            mock.patch.object(moe_layer, "_FINAL_OUT_CACHE", {})
        )
        self._patches.enter_context(
            mock.patch.object(moe_layer, "_FINAL_OUT_RETIRED", [])
        )
        self.device = torch.device("cuda:0")

    def get(self, capacity):
        return moe_layer._get_or_create_final_out(
            capacity, 32, torch.bfloat16, self.device
        )

    def test_replay_after_growth_preserves_old_owner_and_new_tenants(self):
        x = torch.full((4, 32), 3.0, dtype=torch.bfloat16, device=self.device)
        self.get(4).copy_(x)
        torch.cuda.synchronize()
        old_ref = weakref.ref(self.get(4))
        old_ptr = self.get(4).data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.get(4).copy_(x)
        current = self.get(64)
        current.fill_(7)
        self.assertIsNotNone(old_ref(), "growth freed a captured allocation")
        self.assertEqual(old_ref().data_ptr(), old_ptr)
        sentinels = [
            torch.full((1,), 61, dtype=torch.int32, device=self.device)
            for _ in range(64)
        ]
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(old_ref(), x)
        torch.testing.assert_close(current, torch.full_like(current, 7))
        self.assertTrue(all(t.item() == 61 for t in sentinels))
        graph.reset()

    def test_multiple_graph_sizes_survive_repeated_growth(self):
        records = []
        for size in (1, 2, 4, 8):
            x = torch.full(
                (size, 32), float(size), dtype=torch.bfloat16, device=self.device
            )
            self.get(size).copy_(x)
            torch.cuda.synchronize()
            owner = weakref.ref(self.get(size))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.get(size).copy_(x)
            records.append((graph, x, owner))
        for size in (9, 17, 65, 129):
            self.get(size).fill_(99)
        for graph, x, owner in reversed(records):
            self.assertIsNotNone(
                owner(), "captured owner lost across graph-size growth"
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(owner(), x)
        current = self.get(129)
        self.assertTrue(bool((current == 99).all().item()))
        total = current.numel() + sum(t.numel() for t in moe_layer._FINAL_OUT_RETIRED)
        self.assertLess(total, 2 * current.numel())
        for graph, _, _ in records:
            graph.reset()
        del records
        gc.collect()


if __name__ == "__main__":
    unittest.main()
