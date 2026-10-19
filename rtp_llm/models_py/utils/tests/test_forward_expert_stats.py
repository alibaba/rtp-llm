import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "forward_expert_stats_under_test", Path(__file__).parents[1] / "forward_expert_stats.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ForwardExpertStatsTest(unittest.TestCase):
    def stats(self):
        return SimpleNamespace(log_stats_buf=torch.zeros((2, 8), dtype=torch.int32),
                               gpu_loads_buf=torch.zeros((2, 4), dtype=torch.int32))

    def record(self, ids, layer=0):
        module.record_expert_routes(torch.tensor(ids, dtype=torch.int32),
                                   layer_id=layer, num_experts=8, ep_size=4)

    def test_counts_chunks_padding_and_empty(self):
        stats = self.stats()
        with module.forward_expert_stats(stats):
            self.record([[0, 2, 6], [0, 3, 7]])
            self.record([[7, -1, 8]])
            self.record([])
            self.record([[4, 5]], layer=1)
        self.assertEqual(stats.log_stats_buf.tolist(), [[2, 0, 1, 1, 0, 0, 1, 2], [0, 0, 0, 0, 1, 1, 0, 0]])
        self.assertEqual(stats.gpu_loads_buf.tolist(), [[2, 2, 0, 3], [0, 0, 2, 0]])

    def test_scope_restores_after_exception_and_nested_none(self):
        stats = self.stats()
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with module.forward_expert_stats(stats):
                with module.forward_expert_stats(None):
                    self.record([[1]])
                self.record([[2]])
                raise RuntimeError("sentinel")
        self.record([[3]])
        self.assertEqual(int(stats.log_stats_buf.sum()), 1)
        self.assertIsNone(module._CURRENT_STATS.get())

    def test_independent_handles_and_raw_tp_work(self):
        handles = [self.stats() for _ in range(4)]
        for stats in handles:
            with module.forward_expert_stats(stats):
                self.record([[0, 2, 4, 6]])
        self.assertEqual(sum(int(s.log_stats_buf.sum()) for s in handles), 16)
        self.assertEqual(len({s.gpu_loads_buf.data_ptr() for s in handles}), 4)

    def test_reject_redundant_layout(self):
        with module.forward_expert_stats(self.stats()), self.assertRaises(ValueError):
            module.record_expert_routes(torch.tensor([0]), layer_id=0, num_experts=9, ep_size=4)


if __name__ == "__main__":
    unittest.main()
