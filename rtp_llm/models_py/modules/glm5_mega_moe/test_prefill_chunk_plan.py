"""CPU scheduling and shared-output lifetime tests; no kernel equivalence claim."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "prefill_chunk_plan_test_module", Path(__file__).with_name("prefill_chunk_plan.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PrefillChunkPlan = module.PrefillChunkPlan


class ChunkPlanTest(unittest.TestCase):
    def run_rank(self, rows, plan):
        hidden = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) + 1
        weights = torch.ones((rows, 4), dtype=torch.float32)
        indices = torch.arange(4).expand(rows, 4).contiguous()
        scratch = torch.empty((plan.capacity, 2))
        calls = []

        def forward(h, w, ids):
            calls.append((h.clone(), w.clone(), ids.clone()))
            # Deliberately overwrite all scratch on every call, including dummy.
            scratch.fill_(-777)
            scratch[: len(h)].copy_(h * 3)
            return scratch[: len(h)]

        output = module.run_prefill_chunks(hidden, weights, indices, forward, plan)
        torch.testing.assert_close(output, hidden * 3, rtol=0, atol=0)
        self.assertEqual(len(calls), plan.chunks)
        real_chunks = (rows + plan.capacity - 1) // plan.capacity
        self.assertEqual(sum(len(c[0]) for c in calls[:real_chunks]), rows)
        for h, w, ids in calls[real_chunks:]:
            self.assertEqual(h.shape, (1, 2))
            self.assertEqual(torch.count_nonzero(h).item(), 0)
            self.assertEqual(torch.count_nonzero(w).item(), 0)
            self.assertEqual(ids.tolist(), [[0, 1, 2, 3]])
        return calls

    def test_mixed_rank_boundaries(self):
        capacity = 8448
        for rows in (
            [1, 81920],
            [8448, 8449],
            [0, 1],
            [1] * 8,
            [1, 8448, 8449, 16896, 17, 81920, 0, 3],
        ):
            with self.subTest(rows=rows):
                count = max(module.local_chunk_count(r, capacity) for r in rows)
                plan = PrefillChunkPlan(capacity, count)
                for r in rows:
                    self.run_rank(r, plan)

    def test_short_rank_output_survives_dummy_overwrite(self):
        self.run_rank(1, PrefillChunkPlan(8, 10))

    def test_successive_plans_do_not_retain_large_count(self):
        self.run_rank(9, PrefillChunkPlan(8, 2))
        self.run_rank(1, PrefillChunkPlan(8, 1))
        self.run_rank(0, PrefillChunkPlan(8, 3))

    def test_reject_underprovisioned_plan_before_any_call(self):
        with self.assertRaises(ValueError):
            module.run_prefill_chunks(
                torch.ones(9, 2),
                torch.ones(9, 4),
                torch.zeros(9, 4),
                lambda *args: self.fail("must validate before launching"),
                PrefillChunkPlan(8, 1),
            )

    def test_invalid_plan(self):
        for capacity, count in [(0, 1), (1, 0), (-1, 1)]:
            with self.assertRaises(ValueError):
                PrefillChunkPlan(capacity, count)
        with self.assertRaises(ValueError):
            module.local_chunk_count(-1, 8)


if __name__ == "__main__":
    unittest.main()
