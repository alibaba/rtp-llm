"""Pure planning coverage and budget contracts, without a GPU runtime."""

import importlib.util
import sys
import unittest
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "dsa_kv_chunk_plan.py"
_spec = importlib.util.spec_from_file_location("dsa_kv_chunk_plan", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
plan_dsa_kv_chunks = _module.plan_dsa_kv_chunks
prefill_indexer_score_bytes = _module.prefill_indexer_score_bytes


class DsaKVChunkPlanTest(unittest.TestCase):
    def test_fp8_prefill_score_budget_includes_storage_padding(self):
        # The allocator snapshot has 8 MiB of row padding beyond the logical score.
        self.assertEqual(prefill_indexer_score_bytes(8192, 154624), 4840 * 2**20)

    def test_full_history_that_fits_is_not_split(self):
        plan = plan_dsa_kv_chunks(
            [17, 2048, 4097],
            page_tokens=1024,
            budget_bytes=100000,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertFalse(plan.chunked)
        self.assertEqual(plan.blocks[0].lengths, (17, 2048, 4097))

    def test_ragged_pages_are_consumed_exactly_once(self):
        lengths = [17, 2300, 0, 4097]
        plan = plan_dsa_kv_chunks(
            lengths,
            page_tokens=1024,
            budget_bytes=9500,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertTrue(plan.chunked)
        seen = [set() for _ in lengths]
        for block in plan.blocks:
            self.assertLessEqual(1000 + block.tokens * 4, 9500)
            for request, start, length in zip(
                block.requests, block.starts, block.lengths
            ):
                self.assertEqual(start % 1024, 0)
                positions = set(range(start, start + length))
                self.assertFalse(seen[request] & positions)
                seen[request].update(positions)
        self.assertEqual(seen, [set(range(n)) for n in lengths])
        # Increasing the chosen page capacity by one must violate this budget.
        bigger = plan_dsa_kv_chunks(
            lengths,
            page_tokens=1024,
            budget_bytes=1000000,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertGreater(bigger.page_capacity, plan.page_capacity)

    def test_independent_phase_budgets_and_fixed_live_data(self):
        def plan(budget):
            return plan_dsa_kv_chunks(
                [65536],
                page_tokens=1024,
                budget_bytes=budget,
                fixed_bytes=1000,
                temporary_bytes=lambda b: b.tokens * 4,
            )

        self.assertNotEqual(len(plan(34000).blocks), len(plan(66000).blocks))
        with self.assertRaisesRegex(ValueError, "fixed live"):
            plan(999)
        with self.assertRaisesRegex(ValueError, "one physical KV page"):
            plan(1001)

    def test_empty_history_preserves_fixed_budget(self):
        plan = plan_dsa_kv_chunks(
            [0, 0],
            page_tokens=1024,
            budget_bytes=2048,
            fixed_bytes=1024,
            temporary_bytes=lambda b: b.tokens,
        )
        self.assertEqual(plan.blocks, ())
        self.assertEqual(plan.required_bytes, 1024)


if __name__ == "__main__":
    unittest.main()
