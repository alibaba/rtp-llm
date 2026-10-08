import runpy
import unittest
from pathlib import Path


plan_prefix_chunks = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "mla_prefix_chunk_plan.py")
)["plan_prefix_chunks"]


class MlaPrefixChunkPlanTest(unittest.TestCase):
    def test_default_budget_accounts_for_overlapping_bf16_and_fp8_kv(self):
        plan = plan_prefix_chunks(
            q_lens=(4096,), prefix_lens=(1_000_000,), page_tokens=4096,
            heads=12, qk_dim=192, v_dim=128, operand_bytes=1,
            budget_gib=6.0,
        )
        self.assertTrue(plan.chunked)
        self.assertEqual(plan.bytes_per_token, 12 * (192 + 128) * 3)
        self.assertLessEqual(plan.capacity_tokens * plan.bytes_per_token, 6 * 1024**3)
        self.assertGreater(len(plan.slices), 1)
        self.assertEqual(sum(s.length for s in plan.slices), 1_000_000)
        self.assertTrue(all(s.start % 4096 == 0 for s in plan.slices))

    def test_multi_request_coverage_and_short_last_slice(self):
        plan = plan_prefix_chunks(
            q_lens=(100, 200), prefix_lens=(8192, 5000), page_tokens=4096,
            heads=12, qk_dim=192, v_dim=128, operand_bytes=1,
            budget_gib=0.05,
        )
        self.assertEqual(
            [(s.owner, s.start, s.length) for s in plan.slices],
            [(0, 0, 4096), (0, 4096, 4096), (1, 0, 4096), (1, 4096, 904)],
        )

    def test_full_route_preserves_common_short_prefix(self):
        plan = plan_prefix_chunks(
            q_lens=(65536,), prefix_lens=(65536,), page_tokens=4096,
            heads=12, qk_dim=192, v_dim=128, operand_bytes=1,
            budget_gib=6.0,
        )
        self.assertFalse(plan.chunked)
        self.assertEqual(plan.slices, ())

    def test_budget_must_fit_a_page(self):
        with self.assertRaisesRegex(ValueError, "at least one cache page"):
            plan_prefix_chunks(
                q_lens=(1,), prefix_lens=(8192,), page_tokens=4096,
                heads=12, qk_dim=192, v_dim=128, operand_bytes=1,
                budget_gib=0.001,
            )


if __name__ == "__main__":
    unittest.main()
