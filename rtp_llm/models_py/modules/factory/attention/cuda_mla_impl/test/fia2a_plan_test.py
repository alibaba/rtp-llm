"""CPU coverage of packed-query routing and automatic launch selection."""

import importlib.util
import sys
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "fia2a_plan", Path(__file__).resolve().parents[1] / "fia2a_plan.py"
)
plan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plan
spec.loader.exec_module(plan)


class Fia2aPlanTest(unittest.TestCase):
    def test_packed_query_tiles(self):
        self.assertEqual(plan.compute_q_tile_layout(96, 4), (384, 3, 128))
        self.assertEqual(plan.compute_q_tile_layout(96, 1), (96, 1, 96))

    def test_split_selection(self):
        self.assertEqual(plan.plan_splits(148, 1, 4, 96), 24)
        self.assertEqual(plan.plan_splits(152, 76, 4, 96), 1)
        self.assertEqual(plan.plan_splits(152, 76, 1, 96), 1)
        self.assertEqual(plan.plan_splits(152, 1, 1, 96), 32)
        with self.assertRaises(ValueError):
            plan.plan_splits(152, 0, 4, 96)

    def test_route_ownership_and_coverage(self):
        for heads in (64, 96, 128):
            for world in (2, 4, 8, 16, 32):
                for queries in (1, 4):
                    routing = plan.make_output_routes(heads, queries, world)
                    covered = set()
                    for tile in range((heads * queries + 127) // 128):
                        phase = tile % routing.phase_count
                        query_base = tile // routing.phase_count * routing.queries_per_period
                        for route in routing.routes:
                            rp, cta, row, q, dest, local_head, count, _ = route
                            if rp != phase:
                                continue
                            for offset in range(count):
                                flat = tile * 128 + cta * 64 + row + offset
                                if flat >= heads * queries:
                                    continue
                                self.assertEqual(q + query_base, flat // heads)
                                self.assertEqual(dest, (flat % heads) // (heads // world))
                                self.assertEqual(local_head + offset, flat % (heads // world))
                                self.assertNotIn(flat, covered)
                                covered.add(flat)
                    self.assertEqual(covered, set(range(heads * queries)))

    def test_tma_alignment(self):
        self.assertTrue(plan.make_output_routes(96, 4, 8).tma_3d_eligible)
        self.assertFalse(plan.make_output_routes(96, 4, 16).tma_3d_eligible)
        self.assertTrue(plan.make_output_routes(128, 4, 16).tma_3d_eligible)

    def test_custom_transport_range(self):
        self.assertFalse(plan.use_custom_a2a(16, 128, 96))
        self.assertTrue(plan.use_custom_a2a(8, 128, 96))
        self.assertFalse(plan.use_custom_a2a(8, 4, 96))
        self.assertFalse(plan.use_custom_a2a(8, 513, 96))


if __name__ == "__main__":
    unittest.main()
