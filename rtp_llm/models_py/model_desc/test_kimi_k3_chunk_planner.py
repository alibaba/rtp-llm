from __future__ import annotations

import unittest

from rtp_llm.models_py.model_desc.kimi_k3_chunk_planner import (
    plan_kimi_k3_chunk_rounds,
)


class KimiK3ChunkPlannerTest(unittest.TestCase):
    def test_multi_batch_terminal_tails(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [1369, 1209, 1813], [0, 0, 0], chunk_budget=2048, page_size=512
        )
        self.assertEqual([item.token_count for item in rounds], [1881, 1721, 789])
        terminal = {
            item.original_batch_idx: item.absolute_end
            for round_plan in rounds
            for item in round_plan.slices
            if item.terminal
        }
        self.assertEqual(terminal, {0: 1369, 1: 1209, 2: 1813})
        for round_plan in rounds:
            for item in round_plan.slices:
                if not item.terminal:
                    self.assertEqual(item.absolute_end % 512, 0)

    def test_skips_budget_fragment_for_new_request(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [900, 1536], [0, 0], chunk_budget=1024, page_size=512
        )
        self.assertEqual(
            [[(item.original_batch_idx, item.new_length) for item in plan.slices]
             for plan in rounds],
            [[(0, 900)], [(1, 1024)], [(1, 512)]],
        )

    def test_non_aligned_legacy_prefix_reaches_page_boundary(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [1536], [124], chunk_budget=1000, page_size=512
        )
        self.assertEqual(
            [(item.new_length, item.absolute_end, item.terminal)
             for plan in rounds for item in plan.slices],
            [(900, 1024, False), (636, 1660, True)],
        )

    def test_page_sized_budget_requirement_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover one linear page"):
            plan_kimi_k3_chunk_rounds(
                [1536], [0], chunk_budget=124, page_size=512
            )

    def test_page_aligned_hit_miss_mix(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [700, 900, 300], [1024, 0, 512], chunk_budget=1024, page_size=512
        )
        self.assertTrue(all(plan.token_count <= 1024 for plan in rounds))
        self.assertEqual(
            [
                item.original_batch_idx
                for plan in rounds
                for item in plan.slices
                if item.terminal
            ],
            [0, 2, 1],
        )

    def test_source_ranges_remain_in_original_packed_coordinates(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [70, 130], [0, 0], chunk_budget=128, page_size=64
        )
        self.assertEqual(
            [
                (
                    item.original_batch_idx,
                    item.source_start,
                    item.source_end,
                    item.terminal,
                )
                for plan in rounds
                for item in plan.slices
            ],
            [
                (0, 0, 70, True),
                (1, 70, 198, False),
                (1, 198, 200, True),
            ],
        )

    def test_active_set_shrinks_across_rounds(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [63, 193, 321], [0, 0, 0], chunk_budget=192, page_size=64
        )
        active_sets = [
            [item.original_batch_idx for item in plan.slices] for plan in rounds
        ]
        self.assertEqual(active_sets, [[0, 1], [1, 2], [2], [2]])

    def test_tp_padding_stays_within_divisible_budget(self) -> None:
        tp_size = 8
        budget = 256
        rounds = plan_kimi_k3_chunk_rounds(
            [255, 129, 65], [0, 64, 128], chunk_budget=budget, page_size=64
        )
        for plan in rounds:
            padded = ((plan.token_count + tp_size - 1) // tp_size) * tp_size
            self.assertLessEqual(padded, budget)

    def test_aligned_prefix_continues_at_absolute_page_boundaries(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [200], [128], chunk_budget=128, page_size=64
        )
        self.assertEqual(
            [
                (item.absolute_start, item.absolute_end, item.terminal)
                for plan in rounds
                for item in plan.slices
            ],
            [(128, 256, False), (256, 328, True)],
        )


if __name__ == "__main__":
    unittest.main()
