import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
    get_segment_valid_lengths,
)


class ZigzagCPPlanTest(unittest.TestCase):
    def test_relay_and_halo_follow_the_global_zigzag_order(self):
        for cp_size in (2, 4, 8):
            steps = ZigzagCPPlan(cp_size, 0).relay_steps
            relayed_segments = [
                segment
                for step in steps
                for segment in range(
                    step.first_global_segment,
                    step.first_global_segment + step.segment_count,
                )
            ]
            self.assertEqual(relayed_segments, list(range(2 * cp_size)))

            for step in steps:
                owner_segments = ZigzagCPPlan(
                    cp_size, step.owner_rank
                ).local_global_segments
                self.assertEqual(
                    owner_segments[
                        step.first_local_segment : step.first_local_segment
                        + step.segment_count
                    ],
                    tuple(
                        range(
                            step.first_global_segment,
                            step.first_global_segment + step.segment_count,
                        )
                    ),
                )

            for rank in range(cp_size):
                plan = ZigzagCPPlan(cp_size, rank)
                for segment, source in zip(
                    plan.local_global_segments, plan.halo_sources
                ):
                    if segment == 0:
                        self.assertIsNone(source)
                        continue
                    source_rank, source_local_segment = source
                    self.assertEqual(
                        ZigzagCPPlan(cp_size, source_rank).local_global_segments[
                            source_local_segment
                        ],
                        segment - 1,
                    )

    def test_valid_lengths_exclude_padding_and_validate_capacity(self):
        self.assertEqual(get_segment_valid_lengths(257, 128, 2), (128, 128, 1, 0))
        self.assertEqual(
            get_segment_valid_lengths(511, 64, 4),
            (64, 64, 64, 64, 64, 64, 64, 63),
        )
        with self.assertRaisesRegex(ValueError, "exceeds CP capacity"):
            get_segment_valid_lengths(513, 64, 4)
        with self.assertRaisesRegex(ValueError, "cp_size must be at least 2"):
            ZigzagCPPlan(1, 0)


if __name__ == "__main__":
    unittest.main()
