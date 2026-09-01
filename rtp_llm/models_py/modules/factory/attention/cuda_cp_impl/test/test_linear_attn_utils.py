import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
    get_segment_valid_lengths,
)


class TestZigzagCPPlan(unittest.TestCase):
    def test_relay_covers_every_segment_in_global_order(self):
        for cp_size in (2, 3, 4, 5, 8, 16):
            steps = ZigzagCPPlan(cp_size, 0).relay_steps
            global_segments = [
                segment
                for step in steps
                for segment in range(
                    step.first_global_segment,
                    step.first_global_segment + step.segment_count,
                )
            ]

            self.assertEqual(global_segments, list(range(2 * cp_size)))
            self.assertEqual(len(steps), 2 * cp_size - 1)

    def test_relay_steps_reference_the_owners_local_layout(self):
        for cp_size in (2, 3, 4, 5, 8, 16):
            for step in ZigzagCPPlan(cp_size, 0).relay_steps:
                owner_segments = ZigzagCPPlan(
                    cp_size, step.owner_rank
                ).local_global_segments
                actual = owner_segments[
                    step.first_local_segment : step.first_local_segment
                    + step.segment_count
                ]
                expected = tuple(
                    range(
                        step.first_global_segment,
                        step.first_global_segment + step.segment_count,
                    )
                )
                self.assertEqual(actual, expected)

    def test_halo_source_owns_the_previous_global_segment(self):
        for cp_size in (2, 3, 4, 5, 8, 16):
            for rank in range(cp_size):
                plan = ZigzagCPPlan(cp_size, rank)
                for global_segment, source in zip(
                    plan.local_global_segments, plan.halo_sources
                ):
                    if global_segment == 0:
                        self.assertIsNone(source)
                        continue
                    self.assertIsNotNone(source)
                    source_rank, source_local_segment = source
                    source_global_segment = ZigzagCPPlan(
                        cp_size, source_rank
                    ).local_global_segments[source_local_segment]
                    self.assertEqual(source_global_segment, global_segment - 1)

    def test_invalid_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cp_size must be at least 2"):
            ZigzagCPPlan(1, 0)
        with self.assertRaisesRegex(ValueError, "cp_rank must be in"):
            ZigzagCPPlan(4, 4)

    def test_relay_rejects_incomplete_valid_lengths(self):
        step = ZigzagCPPlan(2, 0).relay_steps[-1]
        with self.assertRaisesRegex(ValueError, "at least 4 entries"):
            step.valid_token_count((64, 64, 64))


class TestSegmentValidLengths(unittest.TestCase):
    def test_padding_is_excluded_for_arbitrary_cp_sizes(self):
        for cp_size in (2, 3, 4, 5, 8, 16):
            segment_tokens = 64
            actual_tokens = (2 * cp_size - 2) * segment_tokens + 17
            expected = (segment_tokens,) * (2 * cp_size - 2) + (17, 0)
            self.assertEqual(
                get_segment_valid_lengths(actual_tokens, segment_tokens, cp_size),
                expected,
            )

    def test_capacity_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds CP capacity"):
            get_segment_valid_lengths(513, segment_tokens=64, cp_size=4)

    def test_invalid_length_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual_tokens must be non-negative"):
            get_segment_valid_lengths(-1, segment_tokens=64, cp_size=2)
        with self.assertRaisesRegex(ValueError, "segment_tokens must be positive"):
            get_segment_valid_lengths(1, segment_tokens=0, cp_size=2)

    def test_relay_consumes_every_real_token_exactly_once(self):
        segment_tokens = 64
        for cp_size in (2, 3, 4, 5, 8, 16):
            capacity = 2 * cp_size * segment_tokens
            for actual_tokens in (1, 63, 64, 65, capacity - 1, capacity):
                valid_lengths = get_segment_valid_lengths(
                    actual_tokens, segment_tokens, cp_size
                )
                relayed_tokens = sum(
                    step.valid_token_count(valid_lengths)
                    for step in ZigzagCPPlan(cp_size, 0).relay_steps
                )
                self.assertEqual(relayed_tokens, actual_tokens)


if __name__ == "__main__":
    unittest.main()
