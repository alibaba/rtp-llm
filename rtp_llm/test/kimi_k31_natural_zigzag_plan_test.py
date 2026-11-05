"""CPU fixtures for natural-shard and shared Zigzag receive geometry."""

import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


# Loading the file directly keeps hybrid.__init__'s Torch/CUDA imports out of
# these standard-library-only tests.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "models_py/modules/hybrid/natural_zigzag_plan.py"
)


def receive(plans, sources, direction):
    """Simulate receives and prove each destination and source row is used once."""
    outputs = [[None] * plan.local_rows for plan in plans]
    source_uses = [[0] * plan.local_rows for plan in plans]
    for rank, plan in enumerate(plans):
        next_destination = 0
        for segment in getattr(plan, direction):
            if segment.destination_offset != next_destination or segment.rows <= 0:
                raise AssertionError("receive segments must cover ordered destination rows")
            if not 0 <= segment.peer < len(plans):
                raise AssertionError("source peer out of range")
            if not 0 <= segment.source_offset <= plan.local_rows - segment.rows:
                raise AssertionError("source interval out of range")
            for offset in range(segment.rows):
                source = segment.source_offset + offset
                destination = segment.destination_offset + offset
                source_uses[segment.peer][source] += 1
                outputs[rank][destination] = sources[segment.peer][source]
            next_destination += segment.rows
        if next_destination != plan.local_rows:
            raise AssertionError("receive segments do not cover every destination")
    if any(count != 1 for counts in source_uses for count in counts):
        raise AssertionError("every source row must be read exactly once")
    return outputs


class NaturalZigzagPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODULE_PATH.is_file():
            return
        spec = importlib.util.spec_from_file_location("natural_zigzag_plan_cpu", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.plan_type = module.NaturalZigzagPlan
        cls.segment_type = module.RowSegment

    def setUp(self):
        self.assertTrue(MODULE_PATH.is_file(), "CPU natural/Zigzag geometry is missing")

    def plans(self, lengths, size):
        return [self.plan_type(lengths, size, rank) for rank in range(size)]

    def assert_coalesced(self, segments):
        for previous, current in zip(segments, segments[1:]):
            self.assertFalse(
                previous.peer == current.peer
                and previous.source_offset + previous.rows == current.source_offset
                and previous.destination_offset + previous.rows == current.destination_offset,
                "adjacent intervals with the same source peer must be coalesced",
            )

    def test_sixteen_row_known_permutation_and_receive_offsets(self):
        # Catches front/back pairing, peer/local-offset and receive-direction bugs.
        plans = self.plans([16], 4)
        natural = [list(range(start, start + 4)) for start in (0, 4, 8, 12)]
        expected = [[0, 1, 14, 15], [2, 3, 12, 13], [4, 5, 10, 11], [6, 7, 8, 9]]
        self.assertEqual(receive(plans, natural, "to_zigzag"), expected)
        self.assertEqual(receive(plans, expected, "to_natural"), natural)
        segment = self.segment_type
        self.assertEqual(plans[0].to_zigzag, (segment(0, 0, 0, 2), segment(3, 2, 2, 2)))
        self.assertEqual(plans[0].to_natural, (segment(0, 0, 0, 2), segment(1, 0, 2, 2)))
        self.assertEqual(plans[0].padded_lengths, (16,))
        self.assertEqual((plans[0].physical_rows, plans[0].local_rows), (16, 4))
        with self.assertRaises(FrozenInstanceError):
            plans[0].to_zigzag[0].rows = 99

    def test_unequal_requests_keep_padding_identity(self):
        # Catches packing requests before padding, or dropping padded rows.
        plans = self.plans([11, 5], 4)
        natural = [list(range(start, start + 6)) for start in (0, 6, 12, 18)]
        expected = [
            [0, 1, 14, 15, 16, 23],
            [2, 3, 12, 13, 17, 22],
            [4, 5, 10, 11, 18, 21],
            [6, 7, 8, 9, 19, 20],
        ]
        self.assertEqual(receive(plans, natural, "to_zigzag"), expected)
        self.assertEqual(receive(plans, expected, "to_natural"), natural)
        for plan in plans:
            self.assertEqual(plan.padded_lengths, (16, 8))
            self.assertEqual((plan.physical_rows, plan.local_rows), (24, 6))
            self.assertIsInstance(plan.to_zigzag, tuple)
            self.assertIsInstance(plan.to_natural, tuple)
            self.assert_coalesced(plan.to_zigzag)
            self.assert_coalesced(plan.to_natural)

    def test_prefix_positions_travel_without_becoming_row_offsets(self):
        # Geometry must preserve metadata values; prefixes are not source offsets.
        plans = self.plans([11, 5], 4)
        first = [(0, position) for position in range(100, 111)]
        first += [("padding", 0, position) for position in range(11, 16)]
        second = [(1, position) for position in range(700, 705)]
        second += [("padding", 1, position) for position in range(5, 8)]
        physical = first + second
        natural = [physical[start:start + 6] for start in (0, 6, 12, 18)]
        zigzag = receive(plans, natural, "to_zigzag")
        self.assertEqual(zigzag[2], [(0, 104), (0, 105), (0, 110), ("padding", 0, 11), (1, 702), ("padding", 1, 5)])
        self.assertEqual(zigzag[3], [(0, 106), (0, 107), (0, 108), (0, 109), (1, 703), (1, 704)])
        self.assertEqual(receive(plans, zigzag, "to_natural"), natural)

    def test_roundtrips_every_row_across_request_and_shard_boundaries(self):
        # A mutually inverse but non-bijective route must not pass a roundtrip.
        for lengths, size in (([1, 0, 7, 2], 3), ([3, 19, 1], 4), ([0, 0], 2), ([5], 1)):
            with self.subTest(lengths=lengths, size=size):
                plans = self.plans(lengths, size)
                rows = plans[0].local_rows
                natural = [list(range(rank * rows, (rank + 1) * rows)) for rank in range(size)]
                zigzag = receive(plans, natural, "to_zigzag")
                self.assertEqual(receive(plans, zigzag, "to_natural"), natural)
                for plan in plans:
                    self.assert_coalesced(plan.to_zigzag)
                    self.assert_coalesced(plan.to_natural)

    def test_sixty_four_k_cp8_uses_two_intervals_per_direction(self):
        # Catches row-expanded plans and incorrectly split contiguous intervals.
        plans = self.plans([65536], 8)
        for rank, plan in enumerate(plans):
            self.assertEqual((plan.physical_rows, plan.local_rows), (65536, 8192))
            self.assertEqual(len(plan.to_zigzag), 2)
            self.assertEqual(len(plan.to_natural), 2)
            self.assertEqual(plan.to_zigzag, (
                self.segment_type(rank // 2, (rank % 2) * 4096, 0, 4096),
                self.segment_type(7 - rank // 2, (1 - rank % 2) * 4096, 4096, 4096),
            ))

    def test_size_one_coalesces_requests_and_accepts_iterable_lengths(self):
        # At CP1 the physical permutation is identity, including padding.
        plan = self.plan_type(iter([3, 0, 5]), 1, 0)
        self.assertEqual(plan.padded_lengths, (4, 0, 6))
        expected = (self.segment_type(0, 0, 0, 10),)
        self.assertEqual(plan.to_zigzag, expected)
        self.assertEqual(plan.to_natural, expected)

    def test_empty_and_zero_requests_have_no_intervals(self):
        for lengths in ([], [0], [0, 0]):
            for plan in self.plans(lengths, 4):
                self.assertEqual(plan.padded_lengths, tuple(0 for _ in lengths))
                self.assertEqual((plan.physical_rows, plan.local_rows), (0, 0))
                self.assertEqual(plan.to_zigzag, ())
                self.assertEqual(plan.to_natural, ())

    def test_invalid_lengths_size_and_rank_are_rejected(self):
        # Catches silent negative rounding, division by zero and invalid peers.
        for lengths, size, rank in (([-1], 4, 0), ([1, -3], 2, 0), ([1], 0, 0), ([1], -2, 0), ([1], 2, -1), ([1], 2, 2), ([], 0, 0)):
            with self.subTest(lengths=lengths, size=size, rank=rank):
                with self.assertRaises(ValueError):
                    self.plan_type(lengths, size, rank)


if __name__ == "__main__":
    unittest.main()
