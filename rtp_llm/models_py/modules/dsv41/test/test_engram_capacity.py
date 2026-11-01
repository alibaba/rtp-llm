# SPDX-License-Identifier: Apache-2.0
"""Capacity arithmetic/policy tests with no allocations or GPU dependencies."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "engram_capacity_cpu_test", Path(__file__).resolve().parents[1] / "engram_capacity.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CapacityTest(unittest.TestCase):
    def test_small_defaults_are_unchanged(self):
        self.assertEqual(c.capacity_from_environ({}), (1 << 30, 8 << 30))

    def test_explicit_local_ceiling_is_supported(self):
        self.assertEqual(c.capacity_from_environ({
            c.MAX_ALLOCATION_ENV: str(128 << 30),
            c.MAX_TOTAL_ALLOCATION_ENV: str(256 << 30),
        }), (128 << 30, 256 << 30))

    def test_each_limit_can_be_lowered_independently(self):
        self.assertEqual(c.validate_capacity(1 << 30, 64 << 20), (1 << 30, 64 << 20))
        self.assertEqual(c.capacity_from_environ({c.MAX_ALLOCATION_ENV: "4096"}),
                         (4096, 8 << 30))

    def test_capacity_types_and_hard_bounds(self):
        for allocation, total in ((0, 1), (1, 0), (-1, 1), (True, 1),
                                  (1, False), (1.0, 1), (1, "1"),
                                  ((128 << 30) + 1, 256 << 30),
                                  (128 << 30, (256 << 30) + 1),
                                  (1 << 64, 1)):
            with self.subTest(allocation=allocation, total=total):
                with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                    c.validate_capacity(allocation, total)

    def test_environment_rejects_ambiguous_or_unbounded_inputs(self):
        for value in ("", "0", "-1", "+1", "01", " 1", "1 ", "1\n", "1GiB", "1e9",
                      "0x1000", "١", "1" * 10000, 1, True, None):
            for key in (c.MAX_ALLOCATION_ENV, c.MAX_TOTAL_ALLOCATION_ENV):
                with self.subTest(key=key, value=str(value)[:20]):
                    with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                        c.capacity_from_environ({key: value})

    def test_bundle_like_keys_cannot_increase_limits(self):
        self.assertEqual(c.capacity_from_environ({"budget_bytes": str(256 << 30),
                                                  "max_allocation_bytes": str(128 << 30)}),
                         (1 << 30, 8 << 30))

    def test_real_layout_padded_capacity_uses_64_bit_offsets(self):
        sizes = [rows * (256 + 8) for rows in (384006168, 384016682)]
        padded = [c.checked_allocation_bytes(size, 2 << 20, 128 << 30) for size in sizes]
        self.assertEqual(sizes, [101377628352, 101380404048])
        self.assertEqual(padded, [101378424832, 101380521984])
        self.assertEqual(sum(padded), 202758946816)
        self.assertLess(sum(padded), 256 << 30)

    def test_rounding_and_exact_cap_boundaries(self):
        limit, granularity = 128 << 30, 2 << 20
        for size, expected in ((1, granularity), (granularity, granularity),
                               ((1 << 32) + 1, (1 << 32) + granularity),
                               (limit - 1, limit), (limit, limit)):
            self.assertEqual(c.checked_allocation_bytes(size, granularity, limit), expected)
        for size, alignment, cap in ((limit + 1, granularity, limit),
                                     (limit, 3, limit), (4097, 4096, 8191),
                                     ((1 << 63) - 1, granularity, limit)):
            with self.assertRaisesRegex(c.CapacityError, "ALLOCATION_LIMIT"):
                c.checked_allocation_bytes(size, alignment, cap)

    def test_rounding_rejects_invalid_numeric_arguments(self):
        for arguments in ((0, 1, 1), (True, 1, 1), (1, 0, 1), (1, True, 1),
                          (1, 1, 0), (1 << 63, 1, 1),
                          (1, (128 << 30) + 1, 1), (1, 1, (128 << 30) + 1)):
            with self.assertRaisesRegex(c.CapacityError, "INVALID_CAPACITY_LIMIT"):
                c.checked_allocation_bytes(*arguments)


if __name__ == "__main__":
    unittest.main()
