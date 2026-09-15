import unittest

from comparison_utils import integer_within_tolerance


class IntegerWithinToleranceTest(unittest.TestCase):
    def test_accepts_exact_and_boundary_values(self):
        self.assertTrue(integer_within_tolerance(44, 44, 1))
        self.assertTrue(integer_within_tolerance(44, 43, 1))
        self.assertTrue(integer_within_tolerance(44, 45, 1))

    def test_rejects_missing_or_out_of_range_values(self):
        self.assertFalse(integer_within_tolerance(44, None, 1))
        self.assertFalse(integer_within_tolerance(44, 46, 1))

    def test_rejects_negative_tolerance(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            integer_within_tolerance(44, 44, -1)


if __name__ == "__main__":
    unittest.main()
