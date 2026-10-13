import unittest

import torch

from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance


class AssertCloseWithMismatchToleranceTest(unittest.TestCase):
    def test_exact_values_match_with_default_tolerances(self):
        values = torch.tensor([0.0, -1.0, 2.5, 100.0])

        assert_close_with_mismatch_tolerance(values, values.clone())

    def test_empty_tensors_match(self):
        assert_close_with_mismatch_tolerance(torch.empty(0), torch.empty(0))

    def test_values_within_absolute_tolerance(self):
        actual = torch.tensor([0.005, -0.005])
        expected = torch.zeros(2)

        assert_close_with_mismatch_tolerance(actual, expected, rtol=0.0, atol=0.01)

    def test_values_within_relative_tolerance(self):
        actual = torch.tensor([100.5, -100.5])
        expected = torch.tensor([100.0, -100.0])

        assert_close_with_mismatch_tolerance(actual, expected, rtol=0.01, atol=0.0)

    def test_value_outside_tolerance_fails_without_allowance(self):
        actual = torch.tensor([1.0, 1.1])
        expected = torch.ones(2)

        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 1 / 2"):
            assert_close_with_mismatch_tolerance(actual, expected, rtol=0.01, atol=0.0)

    def test_zero_tolerance_requires_exact_values(self):
        expected = torch.tensor([1.0, 2.0])
        assert_close_with_mismatch_tolerance(
            expected.clone(), expected, rtol=0.0, atol=0.0
        )

        actual = torch.tensor([1.0, 2.0001])
        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 1 / 2"):
            assert_close_with_mismatch_tolerance(actual, expected, rtol=0.0, atol=0.0)

    def test_mismatch_count_equal_to_allowance_passes(self):
        actual = torch.tensor([1000.0, -1000.0, 1.0])
        expected = torch.ones(3)

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=2
        )

    def test_mismatch_count_above_allowance_fails(self):
        actual = torch.tensor([1000.0, -1000.0, 1.0])
        expected = torch.ones(3)

        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 2 / 3"):
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=1
            )

    def test_float16_inputs_use_float_comparison(self):
        actual = torch.tensor([1.0, 2.0], dtype=torch.float16)
        expected = torch.tensor([1.0, 2.0], dtype=torch.float16)

        assert_close_with_mismatch_tolerance(actual, expected)

    def test_nan_within_mismatch_allowance(self):
        actual = torch.tensor([float("nan"), 1.0, 1.0])
        expected = torch.ones(3)

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=1
        )

    def test_nan_exceeding_mismatch_allowance(self):
        actual = torch.tensor([float("nan"), float("nan"), 1.0])
        expected = torch.ones(3)

        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 2 / 3"):
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=1
            )

    def test_positive_and_negative_inf_consume_mismatch_allowance(self):
        actual = torch.tensor([float("inf"), float("-inf"), 1.0])
        expected = torch.ones(3)

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=2
        )
        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 2 / 3"):
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=1
            )

    def test_equal_inf_consumes_mismatch_allowance(self):
        actual = torch.tensor([float("inf"), 1.0])
        expected = torch.tensor([float("inf"), 1.0])

        with self.assertRaisesRegex(AssertionError, "Non-finite element pairs: 1"):
            assert_close_with_mismatch_tolerance(actual, expected)
        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=1
        )

    def test_expected_non_finite_consumes_mismatch_allowance(self):
        actual = torch.ones(3)
        expected = torch.tensor([float("nan"), float("inf"), 1.0])

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=2
        )

    def test_non_finite_pair_counts_as_one_mismatch(self):
        actual = torch.tensor([float("nan"), float("inf"), 1.0])
        expected = torch.tensor([float("nan"), float("inf"), 1.0])

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=2
        )
        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 2 / 3"):
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=1
            )

    def test_all_nan_exceeds_mismatch_allowance(self):
        actual = torch.full((8,), float("nan"))
        expected = torch.ones(8)

        with self.assertRaisesRegex(AssertionError, "Mismatched elements: 8 / 8"):
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=1
            )

    def test_finite_outlier_within_mismatch_allowance(self):
        actual = torch.tensor([1000.0, 1.0, 1.0])
        expected = torch.ones(3)

        assert_close_with_mismatch_tolerance(
            actual, expected, max_mismatched_elements=1
        )

    def test_failure_reports_non_finite_and_finite_differences(self):
        actual = torch.tensor([float("nan"), 5.0])
        expected = torch.ones(2)

        with self.assertRaises(AssertionError) as context:
            assert_close_with_mismatch_tolerance(actual, expected)

        message = str(context.exception)
        self.assertIn("Mismatched elements: 2 / 2", message)
        self.assertIn("Non-finite element pairs: 1", message)
        self.assertIn("Greatest finite absolute difference: 4", message)

    def test_failure_reports_na_when_no_finite_pairs_exist(self):
        actual = torch.tensor([float("nan"), float("inf")])
        expected = torch.tensor([float("nan"), float("inf")])

        with self.assertRaises(AssertionError) as context:
            assert_close_with_mismatch_tolerance(
                actual, expected, max_mismatched_elements=0
            )

        message = str(context.exception)
        self.assertIn("Non-finite element pairs: 2", message)
        self.assertIn("Greatest finite absolute difference: N/A", message)
        self.assertIn("Greatest finite relative difference: N/A", message)


if __name__ == "__main__":
    unittest.main()
