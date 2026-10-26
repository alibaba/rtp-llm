import copy
import math
import unittest

from asymmetric_tp_writeback_acceptance import compare_handoff


def response(logits):
    winner = max(range(len(logits)), key=logits.__getitem__)
    return {"input_ids": [[10, 11]], "output_ids": [[12, winner]], "logits": [logits]}


class HandoffComparisonTest(unittest.TestCase):
    def test_identical_outputs_still_require_matching_probabilities(self):
        with self.assertRaisesRegex(AssertionError, "probability error"):
            compare_handoff(response([5.0, 0.0, -1.0]), response([0.5, 0.0, -1.0]))

    def test_near_tied_greedy_winners_can_exchange_order(self):
        metrics = compare_handoff(
            response([2.0, 2.01, 0.0]), response([2.01, 2.0, 0.0])
        )
        self.assertFalse(metrics["same_greedy_output"])
        self.assertLess(metrics["total_variation"], 0.02)

    def test_relative_tolerance_matches_smoke_convention(self):
        metrics = compare_handoff(
            response([math.log(0.139), math.log(0.861)]),
            response([math.log(0.15), math.log(0.85)]),
        )
        self.assertGreater(metrics["max_probability_error"], 0.01)
        self.assertLess(metrics["max_tolerance_ratio"], 1)

    def test_material_greedy_divergence_fails(self):
        with self.assertRaisesRegex(AssertionError, "probability error"):
            compare_handoff(response([3.0, 0.0]), response([0.0, 3.0]))

    def test_distributed_probability_error_cannot_hide_below_per_token_limit(self):
        with self.assertRaisesRegex(AssertionError, "Total variation"):
            compare_handoff(response([0.1] * 50 + [-0.1] * 50), response([0.0] * 100))

    def test_different_conditioning_fails_before_logit_comparison(self):
        cold = response([2.0, 1.0])
        for key, replacement in (("input_ids", [[10, 13]]), ("output_ids", [[13, 0]])):
            warm = copy.deepcopy(cold)
            warm[key] = replacement
            with self.subTest(key=key), self.assertRaises(AssertionError):
                compare_handoff(warm, cold)

    def test_wrong_step_logits_are_rejected(self):
        warm = response([2.0, 1.0])
        warm["output_ids"][0][-1] = 1
        with self.assertRaisesRegex(AssertionError, "final greedy token"):
            compare_handoff(warm, response([2.0, 1.0]))

    def test_missing_invalid_and_nonfinite_logits_are_rejected(self):
        for logits in (
            None, [], [[]], [[1.0, float("nan")]], [[1.0, float("inf")]], [[True, 0]]
        ):
            warm = response([2.0, 1.0])
            warm["logits"] = logits
            with self.subTest(logits=logits), self.assertRaises(AssertionError):
                compare_handoff(warm, response([2.0, 1.0]))

    def test_vocabulary_mismatch_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "vocabulary sizes"):
            compare_handoff(response([2.0, 1.0]), response([2.0, 1.0, 0.0]))

    def test_common_logit_shift_preserves_distribution(self):
        metrics = compare_handoff(response([1002.0, 1001.0]), response([2.0, 1.0]))
        self.assertEqual(metrics["total_variation"], 0)


if __name__ == "__main__":
    unittest.main()
