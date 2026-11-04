import unittest

from AnalyzeLatencyDrain import analyze, ready_us, summary
from AnalyzeLatencyFeatureNeighbors import FEATURES


def sample(sequence, ready, actual, prediction, **features):
    return {**dict.fromkeys(FEATURES, 0), **features, "sequence": sequence,
            "batch_id": sequence, "event_time_us": int((ready + actual / 1000) * 1e6),
            "target_ms": actual, "master_prediction_ms": prediction,
            "backend_queue_ms": actual - 100}


class DrainAnalysisTest(unittest.TestCase):
    def test_original_predictions_and_complete_vector_match(self):
        control = sample(1, 300, 30000, 30000)
        tail = sample(2, 599, 10000, 30000)
        result = analyze([control, tail], 0, 600_000_000)
        self.assertEqual(ready_us(tail), 599_000_000)
        self.assertEqual(result["steady"]["signed_error_mean_ms"], 0)
        self.assertEqual(result["tail_finishing_after_cut"]["signed_error_mean_ms"], 20000)
        self.assertEqual(result["matching"]["pair_count"], 1)
        for feature in FEATURES:
            changed = {**tail, feature: 1}
            self.assertEqual(analyze([control, changed], 0, 600_000_000)
                             ["matching"]["pair_count"], 0, feature)

    def test_completed_before_cut_is_not_tail(self):
        result = analyze([sample(1, 550, 10000, 30000)], 0, 600_000_000)
        self.assertEqual(result["tail_finishing_after_cut"], {"count": 0})

    def test_new_arrivals_after_cut_are_flagged(self):
        result = analyze([sample(1, 610, 10000, 30000)], 0, 600_000_000)
        self.assertEqual(result["samples_ready_after_cut_plus_2s"], 1)
        self.assertEqual(result["tail_finishing_after_cut"], {"count": 0})

    def test_no_controls_is_not_evidence_of_matching(self):
        result = analyze([sample(1, 599, 10000, 30000)], 0, 600_000_000)
        self.assertEqual(result["steady"], {"count": 0})
        self.assertEqual(result["matching"]["pair_count"], 0)

    def test_error_sign_and_empty(self):
        self.assertEqual(summary([]), {"count": 0})
        result = summary([sample(1, 0, 30000, 20000)])
        self.assertEqual(result["signed_error_mean_ms"], -10000)
        self.assertEqual(result["mae_ms"], 10000)
        self.assertEqual(result["overprediction_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
