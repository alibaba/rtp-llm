import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from EvaluateLatencyReplay import evaluate


def prediction(host, sequence, target, base, final, logged_base, logged_final,
               event_time_us=0):
    return (f"P\t{host}\t1\t1\t{sequence}\t{event_time_us}\t{target}\t1\t{base}\t{final}"
            f"\t0\t{logged_base}\t{logged_final}\n")


def update(host, sequence, parameter_drift, residual_drift):
    return (f"U\t{host}\t1\t1\t{sequence}\t0\t{parameter_drift}"
            f"\t{residual_drift}\t0\t0\t0\n")


def zero(host, sequence, base, final, event_time_us=0):
    return f"Z\t{host}\t1\t1\t{sequence}\t{event_time_us}\t{base}\t{final}\n"


class EvaluateLatencyReplayTest(unittest.TestCase):
    def test_filters_drift_per_sample_and_zero_update(self):
        host = "inference_cn_MI308X_2TP_spot_new2"
        baseline = [
            prediction(host, 1, 100, 90.05, 95, 90, 95),
            prediction(host, 2, 100, 50, 60, 90, 95),
            prediction(host, 3, 100, 80, 85, 80, 85),
            update(host, 1, 0.01, 0.02), zero(host, 1, 20, 25),
            update(host, 2, 0.2, 0), zero(host, 2, 30, 35),
        ]
        candidate = [
            prediction(host, 1, 100, 92, 96, 90, 95),
            prediction(host, 2, 100, 80, 90, 90, 95),
            update(host, 1, 0, 0), zero(host, 1, 21, 26),
            update(host, 2, 0, 0), zero(host, 2, 31, 36),
        ]
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.tsv"
            candidate_path = Path(directory) / "candidate.tsv"
            baseline_path.write_text("".join(baseline), encoding="utf-8")
            candidate_path.write_text("".join(candidate), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                evaluate(baseline_path, candidate_path, 0.1)
        result = output.getvalue()
        self.assertIn("'MI_matched': 1", result)
        self.assertIn("'MI_baseline_mismatch': 1", result)
        self.assertIn("'MI_missing_candidate': 1", result)
        self.assertIn("'MI_zero_matched': 1", result)
        self.assertIn("'MI_zero_unverified': 1", result)
        self.assertIn("MI n 1 MAE 5.0 4.0 P95 5.0 4.0 P99 5.0 4.0", result)
        self.assertIn("zero_MI_final n 1 mean 25.0 26.0 P95 25.0 26.0 "
                      "drift_MAE 1.0 drift_P95 1.0", result)

    def test_groups_by_running_and_waiting_features(self):
        host = "mi-node"
        baseline = "".join(prediction(host, index, 100, 90, 95, 90, 95)
                           for index in range(1, 4))
        candidate = "".join(prediction(host, index, 100, 90, 96, 90, 95)
                            for index in range(1, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.tsv"
            candidate_path = root / "candidate.tsv"
            samples_dir = root / "samples"
            samples_dir.mkdir()
            baseline_path.write_text(baseline, encoding="utf-8")
            candidate_path.write_text(candidate, encoding="utf-8")
            samples = [(1, 1, 0), (2, 2, 1), (3, 9, 0)]
            with (samples_dir / f"{host}.jsonl").open("w", encoding="utf-8") as stream:
                for sequence, running, waiting in samples:
                    stream.write(json.dumps({"type": "sample", "sequence": sequence,
                                             "features": [1, running, 0, 0, 0, 0,
                                                          waiting]}) + "\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                evaluate(baseline_path, candidate_path, 0.1, samples_dir)
        result = output.getvalue()
        self.assertIn("MI_low_concurrency n 1 MAE 5.0 4.0", result)
        self.assertIn("MI_mid_concurrency n 1 MAE 5.0 4.0", result)
        self.assertIn("MI_high_concurrency n 1 MAE 5.0 4.0", result)
        self.assertNotIn("pressure_missing", result)

    def test_excludes_later_predictions_and_zero_points(self):
        host = "mi-node"
        baseline = (prediction(host, 1, 100, 90, 95, 90, 95) +
                    prediction(host, 2, 100, 50, 60, 90, 95, 1) +
                    zero(host, 1, 20, 25) + zero(host, 2, 30, 35, 1))
        candidate = (prediction(host, 1, 100, 90, 96, 90, 95) +
                     prediction(host, 2, 100, 90, 96, 90, 95, 1) +
                     zero(host, 1, 21, 26) + zero(host, 2, 31, 36, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.tsv"
            candidate_path = root / "candidate.tsv"
            baseline_path.write_text(baseline, encoding="utf-8")
            candidate_path.write_text(candidate, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                evaluate(baseline_path, candidate_path, 0.1,
                         before_event_time_us=1)
        result = output.getvalue()
        self.assertIn("'MI_matched': 1", result)
        self.assertIn("'MI_outside_window': 1", result)
        self.assertIn("'MI_zero_outside_window': 1", result)
        self.assertIn("MI n 1 MAE 5.0 4.0", result)

    def test_excludes_later_points_after_first_large_drift(self):
        host = "mi-node"
        baseline = (prediction(host, 1, 100, 90, 95, 90, 95, 0) +
                    prediction(host, 2, 100, 80, 85, 90, 95, 1) +
                    prediction(host, 3, 100, 90, 95, 90, 95, 2) +
                    zero(host, 1, 20, 25, 0) + zero(host, 3, 30, 35, 2))
        candidate = (prediction(host, 1, 100, 90, 96, 90, 95, 0) +
                     prediction(host, 2, 100, 80, 86, 90, 95, 1) +
                     prediction(host, 3, 100, 90, 96, 90, 95, 2) +
                     zero(host, 1, 21, 26, 0) + zero(host, 3, 31, 36, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.tsv"
            candidate_path = root / "candidate.tsv"
            baseline_path.write_text(baseline, encoding="utf-8")
            candidate_path.write_text(candidate, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                evaluate(baseline_path, candidate_path, 0.1,
                         prefix_drift_ms=5)
        result = output.getvalue()
        self.assertIn("'MI_matched': 1", result)
        self.assertIn("'MI_outside_prefix': 2", result)
        self.assertIn("'MI_zero_outside_prefix': 1", result)
        self.assertIn("'mi-node': 1", result)


if __name__ == "__main__":
    unittest.main()
