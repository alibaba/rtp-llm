#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

from AnalyzeLatencyShortlistRobustness import (
    analyze_candidate,
    fit_control_origin,
    read_trace,
    target_qps,
)


def prediction_line(host, sequence, event_time_us, target_ms, prediction_ms,
                    logged_prediction_ms=None):
    if logged_prediction_ms is None:
        logged_prediction_ms = prediction_ms
    fields = [
        "P", host, "4", "20261011", str(sequence), str(event_time_us),
        str(target_ms), "17", "10", str(prediction_ms), "0", "10",
        str(logged_prediction_ms),
    ]
    return "\t".join(fields) + "\n"


class AnalyzeLatencyShortlistRobustnessTest(unittest.TestCase):
    def test_control_origin_recovers_ticks_and_reports_missing_update(self):
        origin_us = 20_000_000
        updates = []
        for tick in (1, 2, 4, 5):
            updates.append((
                origin_us + tick * 1_000_000 + (tick % 2) * 2000,
                target_qps(tick, 0.01, 400.0, 300.0),
            ))

        audit = fit_control_origin(
            updates, origin_us - 1_000_000, 6.0, 0.01, 400.0, 300.0
        )

        self.assertAlmostEqual(origin_us + 1000, audit["origin_us"], delta=2)
        self.assertEqual([3], audit["missing_ticks"])
        self.assertEqual(0, audit["duplicate_tick_count"])
        self.assertLess(audit["target_qps_max_abs_error"], 1e-9)

    def test_ready_time_controls_cycle_phase_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.tsv"
            candidate_path = Path(directory) / "candidate.tsv"
            baseline_path.write_text(
                prediction_line("cn_MI308_1", 1, 300_100_000, 100.0, 130.0)
                + prediction_line("cn_MI308_1", 2, 460_100_000, 100.0, 80.0),
                encoding="utf-8",
            )
            candidate_path.write_text(
                prediction_line("cn_MI308_1", 1, 300_100_000, 100.0, 110.0)
                + prediction_line("cn_MI308_1", 2, 460_100_000, 100.0, 90.0),
                encoding="utf-8",
            )
            baseline = read_trace(
                baseline_path, 0, 600_000_000, 0, 300.0, 30.0
            )
            candidate = read_trace(
                candidate_path, 0, 600_000_000, 0, 300.0, 30.0
            )
            report = analyze_candidate(baseline, candidate, 1000, 7)

        rows = list(baseline.values())
        self.assertEqual(1, rows[0]["cycle"])
        self.assertEqual(0, rows[0]["phase_bin"])
        self.assertEqual("rising", rows[0]["trend"])
        full = report["architectures"]["MI308"]["full"]
        self.assertEqual(2, full["all"]["count"])
        self.assertLess(full["all"]["mae_change_pct"], 0.0)
        self.assertEqual(1, full["improved_hosts"])

    def test_baseline_log_mismatch_is_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.tsv"
            candidate_path = Path(directory) / "candidate.tsv"
            baseline_path.write_text(
                prediction_line("cn_MI308_1", 1, 20_000, 10.0, 100.0, 101.0),
                encoding="utf-8",
            )
            candidate_path.write_text(
                prediction_line("cn_MI308_1", 1, 20_000, 10.0, 90.0),
                encoding="utf-8",
            )
            baseline = read_trace(
                baseline_path, 0, 1_000_000, 0, 300.0, 30.0
            )
            candidate = read_trace(
                candidate_path, 0, 1_000_000, 0, 300.0, 30.0
            )
            report = analyze_candidate(baseline, candidate, 100, 9)

        self.assertEqual(
            1, report["coverage"]["MI308_baseline_log_mismatch"]
        )
        self.assertEqual(
            0, report["architectures"]["MI308"]["full"]["all"]["count"]
        )


if __name__ == "__main__":
    unittest.main()
