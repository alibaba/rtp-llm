#!/usr/bin/env python3

import pathlib
import tempfile
import unittest

import analyze_queue_timing as audit


def training_line(sequence, actual_ms, master_ms, raw_ms, agent_ms,
                  fit_step, valid=1, include_ready_fields=True):
    ready_fields = ""
    if include_ready_fields:
        ready_fields = (
            " queue_features_valid={} queue_raw_at_ready_ms={}"
            " agent_queue_at_ready_ms={} queue_preceding_batch_count=2"
            " queue_fit_step_at_ready={}"
            " queue_weights_at_ready=[1.0,2.0,3.0]"
        ).format(valid, raw_ms, agent_ms, fit_step)
    return (
        "[10.0.0.1] [2026-09-02 00:00:00.000001] "
        "[navi-latency-training-sample] sequence={} type=batch_finished "
        "batch_id={} master_queue_prediction_ms={} backend_queue_ms={}{}"
    ).format(
        sequence, sequence + 100, master_ms, actual_ms, ready_fields
    )


class QueueTimingAuditTest(unittest.TestCase):
    def parse(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "10_0_0_1.agent.log"
            path.write_text("\n".join(lines) + "\n")
            return audit.parse_training_samples(path)

    def test_ready_time_training_summary_and_stages(self):
        rows = self.parse([
            training_line(1, 120.0, 100.0, 90.0, 110.0, 25),
            training_line(2, 80.0, 90.0, 70.0, 85.0, 150),
        ])

        self.assertEqual("ready_time_frozen", audit.training_semantics(rows))
        summary = audit.summarize_training(rows)
        self.assertEqual(2, summary["agent_queue_at_ready_error_ms"]["count"])
        self.assertAlmostEqual(
            2.5, summary["agent_queue_at_ready_error_ms"]["mean"]
        )
        self.assertAlmostEqual(
            7.5,
            summary["agent_queue_at_ready_absolute_error_ms"]["mean"],
        )
        stages = audit.summarize_training_stages(rows)
        self.assertEqual(1, stages["cold_fit_step_lt_50"]["sample_count"])
        self.assertEqual(1, stages["steady_fit_step_ge_100"]["sample_count"])
        self.assertEqual(0, stages["steady_fit_step_ge_200"]["sample_count"])

    def test_legacy_and_invalid_rows_are_not_used_as_ready_predictions(self):
        rows = self.parse([
            training_line(
                1, 100.0, 90.0, 80.0, 70.0, 20,
                include_ready_fields=False,
            ),
            training_line(2, 100.0, 90.0, 80.0, 70.0, 20, valid=0),
        ])

        self.assertEqual("mixed_or_missing", audit.training_semantics(rows))
        summary = audit.summarize_training(rows)
        self.assertEqual(0, summary["actual_queue_ms"]["count"])

    def test_pure_legacy_log_is_explicitly_marked(self):
        rows = self.parse([
            training_line(
                1, 100.0, 90.0, 80.0, 70.0, 20,
                include_ready_fields=False,
            ),
        ])

        self.assertEqual(
            "legacy_finish_time_recomputed_or_unavailable",
            audit.training_semantics(rows),
        )


if __name__ == "__main__":
    unittest.main()
