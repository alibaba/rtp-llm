#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from AblateLatencyOnlineRates import (
    build_replay_command,
    cycle_index,
    evaluate,
    score_time_us,
    summarize,
)


class AblateLatencyOnlineRatesTest(unittest.TestCase):
    def test_grouped_mi_data_does_not_classify_215_host_as_rtx(self):
        def write_replay(command, **_kwargs):
            Path(command[2]).write_text(
                "P\t215_82_131_240\t4\t0\t1\t1000000\t100\t0\t120\t120\t0\t120\t120\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("AblateLatencyOnlineRates.subprocess.run",
                            side_effect=write_replay):
                result = evaluate(
                    Path("replay"), Path(directory) / "mi", 0, 2_000_000,
                    300, 4, 1, 0, directory,
                )
        self.assertEqual(1, result["MI"]["count"])
        self.assertNotIn("RTX", result)

    def test_grouped_rtx_data_uses_group_not_host_address(self):
        def write_replay(command, **_kwargs):
            Path(command[2]).write_text(
                "P\t215_82_131_240\t4\t0\t1\t1000000\t100\t0\t120\t120\t0\t120\t120\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("AblateLatencyOnlineRates.subprocess.run",
                            side_effect=write_replay):
                result = evaluate(
                    Path("replay"), Path(directory) / "rtx_stable", 0,
                    2_000_000, 300, 4, 1, 0, directory,
                )
        self.assertEqual(1, result["RTX"]["count"])
        self.assertNotIn("MI", result)

    def test_ambiguous_host_without_group_is_rejected(self):
        def write_replay(command, **_kwargs):
            Path(command[2]).write_text(
                "P\t215_82_131_240\t4\t0\t1\t1000000\t100\t0\t120\t120\t0\t120\t120\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("AblateLatencyOnlineRates.subprocess.run",
                            side_effect=write_replay):
                with self.assertRaisesRegex(ValueError, "hardware is unknown"):
                    evaluate(
                        Path("replay"), Path(directory) / "mixed", 0,
                        2_000_000, 300, 4, 1, 0, directory,
                    )

    def test_summary_preserves_signed_error_direction(self):
        summary = summarize([(10.0, 7.0), (10.0, 15.0)])

        self.assertEqual(1.0, summary["bias_ms"])
        self.assertEqual(4.0, summary["mae_ms"])
        self.assertEqual(0.4, summary["wape"])

    def test_cycle_index_uses_configured_period(self):
        start_us = 1_000_000

        self.assertEqual(0, cycle_index(start_us, start_us, 300))
        self.assertEqual(
            0, cycle_index(start_us + 300_000_000 - 1, start_us, 300)
        )
        self.assertEqual(1, cycle_index(start_us + 300_000_000, start_us, 300))
        self.assertEqual(4, cycle_index(start_us + 1_499_999_999, start_us, 300))

    def test_ready_time_and_event_time_have_explicit_boundaries(self):
        event_time_us = 610_000_000
        target_ms = 20_000.0

        ready_time_us = score_time_us(event_time_us, target_ms, "ready")
        completion_time_us = score_time_us(event_time_us, target_ms, "event")

        self.assertEqual(590_000_000, ready_time_us)
        self.assertEqual(610_000_000, completion_time_us)
        self.assertEqual(1, cycle_index(ready_time_us, 0, 300))
        self.assertEqual(2, cycle_index(completion_time_us, 0, 300))

    def test_score_time_rejects_unknown_basis(self):
        with self.assertRaisesRegex(ValueError, "invalid score time basis"):
            score_time_us(1_000_000, 10.0, "unknown")

    def test_command_uses_counterfactual_rates(self):
        command = build_replay_command(
            Path("replay"), Path("input"), Path("output"), 6, 1.7, 0.65
        )

        self.assertEqual(
            [
                "replay",
                "input",
                "output",
                "6",
                "20261008",
                "1.7",
                "0.65",
                "all",
                "0",
                "counterfactual_fixed",
            ],
            command,
        )

    def test_command_accepts_experimental_strategy(self):
        mode = "production_error_rate_hinge:1,2,8,0,0.1"
        command = build_replay_command(
            Path("replay"), Path("input"), Path("output"), 80, 2.55, 0, mode
        )

        self.assertEqual("0", command[6])
        self.assertEqual(mode, command[-1])


if __name__ == "__main__":
    unittest.main()
