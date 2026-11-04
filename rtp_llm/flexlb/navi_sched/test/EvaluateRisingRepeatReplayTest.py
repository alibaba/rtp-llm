#!/usr/bin/env python3
"""Check paired replay error signs and pressure buckets."""

import json
import tempfile
import unittest
from pathlib import Path

from EvaluateRisingRepeatReplay import sample_segments, summarize


class EvaluateRisingRepeatReplayTest(unittest.TestCase):
    def test_signed_error_uses_actual_minus_prediction(self):
        result = summarize([(100.0, 80.0, 110.0, 90.0, 120.0),
                            (200.0, 180.0, 190.0, 220.0, 210.0)])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["base_before"]["bias_ms"], 20.0)
        self.assertEqual(result["final_before"]["bias_ms"], -5.0)
        self.assertEqual(result["final_after"]["mae_ms"], 15.0)

    def test_rising_pressure_only_uses_arrived_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mi-test.jsonl"
            events = []
            for index in range(18):
                features = [0] * 13
                features[6] = 0 if index < 9 else 40
                events.append({"type": "sample", "sequence": index + 1,
                               "event_time_us": (index % 9) * 1_000_000 +
                               (0 if index < 9 else 30_000_000),
                               "features": features})
            path.write_text("".join(json.dumps(event) + "\n" for event in events),
                            encoding="utf-8")
            segments = sample_segments(Path(directory))
            self.assertEqual(segments[("mi-test", 1)], "low")
            self.assertEqual(segments[("mi-test", 18)], "high_rising")


if __name__ == "__main__":
    unittest.main()
