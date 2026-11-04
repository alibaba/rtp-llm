#!/usr/bin/env python3
"""Bounded, recoverable replay-trace transformation checks."""

import json
import tempfile
import unittest
from pathlib import Path

from RepeatRisingReplaySamples import transform_file


class RepeatRisingReplaySamplesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source.jsonl"
        self.destination = Path(self.temporary.name) / "result.jsonl"
        events = []
        for index in range(16):
            features = [0] * 13
            features[6] = 0 if index < 8 else 40
            events.append({"type": "sample", "sequence": index + 1,
                           "history_sequence": index + 1,
                           "event_time_us": (index % 8) * 1_000_000 +
                           (0 if index < 8 else 30_000_000),
                           "features": features})
        events.append({"type": "update", "step_before": 0,
                       "step_after": 2, "fit_batches": [[15], [16]]})
        self.source.write_text("".join(json.dumps(event) + "\n" for event in events),
                               encoding="utf-8")

    def test_extra_updates_only_use_observed_history(self):
        result = transform_file(self.source, self.destination, 2, 2, 100_000, 7)
        event = json.loads(self.destination.read_text().splitlines()[-1])
        self.assertEqual(result["extra_updates"], 4)
        self.assertEqual(event["fit_batches"], [[15], [16], [15], [16], [15], [16]])
        self.assertEqual(event["step_after"], 6)

    def test_within_batch_keeps_update_count(self):
        transform_file(self.source, self.destination, 2, 2, 100_000, 7,
                       mode="within_batch")
        event = json.loads(self.destination.read_text().splitlines()[-1])
        self.assertEqual(event["step_after"], 2)
        self.assertEqual(event["fit_batches"], [[15, 15, 15], [16, 16, 16]])

    def test_capacity_and_corrupt_tail_preserve_last_complete_result(self):
        transform_file(self.source, self.destination, 2, 2, 100_000, 7)
        previous = self.destination.read_bytes()
        with self.assertRaises(ValueError):
            transform_file(self.source, self.destination, 2, 2, 10, 7)
        self.assertEqual(self.destination.read_bytes(), previous)
        self.assertFalse(self.destination.with_name("result.jsonl.tmp").exists())
        with self.source.open("a", encoding="utf-8") as stream:
            stream.write("{corrupt tail\n")
        with self.assertRaises(json.JSONDecodeError):
            transform_file(self.source, self.destination, 2, 2, 100_000, 7)
        self.assertEqual(self.destination.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
