"""Exercise the production event-loading block, including its strict failure paths."""

import copy
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EvictionEventAggregationTest(unittest.TestCase):
    def load(self, events):
        source = Path(__file__).resolve().parents[1] / "stress/aggregate_canvas_run.py"
        text = source.read_text()
        block = text[
            text.index("def validate_eviction_event(") : text.index(
                "per_sec = defaultdict("
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine_events.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in events))
            # Only redirect the input pathname. Execute the real loader unchanged.
            scope = dict(json=json, gzip=gzip, os=os, sys=sys)
            real_open = open
            with patch(
                "builtins.open",
                side_effect=lambda name, *a, **kw: real_open(
                    path if name == "engine_events.jsonl" else name, *a, **kw
                ),
            ), patch(
                "os.path.isfile", side_effect=lambda name: name == "engine_events.jsonl"
            ):
                exec(compile(block, str(source), "exec"), scope)
            return scope

    def chain(self):
        return dict(
            event="evict_chain",
            leaf_key=3,
            chain_keys=[3, 2, 1],
            blocks_freed=2,
            reason="admission",
            timestamp_ms=1000,
            engine_name="p0",
            engine_incarnation="epoch",
            engine_address="127.0.0.1:1",
        )

    def test_valid_chain_does_not_create_or_change_request_completion(self):
        completion = dict(
            event="decode_done",
            rid=10,
            decode_done_ms=1005,
            exec_ms=3,
            engine_arrival_ms=1001,
            decode_start_ms=1002,
        )
        scope = self.load([self.chain(), completion, self.chain()])
        self.assertEqual(scope["decode_done_map"], {10: (1005, 3, 1001, 1002)})
        self.assertEqual(scope["prefill_done_map"], {})

    def test_malformed_chain_still_fails_closed(self):
        for field, value in [
            ("chain_keys", []),
            ("chain_keys", [3, 3]),
            ("blocks_freed", 4),
            ("blocks_freed", True),
            ("leaf_key", 7),
            ("timestamp_ms", 0),
            ("reason", "unknown"),
            ("engine_incarnation", None),
        ]:
            with self.subTest(field=field, value=value):
                event = copy.deepcopy(self.chain())
                event[field] = value
                with self.assertRaisesRegex(SystemExit, "invalid evict_chain"):
                    self.load([event])

    def test_unknown_event_and_incomplete_completion_remain_errors(self):
        with self.assertRaisesRegex(SystemExit, "unknown engine event"):
            self.load([dict(event="unknown", rid=1)])
        with self.assertRaisesRegex(SystemExit, "missing/invalid terminal"):
            self.load([dict(event="decode_done", rid=1)])


if __name__ == "__main__":
    unittest.main()
