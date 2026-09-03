import argparse
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _load_cache_grid_cases,
    parse_args,
)


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_cache_grid_loader_validates_explicit_cases(self):
        payload = {
            "cases": [
                {"case_id": 7, "batch_size": 1, "input_len": 4096, "cache_len": 0},
                {
                    "case_id": 8,
                    "batch_size": 1,
                    "input_len": 4096,
                    "cache_len": 2048,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_cache_grid_cases(str(path)), payload["cases"])

    def test_cache_grid_loader_rejects_duplicate_geometry(self):
        payload = {
            "cases": [
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate cache grid case"):
                _load_cache_grid_cases(str(path))

    def test_parse_args_exposes_cache_runner_controls(self):
        args, remaining = parse_args()
        self.assertEqual(args.cache_measure_runs, 3)
        self.assertGreater(args.cache_request_timeout, 0)
        self.assertEqual(args.cache_grid_json, "")
        self.assertIsInstance(remaining, list)


if __name__ == "__main__":
    unittest.main()
