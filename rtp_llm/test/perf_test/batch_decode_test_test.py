import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _load_cache_grid_cases,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid_runner import (
    CacheGridRunner,
    PrefixPromptFactory,
    _post_prefill,
)


class _WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


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
        self.assertEqual(args.cache_commit_tail_tokens, 4096)
        self.assertEqual(args.cache_grid_json, "")
        self.assertIsInstance(remaining, list)

    def test_generated_cache_grid_uses_independent_seq_and_cache_alignment(self):
        payload = {
            "seq_generation": {
                "kind": "linear_with_dense_prefix",
                "count": 20,
                "max_seq_len": 65535,
            },
            "seq_block_size": 256,
            "cache_block_size": 4096,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cases = _load_cache_grid_cases(str(path))
        self.assertTrue(any(case["input_len"] == 256 for case in cases))
        self.assertTrue(
            all(
                case["cache_len"] == 0 or case["cache_len"] % 4096 == 0
                for case in cases
            )
        )
        self.assertTrue(
            all(
                case["cache_len"] == 0
                or case["cache_len"] + 4096 <= case["input_len"]
                for case in cases
            )
        )

    def test_cache_seed_commits_one_tail_and_preserves_exact_prefix(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        target, prefix, built_len = factory.make_case(7, 32, 16)
        seed = factory.make_seed(7, prefix, 16, 8)
        prefix_ids = tokenizer.encode(prefix)
        self.assertEqual(built_len, 32)
        self.assertEqual(len(prefix_ids), 16)
        self.assertEqual(len(tokenizer.encode(seed)), 24)
        self.assertEqual(tokenizer.encode(target)[:16], prefix_ids)
        self.assertEqual(tokenizer.encode(seed)[:16], prefix_ids)

    def test_case_prefixes_are_isolated(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        _, prefix_a, _ = factory.make_case(1, 32, 16)
        _, prefix_b, _ = factory.make_case(2, 32, 16)
        self.assertNotEqual(tokenizer.encode(prefix_a), tokenizer.encode(prefix_b))

    @patch("rtp_llm.test.perf_test.cache_grid_runner.requests.post")
    def test_post_prefill_records_client_ttft_separately(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "aux_info": {
                "input_len": 1048575,
                "output_len": 1,
                "reuse_len": 0,
                "first_token_cost_time": 173.0,
                "cost_time": 175.0,
                "wait_time": 2.0,
            }
        }
        post.return_value = response
        result = _post_prefill(12345, "prompt", 10, "case:run0")
        self.assertTrue(result["success"])
        self.assertEqual(result["prefill_time_ms"], 173.0)
        self.assertGreaterEqual(result["ttft_ms"], 0.0)
        self.assertEqual(result["ttft_ms"], result["client_wall_time_ms"])
        self.assertEqual(result["ttft_source"], "client_http_wall_max_new_tokens_1")

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_runner_accepts_only_exact_shape_reuse_and_ttft(self, post):
        post.side_effect = [
            {"success": True},
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 12.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 10.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 11.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
            ).run()
        self.assertEqual(rows[0]["status"], "ok")
        self.assertTrue(rows[0]["shape_exact"])
        self.assertTrue(rows[0]["reuse_exact"])
        self.assertTrue(rows[0]["timing_valid"])
        self.assertEqual(rows[0]["median_ttft_ms"], 11.0)


if __name__ == "__main__":
    unittest.main()
