import argparse
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.batch_decode_test import (
    _create_grid_query_dict,
    _effective_grid_max_seq_len,
    _parse_grid_cases,
    _positive_int_arg,
)
from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import ResponseInfo, TestResultMetrics
from rtp_llm.test.perf_test.grid_runner import GridRunner
from rtp_llm.test.perf_test.test_util import target_reuse_len_for_hit_rate


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    @patch("rtp_llm.test.perf_test.batch_decode_test.create_query")
    def test_grid_query_token_offset_preserves_requested_keys(self, create_query):
        create_query.return_value = {4090: "short", 16378: "long"}
        with patch.dict(os.environ, {"PERF_QUERY_TOKEN_OFFSET": "6"}):
            self.assertEqual(
                _create_grid_query_dict([4096, 16384]),
                {4096: "short", 16384: "long"},
            )
        create_query.assert_called_once_with(input_len_list=[4090, 16378])

    def test_grid_query_token_offset_rejects_invalid_values(self):
        with patch.dict(os.environ, {"PERF_QUERY_TOKEN_OFFSET": "-1"}):
            with self.assertRaises(ValueError):
                _create_grid_query_dict([4096])
        with patch.dict(os.environ, {"PERF_QUERY_TOKEN_OFFSET": "4096"}):
            with self.assertRaises(ValueError):
                _create_grid_query_dict([4096])

    def test_parse_grid_cases_dedupes_in_order(self):
        self.assertEqual(
            _parse_grid_cases("1:8192, 2:262144,1:8192"),
            [(1, 8192), (2, 262144)],
        )

    def test_parse_grid_cases_empty_uses_cartesian_grid(self):
        self.assertIsNone(_parse_grid_cases(""))

    def test_parse_grid_cases_rejects_malformed_items(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("1:8192,bad")

    def test_parse_grid_cases_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("0:8192")

    def test_positive_int_arg_parses_forwarded_server_args(self):
        self.assertEqual(
            _positive_int_arg(["--seq_size_per_block", "64"], "seq_size_per_block", 1),
            64,
        )
        self.assertEqual(_positive_int_arg([], "seq_size_per_block", 8), 8)

    def test_target_reuse_len_rounds_to_block(self):
        self.assertEqual(target_reuse_len_for_hit_rate(1048576, 0.85, 64), 891264)

    def test_fixed_batch_capacity_keeps_all_grid_statuses(self):
        batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        contexts = [4096, 16384, 32768, 65536]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "PERF_KV_BLOCK_CAPACITY": "35999",
                "PERF_KV_TOKENS_PER_BLOCK": "64",
                "PERF_VALIDATE_DECODE": "1",
            },
        ), patch.object(GridRunner, "warmup"), patch(
            "rtp_llm.test.perf_test.grid_runner.create_metrics_table", return_value=""
        ), patch(
            "rtp_llm.test.perf_test.grid_runner.BatchPerfImpl"
        ) as impl:

            def make_impl(*args, **kwargs):
                from types import SimpleNamespace

                return SimpleNamespace(
                    validation_errors=[],
                    run=lambda: TestResultMetrics(
                        args[2] * 10, args[2] * 10, 0, avg_input_len=int(args[3])
                    ),
                )

            impl.side_effect = make_impl
            GridRunner(
                1,
                8,
                batches,
                contexts,
                {n: str(n) for n in contexts},
                dump_json_path=directory,
                decode_test_length=30,
            ).run()
            with open(os.path.join(directory, "Grid_Status.json")) as f:
                statuses = json.load(f)
            self.assertEqual(len(statuses), 36)
            self.assertEqual(impl.call_count, 30)
            self.assertEqual(sum(s["status"] == "N/A" for s in statuses), 6)
            self.assertTrue(
                all(s["status"] == "PASS" for s in statuses if s["input_len"] == 4096)
            )

    def test_decode_round_rejects_zero_success_short_output_and_nonfinite_time(self):
        impl = BatchPerfImpl.__new__(BatchPerfImpl)
        impl.is_decode, impl.batch_size, impl.decode_test_length = True, 1, 30
        impl.profile_trace_name, impl.validation_errors = "bs1_seq4096_decode", []
        impl.expected_input_len = 4096
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "PERF_VALIDATE_DECODE": "1",
                "TEST_UNDECLARED_OUTPUTS_DIR": directory,
            },
        ):
            for index, response in enumerate(
                [
                    ResponseInfo({}, False),
                    ResponseInfo({"aux_info": {"output_len": 1}}),
                    ResponseInfo(
                        {"aux_info": {"output_len": 30, "cost_time": float("nan")}}
                    ),
                    ResponseInfo(
                        {
                            "aux_info": {
                                "input_len": 4102,
                                "output_len": 30,
                                "cost_time": 30,
                            }
                        }
                    ),
                ]
            ):
                impl._record_decode_round("formal", index, [response])
            self.assertEqual(len(impl.validation_errors), 4)
            with open(os.path.join(directory, "decode_rounds.jsonl")) as f:
                rows = [
                    json.loads(line, parse_constant=lambda value: self.fail(value))
                    for line in f
                ]
                self.assertTrue(all(not row["valid"] for row in rows))


if __name__ == "__main__":
    unittest.main()
