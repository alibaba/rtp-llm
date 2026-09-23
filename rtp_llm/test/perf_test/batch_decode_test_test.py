import argparse
import unittest
from unittest.mock import MagicMock, patch

import rtp_llm.test.perf_test.batch_decode_test as batch_decode_test
from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _engine_tp_size,
    _ensure_default_role_type,
)
from rtp_llm.test.perf_test.batch_perf_impl import (
    BatchPerfImpl,
    require_complete_measurement,
)
from rtp_llm.test.perf_test.dataclass import ResponseInfo, TestResultMetrics


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_default_role_does_not_override_environment(self):
        args = []
        _ensure_default_role_type(args, {"ROLE_TYPE": "PREFILL"})
        self.assertEqual(args, [])

    def test_default_role_applies_without_cli_or_environment(self):
        args = []
        _ensure_default_role_type(args, {})
        self.assertEqual(args, ["--role_type", "PDFUSION"])

    def test_cli_role_wins_over_environment(self):
        args = ["--role_type", "DECODE"]
        _ensure_default_role_type(args, {"ROLE_TYPE": "PREFILL"})
        self.assertEqual(args, ["--role_type", "DECODE"])

    def test_engine_tp_size_prefers_cli_then_environment(self):
        self.assertEqual(_engine_tp_size(["--tp_size", "4"], {"TP_SIZE": "8"}), 4)
        self.assertEqual(_engine_tp_size([], {"TP_SIZE": "8"}), 8)
        self.assertEqual(_engine_tp_size([], {}), 1)

    @staticmethod
    def _batch_perf(responses):
        runner = BatchPerfImpl.__new__(BatchPerfImpl)
        runner.warmup_runs = 0
        runner.measure_runs = 1
        runner.profile = False
        runner.profile_runs = 0
        runner.profile_trace_name = "unit"
        runner.is_decode = True
        runner._set_concurrency = MagicMock()
        runner._curl_server_responses = MagicMock(return_value=responses)
        return runner

    def test_batch_perf_returns_all_failed_measurements_to_caller(self):
        runner = self._batch_perf([ResponseInfo({}, False), ResponseInfo({}, False)])
        result = runner.run()
        self.assertEqual(result.total_requests, 2)
        self.assertEqual(result.success_requests, 0)
        self.assertEqual(result.fail_requests, 2)

    def test_batch_perf_accepts_partial_measurement_success(self):
        success = ResponseInfo(
            {
                "aux_info": {
                    "input_len": 4,
                    "output_len": 2,
                    "cost_time": 3.0,
                    "first_token_cost_time": 2.0,
                    "wait_time": 1.0,
                }
            }
        )
        result = self._batch_perf([ResponseInfo({}, False), success]).run()
        self.assertEqual(result.success_requests, 1)
        self.assertEqual(result.fail_requests, 1)

    def test_multi_run_with_failed_round_returns_complete_aggregate(self):
        success = ResponseInfo(
            {
                "aux_info": {
                    "input_len": 4,
                    "output_len": 2,
                    "cost_time": 3.0,
                    "first_token_cost_time": 2.0,
                    "wait_time": 1.0,
                }
            }
        )
        runner = self._batch_perf([])
        runner.measure_runs = 3
        runner._curl_server_responses.side_effect = [
            [ResponseInfo({}, False)],
            [ResponseInfo({}, False)],
            [success],
        ]

        result = runner.run()

        self.assertEqual(result.total_requests, 3)
        self.assertEqual(result.success_requests, 1)
        self.assertEqual(result.fail_requests, 2)
        self.assertEqual(result.avg_decode_time, 1.0)

    def test_result_table_contract_rejects_partial_measurement(self):
        metric = TestResultMetrics(
            total_requests=2,
            success_requests=1,
            fail_requests=1,
        )
        with self.assertRaisesRegex(RuntimeError, "grid.*success=1.*failed=1"):
            require_complete_measurement(metric, context="grid")

    def test_main_stops_server_when_runner_raises(self):
        args = argparse.Namespace(
            generate_config="{}",
            result_dir="/tmp/perf-test",
            partial=1,
            dp_size=1,
            decode_test_length=1,
            num_measures=1,
            batch_size="1",
            max_seq_len=8,
        )
        config = argparse.Namespace(
            is_distribution=False,
            input_len_list=[4],
            all_seq_lens=[4],
            max_seq_len=5,
            max_concurrency=1,
        )
        server = MagicMock(port=12345)
        with patch("rtp_llm.config.log_config.setup_logging"), patch.object(
            batch_decode_test, "parse_args", return_value=(args, [])
        ), patch.object(
            batch_decode_test, "resolve_perf_engine_paths", side_effect=lambda x: x
        ), patch.object(
            batch_decode_test, "prepare_config", return_value=config
        ), patch.object(
            batch_decode_test, "EngineServer"
        ) as engine_server, patch.object(
            batch_decode_test, "query_engine_status", return_value={}
        ), patch.object(
            batch_decode_test, "print_config_table"
        ), patch.object(
            batch_decode_test, "create_query", return_value={4: "query"}
        ), patch.object(
            batch_decode_test, "_run_decode", side_effect=RuntimeError("runner failed")
        ), patch.object(
            batch_decode_test, "summarize_and_cleanup_coredumps"
        ), patch.object(
            batch_decode_test.os, "makedirs"
        ):
            engine_server.return_value = server
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                batch_decode_test.main()

        server.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
