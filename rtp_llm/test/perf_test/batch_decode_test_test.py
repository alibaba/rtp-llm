import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import rtp_llm.test.perf_test.batch_decode_test as batch_decode_test
from rtp_llm.test.perf_test.batch_decode_test import (
    _configure_scheduler,
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


class PipelinePrefillPerfTest(unittest.TestCase):
    def args(self, **kw):
        values = dict(
            partial=2,
            dp_size=1,
            batch_size="1",
            target_tpot=0,
            dataset="",
            dataset_name="",
            dataset_path="",
            test_json="",
        )
        return argparse.Namespace(**dict(values, **kw))

    def test_non_pp_keeps_legacy_scheduler(self):
        cli = []
        self.assertTrue(_configure_scheduler(self.args(), cli, {}, {}))
        self.assertEqual(cli, ["--use_batch_decode_scheduler", "1"])

    def test_pp_uses_native_scheduler(self):
        for cli in (
            ["--pp_size", "2"],
            ["--pp_size=2", "--use_batch_decode_scheduler=0"],
        ):
            self.assertFalse(
                _configure_scheduler(self.args(), cli, {}, {"PERF_PROFILE_RUNS": "0"})
            )

    def test_pp_rejects_incompatible_modes_before_server_start(self):
        for kw in (
            {"partial": 1},
            {"dp_size": 2},
            {"batch_size": "2"},
            {"dataset_name": "sharegpt"},
            {"target_tpot": 10},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                _configure_scheduler(
                    self.args(**kw), ["--pp_size", "2"], {}, {"PERF_PROFILE_RUNS": "0"}
                )
        with self.assertRaisesRegex(ValueError, "BatchDecodeScheduler"):
            _configure_scheduler(
                self.args(),
                ["--pp_size", "2", "--use_batch_decode_scheduler", "1"],
                {},
                {},
            )

    def test_pp_rejects_request_and_prearm_profilers(self):
        for env, gen in [
            ({}, {}),
            ({"PERF_PROFILE_RUNS": "0", "PERF_PREARM_PROFILE": "1"}, {}),
            ({"PERF_PROFILE_RUNS": "0"}, {"gen_timeline": True}),
        ]:
            with self.subTest(env=env, gen=gen), self.assertRaisesRegex(
                ValueError, "profiler"
            ):
                _configure_scheduler(self.args(), ["--pp_size", "2"], gen, env)

    def test_native_scheduler_skips_control_api(self):
        runner = BatchPerfImpl.__new__(BatchPerfImpl)
        runner.use_batch_decode_scheduler = False
        runner.is_decode = False
        runner.batch_size = runner.dp_size = 1
        with patch("rtp_llm.test.perf_test.batch_perf_impl.requests.post") as post:
            runner._set_concurrency()
            post.assert_not_called()
            runner.batch_size = 2
            with self.assertRaises(Exception):
                runner._set_concurrency()
            post.assert_not_called()

    def test_legacy_scheduler_still_calls_control_api(self):
        runner = BatchPerfImpl.__new__(BatchPerfImpl)
        runner.use_batch_decode_scheduler = True
        runner.is_decode = True
        runner.batch_size = 8
        runner.dp_size = 4
        runner.base_port = 1234
        with patch("rtp_llm.test.perf_test.batch_perf_impl.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {"status": "ok"}
            runner._set_concurrency()
            self.assertEqual(
                post.call_args.kwargs["json"], {"batch_size": 2, "mode": "decode"}
            )

    def test_pp_fails_on_first_bad_request(self):
        runner = BatchDecodeTest._batch_perf([ResponseInfo({}, False)])
        runner.use_batch_decode_scheduler = False
        runner.measure_runs = 2
        with self.assertRaisesRegex(RuntimeError, "PP request"):
            runner.run()
        self.assertEqual(runner._curl_server_responses.call_count, 1)

    def test_pp_failed_formal_warmup_stops_measurement(self):
        runner = BatchDecodeTest._batch_perf([])
        runner.use_batch_decode_scheduler = False
        runner.warmup_runs = 1
        runner._curl_server = MagicMock(return_value=TestResultMetrics(1, 0, 1))
        with self.assertRaisesRegex(RuntimeError, "PP warmup"):
            runner.run()
        runner._curl_server_responses.assert_not_called()

    def test_grid_threads_native_scheduler_to_warmup_and_measurement(self):
        from rtp_llm.test.perf_test.grid_runner import GridRunner

        with patch("rtp_llm.test.perf_test.grid_runner.BatchPerfImpl") as batch, patch(
            "rtp_llm.test.perf_test.grid_runner.create_metrics_table"
        ), patch.dict(os.environ, {"PERF_GRID_WARMUP_RUNS": "2"}):
            batch.return_value.run.return_value = TestResultMetrics(1, 1, 0)
            GridRunner(
                1234,
                1,
                [1],
                [32],
                {32: "hello"},
                is_decode=False,
                use_batch_decode_scheduler=False,
            ).run()
            self.assertEqual(batch.call_count, 2)
            for call in batch.call_args_list:
                self.assertFalse(call.kwargs["use_batch_decode_scheduler"])
            self.assertEqual(batch.call_args_list[0].kwargs["measure_runs"], 2)

    def test_server_respects_routing_override_and_trace_output(self):
        from rtp_llm.test.perf_test.server import EngineServer

        args = argparse.Namespace(partial=2, dp_size=1, result_dir="/tmp/pp-perf")
        with patch(
            "rtp_llm.test.perf_test.server.MagaServerManager"
        ) as manager, patch.dict(
            os.environ,
            {"FAKE_BALANCE_EXPERT": "0", "DSV4_FWD_PROFILE": "1"},
            clear=True,
        ):
            server = EngineServer(args, [])
            server.start(32832, 1, use_batch_decode_scheduler=False)
            env = manager.call_args.kwargs["env_args"]
            self.assertEqual(env["FAKE_BALANCE_EXPERT"], "0")
            self.assertEqual(env["USE_BATCH_DECODE_SCHEDULER"], "0")
            self.assertEqual(env["DSV4_FWD_TRACE_DIR"], "/tmp/pp-perf")
            server.stop()
        with patch(
            "rtp_llm.test.perf_test.server.MagaServerManager"
        ) as manager, patch.dict(os.environ, {}, clear=True):
            server = EngineServer(args, [])
            server.start(32832, 1)
            self.assertEqual(
                manager.call_args.kwargs["env_args"]["FAKE_BALANCE_EXPERT"], "1"
            )
            self.assertNotIn("DSV4_FWD_TRACE_DIR", manager.call_args.kwargs["env_args"])
            server.stop()

    def test_required_trace_positive_and_negative_gates(self):
        from rtp_llm.test.perf_test.perf_utils import collect_timeline_files

        event = {"cat": "kernel", "ph": "X", "ts": 1.0, "dur": 2.0}
        for case in (
            "valid",
            "missing",
            "cpu_only",
            "zero",
            "nan",
            "stale",
            "overwrite",
        ):
            with self.subTest(
                case=case
            ), tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"PERF_REQUIRED_TRACE": "fwd_rank0_idx24.json"}
            ), patch(
                "rtp_llm.test.perf_test.perf_utils.time.sleep"
            ):
                p = Path(directory) / "fwd_rank0_idx24.json"
                e = dict(event)
                if case == "cpu_only":
                    e["cat"] = "cpu_op"
                if case == "zero":
                    e["ts"] = 0
                if case == "nan":
                    e["dur"] = float("nan")
                if case != "missing":
                    p.write_text(json.dumps({"traceEvents": [e]}))
                if case == "stale":
                    os.utime(p, (1, 1))
                if case == "overwrite":
                    (Path(directory) / "timelines").mkdir()
                    (Path(directory) / "timelines" / p.name).write_text("old")
                if case == "valid":
                    collect_timeline_files(directory, started=2)
                    self.assertTrue((Path(directory) / "timelines" / p.name).is_file())
                else:
                    with self.assertRaises(RuntimeError):
                        collect_timeline_files(directory, started=2)


if __name__ == "__main__":
    unittest.main()
