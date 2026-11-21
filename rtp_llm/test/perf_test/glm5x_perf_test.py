import importlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.engine_paths import extract_arg
from rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.glm5x_legacy.dataclass import ResponseInfo
from rtp_llm.test.perf_test.glm5x_legacy.perf_config import (
    _replace_cli_value,
    parse_args,
    prepare_config,
)
from rtp_llm.test.perf_test.glm5x_legacy.tps_runner import TpsBinarySearchRunner
from rtp_llm.test.perf_test.glm5x_perf import normalize_model_paths, validation_policy


class Glm5xPerfTest(unittest.TestCase):
    def test_model_path_override(self):
        for option in (["--tokenizer_path", "/shared"], ["--tokenizer_path=/shared"]):
            argv = normalize_model_paths(["--tokenizer_path", "/default"] + option)
            self.assertEqual(extract_arg(argv, "tokenizer_path"), "/shared")
            _replace_cli_value(argv, "tokenizer_path", "/resolved")
            self.assertEqual(extract_arg(argv, "tokenizer_path"), "/resolved")

    def test_preserve_other_options(self):
        argv = ["entry", "--test_env", "K=V", "--batch_size", "1,4"]
        self.assertEqual(normalize_model_paths(argv), argv)

    def test_all_model_paths(self):
        for key in ("checkpoint_path", "tokenizer_path", "sp_checkpoint_path"):
            argv = normalize_model_paths([f"--{key}=old", f"--{key}", "new"])
            self.assertEqual(extract_arg(argv, key), "new")

    def test_missing_value(self):
        with self.assertRaises(ValueError):
            normalize_model_paths(["--checkpoint_path"])

    def collect(self, successes, expected=2):
        with patch(
            "rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl.ProcessPoolExecutor"
        ):
            runner = BatchPerfImpl(8000, 1, expected, "prompt")
        runner.executor.submit.return_value.result.return_value = [
            ResponseInfo({}, value) for value in successes
        ]
        return runner._curl_server_responses()

    def test_glm_policy_runs_at_response_boundary(self):
        with validation_policy():
            for responses in (
                [False, False],
                [True, False],
                [True],
                [True, True, True],
            ):
                with self.subTest(responses=responses), self.assertRaises(RuntimeError):
                    self.collect(responses)
            self.assertEqual(len(self.collect([True, True])), 2)

    def test_tps_failed_probe_shrinks_search_and_validates_winner(self):
        calls = []

        def run(impl, **kwargs):
            calls.append((impl.batch_size, impl.validate_responses))
            successes = [impl.batch_size <= 4] * impl.batch_size
            impl.executor.submit.return_value.result.return_value = [
                ResponseInfo({}, success) for success in successes
            ]
            responses = impl._curl_server_responses()
            return SimpleNamespace(
                success_requests=sum(r.success for r in responses),
                total_requests=len(responses),
                avg_decode_time=10.0,
            )

        with validation_policy(), patch(
            "rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl.ProcessPoolExecutor"
        ), patch.object(BatchPerfImpl, "run", run):
            runner = TpsBinarySearchRunner(8000, 1, 30, 16)
            result = runner._binary_search(lambda bs: "prompt", "test")
            self.assertEqual(result.best_bs, 4)
            self.assertTrue(
                any(not step.satisfies_tpot for step in result.search_steps)
            )
            self.assertIn((8, False), calls)
            self.assertEqual(calls[-1], (4, True))
            with self.assertRaises(RuntimeError):
                runner._test_bs(8, "prompt", validate_responses=True)
            # The search exception must not disable the process-wide policy.
            with self.assertRaises(RuntimeError):
                self.collect([True, False])

    def test_tps_measurement_trimming_preserves_failed_and_missing_requests(self):
        good = ResponseInfo({"aux_info": {"output_len": 2, "cost_time": 10}}, True)
        for first_batch in ([ResponseInfo({}, False)], []):
            with self.subTest(first_batch=first_batch), validation_policy(), patch(
                "rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl.ProcessPoolExecutor"
            ), patch.object(BatchPerfImpl, "_set_concurrency"):
                impl = BatchPerfImpl(
                    8000,
                    1,
                    1,
                    "prompt",
                    profile=False,
                    warmup_runs=0,
                    measure_runs=3,
                    validate_responses=False,
                )
                impl.executor.submit.return_value.result.side_effect = [
                    first_batch,
                    [good],
                    [good],
                ]
                result = impl.run()
                self.assertEqual(result.total_requests, 3)
                self.assertEqual(result.success_requests, 2)
                self.assertEqual(result.fail_requests, 1)
                self.assertEqual(result.avg_decode_time, 10)

    def test_default_behavior_and_restore_on_error(self):
        self.assertIsNone(BatchPerfImpl.response_validator)
        self.assertEqual(len(self.collect([False, False])), 2)
        with self.assertRaises(RuntimeError):
            with validation_policy():
                self.collect([False, False])
        self.assertIsNone(BatchPerfImpl.response_validator)
        self.assertEqual(len(self.collect([True, False])), 2)


class Glm5xLegacyGridCompatibilityTest(unittest.TestCase):
    @staticmethod
    def grid_entry():
        # Server construction is the external model boundary; these CPU tests
        # exercise orchestration without importing the compiled engine runtime.
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.test.perf_test.server": SimpleNamespace(EngineServer=object),
                "rtp_llm.test.perf_test.test_util": SimpleNamespace(
                    create_query=lambda **kwargs: {}
                ),
            },
        ):
            return importlib.import_module(
                "rtp_llm.test.perf_test.glm5x_legacy.batch_decode_test"
            )

    def test_grid_configuration_preserves_per_rank_batches_and_sequence_limit(self):
        args, remaining = parse_args(
            [
                "--dp_size",
                "4",
                "--batch_size",
                "1,4,8",
                "--input_len",
                "4096,16384,32768,65536",
                "--max_seq_len",
                "1048576",
                "--decode_test_length",
                "30",
                "--reuse_cache",
                "1",
            ]
        )
        config = prepare_config(args, remaining)
        _effective_grid_max_seq_len = self.grid_entry()._effective_grid_max_seq_len
        self.assertEqual(config.batch_size_list, [1, 4, 8])
        self.assertEqual(config.max_concurrency, 8)
        self.assertEqual(
            _effective_grid_max_seq_len(args, config.input_len_list), 1048576
        )
        self.assertEqual(extract_arg(remaining, "reuse_cache"), "1")

    def test_generic_legacy_modes_are_not_exposed(self):
        for option in ("--target_tpot=30", "--dataset=x", "--cache_grid_json=x"):
            with self.subTest(option=option), patch("sys.stderr"), self.assertRaises(
                SystemExit
            ):
                parse_args([option])

    def test_grid_scales_batches_per_dp_and_keeps_cold_prefill(self):
        from rtp_llm.test.perf_test.glm5x_legacy.grid_runner import GridRunner

        with tempfile.TemporaryDirectory() as directory, patch(
            "rtp_llm.test.perf_test.glm5x_legacy.grid_runner.BatchPerfImpl"
        ) as impl, patch(
            "rtp_llm.test.perf_test.glm5x_legacy.grid_runner.create_metrics_table"
        ):
            GridRunner(
                8000,
                4,
                [1, 4, 8],
                [4096],
                {4096: "prompt"},
                dump_json_path=directory,
            ).run()
            self.assertEqual(
                [call.args[2] for call in impl.call_args_list], [4, 4, 16, 32]
            )
            impl.reset_mock()
            GridRunner(
                8000,
                1,
                [1],
                [4096],
                {4096: "prompt"},
                is_decode=False,
                dump_json_path=directory,
            ).run()
            self.assertEqual(
                [call.args[4] for call in impl.call_args_list], [False, False]
            )

    def test_engine_is_stopped_if_grid_validation_fails(self):
        entry = self.grid_entry()
        args, remaining = parse_args(["--input_len", "16", "--batch_size", "1"])
        with tempfile.TemporaryDirectory() as directory:
            args.result_dir = directory
            with patch.object(
                entry, "parse_args", return_value=(args, remaining)
            ), patch.object(
                entry, "resolve_perf_engine_paths", side_effect=lambda argv: argv
            ), patch.object(
                entry, "EngineServer"
            ) as server, patch.object(
                entry, "create_query", return_value={16: "prompt"}
            ), patch.object(
                entry, "query_engine_status", return_value={}
            ), patch.object(
                entry, "print_config_table"
            ), patch.object(
                entry, "write_test_info"
            ), patch.object(
                entry, "_run_decode", side_effect=RuntimeError("failed batch")
            ), patch.object(
                entry, "summarize_and_cleanup_coredumps"
            ), patch.object(
                importlib.import_module("rtp_llm.config.log_config"), "setup_logging"
            ), patch.dict(
                os.environ, {}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "failed batch"):
                    entry.main()
                server.return_value.stop.assert_called_once()

    def test_upstream_warmup_trim_and_separate_profile_protocol(self):
        with patch(
            "rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl.ProcessPoolExecutor"
        ), patch.dict(os.environ, {"PERF_PREARM_PROFILE": "0"}):
            impl = BatchPerfImpl(
                8000, 1, 1, "prompt", warmup_runs=3, measure_runs=5, profile_runs=1
            )
            responses = [
                [ResponseInfo({"aux_info": {"output_len": 2, "cost_time": value}})]
                for value in (1, 2, 50, 4, 5)
            ]
            with patch.object(impl, "_set_concurrency"), patch.object(
                impl, "_curl_server"
            ) as non_measure, patch.object(
                impl, "_curl_server_responses", side_effect=responses
            ), patch(
                "rtp_llm.test.perf_test.glm5x_legacy.batch_perf_impl._wait_for_trace_flush",
                return_value=0,
            ):
                result = impl.run()
            self.assertAlmostEqual(result.avg_decode_time, 11 / 3)
            self.assertEqual(
                [call.args for call in non_measure.call_args_list],
                [(), (), (), (True,)],
            )


if __name__ == "__main__":
    unittest.main()
