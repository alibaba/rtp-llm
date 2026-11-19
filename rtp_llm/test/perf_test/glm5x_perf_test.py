import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import ResponseInfo
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.glm5x_perf import normalize_model_paths, validation_policy
from rtp_llm.test.perf_test.perf_config import _replace_cli_value
from rtp_llm.test.perf_test.tps_runner import TpsBinarySearchRunner


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
        with patch("rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"):
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
            "rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"
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
                "rtp_llm.test.perf_test.batch_perf_impl.ProcessPoolExecutor"
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


if __name__ == "__main__":
    unittest.main()
