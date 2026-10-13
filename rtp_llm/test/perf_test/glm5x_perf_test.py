import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataclass import ResponseInfo
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.glm5x_perf import normalize_model_paths, validation_policy
from rtp_llm.test.perf_test.perf_config import _replace_cli_value


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
