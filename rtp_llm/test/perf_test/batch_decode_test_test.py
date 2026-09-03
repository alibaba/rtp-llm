import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _engine_arg_argv,
    _parse_name_value,
    _redact_argv,
    parse_args,
)


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_engine_arg_shorthand_is_forwarded(self):
        self.assertEqual(
            _engine_arg_argv(["tp_size=8", "fp8_kv_cache=1"]),
            ["--tp_size", "8", "--fp8_kv_cache", "1"],
        )

    def test_name_value_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            _parse_name_value("tp_size", "--engine_arg")

    def test_parse_args_exposes_runtime_overrides(self):
        args, remaining = parse_args(
            [
                "--engine_arg=tp_size=8",
                "--engine_env=FP8_KV_CACHE=1",
                "--measure_runs=3",
                "--model_type=example_model",
            ]
        )
        self.assertEqual(args.engine_arg, ["tp_size=8"])
        self.assertEqual(args.engine_env, ["FP8_KV_CACHE=1"])
        self.assertEqual(args.measure_runs, 3)
        self.assertIn("--model_type=example_model", remaining)

    def test_redact_argv_hides_embedded_engine_secret(self):
        self.assertEqual(
            _redact_argv(
                [
                    "--engine_env=OSS_ACCESS_KEY_ID=secret",
                    "--engine_arg=tp_size=8",
                ]
            ),
            ["--engine_env=***", "--engine_arg=tp_size=8"],
        )


if __name__ == "__main__":
    unittest.main()
