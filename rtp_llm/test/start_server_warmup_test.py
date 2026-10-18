import argparse
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import SpeculativeType
from rtp_llm.server.server_args.server_args import setup_args


class StartupRealWarmupTest(unittest.TestCase):
    @staticmethod
    def _parse(env, args=()):
        pinned_devices = {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
            if key in os.environ
        }
        with patch.dict(os.environ, {**pinned_devices, **env}, clear=True):
            return setup_args(
                [
                    "--model_type",
                    "deepseek_v4",
                    "--role_type",
                    "PREFILL",
                    "--warm_up",
                    "1",
                    "--model_warm_up",
                    "1",
                    *args,
                ]
            )

    def test_warmup_env_and_cli_reach_startup_policy(self):
        for env, args, expected in (
            ({}, [], True),
            ({"DSV4_STARTUP_REAL_WARMUP": "0"}, [], False),
            ({"DSV4_STARTUP_REAL_WARMUP": "true"}, [], True),
            (
                {"DSV4_STARTUP_REAL_WARMUP": "0"},
                ["--dsv4_startup_real_warmup", "1"],
                True,
            ),
            (
                {"DSV4_STARTUP_REAL_WARMUP": "1"},
                ["--dsv4_startup_real_warmup", "false"],
                False,
            ),
        ):
            with self.subTest(env=env, args=args):
                config = self._parse(env, args)
                self.assertEqual(
                    start_server._should_run_startup_real_warmup(config), expected
                )

    def test_warmup_invalid_values_are_scoped_to_consumers(self):
        env = {"DSV4_STARTUP_REAL_WARMUP": "invalid"}
        with self.assertRaises(argparse.ArgumentTypeError):
            start_server._should_run_startup_real_warmup(self._parse(env))
        for args in (
            ["--role_type", "DECODE"],
            ["--model_type", "qwen"],
            ["--warm_up", "0"],
            ["--model_warm_up", "0"],
        ):
            with self.subTest(args=args):
                self.assertFalse(
                    start_server._should_run_startup_real_warmup(self._parse(env, args))
                )

    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    def test_speculative_reserve_matches_engine(self):
        mtp = self._configs(SpeculativeType.MTP, 3)
        dspark = self._configs(SpeculativeType.DSPARK, 3)

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                7,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048568,
        )


if __name__ == "__main__":
    unittest.main()
