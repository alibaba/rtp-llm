import os
import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.server.server_args.server_args import setup_args

_PINNED_DEVICES = {
    name: os.environ[name]
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
    if name in os.environ
}


def _jit_env(**values):
    return {**_PINNED_DEVICES, "MODEL_TYPE": "fake_model", **values}


class KernelTuningGroupArgsTest(TestCase):
    """The kernel-tuning switches are paired CLI/env arguments.

    Governance requirement: every new environment variable registers with
    env_name= + bind_to= in a server_args group file, so both the CLI and the
    raw environment resolve into the same parsed KernelTuningConfig instead
    of a bare os.environ read at consumption time.
    """

    def test_defaults_keep_the_deterministic_path_disabled(self):
        with patch.dict(os.environ, _jit_env(), clear=True):
            config = setup_args().kernel_tuning_config
        self.assertFalse(config.rocm_fp8_moe_deterministic_reduce)
        self.assertEqual(config.rocm_fp8_moe_deterministic_max_tokens, 128)
        self.assertEqual(config.aiter_fmoe_config_paths, "")

    def test_cli_args_bind_to_the_config(self):
        config = setup_args(
            [
                "--rocm_fp8_moe_deterministic_reduce",
                "true",
                "--rocm_fp8_moe_deterministic_max_tokens",
                "64",
                "--aiter_fmoe_config_paths",
                "/a.csv:/b.csv",
            ]
        ).kernel_tuning_config
        self.assertTrue(config.rocm_fp8_moe_deterministic_reduce)
        self.assertEqual(config.rocm_fp8_moe_deterministic_max_tokens, 64)
        self.assertEqual(config.aiter_fmoe_config_paths, "/a.csv:/b.csv")

    def test_environment_binds_to_the_config(self):
        with patch.dict(
            os.environ,
            _jit_env(
                RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_REDUCE="1",
                RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_MAX_TOKENS="32",
                AITER_CONFIG_FMOE="/overlay.csv",
            ),
            clear=True,
        ):
            config = setup_args().kernel_tuning_config
        self.assertTrue(config.rocm_fp8_moe_deterministic_reduce)
        self.assertEqual(config.rocm_fp8_moe_deterministic_max_tokens, 32)
        self.assertEqual(config.aiter_fmoe_config_paths, "/overlay.csv")


if __name__ == "__main__":
    unittest.main()
