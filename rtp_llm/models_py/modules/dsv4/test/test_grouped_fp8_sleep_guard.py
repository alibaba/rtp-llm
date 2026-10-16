"""CPU coverage for the SM90 strategy's unsupported sleep configuration."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import GroupedFP8Strategy


class GroupedFP8SleepGuardTest(unittest.TestCase):
    def test_sleep_enabled_is_rejected_before_weight_setup(self):
        with mock.patch.dict(os.environ, {"ENABLE_SLEEP_MODE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "grouped_fp8.*sleep mode"):
                GroupedFP8Strategy(SimpleNamespace())

    def test_sleep_disabled_preserves_normal_initialization(self):
        cfg = SimpleNamespace()
        with mock.patch.dict(
            os.environ, {"ENABLE_SLEEP_MODE": "0", "RTP_LLM_WEIGHT_MEMORY_SAVER": "0"}
        ):
            strategy = GroupedFP8Strategy(cfg)
        self.assertIs(strategy.cfg, cfg)
        self.assertFalse(strategy._ll_ok)


if __name__ == "__main__":
    unittest.main()
