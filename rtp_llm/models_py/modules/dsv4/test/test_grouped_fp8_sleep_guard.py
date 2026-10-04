"""CPU coverage for the SM90 strategy's unsupported sleep configuration."""

import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.model_loader import weight_memory_saver
from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import GroupedFP8Strategy


class GroupedFP8SleepGuardTest(unittest.TestCase):
    def test_sleep_enabled_is_rejected_before_weight_setup(self):
        with mock.patch.object(weight_memory_saver, "is_enabled", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "grouped_fp8.*sleep mode"):
                GroupedFP8Strategy(SimpleNamespace())

    def test_sleep_disabled_preserves_normal_initialization(self):
        cfg = SimpleNamespace()
        with mock.patch.object(weight_memory_saver, "is_enabled", return_value=False):
            strategy = GroupedFP8Strategy(cfg)
        self.assertIs(strategy.cfg, cfg)
        self.assertFalse(strategy._ll_ok)


if __name__ == "__main__":
    unittest.main()
