import os
import unittest
from unittest.mock import patch

from rtp_llm.utils.shared_expert_config import shared_expert_weight_shard_enabled

COMMON_ENV = "SHARED_EXPERT_WEIGHT_SHARD"


class SharedExpertConfigTest(unittest.TestCase):
    def test_default_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(shared_expert_weight_shard_enabled())

    def test_common_option(self):
        for value, expected in (("0", False), ("1", True), (" 1 ", True)):
            with self.subTest(value=value), patch.dict(
                os.environ, {COMMON_ENV: value}, clear=True
            ):
                self.assertIs(shared_expert_weight_shard_enabled(), expected)

    def test_invalid_common_option(self):
        for value in ("", " ", "2", "auto"):
            with self.subTest(value=value), patch.dict(
                os.environ, {COMMON_ENV: value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, COMMON_ENV):
                    shared_expert_weight_shard_enabled()


if __name__ == "__main__":
    unittest.main()
