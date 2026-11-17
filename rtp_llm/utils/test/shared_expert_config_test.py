import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils.shared_expert_config import shared_expert_weight_shard_enabled


class SharedExpertConfigTest(unittest.TestCase):
    def test_default_is_disabled(self):
        self.assertFalse(shared_expert_weight_shard_enabled(SimpleNamespace()))

    def test_common_option(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = SimpleNamespace(shared_expert_weight_shard=enabled)
                self.assertIs(shared_expert_weight_shard_enabled(config), enabled)

    def test_invalid_common_option(self):
        for value in ("0", "1", 0, 1, None):
            with self.subTest(value=value):
                config = SimpleNamespace(shared_expert_weight_shard=value)
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    shared_expert_weight_shard_enabled(config)

    def test_retired_environment_is_ignored(self):
        with patch.dict(
            os.environ,
            {
                "SHARED_EXPERT_WEIGHT_SHARD": "1",
                "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD": "1",
            },
            clear=True,
        ):
            self.assertFalse(shared_expert_weight_shard_enabled(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
