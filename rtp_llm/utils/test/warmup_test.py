import os
import unittest
from unittest import mock

from rtp_llm.utils.warmup import (
    configure_warmup,
    global_warm_up_enabled,
    model_warm_up_enabled,
)


class WarmupTest(unittest.TestCase):
    def test_defaults_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(global_warm_up_enabled())
            self.assertTrue(model_warm_up_enabled())

    def test_model_switch_is_subordinate_to_global_switch(self):
        with mock.patch.dict(
            os.environ, {"WARM_UP": "1", "MODEL_WARM_UP": "0"}, clear=True
        ):
            self.assertTrue(global_warm_up_enabled())
            self.assertFalse(model_warm_up_enabled())
        with mock.patch.dict(
            os.environ, {"WARM_UP": "0", "MODEL_WARM_UP": "1"}, clear=True
        ):
            self.assertFalse(global_warm_up_enabled())
            self.assertFalse(model_warm_up_enabled())

    def test_configure_normalizes_values(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_warmup(False, True)
            self.assertEqual(os.environ["WARM_UP"], "0")
            self.assertEqual(os.environ["MODEL_WARM_UP"], "1")
            self.assertFalse(model_warm_up_enabled())


if __name__ == "__main__":
    unittest.main()
