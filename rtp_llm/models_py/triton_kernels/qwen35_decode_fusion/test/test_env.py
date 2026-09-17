"""Master switch for all Qwen3.5 decode fusions."""

from __future__ import annotations

import os
import unittest

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    DECODE_FUSION_ENV,
    is_decode_fusion_enabled,
)


class DecodeFusionEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get(DECODE_FUSION_ENV)
        os.environ.pop(DECODE_FUSION_ENV, None)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop(DECODE_FUSION_ENV, None)
        else:
            os.environ[DECODE_FUSION_ENV] = self._prev

    def test_unset_is_off(self) -> None:
        self.assertFalse(is_decode_fusion_enabled())

    def test_zero_and_false_are_off(self) -> None:
        for raw in ("0", "false", "off", "no"):
            os.environ[DECODE_FUSION_ENV] = raw
            self.assertFalse(is_decode_fusion_enabled(), raw)

    def test_truthy_enables_all(self) -> None:
        for raw in ("1", "true", "TRUE", "on", "yes"):
            os.environ[DECODE_FUSION_ENV] = raw
            self.assertTrue(is_decode_fusion_enabled(), raw)


if __name__ == "__main__":
    unittest.main()
