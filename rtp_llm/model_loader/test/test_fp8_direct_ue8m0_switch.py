import os
import unittest
from unittest.mock import patch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    _direct_ue8m0_weight_quant_enabled,
    _should_use_direct_ue8m0_weight_quant,
)


class Fp8DirectUE8M0SwitchTest(unittest.TestCase):
    ENV_NAME = "RTP_LLM_ENABLE_DIRECT_UE8M0_WEIGHT_QUANT"

    def test_switch_parsing(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self.ENV_NAME, None)
            self.assertTrue(_direct_ue8m0_weight_quant_enabled())

        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ, {self.ENV_NAME: value}, clear=False
            ):
                self.assertTrue(_direct_ue8m0_weight_quant_enabled())

        for value in ("0", "false", "NO", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ, {self.ENV_NAME: value}, clear=False
            ):
                self.assertFalse(_direct_ue8m0_weight_quant_enabled())

        with patch.dict(
            os.environ, {self.ENV_NAME: "legacy"}, clear=False
        ), self.assertRaisesRegex(ValueError, self.ENV_NAME):
            _direct_ue8m0_weight_quant_enabled()

    def test_switch_gates_only_applicable_blackwell_dense_weights(self) -> None:
        with patch.dict(os.environ, {self.ENV_NAME: "0"}, clear=False):
            self.assertFalse(_should_use_direct_ue8m0_weight_quant(True, True, True))

        with patch.dict(os.environ, {self.ENV_NAME: "1"}, clear=False):
            self.assertTrue(_should_use_direct_ue8m0_weight_quant(True, True, True))

        # An invalid Blackwell-only knob must not break unrelated devices or
        # MoE/pre-quantized weights where the direct path is inapplicable.
        with patch.dict(os.environ, {self.ENV_NAME: "invalid"}, clear=False):
            self.assertFalse(_should_use_direct_ue8m0_weight_quant(False, True, True))
            self.assertFalse(_should_use_direct_ue8m0_weight_quant(True, False, True))
            self.assertFalse(_should_use_direct_ue8m0_weight_quant(True, True, False))
            with self.assertRaisesRegex(ValueError, self.ENV_NAME):
                _should_use_direct_ue8m0_weight_quant(True, True, True)


if __name__ == "__main__":
    unittest.main()
