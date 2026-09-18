"""Master switch for all Qwen3.5 decode fusions."""

from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    DECODE_FUSION_ENV,
    fusion_phase,
    is_decode_fusion_enabled,
    quantized_linear_for,
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


class FusionPhaseTest(unittest.TestCase):
    def test_decode_switch_does_not_enable_prefill(self):
        for decode in ("0", "1"):
            with patch.dict(os.environ, {DECODE_FUSION_ENV: decode}):
                with fusion_phase(is_prefill=True):
                    self.assertFalse(is_decode_fusion_enabled())
                with fusion_phase(is_prefill=False):
                    self.assertEqual(is_decode_fusion_enabled(), decode == "1")

    def test_decoder_dispatch(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextDecoderLayer

        stub = SimpleNamespace(
            _forward_with_phase=lambda *args: is_decode_fusion_enabled()
        )
        with patch.dict(os.environ, {DECODE_FUSION_ENV: "1"}):
            for prefill, cp, verify in (
                (True, False, False),
                (True, True, False),
                (True, False, True),
                (False, False, False),
            ):
                result = Qwen3NextDecoderLayer.forward(
                    stub,
                    None,
                    None,
                    None,
                    attention_inputs=SimpleNamespace(is_prefill=prefill),
                    attn_meta=SimpleNamespace(
                        is_cp_linear_attn=cp,
                        is_target_verify=verify,
                    ),
                )
                self.assertEqual(result, not prefill)

    def test_nested_and_exception_restore(self):
        with patch.dict(os.environ, {DECODE_FUSION_ENV: "1"}):
            with fusion_phase(is_prefill=True):
                try:
                    with fusion_phase(is_prefill=False):
                        self.assertTrue(is_decode_fusion_enabled())
                        raise RuntimeError("test")
                except RuntimeError:
                    pass
                self.assertFalse(is_decode_fusion_enabled())
            self.assertTrue(is_decode_fusion_enabled())

    def test_phase_does_not_leak_to_another_thread(self):
        with patch.dict(os.environ, {DECODE_FUSION_ENV: "1"}):
            with fusion_phase(is_prefill=True), ThreadPoolExecutor(1) as pool:
                self.assertTrue(pool.submit(is_decode_fusion_enabled).result())
                self.assertFalse(is_decode_fusion_enabled())


class QuantizedLinearCompatibilityTest(unittest.TestCase):
    def test_requires_ue8m0_and_quantized_entry_point(self):
        from types import SimpleNamespace

        compatible = SimpleNamespace(scale_ue8m0=True, forward_quantized=lambda: None)
        self.assertIs(quantized_linear_for(compatible), compatible)
        for linear in (
            SimpleNamespace(forward_quantized=lambda: None),
            SimpleNamespace(scale_ue8m0=False, forward_quantized=lambda: None),
            SimpleNamespace(scale_ue8m0=True),
            SimpleNamespace(scale_ue8m0=True, forward_quantized=None),
            SimpleNamespace(_deepgemm_linear=compatible),
        ):
            self.assertIsNone(quantized_linear_for(linear))


if __name__ == "__main__":
    unittest.main()
