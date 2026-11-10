"""Whole-sequence BF16 PLE projections preserve every resumable page state."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer


@contextmanager
def record_conv_launches():
    original = ple_conv_triton._ple_conv
    launches = []

    class Recorder:
        def __getitem__(self, grid):
            launch = original[grid]

            def recorded(*args, **kwargs):
                launches.append(
                    (kwargs["BLOCK"], kwargs["num_warps"], kwargs["ROLL_STATE"])
                )
                return launch(*args, **kwargs)

            return recorded

    with patch.object(ple_conv_triton, "_ple_conv", Recorder()):
        yield launches


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PLEBF16PrefillTest(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("This projection path is qualified only on SM90")
        torch.manual_seed(88319)
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        self.layer = Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3, context_len=2),
            torch.randn(10240, 2560, **kw) * 0.02,
            torch.randn(2560, 2560, **kw) * 0.05,
            torch.randn(10240, 1, 4, **kw) * 0.1,
            *[torch.randn(10240, **kw) * 0.07 for _ in range(3)],
            hc_mult=4,
            hidden_size=2560,
            conv_kernel_size=4,
            norm_eps=1e-6,
        )
        self.assertIsNone(self.layer.key_projection)
        self.assertIsNone(self.layer.value_projection)
        for weight in (
            self.layer.key_proj,
            self.layer.value_proj,
            self.layer.conv_weight,
        ):
            self.assertTrue(torch.count_nonzero(weight).item() > 0)

    def inputs(self, length):
        torch.manual_seed(90100 + length)
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        hidden = torch.randn(1, length, 10240, **kw)
        embeddings = torch.randn(1, length, 2560, **kw)
        initial = torch.randn(1, 9, 10240, **kw)
        history = torch.zeros(1, length + 2, device="cuda", dtype=torch.int64)
        for tensor in (hidden, embeddings, initial):
            self.assertTrue(torch.count_nonzero(tensor).item() > 0)
        return hidden, embeddings, initial, history

    def call(self, inputs):
        hidden, embeddings, state, history = inputs
        return self.layer.prefill_page_snapshots(
            hidden,
            history,
            64,
            state,
            embeddings=embeddings,
            add_residual=True,
        )

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def test_exact_output_all_states_and_two_full_gemms(self):
        original_linear = F.linear
        with torch.inference_mode():
            for length in (128, 2048, 8192, 32768):
                with self.subTest(length=length):
                    inputs = self.inputs(length)
                    calls = []

                    def record_linear(x, weight, *args, **kwargs):
                        calls.append((tuple(x.shape), tuple(weight.shape)))
                        return original_linear(x, weight, *args, **kwargs)

                    with patch.dict(
                        os.environ, {"QWEN4_PLE_BF16_PREFILL": "0"}
                    ), patch.object(F, "linear", record_linear):
                        expected, expected_states = self.call(inputs)
                    self.assertEqual(len(calls), 2 * (length // 64))
                    self.assertTrue(all(shape[0] == (1, 64, 2560) for shape in calls))
                    calls.clear()
                    with (
                        patch.dict(os.environ, {"QWEN4_PLE_BF16_PREFILL": "1"}),
                        patch.object(F, "linear", record_linear),
                        record_conv_launches() as conv_launches,
                    ):
                        actual, actual_states = self.call(inputs)
                    self.assertEqual(
                        conv_launches, [(512 if length >= 8192 else 1024, 4, False)]
                    )
                    self.assertEqual(
                        calls,
                        [
                            ((1, length, 2560), (10240, 2560)),
                            ((1, length, 2560), (2560, 2560)),
                        ],
                    )
                    self.assertTrue(torch.count_nonzero(actual).item() > 0)
                    self.assertTrue((actual != inputs[0]).any().item())
                    self.assert_exact(actual, expected)
                    self.assertEqual(len(actual_states), length // 64)
                    self.assertEqual(len(actual_states), len(expected_states))
                    for actual_state, expected_state in zip(
                        actual_states, expected_states
                    ):
                        self.assert_exact(actual_state, expected_state)

    def test_zero_left_context_preserves_output_and_states(self):
        with torch.inference_mode():
            for length in (128, 8192):
                with self.subTest(length=length):
                    inputs = list(self.inputs(length))
                    inputs[2] = None
                    with patch.dict(os.environ, {"QWEN4_PLE_BF16_PREFILL": "0"}):
                        expected, expected_states = self.call(inputs)
                    with patch.dict(
                        os.environ, {"QWEN4_PLE_BF16_PREFILL": "1"}
                    ), record_conv_launches() as conv_launches:
                        actual, actual_states = self.call(inputs)
                    self.assertEqual(
                        conv_launches, [(512 if length >= 8192 else 1024, 4, False)]
                    )
                    self.assert_exact(actual, expected)
                    self.assertEqual(len(actual_states), len(expected_states))
                    for actual_state, expected_state in zip(actual_states, expected_states):
                        self.assert_exact(actual_state, expected_state)

    def test_default_off_and_unqualified_inputs_keep_original_route(self):
        hidden, embeddings, _, _ = self.inputs(128)
        enabled = self.layer._bf16_prefill_enabled
        with patch.dict(os.environ):
            os.environ.pop("QWEN4_PLE_BF16_PREFILL", None)
            self.assertFalse(enabled(hidden, embeddings, 64))
        with patch.dict(os.environ, {"QWEN4_PLE_BF16_PREFILL": "1"}):
            self.assertTrue(enabled(hidden, embeddings, 64))
            self.assertFalse(enabled(hidden, embeddings, 128))
            self.assertFalse(enabled(hidden[:, :127], embeddings[:, :127], 64))
            for length in (129, 32769):
                self.assertFalse(
                    enabled(
                        hidden[:, :1].expand(1, length, 10240),
                        embeddings[:, :1].expand(1, length, 2560),
                        64,
                    )
                )
            self.assertFalse(enabled(hidden[:, :, :-1], embeddings, 64))
            self.assertFalse(enabled(hidden, embeddings[:, :, :-1], 64))
            self.assertFalse(enabled(hidden.float(), embeddings, 64))
            self.assertFalse(enabled(hidden, embeddings.float(), 64))
            self.assertFalse(
                enabled(hidden.expand(2, -1, -1), embeddings.expand(2, -1, -1), 64)
            )
            with patch.object(self.layer, "key_proj", self.layer.key_proj.float()):
                self.assertFalse(enabled(hidden, embeddings, 64))
            with patch.object(self.layer, "value_projection", SimpleNamespace()):
                self.assertFalse(enabled(hidden, embeddings, 64))
            for capability in ((8, 0), (10, 0), (12, 0)):
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ):
                    self.assertFalse(enabled(hidden, embeddings, 64))


if __name__ == "__main__":
    unittest.main()
