"""Kimi's full generic decoder returns the native PyModelOutputs type."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.kimi_k25.test.test_language import (
    TorchCausalMLA,
    complete_checkpoint_cases,
    reference_forward,
)


class KimiNativeForwardTest(unittest.TestCase):
    def test_complete_sharded_vision_language_image_video_and_text(self):
        for method in (NewLoaderLoadMethod.SCRATCH, NewLoaderLoadMethod.FASTSAFETENSORS):
            with self.subTest(method=method.value):
                for model, language, inputs, expected_input in complete_checkpoint_cases(self, method):
                    with torch.inference_mode():
                        actual = model(inputs, fmha_impl=TorchCausalMLA(language)).hidden_states
                        expected = reference_forward(language, expected_input)
                        logits = model.lm_head(actual)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(
                        logits, F.linear(expected, language["lm_head.weight"]), rtol=1e-5, atol=1e-6
                    )
                    self.assertEqual(tuple(logits.shape), (4, 16))


if __name__ == "__main__":
    unittest.main()
