import unittest

import torch
from torch import nn

from rtp_llm.models_py.model_desc.module_base import GptModelBase


class QuantizedLmHeadOutputTest(unittest.TestCase):
    def test_native_output_preserves_fp32_interface(self):
        # A BF16 linear isolates the output contract from the packed-weight
        # kernel. Production NVFP4 head/GEMM coverage is an on-device test.
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.nvfp4_lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.nvfp4_lm_head.weight.copy_(
                torch.arange(128).reshape(16, 8).to(torch.bfloat16) / 128
            )
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for offset in (0.0, 0.25):
                hidden = (torch.arange(24).reshape(3, 8) / 16 + offset).to(dtype)
                expected = nn.functional.linear(
                    hidden.to(torch.bfloat16), model.nvfp4_lm_head.weight
                ).float()
                native = model._compute_quantized_logits_native(hidden)
                public = model.compute_quantized_logits(hidden)
                self.assertEqual(native.dtype, torch.bfloat16)
                self.assertEqual(public.dtype, torch.float32)
                self.assertTrue(torch.equal(native.float(), expected))
                self.assertTrue(torch.equal(public, expected))

    def test_uninitialized_head_fails_explicitly(self):
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.nvfp4_lm_head = None
        for compute in (
            model.compute_quantized_logits,
            model._compute_quantized_logits_native,
        ):
            with self.assertRaisesRegex(ValueError, "No quantized LM head"):
                compute(torch.zeros(1, 8))


if __name__ == "__main__":
    unittest.main()
