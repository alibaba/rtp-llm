import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_down_triton import (
    separate_down_inject_epilogue,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SeparateEpilogueTest(unittest.TestCase):
    def test_preserves_bf16_div_silu_sigmoid_and_double(self):
        for rows in (1, 4, 16, 32, 128, 4096, 12288):
            torch.manual_seed(7328 + rows)
            d = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16) * 8
            i = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16) * 8
            for injection in (None, i):
                actual, gate = separate_down_inject_epilogue(d, injection)
                torch.testing.assert_close(actual, F.silu(d / 4), rtol=0, atol=0)
                if injection is None:
                    self.assertIsNone(gate)
                else:
                    torch.testing.assert_close(
                        gate, 2 * torch.sigmoid(i / 4), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
