import unittest

import torch

from rtp_llm.models_py.triton_kernels.gated_residual.down import (
    separate_down_inject_epilogue,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class InjectProjectionContractTest(unittest.TestCase):

    def test_separate_epilogue_preserves_bf16_div_silu_sigmoid_and_double(self):
        for rows in (1, 4, 16, 32, 128, 4096, 12288):
            torch.manual_seed(7328 + rows)
            d = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16) * 8
            i = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16) * 8
            for injection in (None, i):
                actual, gate = separate_down_inject_epilogue(d, injection)
                torch.testing.assert_close(
                    actual, torch.nn.functional.silu(d / 4), rtol=0, atol=0
                )
                if injection is None:
                    self.assertIsNone(gate)
                else:
                    torch.testing.assert_close(
                        gate, 2 * torch.sigmoid(i / 4), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
