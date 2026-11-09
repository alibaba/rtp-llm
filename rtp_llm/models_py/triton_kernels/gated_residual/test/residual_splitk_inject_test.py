import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.hybrid.test.residual_fixtures import ResidualFp8Fixture
from rtp_llm.models_py.triton_kernels.gated_residual.splitk_inject import (
    try_splitk_down_inject,
)

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.splitk_inject"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualSplitKInjectTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 native numerical contract")
        torch.manual_seed(83001)

    def test_all_small_row_counts_and_bf16_boundaries(self):
        for seed in range(3):
            torch.manual_seed(83001 + seed)
            model = ResidualFp8Fixture().module(True)
            weight = model.inject_projection.compute_weight
            for rows in range(4, 33):
                with self.subTest(seed=seed, rows=rows):
                    x = model._norm(
                        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                    )
                    down = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
                    actual = try_splitk_down_inject(x, weight, down)
                    self.assertIsNotNone(actual)
                    expected = (
                        F.silu(down / 4),
                        2 * torch.sigmoid(F.linear(x, weight) / 4),
                    )
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unsupported_layouts_do_not_dispatch(self):
        weight = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
        for rows in (0, 1, 3, 33, 4092, 4097, 12288):
            x = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            down = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_splitk_down_inject(x, weight, down))
        x = torch.empty(4, 10240, device="cuda", dtype=torch.bfloat16)
        down = torch.empty(4, 320, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(try_splitk_down_inject(x[:, ::2], weight, down))
        self.assertIsNone(try_splitk_down_inject(x, weight.float(), down))
        self.assertIsNone(try_splitk_down_inject(x.requires_grad_(), weight, down))


if __name__ == "__main__":
    unittest.main()
