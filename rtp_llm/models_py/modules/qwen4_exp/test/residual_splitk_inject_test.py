import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_splitk_inject import (
    try_splitk_down_inject,
)
from rtp_llm.models_py.modules.qwen4_exp.test.residual_norm_fp8_test import (
    ResidualNormFp8Test,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.residual.gated_residual_splitk_inject"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualSplitKInjectTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 native numerical contract")
        torch.manual_seed(83001)

    def test_all_small_row_counts_and_bf16_boundaries(self):
        for seed in range(3):
            torch.manual_seed(83001 + seed)
            model = ResidualNormFp8Test().module(True)
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

    def test_complete_consumer_and_graph_live_weight(self):
        for rows in (4, 8, 16, 17, 24, 32, 4093, 4094, 4095, 4096):
            model = ResidualNormFp8Test().module(True)
            model.up_projection.role = "DECODE"
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for gain in (0.01, 0.3, 2.0):
                x.normal_(0, gain)
                model.norm_gamma.normal_(0, gain)
                model.inject_projection.compute_weight.normal_(0, gain)
                graph.replay()
                with patch(MODULE + ".try_splitk_down_inject", return_value=None):
                    expected = model(x)
                ResidualNormFp8Test().assert_outputs(actual, expected)

    def test_prefill_mix_tiles_preserve_complete_consumer_and_live_graph(self):
        import triton

        from rtp_llm.models_py.modules.qwen4_exp.residual import (
            gated_residual_mix_triton as kernels,
        )

        def original_mix(logits, normed, branches):
            rows = logits.shape[0]
            out = torch.empty((rows, 2560), device=logits.device, dtype=logits.dtype)
            kernels._mix_reduce_kernel[(rows, triton.cdiv(2560, 128))](
                logits, normed, out, HIDDEN=2560, BRANCHES=4, BLOCK=128, num_warps=4
            )
            return out

        for rows in (4, 16, 4093, 4094, 4095, 4096, 12288):
            model = ResidualNormFp8Test().module(True)
            model.up_projection.role = "PREFILL"
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x)
            for gain in (0.01, 2.0):
                x.normal_(0, gain)
                model.norm_gamma.normal_(0, gain)
                graph.replay()
                with patch.object(kernels, "fused_mix_reduce", original_mix):
                    expected = model(x)
                ResidualNormFp8Test().assert_outputs(actual, expected)
            graph.reset()

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
