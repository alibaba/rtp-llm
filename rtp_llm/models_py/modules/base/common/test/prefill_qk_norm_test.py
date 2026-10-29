import unittest

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
    exact_head_rms_norm,
)
from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
    try_prefill_qk_rmsnorm_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefillQkNormTest(unittest.TestCase):
    def reference(self, x, qg, kg, heads):
        out = x.clone().reshape(x.shape[0], heads + 2, 256)
        for value, gamma in ((out[:, :heads], qg), (out[:, heads : heads + 1], kg)):
            f = value.float()
            value.copy_(
                (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6))
                * (1 + gamma.float())
            )
        return out.reshape_as(x)

    def test_prefill_shapes_raw_gain_and_untouched_v(self):
        for rows in (33, 127, 128, 4093, 4095, 4096, 12288):
            for heads in (3, 6, 12):
                torch.manual_seed(rows + heads)
                x = torch.randn(
                    rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                qg = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.07
                kg = torch.randn_like(qg) * 0.11
                expected = self.reference(x, qg, kg, heads)
                actual = ZeroCenteredFusedQKRMSNorm(qg, kg, heads, 1, 256)(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_refreshes_input_and_raw_gains(self):
        x = torch.randn(4096, 3584, device="cuda", dtype=torch.bfloat16)
        qg = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
        kg = torch.zeros_like(qg)
        for _ in range(3):
            try_prefill_qk_rmsnorm_(x, qg, kg, 12, 1, 256, 1e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_prefill_qk_rmsnorm_(x, qg, kg, 12, 1, 256, 1e-6)
        for gain in (0.01, 0.2, 2):
            x.normal_()
            qg.normal_(0, gain)
            kg.normal_(0, gain)
            expected = self.reference(x, qg, kg, 12)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_indexer_prefill_norm_with_strided_projection(self):
        for rows in (129, 4096, 12288, 32768):
            torch.manual_seed(7380 + rows)
            projection = torch.randn(1, rows, 640, device="cuda", dtype=torch.bfloat16)
            x = projection[:, :, :512].view(1, rows, 4, 128)
            for dtype in (torch.bfloat16, torch.float32):
                gamma = (torch.randn(128, device="cuda") * 0.07).to(dtype)
                f = x.float()
                expected = (
                    f
                    * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
                    * (1 + gamma.float())
                ).bfloat16()
                torch.testing.assert_close(
                    exact_head_rms_norm(x, gamma, 1e-6), expected, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
