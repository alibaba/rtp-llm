"""Byte parity for TP2/4/8 head groups and fixed-address graph refresh."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
    try_prefill_qk_rmsnorm_,
)


class GroupedQKNormTest(unittest.TestCase):
    def reference(self, x, q, k, heads, eps=1e-6):
        result = x.clone().reshape(x.shape[0], heads + 2, 256)
        for start, stop, gamma in ((0, heads, q), (heads, heads + 1, k)):
            value = result[:, start:stop].float()
            result[:, start:stop].copy_(
                (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))
                * (1.0 + gamma.float())
            )
        return result.reshape_as(x)

    def test_prefill_graph_refreshes_input_and_raw_gains(self):
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


if __name__ == "__main__":
    unittest.main()
