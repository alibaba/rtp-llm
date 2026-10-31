"""Byte parity for TP2/4/8 head groups and fixed-address graph refresh."""

import unittest

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
    exact_head_rms_norm,
)
from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
    is_small_qk_supported,
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

    def test_real_head_groups_exact_bytes_and_v(self):
        torch.manual_seed(251026)
        for heads in (3, 6, 12):
            for dtype in (torch.bfloat16, torch.float32):
                q = torch.randn(256, device="cuda", dtype=dtype) * 0.1
                k = torch.randn_like(q) * 0.1
                module = ZeroCenteredFusedQKRMSNorm(q, k, heads, 1, 256)
                for rows in (1, 3, 4, 7, 8, 16, 17, 28, 31, 32):
                    with self.subTest(heads=heads, gamma=dtype, rows=rows):
                        x = torch.randn(
                            rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                        )
                        self.assertTrue(is_small_qk_supported(x, q, k, heads, 1, 256))
                        expected = self.reference(x, q, k, heads)
                        actual = module(x.clone())
                        self.assertTrue(
                            torch.equal(
                                expected.view(torch.int16), actual.view(torch.int16)
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                actual[:, (heads + 1) * 256 :],
                                x[:, (heads + 1) * 256 :],
                            )
                        )

    def test_graph_refreshes_inputs_and_raw_gains(self):
        for heads in (3, 6, 12):
            for rows in (1, 17, 28):
                x = torch.randn(
                    rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                q = torch.full((256,), 0.003, device="cuda", dtype=torch.bfloat16)
                k = q.clone()
                module = ZeroCenteredFusedQKRMSNorm(q, k, heads, 1, 256)
                for _ in range(3):
                    module(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    module(x)
                for scale in (0.0, 0.001, 1.0, 100.0):
                    source = torch.randn_like(x) * scale
                    q.normal_(0, 0.1)
                    k.normal_(0, 0.1)
                    expected = self.reference(source, q, k, heads)
                    x.copy_(source)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(x.view(torch.int16), expected.view(torch.int16))
                    )

    def test_unsupported_layout_keeps_fallback(self):
        for heads in (3, 6, 12):
            q = torch.randn(256, device="cuda", dtype=torch.bfloat16)
            k = q.clone()
            module = ZeroCenteredFusedQKRMSNorm(q, k, heads, 1, 256)
            for rows in (33, 4093):
                x = torch.randn(
                    rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
                )
                self.assertFalse(is_small_qk_supported(x, q, k, heads, 1, 256))
                self.assertTrue(
                    torch.equal(module(x.clone()), self.reference(x, q, k, heads))
                )

    def test_unsupported_gamma_and_dtype_use_original(self):
        self.q = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        self.k = self.q.clone()
        self.model = ZeroCenteredFusedQKRMSNorm(self.q, self.k, 3, 1, 256)
        self.model.q_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)[::2]
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(7, 1280, device="cuda", dtype=dtype)
            self.assertTrue(
                torch.equal(
                    self.reference(x, self.model.q_weight, self.k, 3),
                    self.model(x.clone()),
                )
            )

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
