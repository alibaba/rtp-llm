import unittest

import torch

from rtp_llm.models_py.triton_kernels.dspark_gemma_rope import dspark_gemma_qk_norm_rope


def reference(x, raw_weight, positions, theta=10000000.0):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    y = y * (1.0 + raw_weight.float())
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, device=x.device).float() / 64))
    angles = positions.float()[:, None] * inv[None, :]
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    a, b = y[..., :32], y[..., 32:64]
    return torch.cat((a * c - b * s, b * c + a * s, y[..., 64:]), -1).bfloat16()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkGemmaRoPETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260930)

    def check(self, q, k, wq, wk, positions, theta=10000000.0):
        before_q, before_k = q.clone(), k.clone()
        oq, ok = dspark_gemma_qk_norm_rope(q, k, wq, wk, positions, rope_theta=theta)
        for actual, expected in (
            (oq, reference(q, wq, positions, theta)),
            (ok, reference(k, wk, positions, theta)),
        ):
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.016)
            if actual.numel():
                relative = (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm()
                self.assertLess(float(relative), 0.0001)
        torch.testing.assert_close(q, before_q, rtol=0, atol=0)
        torch.testing.assert_close(k, before_k, rtol=0, atol=0)
        return oq, ok

    def test_shapes_strides_empty_and_theta(self):
        for tokens, hq, hk in (
            (1, 64, 4),
            (7, 64, 4),
            (21, 16, 2),
            (7, 0, 4),
            (0, 64, 4),
            (7, 0, 0),
        ):
            for theta in (10000000.0, 5000000.0):
                with self.subTest(tokens=tokens, hq=hq, hk=hk, theta=theta):
                    # Token slicing preserves both QKV interleave and row gaps.
                    storage = torch.randn(
                        tokens * 2,
                        hq + 2 * hk,
                        128,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    q, k = storage[::2, :hq], storage[::2, hq : hq + hk]
                    wq, wk = [
                        torch.randn(128, device="cuda").bfloat16() for _ in range(2)
                    ]
                    positions = torch.arange(
                        tokens * 2, device="cuda", dtype=torch.int64
                    )[::2]
                    if tokens:
                        positions[-1] = 1048575
                    self.check(q, k, wq, wk, positions, theta)

    def test_raw_weight_and_no_intermediate_bf16(self):
        q = torch.randn(7, 64, 128, device="cuda", dtype=torch.bfloat16)
        k = q[:, :4]
        # Small raw weights are lost when 1+w is prematurely stored in BF16.
        w = torch.full((128,), 0.001, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(17, 24, device="cuda", dtype=torch.int32)
        oq, _ = self.check(q, k, w, w, positions)
        expected = reference(q, w, positions)
        rounded_weight = (1.0 + w).float() - 1.0
        wrong = reference(q, rounded_weight, positions)
        self.assertGreater(int((expected != wrong).sum()), 0)
        xf = q.float()
        normalized = (
            (
                xf
                * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                * (1.0 + w.float())
            )
            .bfloat16()
            .float()
        )
        inv = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
        angle = positions.float()[:, None] * inv
        c, s = angle.cos()[:, None], angle.sin()[:, None]
        a, b = normalized[..., :32], normalized[..., 32:64]
        separated = torch.cat(
            (a * c - b * s, b * c + a * s, normalized[..., 64:]), -1
        ).bfloat16()
        fused_error = (oq.float() - expected.float()).norm()
        separated_error = (separated.float() - expected.float()).norm()
        self.assertLess(float(fused_error), float(separated_error) * 0.1)

    def test_graph_reloads_positions_and_raw_weights(self):
        qkv = torch.randn(7, 72, 128, device="cuda", dtype=torch.bfloat16)
        q, k = qkv[:, :64], qkv[:, 64:68]
        wq, wk = [torch.randn(128, device="cuda").bfloat16() for _ in range(2)]
        positions = torch.arange(7, device="cuda", dtype=torch.int32)
        for _ in range(3):
            dspark_gemma_qk_norm_rope(q, k, wq, wk, positions)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = dspark_gemma_qk_norm_rope(q, k, wq, wk, positions)
        pointers = [t.data_ptr() for t in (q, k, wq, wk, positions, *captured)]
        qkv.normal_()
        wq.mul_(0.1)
        wk.add_(0.001)
        positions.add_(1040000)
        graph.replay()
        eager = self.check(q, k, wq, wk, positions)
        for a, b in zip(captured, eager):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(
            pointers, [t.data_ptr() for t in (q, k, wq, wk, positions, *captured)]
        )


if __name__ == "__main__":
    unittest.main()
