"""SM90 indexer Norm/RoPE parity and synchronization-free tail validation."""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import validate_mrope_decode_tail
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import try_apply_partial_rope
from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
    try_indexer_rmsnorm,
)


def norm_reference(x, gamma, eps=1e-6):
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * (1.0 + gamma.float())
    ).to(x.dtype)


def rope_reference(x, cos, sin):
    front = x[..., :64]
    rotated = torch.cat((-front[..., 32:], front[..., :32]), dim=-1)
    return torch.cat((front * cos + rotated * sin, x[..., 64:]), dim=-1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90PrefillIndexerTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an actual SM90 GPU")
        torch.manual_seed(20261106)

    def test_complete_mrope_groups_do_not_synchronize(self):
        for seq in (0, 4, 8192, 32768):
            positions = torch.randint(0, 1000, (seq, 3), device="cuda")
            prior = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                validate_mrope_decode_tail(positions, 4)
            finally:
                torch.cuda.set_sync_debug_mode(prior)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            validate_mrope_decode_tail(positions, 4)
            copied = positions + 0
        positions.random_(0, 1000)
        graph.replay()
        torch.testing.assert_close(copied, positions, rtol=0, atol=0)

    def test_unfinished_mrope_tail_validation_is_preserved(self):
        for seq in (1, 2, 3, 5, 6, 7):
            positions = torch.arange(seq, device="cuda")[:, None].expand(-1, 3).clone()
            validate_mrope_decode_tail(positions, 4)
            positions[-1, 1] += 1
            with self.assertRaisesRegex(ValueError, "unfinished MRoPE"):
                validate_mrope_decode_tail(positions, 4)

    def test_norm_rounding_strides_and_prefill_sizes(self):
        for seq in (1, 3, 32, 127, 8192, 32768):
            projected = torch.randn(seq, 640, device="cuda", dtype=torch.bfloat16)
            query = projected[:, :512].view(seq, 4, 128)
            for x in (query, query.unsqueeze(0), projected[:, 512:]):
                for dtype in (torch.bfloat16, torch.float32):
                    gamma = torch.randn(128, device="cuda", dtype=dtype) * 0.1
                    actual = try_indexer_rmsnorm(x, gamma, 1e-6)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(
                        actual, norm_reference(x, gamma), rtol=0, atol=0
                    )

    def test_rope_preserves_products_and_tail(self):
        for seq, heads in ((1, 4), (127, 4), (8192, 1), (8192, 4), (32768, 4)):
            x = torch.randn(1, seq, heads, 128, device="cuda", dtype=torch.bfloat16)
            cos = torch.randn(1, seq, 1, 64, device="cuda", dtype=torch.bfloat16)
            sin = torch.randn_like(cos)
            actual = try_apply_partial_rope(x, cos, sin)
            self.assertIsNotNone(actual)
            torch.testing.assert_close(
                actual, rope_reference(x, cos, sin), rtol=0, atol=0
            )

    def test_norm_rope_graph_refresh_and_rejection(self):
        x = torch.randn(129, 4, 128, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
        cos = torch.randn(129, 1, 64, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn_like(cos)

        def run():
            return try_apply_partial_rope(try_indexer_rmsnorm(x, gamma, 1e-6), cos, sin)

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for _ in range(3):
            x.normal_()
            gamma.normal_(0, 0.1)
            cos.normal_()
            sin.normal_()
            graph.replay()
            torch.testing.assert_close(
                actual,
                rope_reference(norm_reference(x, gamma), cos, sin),
                rtol=0,
                atol=0,
            )
        self.assertIsNone(try_indexer_rmsnorm(x.float(), gamma, 1e-6))
        self.assertIsNone(try_indexer_rmsnorm(x[..., ::2], gamma, 1e-6))
        self.assertIsNone(try_apply_partial_rope(x.transpose(0, 1), cos, sin))
        gamma.requires_grad_(True)
        self.assertIsNone(try_indexer_rmsnorm(x, gamma, 1e-6))


if __name__ == "__main__":
    unittest.main()
