"""Equivalence tests for ``SparseAttnV4DecodeOp`` vs the PyTorch reference
``_sparse_attn``. The op delegates to V4's vendored TileLang kernel when
available, falling back to the reference otherwise.
"""

import os
import sys
import unittest
from unittest.mock import patch

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl
from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn
from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
    SparseAttnV4DecodeOp,
)


def _rel_mean(out_a: torch.Tensor, out_b: torch.Tensor) -> float:
    diff = (out_a.float() - out_b.float()).abs()
    ref_mag = out_a.float().abs().mean().item() + 1e-9
    return diff.mean().item() / ref_mag


class TestSparseAttnDecodeOp(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_tl.tilelang_available(), "no tilelang")
    def test_b1_qlen1_full_window(self):
        torch.manual_seed(0)
        device = "cuda:0"
        B, q_len, H, D = 1, 1, 64, 512
        T_max = 256
        K = 128
        sm_scale = D**-0.5

        q = torch.randn(B, q_len, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_max, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_max, (B, q_len, K), device=device, dtype=torch.int32)

        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        out_op = op.forward(q, kv, sink, topk)
        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)

        self.assertEqual(tuple(out_op.shape), (B, q_len, H, D))
        self.assertEqual(out_op.dtype, torch.bfloat16)
        self.assertLess(_rel_mean(out_ref, out_op), 5e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_tl.tilelang_available(), "no tilelang")
    def test_b8_qlen1_realistic(self):
        torch.manual_seed(1)
        device = "cuda:0"
        B, q_len, H, D = 8, 1, 64, 512
        T_max = 512
        K = 512 + 128  # window + compressed concat
        sm_scale = D**-0.5

        q = torch.randn(B, q_len, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_max, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_max, (B, q_len, K), device=device, dtype=torch.int32)

        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        out_op = op.forward(q, kv, sink, topk)
        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)

        self.assertEqual(tuple(out_op.shape), (B, q_len, H, D))
        self.assertEqual(out_op.dtype, torch.bfloat16)
        self.assertLess(_rel_mean(out_ref, out_op), 5e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_tl.tilelang_available(), "no tilelang")
    def test_b4_qlen1_with_invalid_idxs(self):
        torch.manual_seed(2)
        device = "cuda:0"
        B, q_len, H, D = 4, 1, 64, 512
        T_max = 256
        K = 256
        sm_scale = D**-0.5

        q = torch.randn(B, q_len, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_max, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_max, (B, q_len, K), device=device, dtype=torch.int32)
        # Punch ~30% of slots to -1 to exercise the mask path.
        invalid = torch.rand(B, q_len, K, device=device) < 0.30
        topk = torch.where(invalid, torch.full_like(topk, -1), topk)

        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        out_op = op.forward(q, kv, sink, topk)
        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)

        self.assertEqual(tuple(out_op.shape), (B, q_len, H, D))
        self.assertLess(_rel_mean(out_ref, out_op), 5e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_tl.tilelang_available(), "no tilelang")
    def test_h_padding(self):
        """H<16 forces the wrapper's pad-to-16 + trim path."""
        torch.manual_seed(3)
        device = "cuda:0"
        B, q_len, H, D = 1, 1, 4, 512
        T_max = 64
        K = 64
        sm_scale = D**-0.5

        q = torch.randn(B, q_len, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_max, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32)
        topk = torch.randint(0, T_max, (B, q_len, K), device=device, dtype=torch.int32)

        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        out_op = op.forward(q, kv, sink, topk)
        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)

        self.assertEqual(tuple(out_op.shape), (B, q_len, H, D))
        self.assertLess(_rel_mean(out_ref, out_op), 5e-3)


class TestSparseAttnDecodeOpReferenceFallback(unittest.TestCase):
    """When tilelang is unavailable the op routes to ``_sparse_attn``.
    These tests verify the fallback contract (shape + dispatch), so we
    have CPU/CUDA-agnostic coverage even on hosts without tilelang.
    """

    def _run(self, device: str):
        torch.manual_seed(7)
        B, q_len, H, D = 2, 1, 4, 32
        T_max, K = 16, 8
        sm_scale = D**-0.5
        q = torch.randn(B, q_len, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_max, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_max, (B, q_len, K), device=device, dtype=torch.int32)
        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        with patch.object(_tl, "tilelang_available", return_value=False):
            out_op = op.forward(q, kv, sink, topk)
        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)
        self.assertEqual(tuple(out_op.shape), (B, q_len, H, D))
        self.assertEqual(out_op.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out_op, out_ref))

    def test_cpu_reference(self):
        self._run("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    def test_cuda_reference_or_kernel(self):
        self._run("cuda:0")


if __name__ == "__main__":
    unittest.main()
