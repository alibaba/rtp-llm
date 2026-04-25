"""Numerical-equivalence test for V4's native TileLang ``sparse_attn``
(vendored from DeepSeek-V4 ``inference/kernel.py``) vs the Python
reference ``_sparse_attn``.

V4 is MQA + Q-LoRA (NOT MLA). This kernel is the V4-author's own
implementation for exactly this shape; no MLA naming ambiguity.

Skipped when tilelang isn't installed.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl


class TestSparseAttnTileLang(unittest.TestCase):
    @unittest.skipUnless(_tl.tilelang_available(), "tilelang not importable")
    def test_matches_reference_v4_shapes(self):
        """V4-realistic shapes: d=512, H=64, h_kv=1 MQA, per-head
        learned attn_sink."""
        from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn

        torch.manual_seed(0)
        device = "cuda:0"
        B, S, H, D = 1, 10, 64, 512
        T_kv = 256
        K = 128   # SWA window default

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_kv, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_kv, (B, S, K), device=device, dtype=torch.long)
        # Randomly set 10% of entries to -1 for the mask path
        invalid = torch.rand(B, S, K, device=device) > 0.9
        topk = torch.where(invalid, torch.tensor(-1, device=device), topk)
        sm_scale = D ** -0.5

        out_ref = _sparse_attn(q, kv, sink, topk, sm_scale)
        out_tl = _tl.sparse_attn(q, kv, sink, topk, sm_scale)

        self.assertEqual(tuple(out_tl.shape), tuple(out_ref.shape))
        self.assertEqual(out_tl.dtype, out_ref.dtype)

        diff = (out_ref.float() - out_tl.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        self.assertLess(rel_mean, 5e-3,
                        f"rel diff {rel_mean:.3e} exceeds 5e-3")

    @unittest.skipUnless(_tl.tilelang_available(), "tilelang not importable")
    def test_head_pad_when_fewer_than_16(self):
        """When H < 16 the wrapper pads heads to 16 for kernel
        efficiency then trims after. Verify behavior for H=4."""
        from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn

        torch.manual_seed(1)
        device = "cuda:0"
        B, S, H, D = 1, 4, 4, 512
        T_kv = 64
        K = 64

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_kv, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32)
        topk = torch.randint(0, T_kv, (B, S, K), device=device, dtype=torch.long)

        out_ref = _sparse_attn(q, kv, sink, topk, D ** -0.5)
        out_tl = _tl.sparse_attn(q, kv, sink, topk, D ** -0.5)
        self.assertEqual(tuple(out_tl.shape), (B, S, H, D))
        diff = (out_ref.float() - out_tl.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        self.assertLess(diff.mean().item() / ref_mag, 5e-3)


if __name__ == "__main__":
    unittest.main()
