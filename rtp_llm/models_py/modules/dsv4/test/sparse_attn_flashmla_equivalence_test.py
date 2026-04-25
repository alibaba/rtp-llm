"""Numerical-equivalence test for ``_sparse_attn_flashmla`` vs the
Python reference ``_sparse_attn``.

V4 uses MQA + Q-LoRA, *not* MLA. This test documents that
``flash_mla.flash_mla_sparse_fwd`` — despite its name — is a generic
sparse attention kernel that produces V4's expected output when
``d_qk == d_v`` and ``h_kv == 1`` (MQA degenerate case).

Skipped when flash_mla is not importable.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_HAS_FLASHMLA = False
try:
    import flash_mla  # noqa: F401

    _HAS_FLASHMLA = True
except ImportError:
    pass


class TestSparseAttnFlashMLAEquivalence(unittest.TestCase):
    @unittest.skipUnless(_HAS_FLASHMLA, "flash_mla not importable")
    def test_matches_reference_on_v4_shapes(self):
        """
        V4-realistic shapes: d_qk == d_v == 512, h_kv == 1 (MQA),
        per-head fp32 attn_sink, variable topk with -1 masking.
        """
        from rtp_llm.models_py.modules.dsv4.attention import (
            _sparse_attn,
            _sparse_attn_flashmla,
        )

        torch.manual_seed(0)
        device = "cuda:0"
        B, S, H, D = 1, 10, 64, 512
        T_kv = 256

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_kv, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5

        # Try topk == 128 (SWA layer default). Include some -1 padding.
        K = 128
        topk_idxs = torch.randint(0, T_kv, (B, S, K), device=device, dtype=torch.long)
        # Randomly set ~10% of entries to -1
        invalid_mask = torch.rand(B, S, K, device=device) > 0.9
        topk_idxs = torch.where(invalid_mask, torch.tensor(-1, device=device), topk_idxs)
        sm_scale = D ** -0.5

        out_ref = _sparse_attn(q, kv, sink, topk_idxs, sm_scale)
        out_mla = _sparse_attn_flashmla(q, kv, sink, topk_idxs, sm_scale)

        # Shape & dtype preserved
        self.assertEqual(tuple(out_mla.shape), tuple(out_ref.shape))
        self.assertEqual(out_mla.dtype, out_ref.dtype)

        # Numerical match at BF16 precision. The kernel accumulates in a
        # different order than the Python reference, so a few ULPs of
        # divergence are expected.
        diff = (out_ref.float() - out_mla.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        # Empirically ~1e-3; bound at 5e-3 to allow for GEMM accumulation
        # variations across CUDA versions.
        self.assertLess(rel_mean, 5e-3,
                        f"rel diff {rel_mean:.3e} exceeds 5e-3")

    @unittest.skipUnless(_HAS_FLASHMLA, "flash_mla not importable")
    def test_non_multiple_of_64_topk_is_padded(self):
        """FlashMLA requires ``topk % 64 == 0``. V4 layers with HCA can
        produce non-multiple topks (e.g. 132 = 128 window + 4 compressed).
        The wrapper must pad with -1."""
        from rtp_llm.models_py.modules.dsv4.attention import (
            _sparse_attn,
            _sparse_attn_flashmla,
        )

        torch.manual_seed(1)
        device = "cuda:0"
        B, S, H, D = 1, 4, 64, 512
        T_kv = 512
        K = 132   # deliberately not a multiple of 64

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(B, T_kv, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.zeros(H, device=device, dtype=torch.float32)
        topk_idxs = torch.randint(0, T_kv, (B, S, K), device=device, dtype=torch.long)
        sm_scale = D ** -0.5

        out_ref = _sparse_attn(q, kv, sink, topk_idxs, sm_scale)
        out_mla = _sparse_attn_flashmla(q, kv, sink, topk_idxs, sm_scale)
        diff = (out_ref.float() - out_mla.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        self.assertLess(diff.mean().item() / ref_mag, 5e-3)


if __name__ == "__main__":
    unittest.main()
