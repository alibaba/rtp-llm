"""gpt-oss attention-sink fallback correction UT.

Backends without native sink support (PyFlashinfer paged/ragged/decode on
sm90/sm12x) correct post-hoc: ``y' = y * sigmoid(lse - sink_h)`` — the sink
column is denominator-only. This test pins:

  1. the closed form against the direct sink-softmax reference (attention
     with an extra denominator column e^{sink_h}, no numerator term),
  2. the _LayerSinksMixin registration/lookup gating,
  3. the TP semantics: sinks slice to the rank's head window (sp_0 split).
"""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    _LayerSinksMixin,
)


def _reference_sink_attention(q, k, v, sinks):
    """Direct reference: softmax over [logits | sink] columns, numerator over
    real keys only (the sink contributes e^{sink} to the denominator and
    nothing to the numerator). q [h, d], k/v [n, h, d], sinks [h]."""
    logits = torch.einsum("hd,nhd->hn", q, k)  # [h, n]
    denom = torch.logsumexp(logits, dim=-1)  # log Z
    full = torch.cat([logits, sinks[:, None]], dim=-1)  # [h, n+1]
    probs = torch.softmax(full, dim=-1)[:, :-1]  # sink column dropped
    out = torch.einsum("hn,nhd->hd", probs, v)
    return out, denom


class SinkFallbackCorrectionTest(unittest.TestCase):
    def test_closed_form_matches_reference(self):
        torch.manual_seed(0)
        h, d, n = 8, 64, 96
        q = torch.randn(h, d, dtype=torch.float32)
        k = torch.randn(n, h, d, dtype=torch.float32)
        v = torch.randn(n, h, d, dtype=torch.float32)
        sinks = torch.randn(h, dtype=torch.float32) * 4

        ref, denom = _reference_sink_attention(q, k, v, sinks)
        # plain attention output + lse (what flashinfer returns)
        logits = torch.einsum("hd,nhd->hn", q, k)
        y0 = torch.einsum(
            "hn,nhd->hd", torch.softmax(logits, dim=-1), v
        )
        lse = torch.logsumexp(logits, dim=-1)

        # y' = y0 * sigmoid(lse - sink_h): broadcast lse [h] over d.
        corrected = y0 * torch.sigmoid(lse - sinks)[:, None]
        self.assertTrue(
            torch.allclose(corrected, ref, atol=1e-5, rtol=1e-5),
            f"closed form mismatch: max={(corrected - ref).abs().max().item():.3e}",
        )

    def test_mixin_helper_shape_and_math(self):
        torch.manual_seed(1)
        rows, heads, dim = 5, 4, 32
        out = torch.randn(rows, heads, dim, dtype=torch.bfloat16)
        lse = torch.randn(rows, heads, dtype=torch.float32) * 3
        sinks = torch.randn(heads, dtype=torch.float32) * 4
        mixin = _LayerSinksMixin()
        mixin.register_layer_sinks({2: sinks, 3: None})
        self.assertIsNone(mixin._sinks_for(-1))
        self.assertIsNone(mixin._sinks_for(0))
        self.assertIsNone(mixin._sinks_for(3))
        got = mixin._apply_layer_sinks(out, lse, mixin._sinks_for(2))
        expected = (
            out.float() * torch.sigmoid(lse - sinks)[..., None]
        ).to(torch.bfloat16)
        self.assertEqual(got.shape, out.shape)
        self.assertEqual(got.dtype, out.dtype)
        self.assertTrue(
            torch.allclose(got.float(), expected.float(), atol=1e-2),
        )

    def test_sinks_slice_to_local_heads(self):
        """TP sp_0 semantics: the loaded sink is already the rank's contiguous
        head window; _apply_layer_sinks uses only the first local_head_num
        entries (sinks[:heads] in the helper)."""
        torch.manual_seed(2)
        full_heads, local_heads = 64, 16
        rows, dim = 3, 8
        sinks_full = torch.randn(full_heads, dtype=torch.float32)
        # rank 2 of tp 4: rows [32, 48)
        local = sinks_full[32:48]
        out = torch.randn(rows, local_heads, dim, dtype=torch.float32)
        lse = torch.randn(rows, local_heads, dtype=torch.float32)
        mixin = _LayerSinksMixin()
        mixin.register_layer_sinks({0: local})
        got = mixin._apply_layer_sinks(out, lse, mixin._sinks_for(0))
        expected = out * torch.sigmoid(lse - local)[..., None]
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
