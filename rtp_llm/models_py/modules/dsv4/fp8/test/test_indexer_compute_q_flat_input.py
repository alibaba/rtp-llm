"""UT: ``IndexerFP8._compute_indexer_q`` flat 2D input parity.

Phase-3a part 3 lets the indexer accept either ``[T_total, q_lora]``
(varlen-native) or ``[B=1, S, q_lora]`` (legacy) for the qr argument.
The two paths must produce byte-equal q output (after a leading-dim
flatten), since downstream consumers only use the (M, n_heads,
head_dim) flat view.

The interesting wrinkle is RoPE: ``apply_rotary_emb`` only natively
handles ``[B, S, K]`` / ``[B, S, H, K]``, so the flat 3D ``[T, H, rope]``
path wraps a fake ``S=1`` dim before calling. This UT proves that
fake-S wrap doesn't shift the RoPE angles — the underlying storage
gets the same bytes as the legacy 4D path.

We use a stub with a plain ``nn.Linear`` for ``wq_b`` so the test
doesn't need a real CudaFp8DeepGEMMLinear (which would require FP8
weights + DeepGEMM). The RoPE math is independent of wq_b's quant
scheme — what we're testing is the shape plumbing.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_indexer_compute_q_flat_input \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

DEVICE = "cuda"


class _StubIndexerFP8(nn.Module):
    """Minimal stand-in: only the attrs ``_compute_indexer_q`` reads."""

    def __init__(
        self, q_lora: int, n_heads: int, head_dim: int, rope_head_dim: int
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        # Plain nn.Linear (fp32) — _compute_indexer_q only cares about
        # the (n_heads*head_dim) output shape, not the quant scheme.
        self.wq_b = nn.Linear(q_lora, n_heads * head_dim, bias=False).to(DEVICE)

    _compute_indexer_q = IndexerFP8._compute_indexer_q


class IndexerComputeQFlatInputTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.q_lora = 32
        self.n_heads = 4
        self.head_dim = 16
        self.rope_head_dim = 8
        self.stub = _StubIndexerFP8(
            self.q_lora, self.n_heads, self.head_dim, self.rope_head_dim
        )

    def _make_freqs_cis(self, T: int) -> torch.Tensor:
        """GPT-J style complex64 ``[T, rope_head_dim/2]``. Use random
        non-trivial unit-norm rotations so a missing RoPE call would
        produce a numerically distinguishable result (vs all-ones)."""
        # Build random angles, then exp(i*theta) → unit-norm complex.
        theta = torch.randn(
            T, self.rope_head_dim // 2, dtype=torch.float32, device=DEVICE
        )
        return torch.complex(theta.cos(), theta.sin())

    def test_3d_vs_2d_qr_byte_equal_q(self) -> None:
        """Same qr fed as ``[1, T, q_lora]`` vs ``[T, q_lora]`` →
        byte-equal q output (modulo a leading-dim squeeze)."""
        T = 16
        torch.manual_seed(123)
        qr_flat = torch.randn(T, self.q_lora, dtype=torch.float32, device=DEVICE)
        freqs_cis = self._make_freqs_cis(T)

        q_3d = self.stub._compute_indexer_q(
            qr_flat.unsqueeze(0), freqs_cis, batched_rope=False
        )
        q_2d = self.stub._compute_indexer_q(qr_flat, freqs_cis, batched_rope=False)

        self.assertEqual(q_3d.shape, (1, T, self.n_heads, self.head_dim))
        self.assertEqual(q_2d.shape, (T, self.n_heads, self.head_dim))
        # Byte-equal after squeezing the legacy leading dim.
        self.assertTrue(
            torch.equal(q_3d.squeeze(0), q_2d),
            "flat 2D qr produced a different q tensor than the 3D path",
        )

    def test_rope_is_actually_applied_in_flat_path(self) -> None:
        """Negative control: if RoPE were silently no-op for the flat
        path (e.g. our unsqueeze workaround broke), q would equal the
        pre-RoPE projection. Make sure that's NOT the case."""
        T = 8
        torch.manual_seed(456)
        qr_flat = torch.randn(T, self.q_lora, dtype=torch.float32, device=DEVICE)
        freqs_cis = self._make_freqs_cis(T)
        q_with_rope = self.stub._compute_indexer_q(
            qr_flat, freqs_cis, batched_rope=False
        )
        # No-RoPE reference: same proj, but identity rotations.
        identity_freqs = torch.ones_like(freqs_cis)
        q_no_rope = self.stub._compute_indexer_q(
            qr_flat, identity_freqs, batched_rope=False
        )
        # The rope half MUST differ; the no-rope half MUST match.
        rope_h = self.rope_head_dim
        diff_rope = (q_with_rope[..., -rope_h:] - q_no_rope[..., -rope_h:]).abs().max()
        self.assertGreater(
            float(diff_rope.item()),
            1e-3,
            "RoPE appears to be a no-op in the flat path (expected real rotation)",
        )
        same_nope = (q_with_rope[..., :-rope_h] - q_no_rope[..., :-rope_h]).abs().max()
        self.assertEqual(
            float(same_nope.item()),
            0.0,
            "non-RoPE half should be untouched by rotation",
        )


if __name__ == "__main__":
    unittest.main()
