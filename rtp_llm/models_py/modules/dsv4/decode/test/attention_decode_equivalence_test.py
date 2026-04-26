"""Equivalence: Attention.forward (prefill-then-decode-step) vs
Attention.forward_decode for B=1 single-step. Validates that the new
Phase 2 decode path produces the same hidden state as the existing
prefill+decode arms when wired through the metadata builder.

Synthetic random weights — no ckpt load. Runs on CPU when no CUDA.
Requires tilelang to fully validate sparse_attn output match; otherwise
both paths use the same _sparse_attn reference and equivalence is a
trivially-true check on the wiring (still useful — catches RoPE
indexing bugs, KV write-slot mismatches, etc.).
"""

import copy
import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    build_decode_metadata,
)


def _make_attention(
    compress_ratio: int,
    dim: int = 256,
    n_heads: int = 4,
    head_dim: int = 32,
    max_batch_size: int = 4,
    max_seq_len: int = 64,
    window_size: int = 8,
) -> Attention:
    """Tiny synthetic Attention. Avoids the factory/QuantizedLinear path
    (weights=None) so all sub-linears are vanilla nn.Linear under
    QuantizedLinear's BF16 fallback. Build under bf16 default dtype so
    nn.Linear sub-modules match the bf16 input (production loads bf16
    weights from ckpt)."""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = Attention(
            layer_id=0,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=64,
            head_dim=head_dim,
            rope_head_dim=8,
            o_lora_rank=64,
            o_groups=2,
            window_size=window_size,
            compress_ratio=compress_ratio,
            compress_rope_theta=10000.0,
            rope_theta=10000.0,
            rope_factor=1.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=0,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            index_n_heads=4,
            index_head_dim=16,
            index_topk=4,
            norm_eps=1e-6,
            weights=None,
            prefix="",
            tp_size=1,
            tp_rank=0,
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    return attn


def _seed_attn_random(attn: Attention, seed: int = 0) -> None:
    """Replace meta-init Linear weights / norm scales / sink with random
    bf16/fp32 values so forward produces deterministic finite output."""
    g = torch.Generator().manual_seed(seed)
    for name, p in attn.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        # CPU has no normal_kernel for FP8 dtypes — generate fp32 then cast.
        try:
            p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05
        except (NotImplementedError, RuntimeError):
            tmp = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05
            p.data = tmp.to(p.dtype)
    # Norm weight should average ~1 (RMSNorm scale)
    if attn.q_norm.weight is not None:
        attn.q_norm.weight.data.fill_(1.0)
    attn.kv_norm.weight.data.fill_(1.0)
    if attn.compressor is not None:
        attn.compressor.norm.weight.data.fill_(1.0)
    attn.attn_sink.data.zero_()  # zero sink for cleaner equivalence
    # The QuantizedLinear scale buffers / FP8 weight stays meta unless we
    # zero it; but BF16 fallback path doesn't read scale, so OK.


class TestAttentionDecodeEquivalence(unittest.TestCase):
    """Run Attention prefill-then-decode (existing forward) vs
    Attention.forward_decode (Phase 2 path) and verify same output for
    the new token's hidden state."""

    def _run_equivalence(self, compress_ratio: int):
        torch.manual_seed(42)
        device = "cpu"  # numerical equivalence on CPU is strongest
        attn = _make_attention(compress_ratio).to(device).eval()
        _seed_attn_random(attn, seed=1)
        attn.reset_rope_cache(device=device)

        S = 6  # short prefill
        dim = 256
        x_prefill = torch.randn(1, S, dim, dtype=torch.bfloat16, device=device) * 0.1
        x_decode = torch.randn(1, 1, dim, dtype=torch.bfloat16, device=device) * 0.1

        # --- Path A: existing forward, prefill then single decode step ---
        attn_a = copy.deepcopy(attn)
        attn_a.kv_cache.zero_()
        if attn_a.compressor is not None:
            attn_a.compressor.kv_cache = None
            attn_a.compressor.kv_state.zero_()
            attn_a.compressor.score_state.fill_(float("-inf"))
            if attn_a.compressor.kv_cache is None:
                pass  # rebound on first forward
        if attn_a.indexer is not None:
            attn_a.indexer.kv_cache.zero_()
            attn_a.indexer.compressor.kv_state.zero_()
            attn_a.indexer.compressor.score_state.fill_(float("-inf"))

        with torch.inference_mode():
            _ = attn_a.forward(x_prefill, start_pos=0)
            out_a = attn_a.forward(x_decode, start_pos=S)

        # --- Path B: same prefill via existing forward, decode via forward_decode ---
        attn_b = copy.deepcopy(attn)
        attn_b.kv_cache.zero_()
        if attn_b.compressor is not None:
            attn_b.compressor.kv_cache = None
            attn_b.compressor.kv_state.zero_()
            attn_b.compressor.score_state.fill_(float("-inf"))
        if attn_b.indexer is not None:
            attn_b.indexer.kv_cache.zero_()
            attn_b.indexer.compressor.kv_state.zero_()
            attn_b.indexer.compressor.score_state.fill_(float("-inf"))

        with torch.inference_mode():
            _ = attn_b.forward(x_prefill, start_pos=0)

        meta = build_decode_metadata(
            start_pos=torch.tensor([S], dtype=torch.int32),
            q_len=1,
            window_size=attn_b.window_size,
            head_dim=attn_b.head_dim,
            max_seq_len=64,
            compress_ratios=[compress_ratio] if compress_ratio else [0],
            index_topk=attn_b.indexer.index_topk if attn_b.indexer is not None else 4,
            device=torch.device(device),
        )
        with torch.inference_mode():
            out_b = attn_b.forward_decode(x_decode, meta)

        self.assertEqual(tuple(out_a.shape), tuple(out_b.shape))
        # Both paths use the same _sparse_attn reference (no tilelang here).
        # If math is wired correctly, they should agree closely. Allow a
        # small tolerance for any RoPE / norm fp casting drift.
        diff = (out_a.float() - out_b.float()).abs()
        ref_mag = out_a.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        self.assertLess(
            rel_mean, 5e-2, f"compress_ratio={compress_ratio}: rel_mean={rel_mean:.3e}"
        )

    def test_swa_only_layer(self):
        """compress_ratio=0 — pure SWA, no compressor or indexer."""
        self._run_equivalence(compress_ratio=0)

    def test_hca_layer(self):
        """compress_ratio=128 — Compressor (overlap=False), no indexer."""
        # max_seq_len=64 is too small for ratio=128 to ever compress; skip.
        self.skipTest("HCA needs max_seq_len >= 128; covered by smoke")

    def test_csa_layer(self):
        """compress_ratio=4 — Compressor (overlap=True) + Indexer."""
        # Equivalence here requires the indexer path to score against the
        # SAME compressed entries, which requires deterministic compression
        # boundaries. With S=6, prefill compresses 1 entry (positions 0-3),
        # remainder = 2 (positions 4-5). Decode at start_pos=6 (no boundary).
        self._run_equivalence(compress_ratio=4)


class TestAttentionDecodeVectorizedEquivalence(unittest.TestCase):
    """Stage 3B end-to-end check: Attention.forward_decode with the
    metadata's ``is_cuda_graph=True`` flag (which routes through
    vectorized Compressor/Indexer paths) produces equivalent output to
    the eager (``is_cuda_graph=False``) path.

    This is the smoke-equivalent check for Stage 3B — proves the
    vectorized variants integrate end-to-end through the model, not
    just in isolation.
    """

    def _run(self, compress_ratio: int):
        from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
            build_decode_metadata,
        )

        torch.manual_seed(33)
        device = "cpu"
        attn = _make_attention(compress_ratio).to(device).eval()
        _seed_attn_random(attn, seed=2)
        attn.reset_rope_cache(device=device)

        S = 8
        x_prefill = torch.randn(1, S, 256, dtype=torch.bfloat16, device=device) * 0.1
        x_decode = torch.randn(1, 1, 256, dtype=torch.bfloat16, device=device) * 0.1

        # Path A: eager (is_cuda_graph=False)
        attn_a = copy.deepcopy(attn)
        attn_a.kv_cache.zero_()
        if attn_a.compressor is not None:
            attn_a.compressor.kv_cache = None
            attn_a.compressor.kv_state.zero_()
            attn_a.compressor.score_state.fill_(float("-inf"))
        if attn_a.indexer is not None:
            attn_a.indexer.kv_cache.zero_()
            attn_a.indexer.compressor.kv_state.zero_()
            attn_a.indexer.compressor.score_state.fill_(float("-inf"))

        with torch.inference_mode():
            _ = attn_a.forward(x_prefill, start_pos=0)

        meta_a = build_decode_metadata(
            start_pos=torch.tensor([S], dtype=torch.int32),
            q_len=1,
            window_size=attn_a.window_size,
            head_dim=attn_a.head_dim,
            max_seq_len=64,
            compress_ratios=[compress_ratio] if compress_ratio else [0],
            index_topk=attn_a.indexer.index_topk if attn_a.indexer is not None else 4,
            device=torch.device(device),
        )
        # eager path: is_cuda_graph=False (default from build_decode_metadata)
        assert meta_a.is_cuda_graph is False
        with torch.inference_mode():
            out_a = attn_a.forward_decode(x_decode, meta_a)

        # Path B: vectorized (is_cuda_graph=True)
        attn_b = copy.deepcopy(attn)
        attn_b.kv_cache.zero_()
        if attn_b.compressor is not None:
            attn_b.compressor.kv_cache = None
            attn_b.compressor.kv_state.zero_()
            attn_b.compressor.score_state.fill_(float("-inf"))
        if attn_b.indexer is not None:
            attn_b.indexer.kv_cache.zero_()
            attn_b.indexer.compressor.kv_state.zero_()
            attn_b.indexer.compressor.score_state.fill_(float("-inf"))

        with torch.inference_mode():
            _ = attn_b.forward(x_prefill, start_pos=0)

        meta_b = build_decode_metadata(
            start_pos=torch.tensor([S], dtype=torch.int32),
            q_len=1,
            window_size=attn_b.window_size,
            head_dim=attn_b.head_dim,
            max_seq_len=64,
            compress_ratios=[compress_ratio] if compress_ratio else [0],
            index_topk=attn_b.indexer.index_topk if attn_b.indexer is not None else 4,
            device=torch.device(device),
        )
        meta_b.is_cuda_graph = True  # force vectorized dispatch
        with torch.inference_mode():
            out_b = attn_b.forward_decode(x_decode, meta_b)

        self.assertEqual(tuple(out_a.shape), tuple(out_b.shape))
        diff = (out_a.float() - out_b.float()).abs()
        ref_mag = out_a.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        self.assertLess(
            rel_mean,
            5e-2,
            f"compress_ratio={compress_ratio}: vec vs eager "
            f"rel_mean={rel_mean:.3e}",
        )

    def test_swa_only_layer_vec_eq_eager(self):
        self._run(compress_ratio=0)

    def test_csa_layer_vec_eq_eager(self):
        self._run(compress_ratio=4)


if __name__ == "__main__":
    unittest.main()
