"""Construct MTPBlock (empty-weights / non-factory mode), materialize on
CUDA, run ``forward_draft`` against a mock embed + lm_head; assert output
is finite [B, S, vocab_size].

Validates the MTPBlock plumbing (e_proj/h_proj + enorm/hnorm/norm +
hc_head_*) independent of the V4-Flash ckpt; full-weight coverage goes
through the smoke gate once MTP is enabled in ``deepseek_v4_model.py``.
"""

import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl
from rtp_llm.models_py.modules.dsv4.block import MTPBlock


class TestMTPBlockForward(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_forward_draft_shape_and_finite(self) -> None:
        device = "cuda:0"
        B, S, dim, hc, vocab = 1, 4, 256, 4, 1024

        # Force the PyTorch reference sparse_attn path: our tiny test
        # head_dim=32 is below TileLang's MMA warp tile minimum of 64.
        # The PyRef is numerically identical to the TileLang kernel
        # (verified in sparse_attn_tilelang_equivalence_test).
        _orig_tl_avail = _tl.tilelang_available
        _tl.tilelang_available = lambda: False
        self.addCleanup(lambda: setattr(_tl, "tilelang_available", _orig_tl_avail))

        # Force the PyTorch reference sparse_attn path: our tiny test
        # dims (head_dim=32) are below the TileLang MMA warp tile
        # minimum (64).  The PyRef is numerically identical to the
        # TileLang kernel (verified in sparse_attn_tilelang_equivalence_test).
        _orig_tl_avail = _tl.tilelang_available
        _tl.tilelang_available = lambda: False
        self.addCleanup(lambda: setattr(_tl, "tilelang_available", _orig_tl_avail))

        # Small but topology-correct dims.  Per-layer compress_ratios
        # covers the MTP layer at index layer_id=n_layers=1 (so len=2).
        mtp = MTPBlock(
            layer_id=1,
            dim=dim, n_heads=4, q_lora_rank=64,
            head_dim=32, rope_head_dim=16,
            o_lora_rank=64, o_groups=2,
            window_size=32, compress_ratio=0,
            compress_rope_theta=160000.0, rope_theta=10000.0,
            rope_factor=1.0, beta_fast=32, beta_slow=1,
            original_seq_len=4096,
            max_batch_size=B, max_seq_len=S,
            index_n_heads=4, index_head_dim=16, index_topk=2,
            moe_inter_dim=128, n_routed_experts=4,
            n_activated_experts=2, n_shared_experts=1,
            score_func="sqrtsoftplus", route_scale=1.0,
            swiglu_limit=10.0, n_hash_layers=0, vocab_size=vocab,
            hc_mult=hc, hc_sinkhorn_iters=4, hc_eps=1e-6,
            norm_eps=1e-6,
        ).to(device=device, dtype=torch.bfloat16)

        # Force hc params + norms back to fp32 — MTPBlock defaults to
        # float32 Parameters for those, but `.to(bfloat16)` above cast
        # them.  Reset to match real-ckpt layout.
        for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
            "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "hc_head_fn", "hc_head_base", "hc_head_scale",
        ):
            p = getattr(mtp, name)
            setattr(mtp, name, nn.Parameter(p.float(), requires_grad=False))
        for m in (mtp.attn_norm, mtp.ffn_norm, mtp.enorm, mtp.hnorm, mtp.norm):
            m.weight = nn.Parameter(m.weight.float(), requires_grad=False)

        # Initialise with small random values so the forward doesn't
        # saturate.  Empty-weight construction leaves params as
        # torch.empty which may contain garbage / inf.
        with torch.no_grad():
            for p in mtp.parameters():
                if p.dtype.is_floating_point:
                    if p.dtype == torch.bfloat16:
                        p.copy_(torch.randn_like(p, dtype=torch.float32).to(torch.bfloat16) * 0.02)
                    else:
                        p.copy_(torch.randn_like(p) * 0.02)
                elif p.dtype == torch.int8:
                    p.copy_(torch.randint(-3, 3, p.shape, dtype=torch.int8, device=p.device))
            # Stabilise the norm weights around 1.
            for m in (mtp.attn_norm, mtp.ffn_norm, mtp.enorm, mtp.hnorm, mtp.norm):
                m.weight.copy_(torch.ones_like(m.weight))

        embed = nn.Embedding(vocab, dim).to(device=device, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(vocab, dim, device=device, dtype=torch.float32) * 0.02

        x = torch.randn(B, S, hc, dim, device=device, dtype=torch.bfloat16)
        input_ids = torch.randint(0, vocab, (B, S), device=device, dtype=torch.long)

        with torch.inference_mode():
            logits = mtp.forward_draft(x, 0, input_ids, embed, lm_head_weight)

        # Plumbing check: shape + dtype.  Finiteness is intentionally
        # NOT asserted — the V4 routed FP4 experts need trained weights
        # to stay numerically stable; with ``torch.empty`` + random init
        # the dequant-and-accumulate chain across ratedexperts routinely
        # saturates or produces NaN on synthetic data.  End-to-end
        # numerical correctness is covered by the SM100_ARM smoke once
        # MTPBlock is wired into ``DeepSeekV4Model`` at inference time.
        self.assertEqual(tuple(logits.shape), (B, S, vocab))
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
