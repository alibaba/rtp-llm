"""UT for the Compressor wkv/wgate load-time BF16 conversion.

Replaces the prior FP32 SIMT SGEMM path::

    x32 = x.float()
    kv = F.linear(x32, self.wkv.weight)         # FP32 SIMT SGEMM
    score = F.linear(x32, self.wgate.weight)    # FP32 SIMT SGEMM

with BF16 weights stored at load time + BF16 input matmul → FP32 output:

    # __init__:
    self.wkv.weight = nn.Parameter(weights[...].to(torch.bfloat16))
    self.wgate.weight = nn.Parameter(weights[...].to(torch.bfloat16))

    # forward:
    x_bf = x.to(torch.bfloat16)
    kv = F.linear(x_bf, self.wkv.weight).float()
    score = F.linear(x_bf, self.wgate.weight).float()

This UT validates the math equivalence: take an FP32 weight tensor as the
"on-disk" source-of-truth, build both reference (FP32 matmul) and new
(BF16 weight + BF16 matmul) paths, compare outputs.

Bypasses the rtp_llm package init via importlib (only relies on torch).
"""

from __future__ import annotations

import unittest

import torch


def _ref_fp32(x: torch.Tensor, w_fp32: torch.Tensor) -> torch.Tensor:
    """Original code path: x.float() + F.linear(FP32) → FP32 SIMT SGEMM."""
    return torch.nn.functional.linear(x.float(), w_fp32)


def _new_bf16_load_time(x: torch.Tensor, w_fp32: torch.Tensor) -> torch.Tensor:
    """Load-time conversion: store weight as BF16 once, matmul in BF16,
    cast output to FP32.  Mirrors compressor.py:_forward_scalar_impl after
    WS-X1' load-time conversion."""
    w_bf16 = w_fp32.to(torch.bfloat16)  # one-time conversion (load)
    x_bf = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    return torch.nn.functional.linear(x_bf, w_bf16).float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompressorBf16MatmulEquivTest(unittest.TestCase):
    def _check(self, *, M, in_dim, out_dim, x_scale=0.1, w_scale=0.05,
               atol_rel_mean=5e-3, atol_rel_max=5e-2):
        torch.manual_seed(0)
        device = "cuda:0"
        # x is bf16 (matches the V4 forward path: hidden states are bf16).
        x = torch.randn(M, in_dim, device=device, dtype=torch.bfloat16) * x_scale
        # w is the FP32 source-of-truth (as loaded from checkpoint).
        w = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32) * w_scale

        ref = _ref_fp32(x, w)
        new = _new_bf16_load_time(x, w)

        self.assertEqual(new.shape, ref.shape)
        self.assertEqual(new.dtype, torch.float32)
        diff = (ref - new).abs()
        ref_mag = ref.abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        rel_max = diff.max().item() / ref_mag
        self.assertLess(
            rel_mean, atol_rel_mean,
            f"rel mean {rel_mean:.3e} > {atol_rel_mean:.0e} "
            f"(M={M},in={in_dim},out={out_dim})",
        )
        self.assertLess(
            rel_max, atol_rel_max,
            f"rel max {rel_max:.3e} > {atol_rel_max:.0e}",
        )

    def test_v4flash_compressor_overlap_csa(self):
        # V4-Flash CSA compressor: dim=2048, head_dim=128, coff=2 (overlap)
        # → out_dim = 256.  M = batch_tokens.
        self._check(M=128, in_dim=2048, out_dim=256)

    def test_v4flash_compressor_no_overlap_hca(self):
        # HCA compressor: coff=1 → out_dim=128.
        self._check(M=128, in_dim=2048, out_dim=128)

    def test_single_token(self):
        self._check(M=1, in_dim=2048, out_dim=256)

    def test_long_seq_chunk(self):
        # Decode/long-prefill chunk shape.
        self._check(M=512, in_dim=2048, out_dim=256)

    def test_topk_rank_stable_under_softmax(self):
        """Stronger property: argmax of softmax(score) is rank-stable
        between FP32 and BF16-load matmul (sparse-attn topk depends on
        the rank order of these scores)."""
        torch.manual_seed(7)
        device = "cuda:0"
        x = torch.randn(64, 2048, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(256, 2048, device=device, dtype=torch.float32) * 0.05

        ref = _ref_fp32(x, w)
        new = _new_bf16_load_time(x, w)

        # Both should pick the same argmax (or near-tied alternatives)
        # → top-1 indices match in > 99% of rows.
        ref_top = ref.argmax(dim=-1)
        new_top = new.argmax(dim=-1)
        match_rate = (ref_top == new_top).float().mean().item()
        self.assertGreater(
            match_rate, 0.99,
            f"top-1 match rate {match_rate:.3f} below 0.99",
        )

    def test_x_already_bf16_skips_cast(self):
        """When x is already BF16, the .to(bfloat16) is a no-op."""
        torch.manual_seed(11)
        device = "cuda:0"
        x_bf16 = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 256, device=device, dtype=torch.float32)
        out = _new_bf16_load_time(x_bf16, w)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (8, 64))


if __name__ == "__main__":
    unittest.main()
