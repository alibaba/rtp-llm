"""UT for MegaMoE fused router gate + input pack kernels.

The reference path is:

  non-hash: scores_bf16.float() -> fused_sqrtsoftplus_gate -> optimized packer
  hash: eager hash gather/normalize -> optimized packer

The candidate writes the same final MegaMoE buffer fields directly.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _load_module(name: str, rel_path: str):
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, rel_path))
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_GATE = _load_module("_v4_gate_fused", "../_gate_fused_triton.py")
_PACK = _load_module("_mega_input_pack_triton", "../moe/_mega_input_pack_triton.py")
_GATE_PACK = _load_module("_mega_gate_pack_triton", "../moe/_mega_gate_pack_triton.py")


def _make_buf(tokens: int, dim: int, topk: int, device: str):
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


def _pack_reference(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
    tokens, dim = x.shape
    topk = weights.shape[1]
    buf = _make_buf(tokens, dim, topk, "cuda")
    _PACK.fused_pack_mega_moe_inputs_optimized(
        x, weights, indices, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights
    )
    return buf


def _assert_buf_equal(test: unittest.TestCase, ref, got) -> None:
    test.assertTrue(torch.equal(ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu()))
    test.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
    test.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
    diff = (ref.topk_weights - got.topk_weights).abs()
    denom = ref.topk_weights.abs().mean().item() + 1.0e-9
    rel = diff.max().item() / denom
    test.assertLess(rel, 1.0e-4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MegaMoeGatePackTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        torch.cuda.set_device(0)

    def _case_nonhash(self, tokens: int, dim: int = 512, experts: int = 256, topk: int = 6):
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.3
        scores_bf16 = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1
        weights, indices = _GATE.fused_sqrtsoftplus_gate(
            scores_bf16.float().contiguous(),
            bias.contiguous(),
            topk=topk,
            route_scale=2.5,
            norm_eps=1.0e-12,
        )
        ref = _pack_reference(x, weights, indices)
        got = _make_buf(tokens, dim, topk, "cuda")
        _GATE_PACK.fused_mega_moe_gate_pack_nonhash(
            x,
            scores_bf16.contiguous(),
            bias.contiguous(),
            got.x,
            got.x_sf,
            got.topk_idx,
            got.topk_weights,
            route_scale=2.5,
            norm_eps=1.0e-12,
        )
        torch.cuda.synchronize()
        _assert_buf_equal(self, ref, got)

    def _case_hash(
        self,
        tokens: int,
        dim: int = 512,
        experts: int = 256,
        topk: int = 6,
        vocab: int = 320,
    ):
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16) * 0.3
        scores_bf16 = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        input_ids = torch.randint(0, vocab, (tokens,), device="cuda", dtype=torch.int64)
        tid2eid = torch.empty((vocab, topk), device="cuda", dtype=torch.int64)
        for token in range(vocab):
            tid2eid[token] = torch.randperm(experts, device="cuda")[:topk]

        scores = F.softplus(scores_bf16.float()).sqrt()
        indices = tid2eid[input_ids].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-12) * 2.5
        ref = _pack_reference(x, weights, indices)
        got = _make_buf(tokens, dim, topk, "cuda")
        _GATE_PACK.fused_mega_moe_gate_pack_hash(
            x,
            scores_bf16.contiguous(),
            input_ids.contiguous(),
            tid2eid.contiguous(),
            got.x,
            got.x_sf,
            got.topk_idx,
            got.topk_weights,
            route_scale=2.5,
            norm_eps=1.0e-12,
        )
        torch.cuda.synchronize()
        _assert_buf_equal(self, ref, got)

    def test_nonhash_small(self):
        self._case_nonhash(tokens=17)

    def test_nonhash_large(self):
        self._case_nonhash(tokens=257)

    def test_hash_small(self):
        self._case_hash(tokens=19)

    def test_hash_large(self):
        self._case_hash(tokens=257)


if __name__ == "__main__":
    unittest.main()
