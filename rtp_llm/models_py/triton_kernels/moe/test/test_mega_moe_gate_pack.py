"""Equivalence tests for the generic MegaMoE route-and-pack kernel."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


def _make_buf(tokens: int, dim: int, topk: int):
    device = "cuda:0"
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MegaMoeGatePackEquivTest(unittest.TestCase):
    def _assert_matches_separate_pack(self, ref, got):
        self.assertTrue(torch.equal(ref.x.view(torch.uint8), got.x.view(torch.uint8)))
        self.assertTrue(torch.equal(ref.x_sf, got.x_sf))
        self.assertTrue(torch.equal(ref.topk_idx, got.topk_idx))
        self.assertTrue(
            torch.allclose(
                ref.topk_weights,
                got.topk_weights,
                rtol=1.0e-4,
                atol=1.0e-6,
            )
        )

    def _check(self, tokens: int, hash_routing: bool):
        from rtp_llm.models_py.triton_kernels.moe.gate_fused import (
            fused_sqrtsoftplus_gate,
            fused_sqrtsoftplus_hash_gate,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
            fused_pack_mega_moe_inputs_optimized,
        )

        torch.manual_seed(11)
        dim, experts, topk = 512, 256, 6
        x = torch.randn(tokens, dim, device="cuda:0", dtype=torch.bfloat16) * 0.3
        scores = torch.randn(tokens, experts, device="cuda:0", dtype=torch.bfloat16)
        kwargs = {}
        if hash_routing:
            vocab = 320
            input_ids = torch.randint(
                vocab, (tokens,), device="cuda:0", dtype=torch.int64
            )
            tid2eid = torch.stack(
                [torch.randperm(experts, device="cuda:0")[:topk] for _ in range(vocab)]
            ).contiguous()
            weights, indices = fused_sqrtsoftplus_hash_gate(
                scores.contiguous(),
                input_ids,
                tid2eid,
                route_scale=2.5,
            )
            kwargs.update(input_ids=input_ids, tid2eid=tid2eid)
        else:
            bias = torch.randn(experts, device="cuda:0", dtype=torch.float32) * 0.1
            weights, indices = fused_sqrtsoftplus_gate(
                scores.float().contiguous(),
                bias.contiguous(),
                topk=topk,
                route_scale=2.5,
            )
            kwargs["bias"] = bias

        ref = _make_buf(tokens, dim, topk)
        fused_pack_mega_moe_inputs_optimized(
            x,
            weights,
            indices,
            ref.x,
            ref.x_sf,
            ref.topk_idx,
            ref.topk_weights,
        )
        got = _make_buf(tokens, dim, topk)
        fused_pack_mega_moe_gate_inputs(
            x,
            scores.contiguous(),
            got.x,
            got.x_sf,
            got.topk_idx,
            got.topk_weights,
            topk=topk,
            score_func="sqrtsoftplus",
            route_scale=2.5,
            **kwargs,
        )
        torch.cuda.synchronize()
        self._assert_matches_separate_pack(ref, got)

    def test_nonhash_small(self):
        self._check(tokens=17, hash_routing=False)

    def test_nonhash_large(self):
        self._check(tokens=257, hash_routing=False)

    def test_hash_small(self):
        self._check(tokens=19, hash_routing=True)

    def test_hash_large(self):
        self._check(tokens=257, hash_routing=True)


if __name__ == "__main__":
    unittest.main()
