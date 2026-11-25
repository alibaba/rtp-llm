"""Equivalence tests for the generic MegaMoE route-and-pack kernel."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


def _make_buf(tokens: int, dim: int, topk: int, *, sentinel: bool = False):
    device = "cuda:0"
    result = SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )
    if sentinel:
        result.x.view(torch.uint8).fill_(0xFF)
        result.x_sf.fill_(-123)
        result.topk_idx.fill_(-1)
        result.topk_weights.fill_(float("nan"))
    return result


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

    def _check(
        self,
        tokens: int,
        hash_routing: bool,
        *,
        dim: int = 512,
        experts: int = 256,
        topk: int = 6,
    ):
        from rtp_llm.models_py.triton_kernels.moe.gate_fused import (
            fused_sqrtsoftplus_gate,
            fused_sqrtsoftplus_hash_gate,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
            fused_pack_mega_moe_inputs_optimized,
        )

        torch.manual_seed(11)
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
        got = _make_buf(tokens, dim, topk, sentinel=True)
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
        self.assertFalse((got.x_sf == -123).any().item())
        self.assertFalse((got.topk_idx == -1).any().item())
        self.assertTrue(torch.isfinite(got.topk_weights).all().item())

    def _check_empty(self, hash_routing: bool):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        tokens, dim, experts, topk = 0, 128, 64, 1
        x = torch.empty((tokens, dim), device="cuda:0", dtype=torch.bfloat16)
        scores = torch.empty((tokens, experts), device="cuda:0", dtype=torch.bfloat16)
        got = _make_buf(tokens, dim, topk, sentinel=True)
        kwargs = {}
        if hash_routing:
            kwargs.update(
                input_ids=torch.empty(0, device="cuda:0", dtype=torch.int64),
                tid2eid=torch.zeros((8, topk), device="cuda:0", dtype=torch.int64),
            )
        else:
            kwargs["bias"] = torch.zeros(experts, device="cuda:0", dtype=torch.float32)

        fused_pack_mega_moe_gate_inputs(
            x,
            scores,
            got.x,
            got.x_sf,
            got.topk_idx,
            got.topk_weights,
            topk=topk,
            score_func="sqrtsoftplus",
            route_scale=2.5,
            **kwargs,
        )
        self.assertEqual(tuple(got.x.shape), (0, dim))
        self.assertEqual(tuple(got.x_sf.shape), (0, 1))
        self.assertEqual(tuple(got.topk_idx.shape), (0, topk))
        self.assertEqual(tuple(got.topk_weights.shape), (0, topk))

    def _check_nonfinite_fallback(self, hash_routing: bool):
        from rtp_llm.models_py.triton_kernels.moe.gate_fused import (
            fused_sqrtsoftplus_gate,
            fused_sqrtsoftplus_hash_gate,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
            fused_pack_mega_moe_inputs_optimized,
        )

        torch.manual_seed(17)
        tokens, dim, experts, topk = 3, 256, 16, 4
        x = torch.randn(tokens, dim, device="cuda:0", dtype=torch.bfloat16)
        x[0, 0] = float("nan")
        x[1, 1] = float("inf")
        scores = torch.randn(tokens, experts, device="cuda:0", dtype=torch.bfloat16)
        kwargs = {}
        route_scale = 2.5
        if hash_routing:
            input_ids = torch.tensor([0, 1, 2], device="cuda:0", dtype=torch.long)
            tid2eid = torch.stack(
                [torch.randperm(experts, device="cuda:0")[:topk] for _ in range(tokens)]
            ).contiguous()
            scores[0, tid2eid[0, 0]] = float("nan")
            scores[1, tid2eid[1, 1]] = float("inf")
            kwargs.update(input_ids=input_ids, tid2eid=tid2eid)
            safe_scores = torch.nan_to_num(
                scores, nan=0.0, posinf=0.0, neginf=0.0
            ).contiguous()
            ref_weights, ref_indices = fused_sqrtsoftplus_hash_gate(
                safe_scores,
                input_ids,
                tid2eid,
                route_scale=route_scale,
            )
            selected = scores.gather(1, tid2eid.index_select(0, input_ids))
            fallback_rows = ~torch.isfinite(selected).all(dim=1)
        else:
            scores[0, 0] = float("nan")
            scores[1, 1] = float("inf")
            bias = torch.zeros(experts, device="cuda:0", dtype=torch.float32)
            kwargs["bias"] = bias
            safe_scores = torch.nan_to_num(
                scores.float(), nan=0.0, posinf=0.0, neginf=0.0
            ).contiguous()
            ref_weights, ref_indices = fused_sqrtsoftplus_gate(
                safe_scores,
                bias,
                topk=topk,
                route_scale=route_scale,
            )
            fallback_rows = ~torch.isfinite(scores).all(dim=1)

        ref_weights[fallback_rows] = route_scale / topk
        if not hash_routing:
            ref_indices[fallback_rows] = torch.arange(
                topk, device="cuda:0", dtype=torch.long
            )

        ref = _make_buf(tokens, dim, topk)
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        fused_pack_mega_moe_inputs_optimized(
            safe_x,
            ref_weights,
            ref_indices,
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
            route_scale=route_scale,
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

    def test_empty_boundaries_for_hash_and_nonhash(self):
        for hash_routing in (False, True):
            with self.subTest(hash_routing=hash_routing):
                self._check_empty(hash_routing)

    def test_one_token_minimum_dim_and_topk_boundaries(self):
        for hash_routing in (False, True):
            for topk in (1, 32):
                with self.subTest(hash_routing=hash_routing, topk=topk):
                    self._check(
                        tokens=1,
                        hash_routing=hash_routing,
                        dim=128,
                        experts=64,
                        topk=topk,
                    )

    def test_invalid_boundaries_are_rejected(self):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        device = "cuda:0"

        def invoke(*, tokens=1, dim=128, experts=64, topk=1, **kwargs):
            x = torch.zeros((tokens, dim), device=device, dtype=torch.bfloat16)
            scores = torch.zeros((tokens, experts), device=device, dtype=torch.bfloat16)
            out = _make_buf(tokens, dim, max(topk, 0))
            fused_pack_mega_moe_gate_inputs(
                x,
                scores,
                out.x,
                out.x_sf,
                out.topk_idx,
                out.topk_weights,
                topk=topk,
                score_func="sqrtsoftplus",
                route_scale=1.0,
                **kwargs,
            )

        for topk in (0, 33):
            with self.subTest(topk=topk), self.assertRaisesRegex(
                ValueError, "1 <= topk <= 32"
            ):
                invoke(
                    topk=topk,
                    bias=torch.zeros(64, device=device, dtype=torch.float32),
                )
        with self.assertRaisesRegex(ValueError, "D % 128"):
            invoke(
                dim=127,
                bias=torch.zeros(64, device=device, dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "requires both"):
            invoke(input_ids=torch.zeros(1, device=device, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "tid2eid must be"):
            invoke(
                input_ids=torch.zeros(1, device=device, dtype=torch.int64),
                tid2eid=torch.zeros((8, 2), device=device, dtype=torch.int64),
            )

    def test_nonhash_nonfinite_inputs_use_finite_fallback(self):
        self._check_nonfinite_fallback(hash_routing=False)

    def test_hash_nonfinite_inputs_use_finite_fallback(self):
        self._check_nonfinite_fallback(hash_routing=True)

    def test_parallel_and_strided_routing_high_token_counts(self):
        for hash_routing in (False, True):
            for dim in (512, 4096):
                with self.subTest(hash_routing=hash_routing, dim=dim):
                    self._check(
                        tokens=2049,
                        hash_routing=hash_routing,
                        dim=dim,
                        experts=256,
                        topk=6,
                    )

    def test_inline_shared_l1_fp8_scale_staging_matches_standalone(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
            stage_shared_fp8_input_scales,
        )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        device = "cuda:0"
        dim, experts, topk = 4096, 256, 6
        packed_k = dim // 128
        capacity_rows = 8192

        for hash_routing in (False, True):
            with self.subTest(hash_routing=hash_routing):
                torch.manual_seed(23)
                got_shared_sf = torch.empty_strided(
                    (capacity_rows, packed_k),
                    (1, capacity_rows),
                    dtype=torch.int32,
                    device=device,
                )

                for tokens, se_block_m in ((257, 192), (33, 16), (0, 16)):
                    got_shared_sf.fill_(7654321)
                    x = (
                        torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
                        * 0.3
                    )
                    scores = torch.randn(
                        tokens, experts, device=device, dtype=torch.bfloat16
                    )
                    buf = _make_buf(tokens, dim, topk, sentinel=True)
                    extra = {}
                    if hash_routing:
                        vocab = 64
                        extra["input_ids"] = torch.randint(
                            vocab, (tokens,), device=device, dtype=torch.int64
                        )
                        extra["tid2eid"] = torch.stack(
                            [
                                torch.randperm(experts, device=device)[:topk]
                                for _ in range(vocab)
                            ]
                        ).contiguous()
                    else:
                        extra["bias"] = (
                            torch.randn(experts, device=device, dtype=torch.float32)
                            * 0.1
                        )

                    fused_pack_mega_moe_gate_inputs(
                        x,
                        scores,
                        buf.x,
                        buf.x_sf,
                        buf.topk_idx,
                        buf.topk_weights,
                        topk=topk,
                        score_func="sqrtsoftplus",
                        route_scale=2.5,
                        out_shared_l1_sf=got_shared_sf,
                        shared_l1_block_m=se_block_m,
                        **extra,
                    )

                    ref_shared_sf = torch.empty_like(got_shared_sf)
                    ref_shared_sf.fill_(7654321)
                    stage_shared_fp8_input_scales(
                        buf.x_sf, ref_shared_sf, tokens, se_block_m
                    )

                    if tokens == 0:
                        self.assertEqual(torch.count_nonzero(got_shared_sf).item(), 0)
                    else:
                        aligned_blk = ((se_block_m + 127) // 128) * 128
                        active_rows = (
                            (tokens + se_block_m - 1) // se_block_m
                        ) * aligned_blk
                        torch.testing.assert_close(
                            got_shared_sf[:active_rows],
                            ref_shared_sf[:active_rows],
                            rtol=0,
                            atol=0,
                        )
                        self.assertTrue(
                            torch.all(got_shared_sf[active_rows:] == 7654321).item()
                        )

                # Shrinking batch reuse without buffer refill must overwrite [0:active_rows] cleanly.
                got_shared_sf.fill_(7654321)
                for shrink_tokens, shrink_block_m in ((257, 192), (33, 16)):
                    x_s = (
                        torch.randn(
                            shrink_tokens, dim, device=device, dtype=torch.bfloat16
                        )
                        * 0.3
                    )
                    scores_s = torch.randn(
                        shrink_tokens, experts, device=device, dtype=torch.bfloat16
                    )
                    buf_s = _make_buf(shrink_tokens, dim, topk, sentinel=True)
                    extra_s = {}
                    if hash_routing:
                        vocab = 64
                        extra_s["input_ids"] = torch.randint(
                            vocab, (shrink_tokens,), device=device, dtype=torch.int64
                        )
                        extra_s["tid2eid"] = torch.stack(
                            [
                                torch.randperm(experts, device=device)[:topk]
                                for _ in range(vocab)
                            ]
                        ).contiguous()
                    else:
                        extra_s["bias"] = (
                            torch.randn(experts, device=device, dtype=torch.float32)
                            * 0.1
                        )
                    fused_pack_mega_moe_gate_inputs(
                        x_s,
                        scores_s,
                        buf_s.x,
                        buf_s.x_sf,
                        buf_s.topk_idx,
                        buf_s.topk_weights,
                        topk=topk,
                        score_func="sqrtsoftplus",
                        route_scale=2.5,
                        out_shared_l1_sf=got_shared_sf,
                        shared_l1_block_m=shrink_block_m,
                        **extra_s,
                    )
                    if shrink_tokens == 33:
                        ref_shrink = torch.empty_like(got_shared_sf)
                        ref_shrink.fill_(7654321)
                        stage_shared_fp8_input_scales(
                            buf_s.x_sf, ref_shrink, shrink_tokens, shrink_block_m
                        )
                        shrink_aligned = ((shrink_block_m + 127) // 128) * 128
                        shrink_active = (
                            (shrink_tokens + shrink_block_m - 1) // shrink_block_m
                        ) * shrink_aligned
                        torch.testing.assert_close(
                            got_shared_sf[:shrink_active],
                            ref_shrink[:shrink_active],
                            rtol=0,
                            atol=0,
                        )

    def test_inline_shared_l1_scale_validation_errors(self):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        device = "cuda:0"
        tokens, dim, experts, topk = 4, 256, 64, 2
        x = torch.zeros((tokens, dim), device=device, dtype=torch.bfloat16)
        scores = torch.zeros((tokens, experts), device=device, dtype=torch.bfloat16)
        bias = torch.zeros((experts,), device=device, dtype=torch.float32)
        out = _make_buf(tokens, dim, topk)

        with self.assertRaisesRegex(ValueError, "must both be set"):
            fused_pack_mega_moe_gate_inputs(
                x,
                scores,
                out.x,
                out.x_sf,
                out.topk_idx,
                out.topk_weights,
                topk=topk,
                score_func="sqrtsoftplus",
                route_scale=1.0,
                bias=bias,
                out_shared_l1_sf=torch.empty(
                    (128, dim // 128), device=device, dtype=torch.int32
                ),
            )
        with self.assertRaisesRegex(ValueError, "too small"):
            fused_pack_mega_moe_gate_inputs(
                x,
                scores,
                out.x,
                out.x_sf,
                out.topk_idx,
                out.topk_weights,
                topk=topk,
                score_func="sqrtsoftplus",
                route_scale=1.0,
                bias=bias,
                out_shared_l1_sf=torch.empty(
                    (64, dim // 128), device=device, dtype=torch.int32
                ),
                shared_l1_block_m=16,
            )


if __name__ == "__main__":
    unittest.main()
