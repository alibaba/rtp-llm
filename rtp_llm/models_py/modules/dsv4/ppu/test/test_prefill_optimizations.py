"""Native Prefill operator numerical acceptance on PPU."""

import unittest
from types import SimpleNamespace

import torch
from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc import PpuHCUnit


class PrefillOperatorsTest(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_cache_rope_keeps_local_kv_and_nope_exact(self):
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
            fused_rmsnorm_rope,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )
        from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

        raw = torch.linspace(-2, 3, 4 * 512, device="cuda").reshape(1, 4, 512)
        raw = raw.bfloat16()
        weight = torch.linspace(0.5, 1.5, 512, device="cuda").bfloat16()
        phase = torch.linspace(0.1, 2, 4 * 32, device="cuda").reshape(4, 32)
        freqs = torch.polar(torch.ones_like(phase), phase)
        x = raw.float()
        expected = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
        ).bfloat16()
        apply_rotary_emb(expected[..., -64:], freqs)

        cache_rope = raw.new_empty((4, 64))
        baseline = fused_rmsnorm_rope(raw, weight, freqs, 64)
        local = fused_rmsnorm_rope(raw, weight, freqs, 64, cache_rope_out=cache_rope)
        self.assertTrue(torch.equal(local, baseline))
        self.assertTrue(torch.equal(cache_rope, expected[0, :, -64:]))
        self.assertFalse(torch.equal(cache_rope, baseline[0, :, -64:]))

        slots = torch.tensor([-1, 1, 2, 3], device="cuda")
        old_cache = torch.zeros((1, 8, 584), device="cuda", dtype=torch.uint8)
        new_cache = torch.zeros_like(old_cache)
        quantize_and_insert_k_cache(baseline[0], old_cache, slots)
        quantize_and_insert_k_cache(local[0], new_cache, slots, cache_rope=cache_rope)
        old_raw, new_raw = (cache.view(-1) for cache in (old_cache, new_cache))
        old_data, new_data = (raw[: 8 * 576].view(8, 576) for raw in (old_raw, new_raw))
        self.assertTrue(torch.equal(old_data[:, :448], new_data[:, :448]))
        self.assertTrue(torch.equal(old_raw[8 * 576 :], new_raw[8 * 576 :]))
        self.assertEqual(torch.count_nonzero(new_data[0]).item(), 0)
        packed_rope = new_data[1:4, 448:].contiguous().view(torch.bfloat16)
        self.assertTrue(torch.equal(packed_rope, cache_rope[1:]))

    @torch.inference_mode()
    def test_prefill_input_quantization_matches_ue8m0(self):
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.prefill_input_quant import (
            quantize_prefill_input,
        )

        torch.manual_seed(890430)
        for rows in (0, 1, 185):
            with self.subTest(rows=rows):
                x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
                groups = x.float().reshape(rows, 32, 128)
                max_abs = groups.abs().amax(dim=-1).clamp_min(1.0e-4)
                expected_scale = torch.exp2(torch.ceil(torch.log2(max_abs / 448.0)))
                expected_payload = (
                    (groups / expected_scale.unsqueeze(-1))
                    .clamp(-448.0, 448.0)
                    .to(torch.float8_e4m3fn)
                    .view_as(x)
                )
                payload, scale = quantize_prefill_input(x)
                self.assertEqual(scale.stride(), (32, 1))
                torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
                torch.testing.assert_close(
                    payload.view(torch.uint8),
                    expected_payload.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )

    @torch.inference_mode()
    def test_hc_prenorm_against_fp32(self):
        from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_prenorm import (
            tf32_hc_prenorm_gemm,
        )

        torch.manual_seed(890431)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            weight = torch.randn((24, 16384), device="cuda") / 128
            # Include the former slow-conversion regime and partial M tiles.
            for rows in (1, 65, 257, 512, 513, 1024, 2049, 8192):
                with self.subTest(rows=rows):
                    x = torch.randn((rows, 16384), device="cuda", dtype=torch.bfloat16)
                    out = torch.empty((1, rows, 24), device="cuda")
                    squares = torch.empty((1, rows), device="cuda")
                    tf32_hc_prenorm_gemm(x, weight, out, squares)
                    torch.cuda.synchronize()
                    reference = x.float() @ weight.t()
                    square_reference = x.float().square().sum(dim=1)
                    self.assertTrue(bool(out.isfinite().all()))
                    self.assertLess(
                        float((out[0] - reference).norm() / reference.norm()), 0.002
                    )
                    self.assertLess(
                        float(
                            (squares[0] - square_reference).norm()
                            / square_reference.norm()
                        ),
                        1e-5,
                    )
                    previous = out.clone(), squares.clone()
                    tf32_hc_prenorm_gemm(x, weight, out, squares)
                    torch.testing.assert_close(out, previous[0], rtol=0, atol=0)
                    torch.testing.assert_close(squares, previous[1], rtol=0, atol=0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @torch.inference_mode()
    def test_hc_quantization_and_post_match_fallback(self):
        torch.manual_seed(890429)
        args = (
            torch.randn((24, 16384), device="cuda") / 128,
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
        )
        kwargs = dict(
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            tp_size=4,
            tp_rank=0,
            fuse_norm=True,
        )
        optimized = PpuHCUnit(*args, options={}, **kwargs)
        fallback = PpuHCUnit(
            *args,
            options={"DSV4_PPU_PREFILL_HC_QUANT": "0", "DSV4_PPU_PREFILL_HC_POST": "0"},
            **kwargs
        )
        norm = SimpleNamespace(
            weight=torch.linspace(
                0.25, 1.25, 4096, device="cuda", dtype=torch.bfloat16
            ),
            variance_epsilon=1e-6,
        )
        for rows in (1024, 2049):
            with self.subTest(rows=rows):
                x = torch.randn((rows, 4, 4096), device="cuda", dtype=torch.bfloat16)
                y, post, comb = fallback.pre_norm(x, norm, tp_size=4, tp_rank=0)
                actual = optimized.prefill_fast_pre_norm_quantized(
                    x, norm, tp_size=4, tp_rank=0
                )
                for value, reference in zip(actual[:3], (y, post, comb)):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                groups = y.float().reshape(rows, 32, 128)
                scale = groups.abs().amax(dim=-1).clamp_min(1.0e-4) / 448.0
                scale = torch.exp2(torch.ceil(torch.log2(scale)))
                q = (groups / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
                q = q.to(torch.float8_e4m3fn).view_as(y)
                self.assertTrue(
                    torch.equal(actual[3][0].view(torch.uint8), q.view(torch.uint8))
                )
                torch.testing.assert_close(actual[3][1], scale, rtol=0, atol=0)
                self.assertTrue(actual[3][1].is_contiguous())
                self.assertIsNone(
                    fallback.prefill_fast_pre_norm_quantized(
                        x, norm, tp_size=4, tp_rank=0
                    )
                )
                inputs = (
                    y.unsqueeze(0),
                    x.unsqueeze(0),
                    post.unsqueeze(0),
                    comb.unsqueeze(0),
                )
                a = optimized._post_operator(*inputs, hc_mult=4)
                b = fallback._post_operator(*inputs, hc_mult=4)
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    @torch.inference_mode()
    def test_hc_post_repeated_against_exact_reference(self):
        from rtp_llm.models_py.kernels.ppu.hc.post_kernel import mhc_post_fwd

        # Small integers make every FP32 operation and BF16 output exact;
        # each hidden tile differs so stale shared-memory reads are visible.
        torch.manual_seed(890430)
        for rows in (1, 1024, 2049):
            x = torch.randint(-4, 5, (1, rows, 4096), device="cuda").bfloat16()
            residual = torch.randint(
                -4, 5, (1, rows, 4, 4096), device="cuda"
            ).bfloat16()
            post = torch.randint(-2, 3, (1, rows, 4, 1), device="cuda").float()
            comb = torch.randint(-2, 3, (1, rows, 4, 4), device="cuda").float()
            reference = post * x.float().unsqueeze(-2)
            for src in range(4):
                reference += comb[..., src, :].unsqueeze(-1) * residual[
                    ..., src, :
                ].float().unsqueeze(-2)
            reference = reference.bfloat16()
            out = torch.empty_like(residual)
            for tuned in (False, True):
                with self.subTest(rows=rows, tuned=tuned):
                    for repeat in range(256):
                        out.fill_(float("nan"))
                        mhc_post_fwd(
                            x,
                            residual,
                            post,
                            comb,
                            out=out,
                            enable_pdl=False,
                            prefill_tuned=tuned,
                        )
                        torch.testing.assert_close(out, reference, rtol=0, atol=0)
                        if repeat < 32:
                            out.copy_(residual)
                            returned = mhc_post_fwd(
                                x,
                                out,
                                post,
                                comb,
                                out=out,
                                enable_pdl=False,
                                prefill_tuned=tuned,
                            )
                            self.assertIs(returned, out)
                            torch.testing.assert_close(out, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_hc_post_against_ordered_cpu_fma_reference(self):
        from rtp_llm.models_py.kernels.ppu.hc.post_kernel import mhc_post_fwd

        torch.manual_seed(890433)
        for rows in (1, 17):
            x = torch.randn((1, rows, 4096), dtype=torch.bfloat16)
            residual = torch.randn((1, rows, 4, 4096), dtype=torch.bfloat16)
            post = torch.randn((1, rows, 4, 1))
            comb = torch.randn((1, rows, 4, 4))
            # Round the first residual product, then each ordered FP32 FMA.
            first = comb[..., 0, :].unsqueeze(-1) * residual[
                ..., 0, :
            ].float().unsqueeze(-2)
            reference = (
                post.double() * x.double().unsqueeze(-2) + first.double()
            ).float()
            for lane in range(1, 4):
                reference = (
                    comb[..., lane, :].double().unsqueeze(-1)
                    * residual[..., lane, :].double().unsqueeze(-2)
                    + reference.double()
                ).float()
            actual = mhc_post_fwd(
                x.cuda(), residual.cuda(), post.cuda(), comb.cuda(), enable_pdl=False
            )
            torch.testing.assert_close(
                actual.cpu(), reference.bfloat16(), rtol=0, atol=0
            )

    @torch.inference_mode()
    def test_hc_post_sum_preserves_materialized_bf16_sum(self):
        from rtp_llm.models_py.kernels.ppu.hc.post_kernel import mhc_post_fwd
        from rtp_llm.models_py.kernels.ppu.hc.ppu_hc_sum import hc_post_sum

        torch.manual_seed(890432)
        for rows in (1, 8, 32, 512):
            routed = torch.randn((1, rows, 4096), device="cuda", dtype=torch.bfloat16)
            shared = torch.randn_like(routed)
            residual = torch.randn(
                (1, rows, 4, 4096), device="cuda", dtype=torch.bfloat16
            )
            post = torch.randn((1, rows, 4, 1), device="cuda")
            comb = torch.randn((1, rows, 4, 4), device="cuda")
            reference = mhc_post_fwd(
                (routed + shared).bfloat16(), residual, post, comb, enable_pdl=False
            )
            actual = hc_post_sum(routed, shared, residual, post, comb)
            self.assertIs(actual, residual)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
