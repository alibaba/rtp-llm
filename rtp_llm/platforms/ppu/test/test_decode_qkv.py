"""Merged projection ownership, provider options, and dynamic norm Graphs."""

import unittest
from unittest.mock import patch

import torch
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.modules.linear.fp8_linear import (
    PpuFp8Linear,
    concatenate_ppu_fp8_linears,
)


class DecodeQKVContractTest(unittest.TestCase):
    def test_published_attention_composition_is_frozen(self):
        options = dict(DECODE_EXECUTION_OPTIONS)
        provider = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_QKV"] = "separate"
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_module_provider.PpuModuleProvider.build_attention"
        ) as build:
            provider.build_attention(object)
            self.assertEqual(build.call_args.kwargs["decode_qkv_mode"], "merged")
            self.assertEqual(build.call_args.kwargs["decode_indexer_mode"], "overlap")
            self.assertIs(
                build.call_args.kwargs["decode_stream_pool"], provider.stream_pool
            )
        for key in DECODE_EXECUTION_OPTIONS:
            with self.subTest(option=key), self.assertRaisesRegex(ValueError, key):
                PpuDecodeProvider({**DECODE_EXECUTION_OPTIONS, key: "unsupported"})

    def test_concat_rejects_nonmatching_types_without_cuda(self):
        for parts in ((), (object(),), (object(), object())):
            with self.assertRaises(TypeError):
                concatenate_ppu_fp8_linears(parts)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires M890P",
)
class DecodeQKVGpuTest(unittest.TestCase):
    @torch.inference_mode()
    def test_bound_compressor_linear_preserves_fp32_and_leading_dimensions(self):
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32
        from rtp_llm.models_py.modules.dsv4.platform_provider import (
            build_dsv4_bf16_fp32_linear,
        )

        def unexpected_fallback(*args):
            self.fail("PPU operation was not bound")

        operation = build_dsv4_bf16_fp32_linear(
            unexpected_fallback,
            platform_provider=PpuDecodeProvider(DECODE_EXECUTION_OPTIONS),
        )
        torch.manual_seed(890440)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) / 32
        for shape in ((0, 256), (1, 256), (2, 3, 256)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 32
            expected = (x.cpu().double() @ weight.cpu().double().t()).float()
            actual = _linear_bf16_bf16_fp32(x, weight, linear_op=operation)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.shape, (*shape[:-1], 128))
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_owned_concat_snapshot_and_quantization_contract(self):
        torch.manual_seed(890437)

        def linear(n, *, quant="v2_column"):
            weight = torch.randn(n, 256, device="cuda").to(torch.float8_e4m3fn)
            scale = torch.ones(n // 128, 2, device="cuda").to(torch.float8_e8m0fnu)
            return PpuFp8Linear(weight, scale, quantization=quant)

        parts = (linear(256), linear(128))
        merged = concatenate_ppu_fp8_linears(parts)
        expected_weight = torch.cat([p.weight.view(torch.uint8) for p in parts])
        self.assertTrue(torch.equal(merged.weight.view(torch.uint8), expected_weight))
        self.assertNotEqual(merged.weight.data_ptr(), parts[0].weight.data_ptr())
        for batch in (0, 1, 8, 128):
            x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(
                merged(x), torch.cat([p(x) for p in parts], -1), rtol=0, atol=0
            )
        for other in (linear(128, quant="auto"),):
            with self.assertRaises(ValueError):
                concatenate_ppu_fp8_linears((parts[0], other))

    @torch.inference_mode()
    def test_dynamic_normalization_graph_and_empty_batch(self):
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
            fused_rmsnorm_rope,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_qkv_norm import normalize_decode_qkv

        torch.manual_seed(890438)
        wq = torch.linspace(0.5, 1.5, 1024, device="cuda", dtype=torch.bfloat16)
        wk = torch.linspace(0.75, 1.25, 512, device="cuda", dtype=torch.bfloat16)
        for batch in (0, 1, 3, 8, 32, 128):
            raw = torch.empty(batch, 1, 1536, device="cuda", dtype=torch.bfloat16)
            freqs = torch.polar(
                torch.ones(batch, 32, device="cuda"),
                torch.randn(batch, 32, device="cuda"),
            )
            raw.normal_()
            if batch == 0:
                qr, kv = normalize_decode_qkv(raw, wq, wk, freqs, 1e-6)
                self.assertEqual(qr.shape, (0, 1, 1024))
                self.assertEqual(kv.shape, (0, 1, 512))
                continue
            for _ in range(3):
                normalize_decode_qkv(raw, wq, wk, freqs, 1e-6)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                qr, kv = normalize_decode_qkv(raw, wq, wk, freqs, 1e-6)
            for magnitude in (0, 0.001, 0.1, 1, 100):
                raw.normal_(std=magnitude)
                freqs.copy_(
                    torch.polar(
                        torch.ones_like(freqs.real), torch.randn_like(freqs.real)
                    )
                )
                qr.fill_(float("nan"))
                kv.fill_(float("nan"))
                graph.replay()
                q64 = raw[..., :1024].double()
                expected_q = (
                    q64
                    * torch.rsqrt(q64.square().mean(-1, keepdim=True) + 1e-6)
                    * wq.double()
                ).bfloat16()
                expected_k = fused_rmsnorm_rope(
                    raw[..., 1024:].contiguous(), wk, freqs, 64, eps=1e-6
                )
                self.assertTrue(
                    bool(torch.isfinite(qr).all()) and bool(torch.isfinite(kv).all())
                )
                torch.testing.assert_close(qr, expected_q, rtol=1 / 128, atol=0)
                torch.testing.assert_close(kv, expected_k, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                normalize_decode_qkv(raw, wq, wk, freqs, float("nan"))


if __name__ == "__main__":
    unittest.main()
