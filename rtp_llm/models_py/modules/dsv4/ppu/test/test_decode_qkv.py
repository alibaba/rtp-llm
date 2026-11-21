"""Merged projection ownership, provider options, and dynamic norm Graphs."""

import unittest
from unittest.mock import patch

import torch
from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.dsv4.ppu.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import (
    PpuFp8Linear,
    concatenate_ppu_fp8_linears,
)


class DecodeQKVContractTest(unittest.TestCase):
    def test_published_attention_composition_is_frozen(self):
        options = dict(DECODE_EXECUTION_OPTIONS)
        provider = PpuDecodeProvider(options)
        options["DSV4_PPU_DECODE_QKV"] = "separate"
        with patch(
            "rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider.PpuModuleProvider.build_attention"
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
    is_ppu(),
    "requires M890P",
)
class DecodeQKVGpuTest(unittest.TestCase):
    @torch.inference_mode()
    def test_q_norm_official_fixture(self):
        import hashlib
        import os
        from pathlib import Path

        from rtp_llm.models_py.kernels.ppu.attention.ppu_q_norm_rope import q_norm_rope

        path = os.environ.get("RTP_Q_NORM_FIXTURE")
        if not path:
            self.skipTest("RTP_Q_NORM_FIXTURE required for official all-layer replay")
        data = Path(path).read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(), os.environ["RTP_Q_NORM_FIXTURE_SHA256"]
        )
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual([c["layer"] for c in fixture["cases"]], list(range(43)))
        for case in fixture["cases"]:
            with self.subTest(layer=case["layer"]):
                actual = q_norm_rope(case["q_b"].cuda(), case["freq"].cuda())
                self.assertTrue(torch.equal(actual.cpu(), case["q"]), case["layer"])
        print("Q_NORM_OFFICIAL_43_LAYERS_EXACT", flush=True)

    @torch.inference_mode()
    def test_q_norm_fixture_reaches_prefill_and_decode_entries(self):
        """Verify post-q_b dispatch with a captured operand, not a fake GEMM claim."""
        import hashlib
        import os
        from pathlib import Path
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillQKV
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.attention import (
            AttentionFP8,
        )
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.compute_qkv import (
            decode_compute_q_b,
        )

        path = os.environ.get("RTP_Q_NORM_FIXTURE")
        if not path:
            self.skipTest("RTP_Q_NORM_FIXTURE required for dispatch qualification")
        self.assertEqual(
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            os.environ["RTP_Q_NORM_FIXTURE_SHA256"],
        )
        case = torch.load(path, map_location="cpu", weights_only=True)["cases"][0]
        self.assertEqual(case["layer"], 0)
        raw, expected, freq = (case[k].cuda() for k in ("q_b", "q", "freq"))
        raw = raw.reshape(1, 1, 64, 512)
        expected = expected.reshape_as(raw)
        # Prefill TP4 rank0 owns 16 heads; target Decode EP8 owns all 64.
        checked_heads = []
        for heads in (16, 64):
            with self.subTest(heads=heads):
                operand = raw[:, :, :heads].contiguous()
                oracle = expected[:, :, :heads].contiguous()
                output = torch.empty_like(operand)
                seen = []

                def linear(weight, qr, out=None):
                    seen.append(out is not None)
                    value = operand.flatten(2)
                    if out is not None:
                        out.copy_(value.reshape_as(out))
                        return out.view_as(value)
                    return value.clone()

                attn = SimpleNamespace(
                    wq_b=object(),
                    _lin=linear,
                    n_heads=heads,
                    head_dim=512,
                    rope_head_dim=64,
                    eps=1e-6,
                )
                qr = raw.new_zeros((1, 1, 1024))
                decode = decode_compute_q_b(attn, qr, freq)
                qkv = PrefillQKV(qr=qr[0], q=None, kv_full=None, cache_rope=None)
                common = SimpleNamespace(
                    rd=64,
                    freqs_cis=freq,
                    workspace=SimpleNamespace(prefill_q=lambda rows: output),
                )
                prefill = AttentionFP8._materialize_prefill_q(attn, qkv, common)
                self.assertTrue(torch.equal(decode, oracle))
                self.assertTrue(torch.equal(prefill.q, oracle[0]))
                self.assertEqual(prefill.q.data_ptr(), output.data_ptr())
                self.assertEqual(seen, [False, True])
                # Already-materialized Q must stay passive and retain workspace.
                self.assertIs(
                    AttentionFP8._materialize_prefill_q(attn, prefill, common), prefill
                )
                self.assertEqual(seen, [False, True])
                checked_heads.append(heads)
        self.assertEqual(checked_heads, [16, 64])
        print("Q_NORM_LAYER0_PREFILL_DECODE_ENTRY_EXACT", flush=True)

    @torch.inference_mode()
    def test_q_norm_graph_and_shapes(self):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_q_norm_rope import q_norm_rope
        from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

        torch.manual_seed(7311199)
        for rows, heads in ((0, 64), (1, 16), (6, 64), (8, 64)):
            x = torch.randn(1, rows, heads, 512, device="cuda", dtype=torch.bfloat16)
            f = torch.polar(
                torch.ones(rows, 32, device="cuda"),
                torch.randn(rows, 32, device="cuda"),
            )
            if not rows:
                self.assertEqual(q_norm_rope(x, f).shape, x.shape)
                continue
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    q_norm_rope(x, f)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = q_norm_rope(x, f)
            torch.cuda.current_stream().wait_stream(stream)
            for magnitude in (0, 0.001, 1, 100):
                x.normal_(std=magnitude)
                expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
                apply_rotary_emb(expected[..., -64:], f)
                eager = q_norm_rope(x, f)
                graph.replay()
                self.assertTrue(torch.equal(actual, eager))
                torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-6)

    @torch.inference_mode()
    def test_prefill_q_norm_inplace_workspace(self):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_q_norm_rope import q_norm_rope
        from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

        torch.manual_seed(731185)
        for rows, heads in ((0, 16), (185, 16), (512, 8)):
            raw = torch.randn(1, rows, heads, 512, device="cuda", dtype=torch.bfloat16)
            f = torch.polar(
                torch.ones(rows, 32, device="cuda"),
                torch.randn(rows, 32, device="cuda"),
            )
            expected = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
            if rows:
                apply_rotary_emb(expected[..., -64:], f)
            inplace = raw.clone()
            actual = q_norm_rope(inplace, f, out=inplace)
            self.assertEqual(actual.data_ptr(), inplace.data_ptr())
            self.assertTrue(torch.equal(actual, q_norm_rope(raw, f)))
            torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-6)
            if not rows:
                continue
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    q_norm_rope(inplace, f, out=inplace)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    q_norm_rope(inplace, f, out=inplace)
            torch.cuda.current_stream().wait_stream(stream)
            for magnitude in (0, 0.001, 1, 100):
                raw.normal_(std=magnitude)
                eager = q_norm_rope(raw, f)
                inplace.copy_(raw)
                graph.replay()
                self.assertTrue(torch.equal(inplace, eager))

    @torch.inference_mode()
    def test_bound_compressor_linear_preserves_fp32_and_leading_dimensions(self):
        from functools import partial

        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.compressor import (
            _linear_bf16_bf16_fp32,
        )

        def unexpected_fallback(*args):
            self.fail("PPU operation was not bound")

        operation = partial(
            PpuDecodeProvider(DECODE_EXECUTION_OPTIONS).run_bf16_fp32_linear,
            unexpected_fallback,
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
        # 5/6 are the DSpARK proposal block and its gamma+1 verify span.
        for width in (1, 2, 3, 4, 5, 6):
            with self.subTest(query_len=width):
                self._check_dynamic_normalization(width)

    def _check_dynamic_normalization(self, width):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_qkv_norm import (
            normalize_decode_qkv,
        )
        from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

        torch.manual_seed(890438)
        wq = torch.linspace(0.5, 1.5, 1024, device="cuda", dtype=torch.bfloat16)
        wk = torch.linspace(0.75, 1.25, 512, device="cuda", dtype=torch.bfloat16)
        for batch in (0, 1, 3, 8, 32, 128):
            raw = torch.empty(batch, width, 1536, device="cuda", dtype=torch.bfloat16)
            freqs = torch.polar(
                torch.ones(batch * width, 32, device="cuda"),
                torch.randn(batch * width, 32, device="cuda"),
            )
            raw.normal_()
            if batch == 0:
                qr, kv = normalize_decode_qkv(raw, wq, wk, freqs, 1e-6)
                self.assertEqual(qr.shape, (0, width, 1024))
                self.assertEqual(kv.shape, (0, width, 512))
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
                # Official inference rounds weighted RMSNorm to BF16 before RoPE.
                k = raw[..., 1024:].float()
                expected_k = (
                    k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * wk
                ).bfloat16()
                apply_rotary_emb(
                    expected_k[..., -64:].reshape(1, batch * width, 64), freqs
                )
                self.assertTrue(
                    bool(torch.isfinite(qr).all()) and bool(torch.isfinite(kv).all())
                )
                torch.testing.assert_close(qr, expected_q, rtol=1 / 128, atol=0)
                torch.testing.assert_close(kv, expected_k, rtol=1 / 128, atol=1e-6)
            with self.assertRaises(ValueError):
                normalize_decode_qkv(raw, wq, wk, freqs, float("nan"))


if __name__ == "__main__":
    unittest.main()
