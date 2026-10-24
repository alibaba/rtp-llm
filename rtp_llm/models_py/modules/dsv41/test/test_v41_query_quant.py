"""Check decode Q fusion against the actual norm and CUDA quant producers."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import strided_q_rmsnorm
from rtp_llm.models_py.modules.dsv41._v41_query_quant import (
    query_norm_quant,
    try_project_quantized_qkv,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


@triton.jit
def _delayed_qa(source, output, size: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.inline_asm_elementwise(
        "nanosleep.u32 100000; // $0", "=r", [], tl.int32, is_pure=False, pack=1
    )
    tl.store(
        output + offsets, tl.load(source + offsets, offsets < size, 0), offsets < size
    )


def reference(x, weight):
    normalized = strided_q_rmsnorm(x, weight, 1e-6)
    q, sf = sgl_per_token_group_quant_fp8(
        normalized.reshape(-1, x.shape[-1]),
        32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return normalized, (q.view(x.shape), sf)


class QueryQuantGateTest(unittest.TestCase):
    def test_shared_fp8_input_and_pdl_are_independent(self):
        x = torch.empty(1, 2, 512, dtype=torch.bfloat16)
        projected = torch.randn(1, 2, 640, dtype=torch.bfloat16)
        norm = torch.empty(1, 2, 128, dtype=torch.bfloat16)
        quant = (object(), object())
        for shared in (None, (object(), object())):
            for pdl in (False, True):
                linear = Mock(return_value=projected, N=640)
                linear.forward_quantized.return_value = projected.flatten(0, 1)
                attn = SimpleNamespace(
                    wq_b=Mock(spec=V41MXFP8Linear),
                    wq_a_wkv=linear,
                    q_norm=torch.ones(128, dtype=torch.bfloat16),
                    q_lora_rank=128,
                    eps=1e-6,
                )
                with self.subTest(shared=shared is not None, pdl=pdl), patch.dict(
                    os.environ,
                    {"DSV41_FUSED_QUERY_QUANT": "1", "DSV4_FP8_QUANT_KERNEL": "auto"},
                ), patch(
                    "rtp_llm.models_py.modules.dsv41._v41_query_quant.is_supported",
                    return_value=True,
                ), patch(
                    "rtp_llm.models_py.modules.dsv41._v41_query_quant.query_norm_quant",
                    return_value=(norm, quant),
                ) as normalize:
                    result = try_project_quantized_qkv(attn, x, shared, pdl=pdl)
                if shared is None:
                    linear.assert_called_once_with(x)
                    linear.forward_quantized.assert_not_called()
                else:
                    linear.assert_not_called()
                    linear.forward_quantized.assert_called_once_with(*shared)
                self.assertEqual(normalize.call_args.kwargs, {"pdl": pdl})
                torch.testing.assert_close(
                    normalize.call_args.args[0], projected[..., :128]
                )
                self.assertIs(result[0], norm)
                self.assertIs(result[2], quant)
                torch.testing.assert_close(result[1], projected[..., 128:])

    def test_disabled_does_not_access_attention(self):
        with patch.dict(os.environ, {"DSV41_FUSED_QUERY_QUANT": "0"}):
            self.assertIsNone(try_project_quantized_qkv(None, None))

    def test_non_native_consumer_keeps_original_path(self):
        attn = SimpleNamespace(wq_b=torch.nn.Identity())
        self.assertIsNone(try_project_quantized_qkv(attn, None))

    def test_quant_backend_selection_falls_back_before_projection(self):
        attn = SimpleNamespace(
            wq_b=Mock(spec=V41MXFP8Linear),
            wq_a_wkv=Mock(),
            q_norm=None,
            q_lora_rank=1280,
        )
        with patch(
            "rtp_llm.models_py.modules.dsv41._v41_query_quant.is_supported",
            return_value=True,
        ):
            for setting in ("v2", " V2 ", "invalid"):
                with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": setting}):
                    self.assertIsNone(
                        try_project_quantized_qkv(attn, torch.empty(4, 128))
                    )
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "auto"}):
                self.assertIsNone(
                    try_project_quantized_qkv(attn, torch.empty(4096, 128))
                )
        attn.wq_a_wkv.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class QueryQuantCudaTest(unittest.TestCase):
    @torch.inference_mode()
    def test_model_initialization_configures_real_pdl_policy(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("PDL requires SM90 or newer")
        import deep_gemm

        from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
            DeepSeekV41DSparkModel,
        )
        from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
        from rtp_llm.models_py.modules.dsv41 import dsv41_kernel_jit_warmup as warmup
        from rtp_llm.models_py.modules.dsv41.fp8.decode.full_overlap import (
            _qb_pdl_enabled,
        )

        self.addCleanup(deep_gemm.set_pdl, deep_gemm.get_pdl())
        x = torch.randn(1, 64, 1280, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        expected = reference(x, weight)
        for model_class in (DeepSeekV41Model, DeepSeekV41DSparkModel):
            for decode in (False, True):
                for setting in (None, "0"):
                    with self.subTest(
                        model=model_class.__name__, decode=decode, setting=setting
                    ), patch.dict(os.environ):
                        os.environ.pop("DSV41_PDL", None)
                        os.environ.pop("DSV41_FULL_DECODE_QB_PDL", None)
                        if setting is not None:
                            os.environ["DSV41_PDL"] = setting
                        model = model_class.__new__(model_class)
                        commit_only = (
                            model_class is DeepSeekV41DSparkModel and not decode
                        )
                        model._v4_args = SimpleNamespace(
                            v41_config={}, commit_only=commit_only
                        )
                        model._materialized = True
                        model._is_decode_role = decode
                        model.v4 = SimpleNamespace(embed=SimpleNamespace(weight=x))
                        model.main_proj = SimpleNamespace(weight=x)

                        def check_policy(*args, **kwargs):
                            self.assertEqual(deep_gemm.get_pdl(), setting != "0")
                            self.assertEqual(
                                _qb_pdl_enabled(SimpleNamespace(merged_qb=object())),
                                setting != "0",
                            )
                            # No explicit pdl argument: use the model's runtime policy.
                            self.assert_same(
                                query_norm_quant(x, weight, 1e-6), expected
                            )

                        with patch.object(
                            warmup,
                            (
                                "warmup_v41_decode_jit"
                                if decode
                                else "warmup_v41_prefill_jit"
                            ),
                            side_effect=check_policy,
                        ) as warm:
                            self.assertTrue(
                                model.initialize(SimpleNamespace(kv_cache=None))
                            )
                        warm.assert_called_once()

    @torch.inference_mode()
    def test_pdl_qa_norm_qb_kv_graph_dependencies(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("native MXFP8 GEMM requires SM10x")
        import deep_gemm

        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )

        previous_pdl = deep_gemm.get_pdl()
        self.addCleanup(deep_gemm.set_pdl, previous_pdl)
        torch.manual_seed(41)
        for rows, q_rope, padded_heads in (
            (64, False, None),
            (96, False, None),
            (384, False, None),
            (64, True, None),
            (64, False, 128),
        ):
            source = torch.randn(1, rows, 1792, device="cuda", dtype=torch.bfloat16)
            raw = torch.empty_like(source)
            q_weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
            kv_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            weight = (torch.randn(32768, 1280, device="cuda") * 0.02).to(
                torch.float8_e4m3fn
            )
            scales = torch.ones(1024, 40, device="cuda").to(torch.float8_e8m0fnu)
            linear = V41MXFP8Linear(weight, scales)
            angles = torch.randn(1024, 32, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            positions = torch.arange(rows, device="cuda", dtype=torch.int64)
            slots = positions.clone()
            slots[0] = -1
            pool = torch.empty(4, 128, 528, device="cuda", dtype=torch.uint8)

            def chain(pdl):
                _delayed_qa[(triton.cdiv(raw.numel(), 256),)](
                    source, raw, raw.numel(), launch_pdl=True
                )
                norm, pair = query_norm_quant(raw[..., :1280], q_weight, 1e-6, pdl=pdl)
                q = linear.forward_quantized(*pair).view(1, rows, 64, 512)
                result = try_fused_qkv_rope_cache(
                    q,
                    raw[..., 1280:],
                    kv_weight,
                    positions,
                    freqs,
                    pool,
                    slots,
                    apply_q_rope=q_rope,
                    pad_query_heads=padded_heads,
                    kv_ready=pdl,
                )
                self.assertIsNotNone(result)
                return (
                    norm,
                    pair[0].view(torch.uint8),
                    pair[1],
                    result.q,
                    result.kv,
                    result.freqs_cis,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                deep_gemm.set_pdl(True)
                for _ in range(3):
                    chain(True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = chain(True)
                for iteration in range(6):
                    source.normal_(std=(0.0, 1e-8, 1.0, 100.0)[iteration % 4])
                    positions.add_(3)
                    slots.copy_(slots.roll(1))
                    pool.fill_(173)
                    deep_gemm.set_pdl(False)
                    expected = chain(False)
                    expected_pool = pool.clone()
                    pool.fill_(173)
                    raw.zero_()
                    graph.replay()
                    for got, ref in zip(actual, expected):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
                    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
            torch.cuda.current_stream().wait_stream(stream)

    def assert_same(self, got, ref):
        torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(
            got[1][0].view(torch.uint8), ref[1][0].view(torch.uint8)
        )
        rows = got[0].numel() // got[0].shape[-1]
        groups = got[0].shape[-1] // 32
        for left, right in zip(
            got[1][1].T.contiguous().view(torch.uint8),
            ref[1][1].T.contiguous().view(torch.uint8),
        ):
            torch.testing.assert_close(
                left[: rows * 4], right[: rows * 4], rtol=0, atol=0
            )
        self.assertEqual(got[1][1].shape, ref[1][1].shape)
        self.assertEqual(got[1][1].stride(), ref[1][1].stride())
        self.assertEqual(groups % 4, 0)

    def test_shapes_strides_and_dynamic_range(self):
        torch.manual_seed(13)
        for rows in (0, 1, 4, 7, 20, 24, 40, 48, 97):
            for width in (128, 1280, 2560):
                with self.subTest(rows=rows, width=width):
                    packed = torch.randn(
                        rows, width + 512, device="cuda", dtype=torch.bfloat16
                    )
                    x = packed[:, :width]
                    w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                    self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))
        for scale in (0.0, 1e-25, 1e-10, 1e-4, 1.0, 1000.0):
            x = torch.randn(7, 1280, device="cuda", dtype=torch.bfloat16) * scale
            w = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
            self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))

    def test_quantization_thresholds(self):
        x = torch.ones(8, 1280, device="cuda", dtype=torch.bfloat16)
        values = torch.tensor(
            [0, 1e-10, 0.4375, 0.439453125, 448, 450, 896, 900], device="cuda"
        )
        w = values.repeat_interleave(160).bfloat16()
        self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))

    def test_non_default_stream_graph_replay_updates_input(self):
        x = torch.randn(24, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
        w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                query_norm_quant(x, w, 1e-6)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = query_norm_quant(x, w, 1e-6)
        for scale in (0.0, 1.0, 100.0):
            x.copy_(torch.randn_like(x) * scale)
            graph.replay()
            self.assert_same(result, reference(x, w))

    def test_native_consumer_and_output_buffer(self):
        x = torch.randn(4, 6, 1280, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        norm, quant = query_norm_quant(x, w, 1e-6)
        weight = (torch.randn(256, 1280, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        scales = torch.ones(8, 40, device="cuda").to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(weight, scales)
        expected = linear(norm)
        output = torch.empty_like(expected)
        actual = linear.forward_quantized(*quant, out=output)
        self.assertEqual(actual.data_ptr(), output.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("V41_QUERY_QUANT_BENCH"), "opt-in benchmark")
    def test_benchmark(self):
        def elapsed(fn):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(50):
                    fn()
            for _ in range(10):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)

        results = []
        for rows in (1, 4, 7, 20, 24, 40, 48, 97, 257):
            x = torch.randn(rows, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
            w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
            ref_us = elapsed(lambda: reference(x, w))
            fused_us = elapsed(lambda: query_norm_quant(x, w, 1e-6))
            results.append(
                dict(
                    rows=rows,
                    baseline_us=ref_us,
                    fused_us=fused_us,
                    speedup=ref_us / fused_us,
                )
            )
        result = json.dumps(results, indent=2)
        print(result)
        destination = os.environ.get("V41_QUERY_QUANT_BENCH_JSON")
        if destination:
            Path(destination).write_text(result + "\n")


if __name__ == "__main__":
    unittest.main()
