"""SM90 direct-FP32-scale quantization: exact contracts before clean timing.

Requires a parent-owned GPU. The benchmark includes quantization, scales and
the unchanged Humming GEMM; it does not time a copy under a new name.
"""

import json
import math
import os
import statistics
import struct
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _sm90_quant_scale_fused as fused


def _baseline(x):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales

    q, packed = sgl_per_token_group_quant_fp8(
        x.contiguous(),
        group_size=32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return q, unpack_ue8m0_input_scales(packed)[:, : x.shape[1] // 32]


def _bytes_equal(actual, expected):
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _finite_oracle(x, v2):
    """Independent Torch group math; no candidate helper or packed decoder."""
    rows, width = x.shape
    values = x.float().reshape(rows, width // 32, 32)
    maxima = values.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    raw = maxima / 448.0
    if v2:
        raw = torch.where(raw < torch.finfo(torch.float32).tiny, 0.0, raw)
    else:
        raw = raw.clamp_min(1e-10)
    exponent = torch.where(raw > 0, torch.ceil(torch.log2(raw)) + 127, 0).int()
    # Each finite UE8M0 value is an exactly representable IEEE FP32 power
    # of two. CUDA torch.exp2(float32(-127)) can approximate one ULP below
    # 2**-127; that transcendental approximation is not the scale ABI.
    # Construct an independent mathematical table in host binary64, then
    # convert to FP32 exactly. This does not reuse the candidate bit decoder.
    powers = torch.tensor(
        [math.ldexp(1.0, code - 127) for code in range(255)],
        dtype=torch.float32,
        device=x.device,
    )
    scale = powers[exponent.long()]
    if v2:
        bits = ((254 - exponent) << 23).contiguous()
        inverse = bits.view(torch.float32)
        # CUDA v2 enables flush-to-zero on the multiply and its operands.
        values = torch.where(
            values.abs() < torch.finfo(torch.float32).tiny,
            torch.copysign(torch.zeros_like(values), values),
            values,
        )
        quantized = values * inverse[..., None]
        quantized = torch.where(
            quantized.abs() < torch.finfo(torch.float32).tiny,
            torch.copysign(torch.zeros_like(quantized), quantized),
            quantized,
        )
    else:
        quantized = values / scale[..., None]
    quantized = torch.where(torch.isnan(quantized), -448.0, quantized)
    return (
        quantized.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(rows, width),
        scale,
    )


def _inverse_rope(o, freqs):
    result = o.clone()
    rows = o.numel() // (o.shape[-2] * 512)
    flat = result.view(rows, o.shape[-2], 512)
    expanded = freqs.repeat_interleave(rows // freqs.shape[0], dim=0)
    tail = torch.view_as_complex(flat[..., 448:].float().reshape(rows, -1, 32, 2))
    rotated = torch.view_as_real(tail * expanded.conj()[:, None]).flatten(-2)
    flat[..., 448:].copy_(rotated.bfloat16())
    return flat


class QuantizationPolicyCPU(unittest.TestCase):
    def test_independent_finite_scale_values_are_exact_ieee_powers(self):
        for code in range(255):
            power = math.ldexp(1.0, code - 127)
            bits = struct.unpack("<I", struct.pack("<f", power))[0]
            expected = 0x00400000 if code == 0 else code << 23
            self.assertEqual(bits, expected)

    def test_dispatch_uses_original_per_linear_threshold(self):
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                self.assertEqual(fused.use_v2(1023, 4096), policy == "v2")
                self.assertEqual(fused.use_v2(1024, 4096), policy != "legacy")
                self.assertEqual(fused.use_v2(0, 4096), policy == "v2")
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "invalid"}):
            with self.assertRaises(ValueError):
                fused.use_v2(1, 4096)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90QuantScaleFusedTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires SM90")
        torch.manual_seed(241016)

    def _compare(self, x, *, oracle=True):
        before = x.clone()
        actual = fused.quantize_g32(x)
        reference = _baseline(x)
        for a, b in zip(actual, reference):
            _bytes_equal(a, b)
        _bytes_equal(x, before)
        self.assertTrue(all(value.is_contiguous() for value in actual))
        if oracle:
            independent = _finite_oracle(x, fused.use_v2(*x.shape))
            for a, b in zip(actual, independent):
                _bytes_equal(a, b)
        return actual

    def test_payload_scales_decode_prefill_and_row_strides(self):
        shapes = (
            (0, 512),
            (1, 32),
            (1, 96),
            (1, 512),
            (1, 5120),
            (3, 4096),
            (16, 2304),
            (17, 8192),
            (128, 5120),
            (819, 5120),
            (820, 5120),
            (1023, 4096),
            (1024, 4096),
        )
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                for rows, width in shapes:
                    with self.subTest(policy=policy, rows=rows, width=width):
                        x = torch.randn(
                            (rows, width + 32), device="cuda", dtype=torch.bfloat16
                        )[:, :width]
                        self._compare(x)

    def test_zeros_ties_subnormals_extremes_and_nonfinite_match_original(self):
        # Keep the failed-oracle root cause observable in the GPU test log;
        # the byte gate below still compares the candidate with original CUDA.
        approximate = torch.exp2(
            torch.full((9, 16), -127.0, device="cuda", dtype=torch.float32)
        )
        print(
            "SM90_SCALE_ORACLE_SUBNORMAL "
            + json.dumps(
                dict(
                    torch_exp2_bits=hex(approximate.view(torch.int32)[0, 0].item()),
                    exact_ieee_bits=hex(
                        struct.unpack("<I", struct.pack("<f", math.ldexp(1.0, -127)))[0]
                    ),
                )
            ),
            flush=True,
        )
        x = torch.randn((12, 512), device="cuda", dtype=torch.bfloat16)
        x[0].zero_()
        x[1].fill_(-0.0)
        tie = torch.tensor(
            [448, 1.0625, 1.1875, 0.015625, -1.0625, -1.1875, -0.015625, -448],
            device="cuda",
            dtype=torch.bfloat16,
        )
        x[2] = tie.repeat(64)
        x[3] = tie.repeat(64) * (2**-120)
        x[4] = tie.repeat(64) * (2**-126)
        x[5].fill_(torch.finfo(torch.bfloat16).tiny)
        x[6].fill_(2**-133)
        x[7].fill_(torch.finfo(torch.bfloat16).max)
        x[8].fill_(-torch.finfo(torch.bfloat16).max)
        x[9, :32].fill_(float("inf"))
        x[9, 32:64].fill_(float("-inf"))
        x[10, :32].fill_(float("nan"))
        x[10, 32] = float("nan")
        x[11, ::8] = float("inf")
        for policy in ("legacy", "v2", "auto"):
            with self.subTest(policy=policy), patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}
            ):
                self._compare(x, oracle=False)
                self._compare(x[:9])

    def test_unpack_all_exponent_bytes_including_strided_packed_input(self):
        from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales

        packed = (
            torch.arange(256, device="cuda", dtype=torch.int32)
            .to(torch.uint8)
            .view(torch.int32)
        )
        backing = torch.empty((64, 12), device="cuda", dtype=torch.int32)
        backing[:, 1:10:2] = packed[:, None]
        for words in (backing[:, 1:10:2], backing.T, backing[:1, :1], backing[:0]):
            _bytes_equal(
                fused.unpack_ue8m0_to_fp32(words), unpack_ue8m0_input_scales(words)
            )

    def test_out_and_invalid_shapes(self):
        x = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((3, 16), device="cuda")
        result = fused.quantize_g32(x, out_q=q, out_scale=scale)
        self.assertEqual(result[0].data_ptr(), q.data_ptr())
        self.assertEqual(result[1].data_ptr(), scale.data_ptr())
        for invalid in (x[:, :511], x[:, ::2], x.float(), x.detach().requires_grad_()):
            with self.assertRaises(ValueError):
                fused.quantize_g32(invalid)
        with self.assertRaises(ValueError):
            fused.quantize_g32(x, out_q=x.view(torch.float8_e4m3fn)[:, :512])

    def test_wo_a_inverse_rope_group_layout_and_threshold(self):
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                for rows, groups, heads in (
                    (1, 8, 8),
                    (7, 8, 8),
                    (16, 1, 8),
                    (1023, 1, 8),
                    (1024, 1, 8),
                ):
                    with self.subTest(policy=policy, rows=rows, groups=groups):
                        backing = torch.randn(
                            (rows, groups * heads, 544),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        o = backing[..., :512]
                        phase = torch.randn((rows, 32), device="cuda")
                        freqs = torch.polar(torch.ones_like(phase), phase)
                        q, scale = fused.quantize_wo_a_g32(
                            o, freqs, n_groups=groups, heads_per_group=heads
                        )
                        independent = _inverse_rope(o, freqs).reshape(rows, groups, -1)
                        for g in range(groups):
                            expected_q, expected_scale = _baseline(independent[:, g])
                            _bytes_equal(q[:, g], expected_q)
                            _bytes_equal(scale[:, g], expected_scale)
                        self.assertTrue(q.transpose(0, 1).is_contiguous())
                        self.assertTrue(scale.transpose(0, 1).is_contiguous())
        o = torch.randn((2, 3, 8, 512), device="cuda", dtype=torch.bfloat16)
        freqs = torch.ones((2, 32), device="cuda", dtype=torch.complex64)
        q, scale = fused.quantize_wo_a_g32(o, freqs, n_groups=1, heads_per_group=8)
        reference = _baseline(_inverse_rope(o, freqs).reshape(6, 4096))
        _bytes_equal(q[:, 0], reference[0])
        _bytes_equal(scale[:, 0], reference[1])

    def test_graph_changed_inputs_and_frequencies(self):
        x = torch.randn((3, 5120), device="cuda", dtype=torch.bfloat16)
        o = torch.randn((3, 64, 512), device="cuda", dtype=torch.bfloat16)
        freqs = torch.ones((3, 32), device="cuda", dtype=torch.complex64)
        for _ in range(3):
            fused.quantize_g32(x)
            fused.quantize_wo_a_g32(o, freqs, n_groups=8, heads_per_group=8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dense = fused.quantize_g32(x)
            woa = fused.quantize_wo_a_g32(o, freqs, n_groups=8, heads_per_group=8)
        for _ in range(3):
            x.normal_()
            o.normal_()
            phase = torch.randn((3, 32), device="cuda")
            freqs.copy_(torch.polar(torch.ones_like(phase), phase))
            graph.replay()
            for a, b in zip(dense, _baseline(x)):
                _bytes_equal(a, b)
            ref = _inverse_rope(o, freqs).reshape(3, 8, 4096)
            for g in range(8):
                for a, b in zip((woa[0][:, g], woa[1][:, g]), _baseline(ref[:, g])):
                    _bytes_equal(a, b)

    @unittest.skipUnless(
        os.environ.get("DSV41_QUANT_SCALE_BENCH") == "1", "explicit clean benchmark"
    )
    def test_clean_wo_a_humming_chain_benchmark(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import (
            Sm90GroupedOutputProjection,
        )

        def time_graph(graph):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) * 10

        for groups, rows in ((8, 1), (8, 16), (1, 1)):
            width, output = 4096, 1024
            o = torch.randn(
                (rows, groups * 8, 512), device="cuda", dtype=torch.bfloat16
            )
            phase = torch.randn((rows, 32), device="cuda")
            freqs = torch.polar(torch.ones_like(phase), phase)
            weight = (torch.randn((groups, output, width), device="cuda") * 0.01).to(
                torch.float8_e4m3fn
            )
            block_scales = torch.ones(
                (groups, output // 32, width // 32), device="cuda"
            ).to(torch.float8_e8m0fnu)
            linear = Sm90GroupedOutputProjection(weight, block_scales)
            rotated = _inverse_rope(o, freqs).reshape(rows, groups, width)
            q, sf = fused.quantize_wo_a_g32(
                o, freqs, n_groups=groups, heads_per_group=8
            )
            oracle_parts = []
            for g in range(groups):
                reference_q, reference_sf = _baseline(rotated[:, g])
                _bytes_equal(q[:, g], reference_q)
                _bytes_equal(sf[:, g], reference_sf)
                decoded = (
                    reference_q.float().view(rows, width // 32, 32)
                    * reference_sf[..., None]
                ).flatten(1)
                oracle_parts.append((decoded @ weight[g].float().T).bfloat16())
            oracle = torch.stack(oracle_parts, dim=1).flatten(1)

            def before():
                return linear.forward_quantized(*linear.quantize(o, freqs))

            def after():
                payload, scales = fused.quantize_wo_a_g32(
                    o, freqs, n_groups=groups, heads_per_group=8
                )
                result = linear._forward(
                    linear.config,
                    inputs=payload.transpose(0, 1).view(groups * rows, width),
                    input_scale=scales.transpose(0, 1).view(groups * rows, width // 32),
                    weight=linear.weight,
                    weight_scale=linear.weight_scale,
                    weight_scale_2=linear.weight_scale_2,
                    expert_layout=None if groups == 1 else linear._offsets(rows),
                    locks=linear.locks,
                    compute_config=linear.compute_config,
                )
                return (
                    result.view(groups, rows, output)
                    .transpose(0, 1)
                    .reshape(rows, groups * output)
                )

            for fn in (before, after):
                torch.testing.assert_close(fn(), oracle, rtol=0.015, atol=0.02)
            _bytes_equal(before(), after())
            graphs = []
            for fn in (before, after):
                for _ in range(3):
                    fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs.append(graph)
            samples = [[], []]
            for iteration in range(5):
                for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    samples[arm].append(time_graph(graphs[arm]))
            print(
                "SM90_WOA_QUANT_SCALE_BENCH "
                + json.dumps(
                    dict(
                        rows=rows,
                        groups=groups,
                        k=width,
                        n=output,
                        old_us=samples[0],
                        new_us=samples[1],
                        speedup=statistics.median(samples[0])
                        / statistics.median(samples[1]),
                    )
                ),
                flush=True,
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_QUANT_SCALE_BENCH") == "1", "explicit clean benchmark"
    )
    def test_clean_quant_and_humming_chain_benchmark(self):
        from rtp_llm.models_py.modules.dsv41.sm90_dense import Sm90MXFP8Linear

        def time_graph(graph):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(100):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) * 10  # ms / 100 -> us

        for rows, width, output in (
            (1, 5120, 1536),
            (1, 1536, 32768),
            (1, 8192, 5120),
            (16, 5120, 1536),
            (820, 5120, 1536),
        ):
            x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
            weight = (torch.randn((output, width), device="cuda") * 0.01).to(
                torch.float8_e4m3fn
            )
            scales = torch.ones((output // 32, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            linear = Sm90MXFP8Linear(weight, scales)
            before = lambda: _baseline(x)
            after = lambda: fused.quantize_g32(x)
            for a, b in zip(before(), after()):
                _bytes_equal(a, b)

            def gemm(quantizer):
                q, sf = quantizer()
                return linear._forward(
                    linear.config,
                    inputs=q,
                    input_scale=sf,
                    weight=linear.weight,
                    weight_scale=linear.weight_scale,
                    weight_scale_2=linear.weight_scale_2,
                    locks=linear.locks,
                    compute_config=linear.compute_config,
                )

            # Independent FP32 gate before timing the unchanged GEMM.
            q, sf = before()
            decoded = (q.float().view(rows, width // 32, 32) * sf[..., None]).flatten(1)
            oracle = (decoded @ weight.float().T).bfloat16()
            for arm in (before, after):
                torch.testing.assert_close(gemm(arm), oracle, rtol=0.015, atol=0.02)
            _bytes_equal(gemm(before), gemm(after))
            for chain in (False, True):
                funcs = (
                    ((lambda: gemm(before)), (lambda: gemm(after)))
                    if chain
                    else (before, after)
                )
                graphs = []
                for fn in funcs:
                    for _ in range(3):
                        fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    graphs.append(graph)
                samples = [[], []]
                for iteration in range(5):
                    for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                        samples[arm].append(time_graph(graphs[arm]))
                print(
                    "SM90_QUANT_SCALE_BENCH "
                    + json.dumps(
                        dict(
                            rows=rows,
                            k=width,
                            n=output,
                            includes_gemm=chain,
                            old_us=samples[0],
                            new_us=samples[1],
                            speedup=statistics.median(samples[0])
                            / statistics.median(samples[1]),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
