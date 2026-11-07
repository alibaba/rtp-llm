"""PPU fusions must retain FP8 bytes, scales, and graph replay semantics."""

import json
from pathlib import Path
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import flashinfer
import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.kernels.ppu.fp8_quant import (
    _fp8_quant_128,
    fp8_quantize_128,
    silu_mul_fp8_quantize_128,
)
from rtp_llm.models_py.kernels.ppu.fused_norm_quant import (
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_deepgemm_linear import (
    PpuFp8DeepGemmLinear,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul


@skipUnless(torch.cuda.is_available() and is_ppu(), "requires PPU")
class FusedQuantTest(TestCase):
    ROW_CASES = (1, 7, 128, 512, 4096)
    QUANT_ROW_CASES = (17, 512)

    def assert_quant_equal(self, actual, expected):
        self.assertTrue(
            torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        )
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def baseline(self, x, gate_first):
        if gate_first:
            value = flashinfer.activation.silu_and_mul(x)
        else:
            value = torch.empty(
                x.shape[0], x.shape[1] // 2, device=x.device, dtype=x.dtype
            )
            silu_and_mul(value, x)
        return fp8_quantize_128(value)

    def test_tiled_quant_preserves_original_kernel_and_frozen_golden(self):
        torch.manual_seed(1271)
        for rows in self.QUANT_ROW_CASES:
            for hidden in [128, 1536, 2048, 6144]:
                x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                scales = torch.empty(
                    rows, hidden // 128, device="cuda", dtype=torch.float32
                )
                _fp8_quant_128[(rows, hidden // 128)](x, q, scales, hidden, num_warps=4)
                self.assert_quant_equal(fp8_quantize_128(x), (q, scales))
        cases = json.loads(
            Path(__file__).with_name("test_fp8_quant_sgl_golden.json").read_text()
        )["cases"]
        for case in cases.values():
            x = (
                torch.tensor(case["input_bf16_bits"], dtype=torch.uint16)
                .view(torch.bfloat16)
                .reshape(case["shape"])
                .cuda()
                .repeat(16, 1)
            )
            q = (
                torch.tensor(case["q_bytes"], dtype=torch.uint8, device=x.device)
                .reshape(case["shape"])
                .repeat(16, 1)
            )
            scales = (
                torch.tensor(case["scale_f32_bits"], dtype=torch.uint32)
                .view(torch.float32)
                .reshape(case["shape"][0], -1)
                .cuda()
                .repeat(16, 1)
            )
            actual = fp8_quantize_128(x)
            self.assertTrue(torch.equal(actual[0].view(torch.uint8), q))
            self.assertTrue(torch.equal(actual[1], scales))

    def test_dense_and_routed_layouts_match_installed_operators(self):
        torch.manual_seed(731)
        for rows in self.ROW_CASES:
            for hidden in [128, 2048, 6144, 24576]:
                x = torch.randn(rows, hidden * 2, device="cuda", dtype=torch.bfloat16)
                for gate_first in [False, True]:
                    with self.subTest(rows=rows, hidden=hidden, gate_first=gate_first):
                        self.assert_quant_equal(
                            silu_mul_fp8_quantize_128(x, gate_first=gate_first),
                            self.baseline(x, gate_first),
                        )

    def test_frozen_sgl_activation_golden_in_both_layouts(self):
        case = json.loads(
            Path(__file__).with_name("test_fp8_quant_sgl_golden.json").read_text()
        )["activation"]
        x = (
            torch.tensor(case["input_gate_up_bf16_bits"], dtype=torch.uint16)
            .view(torch.bfloat16)
            .reshape(case["shape"])
            .cuda()
        )
        expected_q = torch.tensor(
            case["q_bytes"], dtype=torch.uint8, device=x.device
        ).reshape(2, 256)
        expected_s = (
            torch.tensor(case["scale_f32_bits"], dtype=torch.uint32)
            .view(torch.float32)
            .reshape(2, 2)
            .cuda()
        )
        for gate_first, values in [
            (True, x),
            (False, torch.cat([x[:, 256:], x[:, :256]], -1)),
        ]:
            q, scales = silu_mul_fp8_quantize_128(values, gate_first=gate_first)
            self.assertTrue(torch.equal(q.view(torch.uint8), expected_q))
            self.assertTrue(torch.equal(scales, expected_s))

    def test_graph_replay_uses_new_values(self):
        x = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)
        for gate_first in [False, True]:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                silu_mul_fp8_quantize_128(x, gate_first=gate_first)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = silu_mul_fp8_quantize_128(x, gate_first=gate_first)
            for scale in [0.0, 1e-7, 1.0, 1000.0]:
                x.normal_().mul_(scale)
                graph.replay()
                self.assert_quant_equal(actual, self.baseline(x, gate_first))

    def test_empty_and_unsupported_input_contracts(self):
        q, scales = silu_mul_fp8_quantize_128(
            torch.empty(0, 256, device="cuda", dtype=torch.bfloat16), gate_first=True
        )
        self.assertEqual(q.shape, (0, 128))
        self.assertEqual(scales.shape, (0, 1))
        for x in [
            torch.empty(2, 250, device="cuda", dtype=torch.bfloat16),
            torch.empty(2, 256, device="cuda", dtype=torch.float32),
            torch.empty(2, 512, device="cuda", dtype=torch.bfloat16)[:, ::2],
        ]:
            with self.assertRaises(ValueError):
                silu_mul_fp8_quantize_128(x, gate_first=True)

    def test_linear_prequantized_input_preserves_dense_output(self):
        torch.manual_seed(991)
        weight = torch.randn(128, 256, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(1, 2, device="cuda", dtype=torch.float32) * 0.01
        bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        with patch.object(PpuFp8DeepGemmLinear, "_maybe_warmup"):
            linear = PpuFp8DeepGemmLinear(
                weight.reshape(256, 128), scales.reshape(2, 1), bias=bias
            )
        for rows in [0, 1, 7, 128]:
            x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
            q, scale = fp8_quantize_128(x)
            expected, actual = linear(x), linear(q, input_scales=scale)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual, expected))
        with self.assertRaises(ValueError):
            linear(q)
        with self.assertRaises(ValueError):
            linear(q, input_scales=scale[:, :1])

    def test_norm_and_residual_match_native_rounding_and_strides(self):
        torch.manual_seed(6117)
        for rows in self.ROW_CASES:
            for hidden in [512, 2048, 6144]:
                x = torch.randn(rows, hidden + 64, device="cuda", dtype=torch.bfloat16)[
                    :, :hidden
                ]
                weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
                for eps in [1e-6, 1e-5]:
                    with self.subTest(rows=rows, hidden=hidden, eps=eps):
                        expected = flashinfer.norm.rmsnorm(
                            x.contiguous(), weight, eps=eps
                        )
                        y, q, scales = (
                            fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                                x, weight, eps
                            )
                        )
                        self.assertTrue(torch.equal(y, expected))
                        self.assert_quant_equal((q, scales), fp8_quantize_128(expected))
                        self.assertTrue(
                            torch.equal(fused_strided_rmsnorm(x, weight, eps), expected)
                        )
                        original_r = torch.randn_like(x)
                        h, r = x.contiguous().clone(), original_r.clone()
                        flashinfer.norm.fused_add_rmsnorm(h, r, weight, eps=eps)
                        residual = original_r.clone()
                        y, q, scales = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                            x, residual, weight, eps
                        )
                        self.assertTrue(torch.equal(y, h))
                        self.assertTrue(torch.equal(residual, r))
                        self.assert_quant_equal((q, scales), fp8_quantize_128(h))

    def test_norm_graph_replay_reads_changed_inputs(self):
        x = torch.randn(7, 6144, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(6144, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_add_rmsnorm_fp8_quant_with_bf16_output(x, residual, weight, 1e-5)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y, q, scales = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                x, residual, weight, 1e-5
            )
        for value in [0.0, 1e-7, 1.0, 1000.0]:
            x.normal_().mul_(value)
            residual.normal_().mul_(value)
            h, r = x.clone(), residual.clone()
            flashinfer.norm.fused_add_rmsnorm(h, r, weight, eps=1e-5)
            graph.replay()
            self.assertTrue(torch.equal(y, h))
            self.assertTrue(torch.equal(residual, r))
            self.assert_quant_equal((q, scales), fp8_quantize_128(h))

    def test_cp8_four_long_requests_norm_shape(self):
        # Four 32k requests have 16k local rows at CP8; this exceeds the
        # single-request 4k-local-row case and exercises the real batch path.
        torch.manual_seed(327684)
        rows = 16384
        for hidden in [512, 2048, 6144]:
            x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
            original_r = torch.randn_like(x)
            h, r = x.clone(), original_r.clone()
            flashinfer.norm.fused_add_rmsnorm(h, r, weight, eps=1e-5)
            y, q, scales = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                x, original_r, weight, 1e-5
            )
            self.assertTrue(torch.equal(y, h))
            self.assertTrue(torch.equal(original_r, r))
            self.assert_quant_equal((q, scales), fp8_quantize_128(h))


if __name__ == "__main__":
    main()
