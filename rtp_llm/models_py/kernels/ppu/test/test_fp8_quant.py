import json
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import m_grouped_fp8_gemm_nt_masked
from rtp_llm.models_py.kernels.ppu.fp8_quant import (
    fp8_quantize_128,
    masked_fp8_quantize_128,
    masked_silu_mul_fp8_quantize_128,
)


class MaskedFp8QuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "This test requires a PPU worker")
        torch.manual_seed(7331)

    def test_dense_column_major_scales_preserve_quantized_values(self):
        for rows in [0, 1, 3, 16, 17, 128, 129, 4096]:
            for hidden in [256, 6144]:
                with self.subTest(rows=rows, hidden=hidden):
                    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                    old_q, old_s = fp8_quantize_128(x)
                    q, scales = fp8_quantize_128(x, column_major_scales=True)
                    self.assertTrue(
                        torch.equal(q.view(torch.uint8), old_q.view(torch.uint8))
                    )
                    self.assertTrue(torch.equal(scales, old_s))
                    self.assertEqual(scales.shape, old_s.shape)
                    self.assertEqual(scales.stride(0), 1)
                    if rows:
                        self.assertEqual(scales.stride(1), rows)

    def test_dense_column_major_graph_reads_new_values(self):
        x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fp8_quantize_128(x, column_major_scales=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q, scales = fp8_quantize_128(x, column_major_scales=True)
        for value in [0.0, 1e-7, 1.0, 1000.0]:
            x.fill_(value)
            graph.replay()
            ref_q, ref_s = fp8_quantize_128(x)
            self.assertTrue(torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)))
            self.assertTrue(torch.equal(scales, ref_s))

    def test_dense_default_sdk_uses_scales_without_a_layout_copy(self):
        from unittest.mock import patch

        import deep_gemm

        import rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_deepgemm_linear as linear

        w = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.ones(2, 2, device="cuda")
        with patch.object(linear.PpuFp8DeepGemmLinear, "_maybe_warmup"):
            layer = linear.PpuFp8DeepGemmLinear(w, ws)
        for rows in [65, 128, 256, 4096]:
            x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
            q, row_scales = fp8_quantize_128(x)
            expected = layer(q, input_scales=row_scales, block_scaled=False)
            observed = []
            original = linear.fp8_gemm_nt

            def inspect_layout(lhs, rhs, output):
                observed.append(lhs[1])
                return original(lhs, rhs, output)

            with patch.object(linear, "fp8_gemm_nt", side_effect=inspect_layout):
                actual = layer(x)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(observed), 1)
            if rows % 4 == 0:
                scales = observed[0]
                self.assertEqual(scales.stride(), (1, rows))
                self.assertEqual(
                    deep_gemm.get_col_major_tma_aligned_tensor(scales).data_ptr(),
                    scales.data_ptr(),
                )
            else:
                self.assertTrue(observed[0].is_contiguous())

    def assert_reference(self, x, counts, q, scales):
        reference_q, reference_s = fp8_quantize_128(x.reshape(-1, x.shape[-1]))
        reference_q = reference_q.view(x.shape)
        reference_s = reference_s.view(*x.shape[:-1], x.shape[-1] // 128)
        active = torch.arange(x.shape[1], device=x.device)[None, :] < counts[:, None]
        self.assertTrue(
            torch.equal(
                q.view(torch.uint8)[active], reference_q.view(torch.uint8)[active]
            )
        )
        self.assertTrue(torch.equal(scales[active], reference_s[active]))
        self.assertEqual(scales.stride(1), 1)

    def test_eager_masked_layout_matches_dense(self):
        for shape in ((3, 16, 256), (32, 512, 6144)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for count in (0, 1, 7, shape[1]):
                with self.subTest(shape=shape, count=count):
                    counts = torch.tensor(
                        [
                            min(count + i % 3, shape[1]) if count else 0
                            for i in range(shape[0])
                        ],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    q, scales = masked_fp8_quantize_128(x, counts, 1)
                    self.assert_reference(x, counts, q, scales)

    def test_graph_replay_reads_changed_counts_and_values(self):
        x = torch.randn((32, 512, 6144), device="cuda", dtype=torch.bfloat16)
        counts = torch.ones(32, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            masked_fp8_quantize_128(x, counts, 1)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q, scales = masked_fp8_quantize_128(x, counts, 1)
        for value, count in ((0.0, 0), (1e-7, 2), (1.0, 9), (1000.0, 512)):
            x.fill_(value)
            counts.fill_(count)
            graph.replay()
            self.assert_reference(x, counts, q, scales)

    def test_column_major_scales_preserve_masked_gemm_output(self):
        x = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([0, 1, 8, 128], device="cuda", dtype=torch.int32)
        weight = torch.randn((4, 256, 256), device="cuda").to(torch.float8_e4m3fn)
        weight_scale = torch.ones((4, 2, 2), device="cuda", dtype=torch.float32)
        old_q, old_s = fp8_quantize_128(x.reshape(-1, 256))
        new_q, new_s = masked_fp8_quantize_128(x, counts, 1)
        old_out, new_out = torch.empty_like(x), torch.empty_like(x)
        m_grouped_fp8_gemm_nt_masked(
            (old_q.view_as(x), old_s.view(4, 128, 2)),
            (weight, weight_scale),
            old_out,
            counts,
            32,
        )
        m_grouped_fp8_gemm_nt_masked(
            (new_q, new_s), (weight, weight_scale), new_out, counts, 32
        )
        active = torch.arange(128, device="cuda")[None, :] < counts[:, None]
        self.assertTrue(torch.equal(old_out[active], new_out[active]))

    def golden(self):
        return json.loads(
            Path(__file__).with_name("test_fp8_quant_sgl_golden.json").read_text()
        )

    def golden_tensors(self, case):
        x = torch.tensor(case["input_bf16_bits"], dtype=torch.uint16, device="cpu")
        x = x.view(torch.bfloat16).reshape(case["shape"]).cuda()
        q = (
            torch.tensor(case["q_bytes"], dtype=torch.uint8, device="cpu")
            .reshape(case["shape"])
            .cuda()
        )
        s = torch.tensor(case["scale_f32_bits"], dtype=torch.uint32, device="cpu")
        s = s.view(torch.float32).reshape(case["shape"][0], -1).cuda()
        return x, q, s

    def test_measured_sgl_golden_dense_and_masked(self):
        for name, case in self.golden()["cases"].items():
            with self.subTest(case=name):
                x, expected_q, expected_s = self.golden_tensors(case)
                q, s = fp8_quantize_128(x)
                self.assertTrue(torch.equal(q.view(torch.uint8), expected_q))
                self.assertTrue(torch.equal(s, expected_s))
                counts = torch.tensor([0, 1], device=x.device, dtype=torch.int32)
                mq, ms = masked_fp8_quantize_128(x[:, None], counts, 1)
                self.assertTrue(torch.equal(mq[1, 0].view(torch.uint8), expected_q[1]))
                self.assertTrue(torch.equal(ms[1, 0], expected_s[1]))
                self.assertEqual(ms.stride(1), 1)

    def test_sgl_golden_graph_changes_values_and_expert_counts(self):
        cases = self.golden()["cases"]
        x = torch.empty((2, 1, 256), dtype=torch.bfloat16, device="cuda")
        x.fill_(1)
        counts = torch.ones(2, dtype=torch.int32, device=x.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            masked_fp8_quantize_128(x, counts, 1)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q, s = masked_fp8_quantize_128(x, counts, 1)
        for name, count_values in (
            ("dense_hidden", [1, 0]),
            ("dense_tiny", [0, 1]),
            ("dense_zero", [0, 0]),
            ("dense_large", [1, 1]),
        ):
            xx, expected_q, expected_s = self.golden_tensors(cases[name])
            x.copy_(xx[:, None])
            counts.copy_(
                torch.tensor(count_values, dtype=counts.dtype, device=counts.device)
            )
            graph.replay()
            for expert, count in enumerate(count_values):
                if count:
                    self.assertTrue(
                        torch.equal(q[expert, 0].view(torch.uint8), expected_q[expert])
                    )
                    self.assertTrue(torch.equal(s[expert, 0], expected_s[expert]))

    def test_fused_silu_matches_sgl_bf16_boundary_and_graph(self):
        case = self.golden()["activation"]
        gate_up = torch.tensor(
            case["input_gate_up_bf16_bits"], dtype=torch.uint16, device="cpu"
        )
        gate_up = gate_up.view(torch.bfloat16).reshape(case["shape"]).cuda()
        up_gate = torch.cat((gate_up[:, 256:], gate_up[:, :256]), dim=-1).contiguous()[
            :, None
        ]
        x = up_gate.clone()
        counts = torch.ones(2, device=x.device, dtype=torch.int32)
        expected_q = torch.tensor(
            case["q_bytes"], dtype=torch.uint8, device=x.device
        ).reshape(2, 256)
        expected_s = torch.tensor(
            case["scale_f32_bits"], dtype=torch.uint32, device="cpu"
        )
        expected_s = expected_s.view(torch.float32).reshape(2, 2).cuda()
        q, s = masked_silu_mul_fp8_quantize_128(x, counts, 1)
        self.assertTrue(torch.equal(q[:, 0].view(torch.uint8), expected_q))
        self.assertTrue(torch.equal(s[:, 0], expected_s))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            masked_silu_mul_fp8_quantize_128(x, counts, 1)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gq, gs = masked_silu_mul_fp8_quantize_128(x, counts, 1)
        for values in ([0, 0], [1, 0], [0, 1], [1, 1]):
            counts.copy_(torch.tensor(values, dtype=counts.dtype, device=counts.device))
            graph.replay()
            for expert, count in enumerate(values):
                if count:
                    self.assertTrue(
                        torch.equal(gq[expert, 0].view(torch.uint8), expected_q[expert])
                    )
                    self.assertTrue(torch.equal(gs[expert, 0], expected_s[expert]))
        x.zero_()
        graph.replay()
        self.assertEqual(int(torch.count_nonzero(gq.view(torch.uint8))), 0)
        zero_bits = self.golden()["cases"]["dense_zero"]["scale_f32_bits"][0]
        zero_scale = (
            torch.tensor([zero_bits], dtype=torch.uint32, device="cpu")
            .view(torch.float32)
            .item()
        )
        self.assertTrue(torch.equal(gs, torch.full_like(gs, zero_scale)))
        x.copy_(up_gate)
        graph.replay()
        self.assertTrue(torch.equal(gq[:, 0].view(torch.uint8), expected_q))
        self.assertTrue(torch.equal(gs[:, 0], expected_s))


if __name__ == "__main__":
    unittest.main()
