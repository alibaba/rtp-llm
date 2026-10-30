import unittest
from unittest.mock import patch

import torch
from flashinfer import fp4_quantize, mm_fp4

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
)
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual


class DynamicNvfp4LinearTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("NVFP4 requires CUDA")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Dynamic NVFP4 requires SM100/SM103")
        self.backend = patch.dict("os.environ", RTP_LLM_FP4_GEMM_BACKEND="cutlass")
        self.backend.start()
        self.addCleanup(self.backend.stop)
        torch.manual_seed(37)
        self.weight = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        self.weight_scale = self.weight.abs().amax().float().clamp_min(1e-8).reshape(
            1
        ) / (6 * 448)
        self.codes, self.scales = fp4_quantize(
            self.weight, self.weight_scale.reciprocal()
        )
        self.linear = CudaFp4GEMMLinear(
            self.codes,
            self.scales,
            quant_config=ModelOptFp4Config(
                bits=4, group_size=16, is_quanted=False, activation_scheme="dynamic"
            ),
            weight_scale_2=self.weight_scale,
        )

    def reference(self, x):
        scale = x.abs().amax().float().clamp_min(1e-8).reshape(1) / (6 * 448)
        codes, scales = fp4_quantize(x.contiguous(), scale.reciprocal())
        return mm_fp4(
            codes,
            self.codes.T,
            scales,
            self.scales.T,
            scale * self.weight_scale,
            torch.bfloat16,
            backend="cutlass",
        )

    def test_dynamic_strided_inputs_against_flashinfer(self):
        for rows in (1, 8, 16, 32):
            with self.subTest(rows=rows):
                storage = torch.randn(rows, 192, dtype=torch.bfloat16, device="cuda")
                x = storage[:, :128]
                torch.testing.assert_close(
                    self.linear(x), self.reference(x), rtol=0.02, atol=0.0625
                )

    def test_replay_refreshes_dynamic_scale(self):
        x = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            self.linear(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.linear(x)
        for magnitude in (0.125, 2.0, 0.0):
            x.copy_(torch.randn_like(x) * magnitude)
            expected = self.reference(x)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)

    @staticmethod
    def logical_block_scales(scales, rows, width):
        # CUTLASS's 128x4 layout is [M/128,K/64,32,4,4]. Compare only
        # real rows; packet padding and unused scale bytes are not an ABI.
        padded_rows = (rows + 127) // 128 * 128
        return (
            scales.view(torch.uint8)
            .reshape(padded_rows // 128, width // 64, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(padded_rows, width // 16)[:rows]
        )

    def packet_reference(
        self, normalized, rowwise, weight_codes, weight_scales, scale_w
    ):
        codes, blocks, scales, outputs = [], [], [], []
        pieces = normalized.split(1) if rowwise else (normalized,)
        for piece in pieces:
            # Native AttnRes scales the rounded BF16 producer by RN float32
            # multiplication with 1/2688, before independent FlashInfer packing.
            scale = piece.float().abs().amax().clamp_min(1e-8).reshape(1) * (
                1 / (6 * 448)
            )
            values, block_scales = fp4_quantize(piece.contiguous(), scale.reciprocal())
            codes.append(values)
            blocks.append(self.logical_block_scales(block_scales, *piece.shape))
            scales.append(scale)
            outputs.append(
                mm_fp4(
                    values,
                    weight_codes.T,
                    block_scales,
                    weight_scales.T,
                    scale * scale_w,
                    torch.bfloat16,
                    backend="cutlass",
                )
            )
        return tuple(torch.cat(parts) for parts in (codes, blocks, scales, outputs))

    def test_attnres_nvfp4_packet_linear_graph_refreshes_inputs_and_scales(self):
        rows, width = 8, 7168
        weight = torch.randn(128, width, dtype=torch.bfloat16, device="cuda")
        scale_w = weight.abs().amax().float().clamp_min(1e-8).reshape(1) / (6 * 448)
        weight_codes, weight_scales = fp4_quantize(weight, scale_w.reciprocal())
        linear = CudaFp4GEMMLinear(
            weight_codes,
            weight_scales,
            quant_config=self.linear.quant_config,
            weight_scale_2=scale_w,
        )
        residual = KimiK3AttentionResidual(
            torch.randn(width, dtype=torch.bfloat16, device="cuda") * 0.02,
            torch.randn(width, dtype=torch.bfloat16, device="cuda") * 0.02,
            1e-5,
        )
        prefix = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
        anchors = torch.randn(rows, 2, width, dtype=torch.bfloat16, device="cuda")
        norm = torch.ones(width, dtype=torch.bfloat16, device="cuda")
        for rowwise, split in ((False, False), (False, True), (True, False)):
            with self.subTest(rowwise=rowwise, split=split):

                def run():
                    # DSA's decode packet retains the SAME producer's BF16
                    # output for its head gate, even with rowwise FP4 packing.
                    # This isolates the packing contract from AttnRes rounding
                    # differences between separate producer implementations.
                    decoded = residual(
                        prefix,
                        anchors,
                        output_norm_weight=norm,
                        output_norm_eps=1e-5,
                        num_blocks=1,
                        block_write_idx=1,
                        quantize_nvfp4=True,
                        split_nvfp4_quant=split,
                        rowwise_nvfp4=rowwise,
                        decode_head_gate_columns=8,
                    )
                    packet = decoded.views()[0]
                    activation = packet.unpack(packet.wire)
                    return (
                        decoded.local_hidden,
                        activation,
                        linear.forward_quantized(activation),
                    )

                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    normalized, activation, actual = run()
                previous_scale = None
                previous_normalized = None
                for magnitude in (0.25, 2.0, 0.0):
                    prefix.normal_()
                    anchors[:, 0].normal_()
                    norm.fill_(magnitude)
                    graph.replay()
                    values, blocks, scale, expected = self.packet_reference(
                        normalized, rowwise, weight_codes, weight_scales, scale_w
                    )
                    torch.testing.assert_close(
                        activation.values, values, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        self.logical_block_scales(activation.block_scales, rows, width),
                        blocks,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(activation.scale, scale, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0625)
                    torch.testing.assert_close(anchors[:, 1], prefix, rtol=0, atol=0)
                    if previous_scale is not None:
                        self.assertFalse(torch.equal(activation.scale, previous_scale))
                        self.assertFalse(torch.equal(normalized, previous_normalized))
                    previous_scale = activation.scale.clone()
                    previous_normalized = normalized.clone()

    def test_static_config_keeps_required_input_scale(self):
        static = ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        self.assertEqual(static.activation_scheme, "static")
        self.assertFalse(
            CudaFp4GEMMLinear.can_handle(
                static, self.codes, self.scales, weight_scale_2=self.weight_scale
            )
        )


if __name__ == "__main__":
    unittest.main()
