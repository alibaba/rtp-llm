"""PPU FP8 policy, scale layout, and native Graph replay contracts."""

import math
import os
import unittest
from unittest.mock import Mock, patch

import torch
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_wo_a import PpuWoAFp8Linear
from rtp_llm.platforms.ppu.modules.linear import fp8_linear as fp8


class DecodeFp8PolicyTest(unittest.TestCase):
    def test_provider_freezes_dense_and_grouped_quantization(self):
        with patch.dict(os.environ, {"DSV4_PPU_DECODE_FP8_QUANT": "auto"}):
            provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        with patch.object(fp8, "PpuFp8Linear") as dense, patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_wo_a.PpuWoAFp8Linear"
        ) as grouped:
            provider.build_fp8_linear(None, "weight", "scale")
            provider.build_wo_a_fp8_linear(None, "weight", "scale")
            self.assertEqual(dense.call_args.kwargs["quantization"], "v2_column")
            self.assertEqual(grouped.call_args.kwargs["quantization"], "v2_row")
        with self.assertRaisesRegex(ValueError, "DSV4_PPU_DECODE_FP8_QUANT"):
            PpuDecodeProvider(
                {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_FP8_QUANT": "unknown"}
            )
        with self.assertRaisesRegex(ValueError, "row-major"):
            PpuWoAFp8Linear(
                None, None, groups=8, k_local=4096, quantization="v2_column"
            )

    def test_quantizer_dispatch_and_empty_layout(self):
        def quantize(x, q, scale, *args):
            q.zero_()
            scale.fill_(1)

        with patch.object(fp8, "_require_m890p"), patch.object(
            fp8, "_resolve_ppu_quant_symbols"
        ) as symbols:
            legacy, modern = Mock(side_effect=quantize), Mock(side_effect=quantize)
            symbols.return_value = legacy, modern
            x = torch.zeros((3, 256), dtype=torch.bfloat16)
            for mode, stride in (
                ("auto", (2, 1)),
                ("v2_row", (2, 1)),
                ("v2_column", (1, 3)),
            ):
                _, scale = fp8.quantize_ppu_fp8_activation(x, quantization=mode)
                self.assertEqual(scale.stride(), stride)
            self.assertEqual(legacy.call_count, 1)
            self.assertEqual(modern.call_count, 2)
            symbols.reset_mock()
            q, scale = fp8.quantize_ppu_fp8_activation(x[:0], quantization="v2_column")
            self.assertEqual(q.shape, (0, 256))
            self.assertEqual(scale.shape, (0, 2))
            symbols.assert_not_called()
            with self.assertRaisesRegex(ValueError, "strides"):
                fp8._validate_activation_scale_layout(
                    torch.empty((3, 4))[:, ::2], "v2_column"
                )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class DecodeFp8GraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_column_graph_matches_row_quantization_and_dense_gemm(self):
        from deep_gemm.jit_kernels.utils import get_col_major_tma_aligned_tensor

        torch.manual_seed(890419)
        weight = torch.randn((1024, 4096), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((8, 32), 124, dtype=torch.uint8, device="cuda").view(
            torch.float8_e8m0fnu
        )
        column = fp8.PpuFp8Linear(
            weight, scale, quantization="v2_column", share_input_quantization=True
        )
        row = fp8.PpuFp8Linear(
            weight, scale, quantization="v2_row", share_input_quantization=True
        )
        self.assertFalse(column.can_share_input_quantization(row))
        peer = fp8.PpuFp8Linear(
            weight, scale, quantization="v2_column", share_input_quantization=True
        )
        self.assertTrue(column.can_share_input_quantization(peer))
        for batch in (1, 3, 8, 32, 64, 128):
            with self.subTest(batch=batch):
                x = torch.randn((batch, 4096), device="cuda", dtype=torch.bfloat16)

                def run():
                    quantized = column.quantize_input(x)
                    output = column.forward_quantized(*quantized)
                    return (*quantized, output)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                torch.cuda.current_stream().wait_stream(stream)
                self.assertIs(get_col_major_tma_aligned_tensor(actual[1]), actual[1])
                for _ in range(5):
                    x.normal_()
                    expected_quant = row.quantize_input(x)
                    expected_out = row.forward_quantized(*expected_quant)
                    actual[0].view(torch.uint8).fill_(255)
                    actual[1].fill_(float("nan"))
                    actual[2].fill_(float("nan"))
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            actual[0].view(torch.uint8),
                            expected_quant[0].view(torch.uint8),
                        )
                    )
                    torch.testing.assert_close(
                        actual[1], expected_quant[1], rtol=0, atol=0
                    )
                    torch.testing.assert_close(actual[2], expected_out, rtol=0, atol=0)
                    torch.testing.assert_close(column(x), expected_out, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "strides"):
                    column.forward_quantized(
                        actual[0],
                        torch.empty_strided((batch, 32), (64, 1), device="cuda"),
                    )


if __name__ == "__main__":
    unittest.main()
