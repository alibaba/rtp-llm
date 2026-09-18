import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.linear import (
    V41Block32Linear,
    is_supported,
    quantize_block32,
)


class Block32LinearTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))
        if not is_supported(torch.empty(0, device=cls.device)):
            raise RuntimeError("V4.1 linear tests require CUDA13/SM100")
        torch.backends.cuda.matmul.allow_tf32 = False

    def activation(self, rows, columns=256):
        values = torch.tensor(
            [448, -448, 1.0625, 1.1875, 0, 32, -32, 0.5] * 4,
            device=self.device,
            dtype=torch.bfloat16,
        )
        decoded = torch.tensor(
            [448, -448, 1, 1.25, 0, 32, -32, 0.5] * 4,
            device=self.device,
            dtype=torch.float32,
        )
        scales = 2.0 ** (
            (torch.arange(rows * (columns // 32), device=self.device) % 6) - 3
        ).reshape(rows, columns // 32)
        source = (values.float().view(1, 1, 32) * scales[..., None]).bfloat16()
        expected = decoded.view(1, 1, 32).expand(rows, columns // 32, 32)
        return source.reshape(rows, columns), expected.reshape(rows, columns), scales

    def test_group32_ue8m0_and_round_to_nearest_even(self):
        source, expected, expected_scales = self.activation(13)
        encoded, scales = quantize_block32(source)
        torch.testing.assert_close(encoded.float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        zero, zero_scales = quantize_block32(torch.zeros_like(source))
        torch.testing.assert_close(
            zero.float(), torch.zeros_like(expected), rtol=0, atol=0
        )
        torch.testing.assert_close(
            zero_scales, torch.full_like(expected_scales, 2.0**-22), rtol=0, atol=0
        )

    def test_packed_activation_scales_match_independent_group32_values(self):
        import deep_gemm

        for rows in (0, 1, 13, 257):
            source, expected, expected_scales = self.activation(rows, columns=5120)
            encoded, scales = quantize_block32(source, packed=True)
            torch.testing.assert_close(encoded.float(), expected, rtol=0, atol=0)
            if rows:
                reference = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                    expected_scales
                )
                torch.testing.assert_close(scales, reference, rtol=0, atol=0)
                self.assertEqual(scales.stride(), (1, ((rows + 3) // 4) * 4))
                self.assertGreaterEqual(
                    scales.untyped_storage().nbytes(),
                    ((rows + 3) // 4) * 4 * (source.shape[1] // 128) * 4,
                )
            else:
                self.assertEqual(scales.shape, (0, 40))

    def test_raw_scale_codec_handles_incomplete_vector_tile(self):
        source, expected, expected_scales = self.activation(3, columns=96)
        encoded, scales = quantize_block32(source)
        torch.testing.assert_close(encoded.float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "K%128"):
            quantize_block32(source, packed=True)

    def test_swizzled_scales_match_group32_values_and_zero_padding(self):
        import flashinfer

        for rows in (0, 1, 6, 127, 128, 129, 1025):
            with self.subTest(rows=rows):
                source, expected, expected_scales = self.activation(rows, columns=1280)
                encoded, scales = quantize_block32(source, swizzled=True)
                torch.testing.assert_close(encoded.float(), expected, rtol=0, atol=0)
                self.assertEqual(scales.dtype, torch.uint8)
                self.assertEqual(scales.numel(), ((rows + 127) // 128) * 128 * 40)
                if rows:
                    reference = flashinfer.block_scale_interleave(
                        expected_scales.to(torch.float8_e8m0fnu).view(torch.uint8)
                    )
                    torch.testing.assert_close(scales, reference, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "only one"):
            quantize_block32(source, packed=True, swizzled=True)

    def fixture(self, rows, *, n=128, k=256):
        source, encoded, scales = self.activation(rows, columns=k)
        columns = (torch.arange(n, device=self.device) * 7) % k
        signs = 1 - 2 * (torch.arange(n, device=self.device) % 2)
        weight = torch.zeros((n, k), dtype=torch.float32, device=self.device)
        weight[torch.arange(n, device=self.device), columns] = signs.float()
        weight = weight.to(torch.float8_e4m3fn)
        weight_scales = 2.0 ** (
            torch.arange((n // 32) * (k // 32), device=self.device) % 5 - 2
        ).reshape(n // 32, k // 32)
        expected = (
            encoded[:, columns]
            * scales[:, columns // 32]
            * signs
            * weight_scales[torch.arange(n, device=self.device) // 32, columns // 32]
        ).bfloat16()
        return (
            V41Block32Linear(weight, weight_scales.to(torch.float8_e8m0fnu)),
            source,
            expected,
        )

    def test_exact_sparse_gemm_raw_block32_scales_and_graph_updates(self):
        for rows in (1, 13, 33):
            with self.subTest(rows=rows):
                linear, source, expected = self.fixture(rows)
                output = torch.empty_like(expected)
                self.assertEqual(
                    linear(source, out=output).data_ptr(), output.data_ptr()
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    linear(source, out=output)
                    linear(source, out=output)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    linear(source, out=output)
                source.neg_()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize(self.device)
                torch.testing.assert_close(output, -expected, rtol=0, atol=0)

    def test_empty_rows_and_ranked_input_preserve_output_shape(self):
        linear, source, expected = self.fixture(6)
        ranked = linear(source.view(2, 3, -1))
        torch.testing.assert_close(ranked, expected.view(2, 3, -1), rtol=0, atol=0)
        self.assertEqual(linear(source[:0]).shape, (0, linear.out_features))

    def test_state_dict_scale_update_changes_actual_projection(self):
        linear, source, expected = self.fixture(13)
        state = linear.state_dict()
        state["weight_scale"] = state["weight_scale"] * 2
        linear.load_state_dict(state)
        torch.testing.assert_close(linear(source), expected * 2, rtol=0, atol=0)

    def test_cudnn_wq_b_exact_sparse_graph_and_scale_reload(self):
        linear, source, expected = self.fixture(2048, n=32768, k=1280)
        output = torch.empty_like(expected)
        self.assertEqual(linear(source, out=output).data_ptr(), output.data_ptr())
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        swizzled = linear._weight_scale_swizzled
        supported = torch.cuda.get_device_capability(self.device) == (10, 3)
        self.assertEqual(linear._use_swizzled_gemm(2048), supported)
        self.assertEqual(linear._use_swizzled_gemm(6), linear._use_cute)
        self.assertFalse(linear._use_swizzled_gemm(257))
        self.assertFalse(linear._use_swizzled_gemm(1024))
        self.assertFalse(linear._use_swizzled_gemm(1536))
        self.assertFalse(linear._use_swizzled_gemm(2049))
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            linear(source, out=output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            linear(source, out=output)
        source.neg_()
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(output, -expected, rtol=0, atol=0)
        state = linear.state_dict()
        state["weight_scale"] = state["weight_scale"] * 2
        packed_ptr = linear._weight_scale_packed.data_ptr()
        linear.load_state_dict(state)
        self.assertEqual(linear._weight_scale_packed.data_ptr(), packed_ptr)
        if supported:
            self.assertEqual(
                linear._weight_scale_swizzled.data_ptr(), swizzled.data_ptr()
            )
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(output, -expected * 2, rtol=0, atol=0)
        self.assertEqual(linear(source[:0]).shape, (0, 32768))
        fallback = V41Block32Linear(
            linear.weight, state["weight_scale"].to(torch.float8_e8m0fnu)
        )
        fallback._use_cudnn = False
        fallback._use_cute = False
        fallback._flashinfer_gemm = None
        fallback._weight_scale_swizzled = None
        self.assertFalse(fallback._use_swizzled_gemm(2048))
        self.assertIsNone(fallback._weight_scale_swizzled)
        torch.testing.assert_close(fallback(source), output, rtol=0, atol=0)

    def test_low_m_wq_b_exact_sparse_graph(self):
        linear, source, expected = self.fixture(6, n=32768, k=1280)
        output = linear(source)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            linear(source, out=output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            linear(source, out=output)
        source.neg_()
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(output, -expected, rtol=0, atol=0)

    def test_rejects_legacy_scale_grid_and_prequantization_dtype(self):
        linear, source, _ = self.fixture(1)
        with self.assertRaisesRegex(ValueError, "checkpoint UE8M0 32x32"):
            V41Block32Linear(
                linear.weight,
                torch.ones((1, 2), device=self.device).to(torch.float8_e8m0fnu),
            )
        with self.assertRaisesRegex(ValueError, "BF16"):
            linear(source.float())
        with self.assertRaisesRegex(ValueError, "checkpoint UE8M0"):
            V41Block32Linear(linear.weight, linear.weight_scale)


if __name__ == "__main__":
    unittest.main()
