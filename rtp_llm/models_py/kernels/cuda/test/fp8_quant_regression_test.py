import os
from unittest import SkipTest, TestCase, main, mock

import torch

import rtp_llm.ops  # isort: skip
from rtp_llm.models_py.kernels.cuda.fp8_quant import (  # isort: skip
    _transform_scale_ue8m0,
    scaled_fp8_per_tensor_quant,
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)


class Fp8QuantRegressionTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_forced_legacy_rejects_v2_only_features(self):
        x = torch.ones((2, 256), dtype=torch.bfloat16, device="cuda")
        with mock.patch.dict(
            os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}, clear=False
        ), self.assertRaisesRegex(ValueError, "legacy does not support"):
            sgl_per_token_group_quant_fp8(x, group_size=128, fuse_silu_and_mul=True)

    def test_forced_v2_rejects_unsupported_group_size(self):
        x = torch.ones((2, 192), dtype=torch.bfloat16, device="cuda")
        with mock.patch.dict(
            os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}, clear=False
        ), self.assertRaisesRegex(ValueError, "v2 does not support"):
            sgl_per_token_group_quant_fp8(x, group_size=96)

    def test_v2_large_offset_matches_reference_at_first_and_last_rows(self):
        if getattr(torch.version, "hip", None) is not None:
            self.skipTest("v2 fp8 kernel path is CUDA-only")

        hidden_dim = 4096
        group_size = 128
        num_tokens = (2**31 // hidden_dim) + 8
        input_bytes = num_tokens * hidden_dim * torch.bfloat16.itemsize
        output_bytes = num_tokens * hidden_dim
        scale_bytes = num_tokens * (hidden_dim // group_size) * 4
        required_bytes = input_bytes + output_bytes + scale_bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes < 16 * 1024**3 or free_bytes < required_bytes + 2 * 1024**3:
            self.skipTest(
                "insufficient GPU memory for large-offset quant test: "
                f"free={free_bytes} required={required_bytes}"
            )

        x = torch.zeros((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
        first_pattern = (
            torch.arange(hidden_dim, device="cuda", dtype=torch.float32)
            .remainder(257)
            .sub(128)
            .div(16)
            .to(torch.bfloat16)
        )
        last_pattern = (
            torch.arange(hidden_dim, device="cuda", dtype=torch.float32)
            .remainder(193)
            .sub(64)
            .div(8)
            .to(torch.bfloat16)
        )
        x[0].copy_(first_pattern)
        x[-1].copy_(last_pattern)

        with mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}):
            quantized, scales = sgl_per_token_group_quant_fp8(x, group_size=group_size)
        torch.cuda.synchronize()

        boundary_input = torch.stack((x[0], x[-1])).float()
        grouped = boundary_input.reshape(2, hidden_dim // group_size, group_size)
        fp8_info = torch.finfo(quantized.dtype)
        expected_scales = grouped.abs().amax(dim=-1).clamp_min(1e-10) / float(
            fp8_info.max
        )
        expected_quantized = (
            grouped.div(expected_scales.unsqueeze(-1))
            .clamp(float(fp8_info.min), float(fp8_info.max))
            .to(quantized.dtype)
            .reshape(2, hidden_dim)
        )

        actual_quantized = torch.stack((quantized[0], quantized[-1]))
        actual_scales = torch.stack((scales[0], scales[-1]))
        torch.testing.assert_close(
            actual_quantized.float(), expected_quantized.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(actual_scales, expected_scales, rtol=1e-5, atol=1e-7)

    def test_per_tensor_empty_input_does_not_resolve_kernel(self):
        input_tensor = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
        with mock.patch(
            "rtp_llm.models_py.kernels.cuda.fp8_quant._resolve_compute_op"
        ) as resolve_op:
            output, dynamic_scale = scaled_fp8_per_tensor_quant(input_tensor)
            static_scale = torch.ones(1, dtype=torch.float32, device="cuda")
            static_output, returned_scale = scaled_fp8_per_tensor_quant(
                input_tensor, static_scale
            )

        resolve_op.assert_not_called()
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(dynamic_scale.shape, (1,))
        self.assertEqual(static_output.shape, input_tensor.shape)
        self.assertIs(returned_scale, static_scale)

    def test_per_token_empty_input_does_not_resolve_kernel(self):
        input_tensor = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
        with mock.patch(
            "rtp_llm.models_py.kernels.cuda.fp8_quant._resolve_compute_op"
        ) as resolve_op:
            output, scale = scaled_fp8_per_token_quant(input_tensor)

        resolve_op.assert_not_called()
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(scale.shape, (0, 1))

    def test_per_token_rejects_zero_width(self):
        input_tensor = torch.empty((2, 0), dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "width must be positive"):
            scaled_fp8_per_token_quant(input_tensor)

    def test_per_token_scale_buffer_contract(self):
        input_tensor = torch.ones((2, 128), dtype=torch.bfloat16, device="cuda")
        scale = torch.empty((2, 1), dtype=torch.float32, device="cuda")
        _, returned_scale = scaled_fp8_per_token_quant(input_tensor, scale)
        self.assertEqual(returned_scale.data_ptr(), scale.data_ptr())

        invalid_scale = torch.empty_like(input_tensor, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "per-token scale must have shape"):
            scaled_fp8_per_token_quant(input_tensor, invalid_scale)

    def test_transform_scale_moves_cpu_input_to_current_cuda_device(self):
        from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor

        mn = 256
        scale_cpu = torch.tensor(
            [[0.25, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 4.0]],
            dtype=torch.float32,
        )
        packed = _transform_scale_ue8m0(scale_cpu, mn)
        current_device = torch.cuda.current_device()
        scale_cuda = scale_cpu.to(device=current_device)
        expanded = scale_cuda.index_select(
            -2, torch.arange(mn, device=scale_cuda.device) // 128
        )
        expected = get_mn_major_tma_aligned_packed_ue8m0_tensor(expanded)

        self.assertTrue(packed.is_cuda)
        self.assertEqual(packed.device.index, current_device)
        self.assertTrue(torch.equal(packed, expected))


if __name__ == "__main__":
    main()
