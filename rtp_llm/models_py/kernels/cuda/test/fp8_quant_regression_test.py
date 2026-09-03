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
