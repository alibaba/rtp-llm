"""CPU-only tests for architecture-specific UE8M0 layout selection."""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    _require_sm100_packed_scale_for_fp8_fp4,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.utils import arch


class TestUe8m0LayoutPolicy(TestCase):
    def setUp(self) -> None:
        self.clear_architecture_caches()

    def tearDown(self) -> None:
        self.clear_architecture_caches()

    def clear_architecture_caches(self) -> None:
        is_deep_gemm_e8m0_used.cache_clear()
        arch._get_sm_for_device.cache_clear()

    def test_blackwell_deepgemm_consumers_use_packed_scales(self):
        for capability, expected in (
            ((10, 0), True),
            ((12, 0), True),
            ((9, 0), False),
        ):
            with (
                self.subTest(capability=capability),
                patch.object(arch, "is_cuda", return_value=True),
                patch.object(deepgemm_wrapper, "is_cuda", return_value=True),
                patch.object(torch.cuda, "current_device", return_value=0),
                patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ),
            ):
                self.clear_architecture_caches()
                self.assertEqual(is_deep_gemm_e8m0_used(), expected)

    def test_layout_policy_is_cached_per_device(self):
        capabilities = {0: (10, 0), 1: (9, 0)}
        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch.object(deepgemm_wrapper, "is_cuda", return_value=True),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda device: capabilities[int(device)],
            ),
        ):
            self.clear_architecture_caches()
            self.assertTrue(is_deep_gemm_e8m0_used(0))
            self.assertFalse(is_deep_gemm_e8m0_used(1))
            self.assertTrue(is_deep_gemm_e8m0_used(0))

    def test_zero_arg_policy_follows_current_device_and_rejects_cpu(self):
        capabilities = {0: (9, 0), 1: (12, 0)}
        current = [0]
        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch.object(deepgemm_wrapper, "is_cuda", return_value=True),
            patch.object(torch.cuda, "current_device", side_effect=lambda: current[0]),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda device: capabilities[int(device)],
            ),
        ):
            self.clear_architecture_caches()
            self.assertFalse(is_deep_gemm_e8m0_used())
            current[0] = 1
            self.assertTrue(is_deep_gemm_e8m0_used())
            self.assertFalse(is_deep_gemm_e8m0_used("cpu"))
            self.assertFalse(is_deep_gemm_e8m0_used(torch.device("cpu")))

    def test_blackwell_fp8_fp4_requires_prepacked_int32_scales(self):
        a_data = torch.zeros((1, 16), dtype=torch.float8_e4m3fn)
        b_data = torch.zeros((1, 8), dtype=torch.int8)
        int32_scale = torch.zeros((1, 1), dtype=torch.int32)
        fp32_scale = torch.zeros((1, 1), dtype=torch.float32)

        for capability in ((10, 0), (12, 0)):
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(
                    torch.cuda,
                    "get_device_capability",
                    side_effect=lambda *_args, _cap=capability: _cap,
                ),
            ):
                _require_sm100_packed_scale_for_fp8_fp4(
                    (a_data, int32_scale), (b_data, int32_scale)
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SM100/SM120 FP8xFP4 DeepGEMM calls require prepacked int32",
                ):
                    _require_sm100_packed_scale_for_fp8_fp4(
                        (a_data, fp32_scale), (b_data, int32_scale)
                    )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SM100/SM120 FP8xFP4 DeepGEMM calls require prepacked int32",
                ):
                    _require_sm100_packed_scale_for_fp8_fp4(
                        (a_data, int32_scale), (b_data, fp32_scale)
                    )

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda *_args: (9, 0),
            ),
        ):
            _require_sm100_packed_scale_for_fp8_fp4(
                (a_data, fp32_scale), (b_data, fp32_scale)
            )


if __name__ == "__main__":
    main()
