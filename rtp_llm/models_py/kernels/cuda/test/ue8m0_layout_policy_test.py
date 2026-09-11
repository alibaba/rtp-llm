"""CPU-only tests for architecture-specific UE8M0 layout selection."""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.utils import arch


class TestUe8m0LayoutPolicy(TestCase):
    def tearDown(self) -> None:
        arch._get_sm_for_device.cache_clear()

    def test_only_sm10x_uses_deepgemm_ue8m0_layout(self):
        for capability, expected in (
            ((10, 0), True),
            ((12, 0), False),
            ((9, 0), False),
        ):
            with self.subTest(capability=capability), patch.object(
                arch, "is_cuda", return_value=True
            ), patch.object(torch.cuda, "current_device", return_value=0), patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                arch._get_sm_for_device.cache_clear()
                self.assertEqual(is_deep_gemm_e8m0_used(), expected)

    def test_layout_policy_is_cached_per_device(self):
        capabilities = {0: (10, 0), 1: (12, 0)}
        with patch.object(arch, "is_cuda", return_value=True), patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=lambda device: capabilities[int(device)],
        ):
            arch._get_sm_for_device.cache_clear()
            self.assertTrue(is_deep_gemm_e8m0_used(0))
            self.assertFalse(is_deep_gemm_e8m0_used(1))
            self.assertTrue(is_deep_gemm_e8m0_used(0))


if __name__ == "__main__":
    main()
