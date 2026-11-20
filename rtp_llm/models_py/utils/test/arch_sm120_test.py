"""SM120 cases split out of arch_test.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.utils import arch


class ArchTestSm120(TestCase):
    """Sibling of ``ArchTest`` carrying the SM120 cases."""

    def test_cuda_ordinal_strings_match_integer_devices(self):
        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda device: {0: (10, 0), 1: (12, 0)}[device],
            ) as capability,
        ):
            for device in ("1", "cuda:1", torch.device("cuda:1"), 1):
                self.assertTrue(arch.is_sm120(device))
                self.assertFalse(arch.is_sm10x(device))
            self.assertTrue(arch.is_sm10x("0"))
            self.assertEqual(capability.call_count, 2)

    def test_mhc_backend_uses_architecture(self):
        with patch.object(
            arch, "is_sm120", side_effect=lambda device=None: device == 1
        ):
            self.assertEqual(arch.mhc_pre_gemm_backend(1), "tilelang_single")
            self.assertEqual(arch.mhc_pre_gemm_backend(0), "deepgemm")
            self.assertEqual(arch.mhc_pre_gemm_backend(torch.device("cpu")), "deepgemm")

    def test_sm120_rejects_non_cuda_without_querying_capability(self):
        with patch.object(arch, "get_sm") as get_sm:
            with patch.object(arch, "is_cuda", return_value=False):
                self.assertFalse(arch.is_sm120())
            with patch.object(arch, "is_cuda", return_value=True):
                for device in ("cpu", torch.device("cpu")):
                    self.assertFalse(arch.is_sm120(device))
            get_sm.assert_not_called()

    def test_sm120_requires_exact_cuda_capability(self):
        for capability in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
            with (
                self.subTest(capability=capability),
                patch.object(arch, "is_cuda", return_value=True),
                patch.object(arch, "get_sm", return_value=capability),
            ):
                self.assertEqual(arch.is_sm120(0), capability == (12, 0))


if __name__ == "__main__":
    import unittest

    unittest.main()
