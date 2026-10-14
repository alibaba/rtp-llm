"""Tests for the can_use_flash_attn arch/version gate.

sm_120 (RTX PRO Blackwell) must only route into flash-attn when the
installed build carries sm_120 cubins (>= 2.8.4); the 2.8.3.post1 r1
wheel imports cleanly on sm_120 and then raises
cudaErrorNoKernelImageForDevice at the first kernel launch.
"""

import unittest
from unittest import mock

import torch

from rtp_llm.utils import flash_attn_utils


def _patch_caps(major, minor):
    return mock.patch.object(
        torch.cuda, "get_device_capability", return_value=(major, minor)
    )


class CanUseFlashAttnGateTest(unittest.TestCase):
    def test_sm8x_and_sm90_pass(self):
        with _patch_caps(8, 9), mock.patch.object(
            flash_attn_utils, "_flash_attn_2_available", return_value=True
        ):
            self.assertTrue(flash_attn_utils.can_use_flash_attn())
        with _patch_caps(9, 0), mock.patch.object(
            flash_attn_utils, "_flash_attn_2_available", return_value=True
        ):
            self.assertTrue(flash_attn_utils.can_use_flash_attn())

    def test_sm12x_requires_new_wheel(self):
        for version, expected in (("2.8.4", True), ("2.8.3.post1", False), ("3.0", True)):
            with _patch_caps(12, 0), mock.patch.object(
                flash_attn_utils, "_flash_attn_2_available", return_value=True
            ), mock.patch.object(
                flash_attn_utils,
                "_flash_attn_has_sm120_kernels",
                return_value=expected,
            ):
                self.assertEqual(
                    flash_attn_utils.can_use_flash_attn(),
                    expected,
                    msg=f"version {version}",
                )

    def test_sm12x_version_probe_parses(self):
        fake = mock.MagicMock()
        fake.__version__ = "2.8.4+b27cdc3.cu132"
        with mock.patch.dict("sys.modules", {"flash_attn": fake}):
            self.assertTrue(flash_attn_utils._flash_attn_has_sm120_kernels())
        fake.__version__ = "2.8.3.post1+cu13torch2.11cxx11abitrue.r1"
        with mock.patch.dict("sys.modules", {"flash_attn": fake}):
            self.assertFalse(flash_attn_utils._flash_attn_has_sm120_kernels())

    def test_unavailable_flash_attn_blocks_all(self):
        with _patch_caps(12, 0), mock.patch.object(
            flash_attn_utils, "_flash_attn_2_available", return_value=False
        ):
            self.assertFalse(flash_attn_utils.can_use_flash_attn())

    def test_other_archs_rejected(self):
        with _patch_caps(10, 0), mock.patch.object(
            flash_attn_utils, "_flash_attn_2_available", return_value=True
        ):
            self.assertFalse(flash_attn_utils.can_use_flash_attn())
        with _patch_caps(11, 0), mock.patch.object(
            flash_attn_utils, "_flash_attn_2_available", return_value=True
        ), mock.patch.object(
            flash_attn_utils, "_flash_attn_has_sm120_kernels", return_value=True
        ):
            self.assertFalse(flash_attn_utils.can_use_flash_attn())


if __name__ == "__main__":
    unittest.main()
