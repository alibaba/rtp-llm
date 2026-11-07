import os
import unittest
from unittest import mock

import torch

from rtp_llm.device.device_type import (
    DeviceType,
    get_device_type,
    is_xpu,
    torch_device_type,
)


def _fake_xpu(available: bool) -> mock.Mock:
    xpu = mock.Mock()
    xpu.is_available.return_value = available
    return xpu


class DeviceTypeTest(unittest.TestCase):
    def setUp(self):
        # Keep the PPU heuristic out of the way on hosts that set PPU_HOME.
        env = mock.patch.dict(os.environ)
        env.start()
        os.environ.pop("PPU_HOME", None)
        self.addCleanup(env.stop)

    def _detect(self, xpu_available: bool, cuda_available: bool, hip=None):
        with mock.patch.object(
            torch, "xpu", _fake_xpu(xpu_available), create=True
        ), mock.patch.object(
            torch.cuda, "is_available", return_value=cuda_available
        ), mock.patch.object(
            torch.version, "hip", hip, create=True
        ), mock.patch.object(
            torch, "__version__", "2.6.0"
        ):
            return get_device_type(), is_xpu(), torch_device_type()

    def test_xpu_detected(self):
        self.assertEqual(self._detect(True, False), (DeviceType.Xpu, True, "xpu"))

    def test_xpu_takes_precedence_over_cuda(self):
        self.assertEqual(self._detect(True, True), (DeviceType.Xpu, True, "xpu"))

    def test_cuda_detected_without_xpu(self):
        self.assertEqual(self._detect(False, True), (DeviceType.Cuda, False, "cuda"))

    def test_rocm_still_reports_cuda_device_string(self):
        # PyTorch-ROCm aliases "cuda" to HIP.
        self.assertEqual(
            self._detect(False, True, hip="6.2"), (DeviceType.ROCm, False, "cuda")
        )

    def test_cpu_otherwise(self):
        device_type, xpu, _ = self._detect(False, False)
        self.assertEqual(device_type, DeviceType.Cpu)
        self.assertFalse(xpu)

    def test_torch_without_xpu_namespace(self):
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.version, "hip", None, create=True
        ), mock.patch.object(
            torch, "__version__", "2.6.0"
        ):
            had_xpu = hasattr(torch, "xpu")
            saved = getattr(torch, "xpu", None)
            if had_xpu:
                del torch.xpu
            try:
                self.assertEqual(get_device_type(), DeviceType.Cuda)
            finally:
                if had_xpu:
                    torch.xpu = saved


if __name__ == "__main__":
    unittest.main()
