import sys
import types
import unittest
from unittest.mock import patch

import rtp_llm.device as device
from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.pluggable.platform import PlatformContext


class PlatformSelectionTest(unittest.TestCase):
    def _detect(self, kind, *, current=0, **kwargs):
        def context(local_rank):
            if current != local_rank:
                raise RuntimeError(
                    "Capture platform only after setting the worker device"
                )
            return PlatformContext(
                kind,
                "ZW-M890P" if kind == DeviceType.Ppu else "test-device",
                local_rank,
                device_string=f"device:{local_rank}",
                capabilities={"test-runtime"},
            )

        fake_device = types.SimpleNamespace(runtime_context=context)
        with (
            patch("rtp_llm.device.runtime.get_device_type", return_value=kind),
            patch.object(device, "get_current_device", return_value=fake_device),
        ):
            return PlatformContext.detect(local_rank=0, **kwargs)

    def test_explicit_and_auto_platform_mapping(self):
        for kind in (DeviceType.Cpu, DeviceType.Cuda, DeviceType.Ppu, DeviceType.ROCm):
            with self.subTest(kind=kind):
                self.assertEqual(self._detect(kind).device_type, kind)
                self.assertEqual(
                    self._detect(kind, requested=kind.name.lower()).device_type, kind
                )
        self.assertEqual(self._detect(DeviceType.Ppu).device_name, "ZW-M890P")

    def test_conflicting_request_or_worker_device_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Requested cuda, detected Ppu"):
            self._detect(DeviceType.Ppu, requested="cuda")
        with self.assertRaisesRegex(RuntimeError, "after setting the worker device"):
            self._detect(DeviceType.Ppu, current=1)

    def test_cached_identity_does_not_redetect_or_construct(self):
        with (
            patch.object(device, "_current_device", None),
            patch.object(device, "_current_device_type", None),
            patch.object(device, "get_device_type", return_value=DeviceType.Ppu),
            patch.object(device, "get_device_cls", return_value=lambda: object()),
        ):
            self.assertIsNone(device.get_cached_device_type())
            instance = device.get_current_device()
            with patch.object(
                device,
                "get_device_type",
                side_effect=AssertionError("unexpected probe"),
            ):
                self.assertIs(device.get_current_device(), instance)
                self.assertEqual(device.get_cached_device_type(), DeviceType.Ppu)
            with self.assertRaisesRegex(RuntimeError, "Created device.*conflicts"):
                self._detect(
                    DeviceType.Cuda, created_device=device.get_cached_device_type()
                )

    def test_cached_object_without_identity_fails_closed(self):
        with (
            patch.object(device, "_current_device", object()),
            patch.object(device, "_current_device_type", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "construction identity"):
                device.get_cached_device_type()


if __name__ == "__main__":
    unittest.main()
