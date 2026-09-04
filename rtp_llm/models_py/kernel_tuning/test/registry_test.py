import unittest
from unittest import mock

from rtp_llm.models_py.kernel_tuning import registry
from rtp_llm.models_py.kernel_tuning.types import KernelTuningStatus


class KernelTuningRegistryTest(unittest.TestCase):
    def test_provider_is_selected_by_compute_architecture(self):
        status = KernelTuningStatus("test", True, "configured")
        provider = mock.Mock(return_value=status)
        with mock.patch.dict(
            registry._PROVIDERS_BY_ARCH, {"gfx942": (provider,)}, clear=True
        ):
            self.assertEqual(registry.configure_kernel_tuning("gfx942"), (status,))
            self.assertEqual(registry.configure_kernel_tuning("gfx950"), ())
        provider.assert_called_once_with()

    def test_current_arch_uses_rocm_device_properties(self):
        properties = mock.Mock(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch.object(
            registry.torch.cuda, "is_available", return_value=True
        ), mock.patch.object(registry.torch.version, "hip", "7.0"), mock.patch.object(
            registry.torch.cuda, "current_device", return_value=2
        ), mock.patch.object(
            registry.torch.cuda,
            "get_device_properties",
            return_value=properties,
        ):
            self.assertEqual(registry._current_rocm_arch(), "gfx942")


if __name__ == "__main__":
    unittest.main()
