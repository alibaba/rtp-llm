import unittest
from unittest import mock

from rtp_llm.config.py_config_modules import KernelTuningConfig
from rtp_llm.models_py.kernel_tuning import registry
from rtp_llm.models_py.kernel_tuning.types import KernelTuningStatus


def _enabled_config(**overrides) -> KernelTuningConfig:
    config = KernelTuningConfig()
    config.rocm_fp8_moe_deterministic_reduce = True
    for field, value in overrides.items():
        setattr(config, field, value)
    return config


class KernelTuningRegistryTest(unittest.TestCase):
    def setUp(self):
        registry._kernel_tuning_config = None
        self.addCleanup(setattr, registry, "_kernel_tuning_config", None)

    def test_provider_is_selected_by_compute_architecture(self):
        status = KernelTuningStatus("test", True, "configured")
        provider = mock.Mock(return_value=status)
        with mock.patch.dict(
            registry._PROVIDERS_BY_ARCH, {"gfx942": (provider,)}, clear=True
        ):
            self.assertEqual(
                registry.configure_kernel_tuning(_enabled_config(), "gfx942"),
                (status,),
            )
            self.assertEqual(
                registry.configure_kernel_tuning(_enabled_config(), "gfx950"),
                (),
            )
        provider.assert_called_once_with()

    def test_providers_are_not_called_when_feature_is_disabled(self):
        provider = mock.Mock()
        with mock.patch.dict(
            registry._PROVIDERS_BY_ARCH, {"gfx942": (provider,)}, clear=True
        ):
            self.assertEqual(
                registry.configure_kernel_tuning(KernelTuningConfig(), "gfx942"),
                (),
            )
        provider.assert_not_called()

    def test_deterministic_helpers_read_the_injected_config(self):
        config = _enabled_config(rocm_fp8_moe_deterministic_max_tokens=64)
        registry.configure_kernel_tuning(config, "gfx950")
        self.assertTrue(registry.is_rocm_fp8_moe_deterministic_reduce_enabled())
        self.assertEqual(registry.rocm_fp8_moe_deterministic_max_tokens(), 64)

        registry.configure_kernel_tuning(KernelTuningConfig(), "gfx950")
        self.assertFalse(registry.is_rocm_fp8_moe_deterministic_reduce_enabled())
        self.assertEqual(registry.rocm_fp8_moe_deterministic_max_tokens(), 128)

    def test_deterministic_helpers_stay_disabled_before_startup_injection(self):
        # Pre-startup callers (import-time or tooling without a parsed server
        # config) must keep the disabled defaults instead of reading the raw
        # process environment.
        self.assertFalse(registry.is_rocm_fp8_moe_deterministic_reduce_enabled())
        self.assertEqual(registry.rocm_fp8_moe_deterministic_max_tokens(), 128)

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
