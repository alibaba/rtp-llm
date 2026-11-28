import sys
import types
import unittest
from unittest.mock import patch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.ops import FMHAConfig

_TRT_MODULE = "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt"


def _stub_trt_module(family_capable: bool) -> types.ModuleType:
    """Stand in for the CUDA-only trt module.

    The real cuda_impl.trt imports CUDA-only ops from rtp_llm.ops.compute_ops
    and cannot even be imported on ROCm/PPU builds. The predicate only
    lazy-imports it after its CUDA device check, so the test injects this
    stub through sys.modules to control the family capability on any platform.
    """
    stub = types.ModuleType(_TRT_MODULE)
    stub.supports_trtllm_fmha_v2 = lambda _attn_configs: family_capable
    return stub


class HasCachelessPrefillCudaGraphBackendTest(unittest.TestCase):
    """Truth table for the embedding prefill graph capability gate.

    The predicate must follow the device/backend capability, not a validated
    device list: non-CUDA devices have no cache-less graph-capable prefill
    family today; on CUDA the family capability function and the config switch
    decide.
    """

    def _check(self, device, family_capable, fmha_config="default"):
        with (
            patch.object(
                attn_factory, "get_device_type", return_value=device
            ),
            patch.dict(
                sys.modules, {_TRT_MODULE: _stub_trt_module(family_capable)}
            ),
        ):
            if fmha_config == "default":
                fmha_config = FMHAConfig()
            return attn_factory.has_cacheless_prefill_cuda_graph_backend(
                object(), fmha_config
            )

    def test_non_cuda_devices_never_capable(self):
        # No ROCm/PPU prefill backend family supports the cache-less
        # (non-causal, no-prefix, non-paged) graph today.
        for device in (DeviceType.ROCm, DeviceType.Ppu, DeviceType.Cpu):
            self.assertFalse(self._check(device, True))

    def test_cuda_capable_family_enabled(self):
        self.assertTrue(self._check(DeviceType.Cuda, True))

    def test_cuda_family_capability_missing(self):
        # SM8.x / SM10.x, CUDA < 12.9, missing flashinfer trt fmha v2 module,
        # or dtype/head dims outside the family whitelist.
        self.assertFalse(self._check(DeviceType.Cuda, False))

    def test_cuda_family_disabled_by_config(self):
        fmha_config = FMHAConfig()
        fmha_config.enable_flashinfer_trt_fmha_v2 = False
        self.assertFalse(self._check(DeviceType.Cuda, True, fmha_config))

    def test_none_fmha_config_keeps_capability(self):
        self.assertTrue(self._check(DeviceType.Cuda, True, None))


if __name__ == "__main__":
    unittest.main()
