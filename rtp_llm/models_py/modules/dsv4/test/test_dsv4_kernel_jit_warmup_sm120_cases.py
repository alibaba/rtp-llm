"""SM120 cases split out of test_dsv4_kernel_jit_warmup.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import os
import sys
import types
import unittest
from unittest import mock

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)
import rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup as warmup_module


class Dsv4KernelJitWarmupTestSm120(unittest.TestCase):
    """Sibling of ``Dsv4KernelJitWarmupTest`` carrying the SM120 cases."""

    def test_mhc_prenorm_warmup_uses_backend_resolver(self):
        device = torch.device("cpu")
        for backend in ("deepgemm", "torch", "cublas"):
            with (
                self.subTest(backend=backend),
                mock.patch.object(
                    warmup_module, "mhc_pre_gemm_backend", return_value=backend
                ) as resolve,
            ):
                self.assertEqual(
                    warmup_module._mhc_prenorm_deepgemm_backend_enabled(device),
                    backend == "deepgemm",
                )
                self.assertEqual(
                    warmup_module._mhc_prenorm_deepgemm_backend_name(device), backend
                )
                self.assertEqual(
                    resolve.call_args_list, [mock.call(device), mock.call(device)]
                )

    def test_mhc_warmup_evaluates_backend_hooks_under_device_scope(self):
        device = torch.device("cuda:3")
        shapes = {(24, 16384): {"name": "attn_hc"}}
        state = {"current": None, "scope_devices": [], "exited": False}
        seen = {"name": [], "enabled": [], "num_sms": []}

        class _RecordingDeviceScope:
            def __init__(self, scoped_device):
                state["scope_devices"].append(scoped_device)
                self._scoped_device = scoped_device

            def __enter__(self):
                state["current"] = self._scoped_device
                return self

            def __exit__(self, *exc):
                state["current"] = None
                state["exited"] = True
                return False

        def _record_name():
            seen["name"].append(state["current"])
            return "tilelang_single"

        def _record_enabled():
            seen["enabled"].append(state["current"])
            return False

        def _record_num_sms(_device):
            seen["num_sms"].append(state["current"])
            return 148

        with (
            mock.patch.object(
                warmup_module, "model_warm_up_enabled", return_value=True
            ),
            mock.patch.object(warmup_module, "_assert_not_capturing"),
            mock.patch.object(
                warmup_module,
                "_get_deep_gemm_num_sms",
                side_effect=_record_num_sms,
            ),
            mock.patch.object(
                warmup_module,
                "_mhc_prenorm_deepgemm_backend_name",
                side_effect=_record_name,
            ),
            mock.patch.object(
                warmup_module,
                "_mhc_prenorm_deepgemm_backend_enabled",
                side_effect=_record_enabled,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device", _RecordingDeviceScope),
        ):
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()
            warmup_module.warmup_mhc_prenorm_gemm_jit(shapes, max_m=0, device=device)

        self.assertEqual(state["scope_devices"], [device])
        self.assertEqual(seen["name"], [device])
        self.assertEqual(seen["enabled"], [device])
        self.assertEqual(seen["num_sms"], [None])
        self.assertTrue(state["exited"])
        self.assertIsNone(state["current"])

    def test_mhc_warmup_skips_device_scope_when_cuda_unavailable(self):
        device = torch.device("cuda:3")
        shapes = {(24, 16384): {"name": "attn_hc"}}
        hook_calls = []

        with (
            mock.patch.object(
                warmup_module, "model_warm_up_enabled", return_value=True
            ),
            mock.patch.object(warmup_module, "_is_cuda_device", return_value=True),
            mock.patch.object(warmup_module, "_assert_not_capturing"),
            mock.patch.object(
                warmup_module, "_get_deep_gemm_num_sms", return_value=148
            ),
            mock.patch.object(
                warmup_module,
                "_mhc_prenorm_deepgemm_backend_name",
                side_effect=lambda: hook_calls.append("name") or "tilelang_single",
            ),
            mock.patch.object(
                warmup_module,
                "_mhc_prenorm_deepgemm_backend_enabled",
                side_effect=lambda: hook_calls.append("enabled") or False,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.cuda, "device") as device_scope,
        ):
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()
            warmup_module.warmup_mhc_prenorm_gemm_jit(shapes, max_m=0, device=device)

        device_scope.assert_not_called()
        self.assertEqual(hook_calls, ["name", "enabled"])


if __name__ == "__main__":
    unittest.main()
