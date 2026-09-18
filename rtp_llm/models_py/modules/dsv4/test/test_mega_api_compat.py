import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

# These host-only API checks do not need the engine or its CUDA-only bindings.
_REPO = Path(__file__).resolve().parents[5]
for _name in (
    "rtp_llm",
    "rtp_llm.models_py",
    "rtp_llm.models_py.modules",
    "rtp_llm.models_py.modules.dsv4",
    "rtp_llm.models_py.modules.dsv4.moe",
    "rtp_llm.models_py.modules.dsv4.moe.strategies",
):
    _package = ModuleType(_name)
    _package.__path__ = [str(_REPO.joinpath(*_name.split(".")))]
    sys.modules.setdefault(_name, _package)

from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
    _MEGA_BUF_CACHE,
    _get_or_create_mega_buf,
    estimate_mega_moe_symm_buffer_bytes,
)
from rtp_llm.models_py.modules.dsv4.moe.mega_jit_warmup import resolve_mega_num_sms
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import (
    _mega_intermediate_size,
    _native_mega_intermediate_supported,
)


class MegaApiCompatibilityTest(unittest.TestCase):
    def test_native_layout_probe_falls_back_only_for_unsupported_geometry(self):
        config = SimpleNamespace(
            moe_inter_dim=2304,
            ep_size=4,
            n_routed_experts=384,
            n_activated_experts=6,
            dim=5120,
        )
        for supports_native, expected in ((True, 2304), (False, 2560)):
            _native_mega_intermediate_supported.cache_clear()
            query = Mock(return_value=(4096, None))
            if not supports_native:
                query.side_effect = RuntimeError("num_bytes % 16 == 0")
            module = SimpleNamespace(
                _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe=query)
            )
            with patch.dict(sys.modules, {"deep_gemm": module}):
                self.assertEqual(_mega_intermediate_size(config), expected)
                self.assertEqual(_mega_intermediate_size(config), expected)
            self.assertEqual(query.call_count, 1)
            self.assertEqual(query.call_args.args[:6], (4, 384, 384, 6, 5120, 2304))
        _native_mega_intermediate_supported.cache_clear()

    def test_public_buffer_api_selects_supported_dispatch_argument(self):
        calls = []
        buffer = SimpleNamespace(buffer=torch.empty(16, dtype=torch.int8))

        def current_api(*, mma_type, **kwargs):
            calls.append((mma_type, kwargs))
            return buffer

        def legacy_api(*, use_fp8_dispatch, **kwargs):
            calls.append((use_fp8_dispatch, kwargs))
            return buffer

        for api, dispatch in ((current_api, "fp8xfp4"), (legacy_api, True)):
            group = SimpleNamespace(size=lambda: 4)
            module = SimpleNamespace(get_symm_buffer_for_mega_moe=api)
            with patch.dict(sys.modules, {"deep_gemm": module}), patch.dict(
                _MEGA_BUF_CACHE, {}, clear=True
            ):
                result = _get_or_create_mega_buf(
                    group, 384, 384, 6, 5120, 2560, True, "swiglu"
                )
            self.assertIs(result, buffer)
            self.assertEqual(calls[-1][0], dispatch)
            self.assertEqual(calls[-1][1]["num_topk"], 6)

    def test_current_size_binding_uses_mma_type_and_shared_count(self):
        get_size = Mock(return_value=(4096, None))
        module = SimpleNamespace(
            _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe=get_size)
        )
        with patch.dict(sys.modules, {"deep_gemm": module}):
            self.assertEqual(
                estimate_mega_moe_symm_buffer_bytes(4, 384, 384, 6, 5120, 2560), 4096
            )
        self.assertEqual(
            get_size.call_args.args,
            (4, 384, 384, 6, 5120, 2560, "fp8xfp4", "swiglu", 0),
        )
        with patch.dict(sys.modules, {"deep_gemm": module}):
            self.assertEqual(
                estimate_mega_moe_symm_buffer_bytes(
                    4, 384, 384, 6, 5120, 2560, use_fp8_dispatch=False
                ),
                4096,
            )
        self.assertEqual(get_size.call_args.args[6], "bf16xbf16")

    def test_legacy_size_binding_retains_bool_api(self):
        calls = []

        def old_size(*args):
            calls.append(args)
            if len(args) != 8 or not isinstance(args[6], bool):
                raise TypeError("legacy signature")
            return (8192, None)

        module = SimpleNamespace(
            _C=SimpleNamespace(get_symm_buffer_size_for_mega_moe=old_size)
        )
        with patch.dict(sys.modules, {"deep_gemm": module}):
            self.assertEqual(
                estimate_mega_moe_symm_buffer_bytes(4, 128, 384, 3, 5120, 2560), 8192
            )
        self.assertEqual(calls[-1], (4, 128, 384, 3, 5120, 2560, True, "swiglu"))

    def test_default_zero_sm_count_resolves_device_without_changing_runtime(self):
        module = SimpleNamespace(get_num_sms=Mock(return_value=0))
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ) as props:
            self.assertEqual(resolve_mega_num_sms(module, "cuda:0"), 148)
            props.assert_called_once_with("cuda:0")
        module.get_num_sms.return_value = 80
        with patch("torch.cuda.get_device_properties") as props:
            self.assertEqual(resolve_mega_num_sms(module), 80)
            props.assert_not_called()


if __name__ == "__main__":
    unittest.main()
