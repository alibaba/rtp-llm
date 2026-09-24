"""Backend isolation contracts; no CUDA allocations needed."""
import os
import sys
import types
from unittest import mock

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import buffer
from rtp_llm.models_py.modules.kimi_k3 import moe_backend


def test_buffers_are_shared_only_within_same_backend():
    group = object()
    old = types.ModuleType("deep_gemm")
    native = types.ModuleType("k3_native_deep_gemm")
    old.get_symm_buffer_for_mega_moe = mock.Mock(side_effect=lambda **kw: object())
    native.get_symm_buffer_for_mega_moe = mock.Mock(side_effect=lambda **kw: object())
    args = (group, 896, 629, 16, 3584, 3072, True, "situ")
    with mock.patch.dict(buffer._MEGA_BUF_CACHE, {}, clear=True), mock.patch.dict(
        sys.modules, {"deep_gemm": old}
    ):
        a = buffer._get_or_create_mega_buf(*args)
        b = buffer._get_or_create_mega_buf(*args, backend=native)
        assert a is not b
        assert buffer._get_or_create_mega_buf(*args) is a
        assert buffer._get_or_create_mega_buf(*args, backend=native) is b
        old.get_symm_buffer_for_mega_moe.assert_called_once()
        native.get_symm_buffer_for_mega_moe.assert_called_once()
        assert (id(group), *args[1:]) in buffer._MEGA_BUF_CACHE


def test_default_k3_does_not_import_optional_backend():
    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
        moe_backend, "_load_native", side_effect=AssertionError("unexpected import")
    ):
        assert moe_backend.get_k3_moe_backend() is None


def test_native_request_never_falls_back():
    moe_backend._load_native.cache_clear()
    with mock.patch.dict(os.environ, {"KIMI_K3_MOE_BACKEND": "vllm_native"}), mock.patch.dict(
        sys.modules, {"k3_native_deep_gemm": None}
    ):
        import pytest
        with pytest.raises(RuntimeError, match="requires the pinned"):
            moe_backend.get_k3_moe_backend()


def test_bad_configuration_fails():
    with mock.patch.dict(os.environ, {"KIMI_K3_MOE_BACKEND": "typo"}):
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            moe_backend.get_k3_moe_backend()


def test_native_manifest_rejects_wrong_source_patch_or_torch(tmp_path):
    import json
    import torch
    import pytest
    backend = types.ModuleType("k3_native_deep_gemm")
    backend.__version__ = "2.8.0"
    backend.__file__ = str(tmp_path / "__init__.py")
    backend.fp8_fp4_mega_moe = lambda activation_alpha=1.0, activation_beta=0.0: None
    manifest = {
        "commit": "a6bbb8000161c0dc3a85a0300a905f76898a7913",
        "patch_sha256": "3685360daa68961c64db6ecce52564fde3cff9c40feafb82095096c64de0e9ef",
        "torch": torch.__version__,
    }
    with mock.patch.dict(sys.modules, {"k3_native_deep_gemm": backend}):
        for key in manifest:
            moe_backend._load_native.cache_clear()
            (tmp_path / "BUILD_INFO.json").write_text(json.dumps({**manifest, key: "wrong"}))
            with pytest.raises(RuntimeError, match="does not match"):
                moe_backend._load_native()
        moe_backend._load_native.cache_clear()
        (tmp_path / "BUILD_INFO.json").write_text(json.dumps(manifest))
        assert moe_backend._load_native() is backend
        moe_backend._load_native.cache_clear()
