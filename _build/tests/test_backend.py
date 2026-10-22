"""PEP 517 dependency hooks; no compiler or Native module is substituted."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _build import backend


def setup_helpers(requirements, skip=False):
    return SimpleNamespace(
        should_skip_bazel_build=lambda: skip,
        get_all_dependencies=lambda: requirements,
    )


@pytest.mark.parametrize("hook", ["wheel", "editable"])
def test_full_hook_delegates_and_adds_platform_build_inputs(hook):
    requirements = [
        "torch @ https://example.invalid/platform/torch.whl",
        "triton==3.6.0",
        "numpy<2",
        "pybind11_stubgen",
        "requests>=2",
    ]
    metadata = {"backend-test": "unchanged"}
    with patch.object(
        backend.importlib.util, "module_from_spec",
        return_value=setup_helpers(requirements),
    ), patch.object(backend.importlib.util, "spec_from_file_location") as spec, patch.object(
        backend._setuptools, "get_requires_for_build_" + hook, return_value=["wheel"]
    ) as upstream:
        spec.return_value.loader.exec_module.side_effect = lambda module: None
        actual = getattr(backend, "get_requires_for_build_" + hook)(metadata)
    upstream.assert_called_once_with(metadata)
    assert actual == ["wheel", *requirements[:4]]


@pytest.mark.parametrize("hook", ["wheel", "editable"])
def test_explicit_deps_only_preserves_setuptools_requirements(hook):
    with patch.object(
        backend.importlib.util, "module_from_spec",
        return_value=setup_helpers([], skip=True),
    ), patch.object(backend.importlib.util, "spec_from_file_location") as spec, patch.object(
        backend._setuptools, "get_requires_for_build_" + hook, return_value=["wheel"]
    ):
        spec.return_value.loader.exec_module.side_effect = lambda module: None
        assert getattr(backend, "get_requires_for_build_" + hook)() == ["wheel"]


@pytest.mark.parametrize(
    "requirements, message",
    [
        (["pybind11_stubgen"], "exactly one platform Torch"),
        (["torch==2", "torch==3", "pybind11_stubgen"], "exactly one platform Torch"),
        (["torch==2"], "requires pybind11_stubgen"),
    ],
)
def test_missing_or_ambiguous_required_inputs_fail_closed(requirements, message):
    with patch.object(
        backend.importlib.util, "module_from_spec",
        return_value=setup_helpers(requirements),
    ), patch.object(backend.importlib.util, "spec_from_file_location") as spec:
        spec.return_value.loader.exec_module.side_effect = lambda module: None
        with pytest.raises(RuntimeError, match=message):
            backend._native_build_requirements()


def test_same_requirement_is_not_duplicated():
    with patch.object(
        backend.importlib.util, "module_from_spec",
        return_value=setup_helpers(["torch==2", "torch==2", "pybind11_stubgen"]),
    ), patch.object(backend.importlib.util, "spec_from_file_location") as spec:
        spec.return_value.loader.exec_module.side_effect = lambda module: None
        assert backend._native_build_requirements() == ["torch==2", "pybind11_stubgen"]


def test_build_and_metadata_hooks_remain_original_setuptools_hooks():
    for name in (
        "build_wheel", "build_editable", "build_sdist",
        "prepare_metadata_for_build_wheel", "prepare_metadata_for_build_editable",
        "get_requires_for_build_sdist",
    ):
        assert getattr(backend, name) is getattr(backend._setuptools, name)


def test_actual_cuda13_owners_supply_exact_build_requirements(monkeypatch):
    monkeypatch.setenv("RTP_BAZEL_CONFIG", "--config=cuda13")
    monkeypatch.delenv("RTP_SKIP_BAZEL_BUILD", raising=False)
    from _build import platform
    monkeypatch.setattr(platform, "_cached_build_config", None)
    before_torch = sys.modules.get("torch")
    actual = backend._native_build_requirements()
    assert actual == [
        "numpy<2.0a0,>=1.25",
        "pybind11_stubgen",
        "torch @ https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
        "triton==3.6.0",
    ]
    assert sys.modules.get("torch") is before_torch
    assert not list((Path(__file__).resolve().parents[2] / "rtp_llm/libs").glob("*.so"))
