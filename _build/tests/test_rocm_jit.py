from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from _build import rocm_jit


def test_prebuilt_modules_are_loaded_without_rebuild():
    core = SimpleNamespace(get_module=Mock(), build_module=Mock())
    rocm_jit.prepare_modules(core)
    assert [call.args[0] for call in core.get_module.call_args_list] == list(
        rocm_jit.MODULES
    )
    core.build_module.assert_not_called()


def test_cold_modules_are_built_and_imported_with_compiler_restored(tmp_path, monkeypatch):
    compiler = tmp_path / "clang"
    compiler.touch()
    monkeypatch.setenv("HIP_CLANG_PATH", "/original/clang")
    built = set()

    def load(name):
        if name not in built:
            raise ModuleNotFoundError(name=name)

    def build_module(md_name, srcs):
        assert rocm_jit.os.environ["HIP_CLANG_PATH"] == str(compiler)
        assert srcs == [md_name + ".cu"]
        built.add(md_name)

    core = SimpleNamespace(
        get_module=Mock(side_effect=load),
        get_args_of_build=lambda name: {
            "srcs": [name + ".cu"], "hip_clang_path": str(compiler)
        },
        build_module=build_module,
    )
    rocm_jit.prepare_modules(core)
    assert built == set(rocm_jit.MODULES)
    assert core.get_module.call_count == 2 * len(rocm_jit.MODULES)
    assert rocm_jit.os.environ["HIP_CLANG_PATH"] == "/original/clang"


def test_compile_failure_stops_preparation_and_restores_environment(monkeypatch):
    monkeypatch.delenv("HIP_CLANG_PATH", raising=False)

    def build_module(md_name):
        raise RuntimeError("compiler failed")

    core = SimpleNamespace(
        get_module=Mock(side_effect=ModuleNotFoundError()),
        get_args_of_build=lambda name: {},
        build_module=build_module,
    )
    with pytest.raises(RuntimeError, match="compiler failed"):
        rocm_jit.prepare_modules(core)
    core.get_module.assert_called_once_with(rocm_jit.MODULES[0])
    assert "HIP_CLANG_PATH" not in rocm_jit.os.environ
