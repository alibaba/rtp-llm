"""Real OSS/internal package resolution contracts; no Native import substitute."""

import importlib
from importlib.machinery import PathFinder
from pathlib import Path

import rtp_llm


def test_oss_registry_import_uses_real_source():
    registry = importlib.import_module("rtp_llm.multimodal.multimodal_mixin_register")
    expected = (
        Path(rtp_llm.__file__).resolve().parent
        / "multimodal/multimodal_mixin_register.py"
    )
    assert Path(registry.__file__).resolve() == expected
    assert callable(registry.register_multimodal_mixin)
    assert callable(registry.get_multimodal_mixin_cls)


def test_multimodal_keeps_each_real_root_package_contribution():
    package = importlib.import_module("rtp_llm.multimodal")
    expected = {
        str((Path(path) / "multimodal").resolve())
        for path in rtp_llm.__path__
        if (Path(path) / "multimodal").is_dir()
    }
    assert {str(Path(path).resolve()) for path in package.__path__} == expected
    # Check each real provider without importing its optional/native dependencies.
    for directory in expected:
        for module in ("multimodal_mixin_register", "greennet"):
            source = Path(directory) / (module + ".py")
            if not source.is_file():
                source = Path(directory) / module / "__init__.py"
            if source.is_file():
                spec = PathFinder.find_spec(
                    "rtp_llm.multimodal." + module, package.__path__
                )
                assert spec is not None
                assert Path(spec.origin).resolve() == source.resolve()


def test_remote_collector_contains_real_initializer_and_registry():
    from rtp_llm.test.remote_tests.remote_exec_rtp import _collect_repo_runtime_files

    root = Path(rtp_llm.__file__).resolve().parents[1]
    files = set(_collect_repo_runtime_files(root, include_libs=False))
    assert "rtp_llm/multimodal/__init__.py" in files
    assert "rtp_llm/multimodal/multimodal_mixin_register.py" in files
    for directory in rtp_llm.__path__:
        path = Path(directory) / "multimodal/__init__.py"
        if path.is_file() and path != root / "rtp_llm/multimodal/__init__.py":
            assert "internal_source/rtp_llm/multimodal/__init__.py" in files
