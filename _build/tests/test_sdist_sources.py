"""Source-selection contracts using real setuptools; no Native compiler fixtures."""
import importlib.util
import shutil
from pathlib import Path

import pytest
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[2]


def load_setup():
    spec = importlib.util.spec_from_file_location("_sdist_sources_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    setup = load_setup()
    root = tmp_path / "source"
    (root / "_build").mkdir(parents=True)
    shutil.copyfile(ROOT / "_build/sdist_sources.txt", root / "_build/sdist_sources.txt")
    monkeypatch.chdir(root)
    monkeypatch.setattr(setup, "get_project_root", lambda: root)
    command = setup.SdistWithProto(Distribution({"name": "source-selection-contract", "version": "0"}))
    return root, command


def write_source(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# source-selection fixture\n")
    return relative


def test_real_build_roots_and_sdk_declarations_are_selected(source_tree):
    root, command = source_tree
    required = [
        "WORKSPACE", "BUILD", "BUILD.pytorch", "BUILD.aiter", "def.bzl", ".bazelrc",
        ".bazeliskrc", ".bazelignore", "cc_test_wrapper.sh", "LICENSE", "NOTICE",
        "_build/backend.py", "_build/oss_optional_extras.toml", "arch_config/arch_select.bzl",
        "bazel/defs.bzl", "deps/git.bzl", "deps/requirements_lock_cuda13.txt",
        "patches/cuda/fix.patch", "3rdparty/cuda_config/cuda_configure.bzl",
        "3rdparty/cutlass/cutlass.BUILD", "3rdparty/flashinfer/flashinfer.h",
        "tools/compile_flags.py", "rtp_llm/cpp/pybind/ConfigInit.cc",
        "rtp_llm/cpp/pybind/BUILD", "rtp_llm/cpp/model_rpc/proto/service.proto",
        "rtp_llm/models_py/kernels/cuda/op.cu", "rtp_llm/models_py/kernels/cuda/op.h",
        "rtp_llm/model_loader/tipc/csrc/ipc.cc", "rtp_llm/utils/BUILD",
        "rtp_llm/libs/BUILD",
        "rtp_llm/dash_sc/proto/model_config.proto", "rtp_llm/dash_sc/proto/predict_v2.proto",
        "rtp_llm/multimodal/kvcm/lookup.proto",
    ]
    for relative in required:
        write_source(root, relative)
    selected = command._source_files([])
    assert set(required) <= set(selected)
    assert "_build/sdist_sources.txt" in selected


def test_prebuilt_private_and_generated_paths_are_excluded_without_reading_them(source_tree):
    root, command = source_tree
    required_build = write_source(root, "rtp_llm/libs/BUILD")
    # Path-only metadata, deliberately no files masquerading as Native libraries.
    omitted = [
        "rtp_llm/libs/kv_cache_manager_server/bin/kv_cache_manager_server",
        "rtp_llm/libs/libth_transformer_config.so", "rtp_llm/models_py/native.so.2",
        "3rdparty/prebuilt.a", "rtp_llm/cpp/old.o", "3rdparty/old.dylib",
        "internal_source/pyproject_internal.toml", "artifacts/diagnostic.py",
        "rtp_llm/models_py/.venv/lib/python.py", "3rdparty/build/generated.cc",
        "3rdparty/.git/config", "rtp_llm/cpp/__pycache__/unused.pyc",
        "rtp_llm/cpp/.pytest_cache/CACHEDIR.TAG", "3rdparty/bazel-out/generated.cc",
        "_build/.env", "_build/.env.local", "_build/.ssh/id_rsa", ".torch_bazelrc",
    ]
    selected = set(command._source_files(omitted))
    assert required_build in selected
    assert not (set(omitted) & selected)


def test_python_resources_and_input_list_are_preserved(source_tree):
    root, command = source_tree
    original = ["./rtp_llm/__init__.py", "rtp_llm/config/model.json", "rtp_llm/ops/config.pyi"]
    for relative in original:
        write_source(root, relative)
    before = original.copy()
    selected = command._source_files(original)
    assert original == before
    assert {str(Path(p)) for p in original} <= set(selected)
    assert len(selected) == len(set(selected))


def test_archive_hook_leaves_egg_info_and_wheel_filelist_unchanged(source_tree, monkeypatch):
    root, command = source_tree
    write_source(root, "BUILD")
    original = [write_source(root, "rtp_llm/__init__.py")]
    captured = []
    monkeypatch.setattr(command.__class__.__mro__[1], "make_release_tree",
                        lambda self, base_dir, files: captured.append((base_dir, files)))
    command.make_release_tree("release", original)
    assert original == ["rtp_llm/__init__.py"]
    assert captured[0][0] == "release"
    assert "BUILD" in captured[0][1]
    assert "rtp_llm/__init__.py" in captured[0][1]


def test_source_manifest_is_required_not_silently_skipped(source_tree):
    root, command = source_tree
    (root / "_build/sdist_sources.txt").unlink()
    with pytest.raises(FileNotFoundError):
        command._source_files([])
