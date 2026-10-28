"""Protect the public preset distribution boundary with real package builders."""

import importlib.util
import json
import re
import shutil
from pathlib import Path
import sys
import zipfile

import pytest
from setuptools import Distribution, find_namespace_packages

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "public_perf_preset_runner", ROOT / "rtp_llm/test/perf_test/run_perf_preset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_presets_do_not_embed_site_paths():
    presets = json.loads(
        (ROOT / "rtp_llm/test/perf_test/perf_presets.json").read_text()
    )
    assert presets
    assert not any(name.startswith("v4_") for name in presets)
    # Scan serialized values, including env and flag=value arguments.
    assert not re.search(r"/(?:home|mnt|data[0-9]*)/", json.dumps(presets))


def test_public_cli_does_not_discover_internal_presets(monkeypatch, capsys):
    runner = load_runner()
    monkeypatch.setattr(sys, "argv", ["preset", "--help"])
    with pytest.raises(SystemExit) as result:
        runner.main()
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "v4_" not in output
    assert "model_perf_test" in output


def test_explicit_registry_preserves_cli_arguments_and_environment(
    tmp_path, monkeypatch
):
    runner = load_runner()
    registry = tmp_path / "presets.json"
    registry.write_text(
        json.dumps(
            {
                "custom": {
                    "main": "batch_decode_test.py",
                    "args": ["--batch_size", "8"],
                    "env": {"WORLD_SIZE": "4"},
                }
            }
        )
    )
    seen = {}

    def execute(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return type("Result", (), {"returncode": 7})()

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(
        sys, "argv", ["preset", "custom", "--", "--decode_test_length", "9"]
    )
    assert runner.main(registry) == 7
    assert seen["argv"][-4:] == ["--batch_size", "8", "--decode_test_length", "9"]
    assert seen["env"]["WORLD_SIZE"] == "4"
    assert seen["env"]["PERF_TEST_NAME"] == "custom"


def test_real_sdist_and_wheel_selection_exclude_internal_presets(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    public = root / "rtp_llm/test/perf_test/perf_presets.json"
    public.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "rtp_llm/test/perf_test/perf_presets.json", public)
    (root / "rtp_llm/__init__.py").write_text("")
    overlay = tmp_path / "private-overlay"
    overlay.mkdir()
    (root / "internal_source").symlink_to(overlay, target_is_directory=True)
    private = root / "internal_source/rtp_llm/test/perf_test/perf_presets.json"
    private.parent.mkdir(parents=True)
    private.write_text('{"internal": {"private_fixture": "INTERNAL_PRESET_SENTINEL"}}')
    (root / "internal_source/__init__.py").write_text("")
    (root / "_build").mkdir()
    shutil.copyfile(
        ROOT / "_build/sdist_sources.txt", root / "_build/sdist_sources.txt"
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    packages = find_namespace_packages(
        where=str(root), include=config["packages"]["find"]["include"]
    )
    assert not any(p.startswith("internal_source") for p in packages)
    spec = importlib.util.spec_from_file_location(
        "privacy_sdist_setup", ROOT / "setup.py"
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    monkeypatch.setattr(setup, "get_project_root", lambda: root)
    monkeypatch.chdir(root)
    distribution = Distribution(
        {
            "name": "privacy-package-fixture",
            "version": "0",
            "packages": packages,
            "package_dir": {"": "."},
            "package_data": config["package-data"],
            "exclude_package_data": config["exclude-package-data"],
        }
    )
    distribution.script_name = "setup.py"
    selected = setup.SdistWithProto(distribution)._source_files(
        [str(public.relative_to(root)), str(private.relative_to(root))]
    )
    assert str(public.relative_to(root)) in selected
    assert str(private.relative_to(root)) not in selected
    # Real bdist_wheel/build_py, with no Native compiler or fake extension.
    wheel = distribution.get_command_obj("bdist_wheel")
    wheel.dist_dir = str(tmp_path / "wheels")
    distribution.run_command("bdist_wheel")
    output = next((tmp_path / "wheels").glob("*.whl"))
    with zipfile.ZipFile(output) as archive:
        paths = archive.namelist()
        assert str(public.relative_to(root)) in paths
        assert not any("internal_source" in name for name in paths)
        assert not any(
            b"INTERNAL_PRESET_SENTINEL" in archive.read(name) for name in paths
        )
        assert json.loads(archive.read(str(public.relative_to(root)))) == json.loads(
            public.read_text()
        )


def test_public_and_internal_cas_preset_selection(tmp_path, monkeypatch):
    from rtp_llm.test.remote_tests import remote_exec_rtp as remote

    public = tmp_path / "rtp_llm/test/perf_test/perf_presets.json"
    private = tmp_path / "internal_source/rtp_llm/test/perf_test/perf_presets.json"
    for path in [public, private]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setattr(remote, "_collect_base_files", lambda root: [])
    monkeypatch.setattr(remote, "_prepare_ppu_runtime_libs", lambda root, items: [])
    # Native library readiness is independent of this data-boundary fixture.
    # Do not fabricate .so files to exercise session manifest selection.
    monkeypatch.setattr(
        remote, "_prepare_runtime_libs_archive", lambda root: "runtime-fixture.tar"
    )

    class Item:
        def __init__(self, path):
            self.fspath = path

        def get_closest_marker(self, name):
            return object() if name == "perf" else None

    public_item = Item(tmp_path / "rtp_llm/test/perf_test/test_perf_dsv4_kernels.py")
    internal_item = Item(
        tmp_path / "internal_source/rtp_llm/test/perf_test/test_perf_dsv4.py"
    )
    assert str(private.relative_to(tmp_path)) not in remote._collect_repo_runtime_files(
        tmp_path, include_libs=False
    )
    assert str(private.relative_to(tmp_path)) not in remote.collect_remote_files(
        tmp_path, [public_item]
    )
    assert str(private.relative_to(tmp_path)) not in remote.collect_session_files(
        tmp_path
    )
    assert str(private.relative_to(tmp_path)) in remote.collect_remote_files(
        tmp_path, [internal_item]
    )
    private.unlink()
    with pytest.raises(FileNotFoundError, match="registry is missing"):
        remote.collect_remote_files(tmp_path, [internal_item])
