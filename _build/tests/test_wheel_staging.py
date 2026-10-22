"""Exercise setuptools packaging with explicit build-output fixtures, not Native execution."""

import importlib.util
import shutil
import stat
from pathlib import Path
from zipfile import ZipFile

import pytest
from setuptools import Distribution

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
TOKENIZERS = (
    "test/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken",
    "test/model_test/fake_test/testdata/llama/fake/hf_source/tokenizer.model",
)


def load_setup():
    spec = importlib.util.spec_from_file_location("_wheel_staging_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config", ["cuda12_6", "cuda12_9", "cuda13", "rocm", "ppu"])
@pytest.mark.parametrize("stale", [False, True], ids=["cold", "previous-build"])
def test_first_wheel_contains_all_fresh_selected_outputs(
    tmp_path, monkeypatch, config, stale
):
    setup = load_setup()
    # Preserve the original complete platform output list. Only the costly
    # compiler boundary is a fixture; these bytes are never imported as .so.
    selected = setup._selected_bazel_staged_outputs(config, ["--config=" + config])
    project = tmp_path / "project"
    package = project / "rtp_llm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "ops").mkdir()
    (package / "ops" / "__init__.py").write_text("")
    for relative in TOKENIZERS:
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "rtp_llm" / relative, destination)
    expected = {}
    test_only = set()
    for kind, _, specs in selected:
        for spec in specs:
            _, relative = setup._normalize_staged_output(spec)
            key = (
                "rtp_llm/"
                + ("" if kind == setup._STAGED_OUTPUT_PYTHON else "libs/")
                + relative
            )
            expected[key] = ("packaging-fixture:" + config + ":" + key).encode()
            if kind == setup._STAGED_OUTPUT_TEST or key.startswith("rtp_llm/libs/test/"):
                test_only.add(key)
    build_lib = tmp_path / "build-lib"
    if stale:
        for root in [package / "libs", build_lib / "rtp_llm/libs"]:
            root.mkdir(parents=True)
            (root / "obsolete.so").write_bytes(b"previous-platform")
            (root / "libth_transformer_config.so").write_bytes(b"previous-version")
        (package / "ops" / "libth_transformer_config.pyi").write_text("OLD_STUB")

    builds = []

    def compiler_fixture():
        builds.append(config)
        setup._wipe_rtp_llm_libs(project)
        for relative, contents in expected.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            path.chmod(0o755)
        (package / "ops" / "libth_transformer_config.pyi").write_text("FRESH_STUB")

    monkeypatch.chdir(project)
    monkeypatch.setattr(setup, "get_project_root", lambda: project)
    monkeypatch.setattr(setup, "detect_build_config", lambda: config)
    monkeypatch.setattr(setup, "should_skip_bazel_build", lambda: False)
    monkeypatch.setattr(setup, "check_build_prerequisites", lambda: None)
    monkeypatch.setattr(setup, "ensure_proto_files_generated", lambda root: None)
    monkeypatch.setattr(setup, "build_all", compiler_fixture)
    monkeypatch.setattr(setup, "_selected_bazel_staged_outputs", lambda config: selected)
    setuptools_config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "setuptools"
    ]
    distribution = Distribution(
        {
            "name": "rtp-wheel-lifecycle-fixture",
            "version": "0",
            "packages": ["rtp_llm", "rtp_llm.ops"],
            "package_data": setuptools_config["package-data"],
            "exclude_package_data": setuptools_config["exclude-package-data"],
            "include_package_data": True,
            "ext_modules": [setup.BazelExtension()],
            "cmdclass": setup.cmdclass,
        }
    )
    distribution.script_name = "setup.py"
    build = distribution.get_command_obj("build")
    build.build_base = str(tmp_path / "build")
    build.build_lib = str(build_lib)
    # PEP517 metadata generation can finalize build_py before compilation.
    build_py = distribution.get_command_obj("build_py")
    build_py.ensure_finalized()
    assert build_py.data_files is not None
    wheel = distribution.get_command_obj("bdist_wheel")
    wheel.dist_dir = str(tmp_path / "dist")
    distribution.run_command("bdist_wheel")
    assert builds == [config]
    archive = next((tmp_path / "dist").glob("*.whl"))
    with ZipFile(archive) as result:
        assert "rtp_llm/libs/obsolete.so" not in result.namelist()
        for relative, contents in expected.items():
            # Complete test outputs are still staged for remote-session tests,
            # but the original release-wheel exclusion must remain effective.
            assert (project / relative).read_bytes() == contents
            if relative in test_only:
                assert relative not in result.namelist()
                continue
            assert result.read(relative) == contents, relative
            assert result.getinfo(relative).external_attr >> 16 & stat.S_IXUSR
        assert result.read("rtp_llm/ops/libth_transformer_config.pyi") == b"FRESH_STUB"
        for relative in TOKENIZERS:
            assert result.read("rtp_llm/" + relative) == (
                ROOT / "rtp_llm" / relative
            ).read_bytes()


def test_missing_staged_output_fails_before_removing_existing_build_lib(tmp_path):
    setup = load_setup()
    source = tmp_path / "source"
    source.mkdir()
    build_lib = tmp_path / "build"
    previous = build_lib / "rtp_llm/libs/previous.so"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="Missing staged wheel output"):
        setup.copy_staged_outputs_to_build_lib(
            source, build_lib, [("core", "//:missing", ("missing.so",))]
        )
    assert previous.read_bytes() == b"previous"


def test_wheel_staging_cannot_remove_source_libraries(tmp_path):
    setup = load_setup()
    with pytest.raises(ValueError, match="separate from the source"):
        setup.copy_staged_outputs_to_build_lib(tmp_path, tmp_path, [])
