import json
from pathlib import Path
import shutil
import subprocess
import sys

from packaging.tags import sys_tags
import pytest
from wheel.wheelfile import WheelFile

from _build import remote_wheels

PIN = "amd-mori @ git+https://github.com/ROCm/mori.git@" + "a" * 40


def make_wheel(directory, *, bindings=True, tag=None):
    directory.mkdir(parents=True, exist_ok=True)
    tag = tag or str(next(sys_tags()))
    path = directory / f"amd_mori-1.2.2-{tag}.whl"
    info = "amd_mori-1.2.2.dist-info"
    with WheelFile(path, "w") as archive:
        archive.writestr("mori/__init__.py", 'SOURCE = "packaging-test-fixture"\n')
        if bindings:
            # This is a packaging fixture, not an executable GPU extension.
            archive.writestr("mori/libmori_pybinds.so", b"fixture-native-payload")
        archive.writestr(
            f"{info}/METADATA",
            "Metadata-Version: 2.1\nName: amd-mori\nVersion: 1.2.2\n",
        )
        archive.writestr(
            f"{info}/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
        )
    return path


def bundle(root, **kwargs):
    wheel = make_wheel(root / remote_wheels.WHEELHOUSE, **kwargs)
    manifest = {
        "requirement": PIN,
        "filename": wheel.name,
        "version": "1.2.2",
        "sha256": remote_wheels.digest(wheel),
    }
    (wheel.parent / remote_wheels.MANIFEST).write_text(json.dumps(manifest))
    return wheel


def test_override_installs_pinned_git_requirement_offline(tmp_path):
    wheel = bundle(tmp_path)
    override = remote_wheels.overrides(tmp_path, PIN)
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    uv = Path(sys.executable).parent / "uv"
    command = [
        str(uv),
        "pip",
        "install",
        "--python",
        str(venv / "bin/python"),
        "--offline",
        "--no-cache",
        PIN,
    ]
    # The source requirement alone cannot resolve in an empty offline cache.
    without = subprocess.run(command, capture_output=True, text=True)
    assert without.returncode != 0
    installed = subprocess.run(
        [*command, "--override", str(override)], capture_output=True, text=True
    )
    assert installed.returncode == 0, installed.stderr
    output = subprocess.check_output(
        [
            str(venv / "bin/python"),
            "-c",
            "import mori; from importlib.metadata import distribution; "
            "assert mori.SOURCE == 'packaging-test-fixture'; "
            "print(distribution('amd-mori').read_text('direct_url.json'))",
        ],
        text=True,
    )
    assert json.loads(output)["url"] == wheel.resolve().as_uri()


@pytest.mark.parametrize("failure", ["missing", "hash", "pin", "version", "traversal"])
def test_invalid_bundle_fails_before_override(tmp_path, failure):
    wheel = bundle(tmp_path)
    manifest_path = wheel.parent / remote_wheels.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if failure == "missing":
        wheel.unlink()
    elif failure == "hash":
        wheel.write_bytes(wheel.read_bytes() + b"corrupt")
    elif failure == "pin":
        manifest["requirement"] = PIN.replace("a" * 40, "b" * 40)
    elif failure == "version":
        manifest["version"] = "0.0.0"
    else:
        manifest["filename"] = "../" + wheel.name
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        remote_wheels.overrides(tmp_path, PIN)
    assert not (wheel.parent / "overrides.txt").exists()


@pytest.mark.parametrize(
    "options", [{"bindings": False}, {"tag": "cp39-cp39-nonexistent_platform"}]
)
def test_unusable_wheel_is_rejected(tmp_path, options):
    bundle(tmp_path, **options)
    with pytest.raises((RuntimeError, KeyError)):
        remote_wheels.overrides(tmp_path, PIN)


def test_build_publishes_validated_bundle_without_loader_path(tmp_path, monkeypatch):
    fixture = make_wheel(tmp_path / "built")
    calls = []

    def pip_wheel(command, **kwargs):
        calls.append(command)
        assert command[-1] == PIN
        assert "--no-build-isolation" in command
        assert "LD_LIBRARY_PATH" not in kwargs["env"]
        destination = Path(command[command.index("--wheel-dir") + 1])
        shutil.copyfile(fixture, destination / fixture.name)

    monkeypatch.setenv("LD_LIBRARY_PATH", "/controller/conda/lib")
    monkeypatch.setattr(remote_wheels.subprocess, "run", pip_wheel)
    built = remote_wheels.build(tmp_path, PIN)
    assert remote_wheels.validated_wheel(tmp_path, PIN) == built
    assert remote_wheels.build(tmp_path, PIN) == built
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["session", "per_test"])
def test_remote_inputs_include_wheel_but_not_absolute_override(tmp_path, mode):
    from rtp_llm.test.remote_tests import remote_exec_rtp

    wheel = bundle(tmp_path)
    override = remote_wheels.overrides(tmp_path, PIN)
    helper = tmp_path / "_build/remote_wheels.py"
    helper.parent.mkdir()
    shutil.copyfile(remote_wheels.__file__, helper)
    if mode == "session":
        libs = tmp_path / "rtp_llm/libs"
        libs.mkdir(parents=True)
        for name in remote_exec_rtp._CORE_RUNTIME_LIBS:
            (libs / name).write_bytes(b"fixture-runtime-payload")
        files = remote_exec_rtp.collect_session_files(tmp_path)
    else:
        files = remote_exec_rtp.collect_remote_files(tmp_path, [])
    assert str(wheel.relative_to(tmp_path)) in files
    assert str((wheel.parent / remote_wheels.MANIFEST).relative_to(tmp_path)) in files
    assert str(helper.relative_to(tmp_path)) in files
    assert str(override.relative_to(tmp_path)) not in files
