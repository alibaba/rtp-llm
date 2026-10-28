"""Build pinned MORI wheels on the CI controller for remote venv installation."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from email.parser import BytesParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from wheel.wheelfile import WheelFile

WHEELHOUSE = Path(".pytest_cache/remote_inputs/mori_wheel")
MANIFEST = "manifest.json"


def source_requirement(root: Path) -> str:
    spec = importlib.util.spec_from_file_location("rtp_wheel_setup", root / "setup.py")
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        requirements = module.get_merged_optional_dependencies().get("rocm", [])
    matches = [
        Requirement(value)
        for value in requirements
        if canonicalize_name(Requirement(value).name) == "amd-mori"
    ]
    if len(matches) != 1 or not matches[0].url:
        raise RuntimeError("ROCm must declare exactly one pinned MORI source")
    requirement = matches[0]
    if not re.fullmatch(r"git\+https://[^\s]+@[0-9a-f]{40}", requirement.url):
        raise RuntimeError("MORI source must be pinned to a full Git commit")
    return str(requirement)


def wheel_version(path: Path) -> str:
    name, version, _, tags = parse_wheel_filename(path.name)
    if name != "amd-mori" or not tags.intersection(sys_tags()):
        raise RuntimeError(
            f"MORI wheel is incompatible with this Python/platform: {path.name}"
        )
    with WheelFile(path) as archive:
        metadata = BytesParser().parsebytes(
            archive.read(f"{archive.dist_info_path}/METADATA")
        )
        if canonicalize_name(metadata["Name"]) != name or metadata["Version"] != str(
            version
        ):
            raise RuntimeError("MORI wheel filename and metadata disagree")
        if "mori/libmori_pybinds.so" not in archive.namelist() or not archive.read(
            "mori/libmori_pybinds.so"
        ):
            raise RuntimeError("MORI wheel has no native bindings")
        # WheelFile verifies RECORD hashes while reading the payload.
        for entry in archive.infolist():
            if not entry.is_dir():
                archive.read(entry.filename)
    return str(version)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validated_wheel(root: Path, requirement: str) -> Path:
    directory = root / WHEELHOUSE
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["requirement"] != requirement:
        raise RuntimeError("MORI wheel source differs from the current dependency pin")
    filename = manifest["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError("Invalid MORI wheel filename in manifest")
    wheel = directory / filename
    if digest(wheel) != manifest["sha256"]:
        raise RuntimeError("MORI wheel SHA256 does not match the manifest")
    if wheel_version(wheel) != manifest["version"]:
        raise RuntimeError("MORI wheel version does not match the manifest")
    return wheel


def build(root: Path, requirement: str) -> Path:
    directory = root / WHEELHOUSE
    directory.mkdir(parents=True, exist_ok=True)
    # A completed, matching bundle can be reused; a partial one is never consumed.
    if (directory / MANIFEST).exists():
        return validated_wheel(root, requirement)
    with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as temporary:
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                temporary,
                requirement,
            ],
            cwd=root,
            env=env,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=True,
        )
        wheels = list(Path(temporary).glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("MORI build must produce exactly one wheel")
        wheel = wheels[0]
        version = wheel_version(wheel)
        manifest = {
            "requirement": requirement,
            "filename": wheel.name,
            "version": version,
            "sha256": digest(wheel),
        }
        destination = directory / wheel.name
        wheel.replace(destination)
        manifest_path = Path(temporary) / MANIFEST
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.replace(directory / MANIFEST)
    return destination


def overrides(root: Path, requirement: str) -> Path:
    wheel = validated_wheel(root, requirement)
    path = root / WHEELHOUSE / "overrides.txt"
    path.write_text(f"amd-mori @ {wheel.resolve().as_uri()}\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "overrides"))
    args = parser.parse_args()
    root = Path.cwd()
    requirement = source_requirement(root)
    result = (
        build(root, requirement)
        if args.action == "build"
        else overrides(root, requirement)
    )
    print(result)


if __name__ == "__main__":
    main()
