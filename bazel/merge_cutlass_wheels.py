"""Assemble the locked CUTLASS wheels into one Bazel installation directory."""

import argparse
import hashlib
import re
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

CUTLASS_DISTRIBUTIONS = {
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
}


def _distribution(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata = [
            name
            for name in archive.namelist()
            if len(PurePosixPath(name).parts) == 2
            and name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ValueError(f"Expected one distribution metadata file: {wheel}")
        headers = BytesParser().parsebytes(archive.read(metadata[0]))
    name = re.sub(r"[-_.]+", "-", headers.get("Name", "")).lower()
    version = headers.get("Version", "")
    if name not in CUTLASS_DISTRIBUTIONS or not version:
        raise ValueError(f"Unexpected CUTLASS distribution: {name!r} in {wheel}")
    return name, version


def merge_wheels(wheels: list[Path], output: Path, version: str) -> None:
    distributions = {}
    for wheel in wheels:
        name, actual = _distribution(wheel)
        if actual != version:
            raise ValueError(f"{name}: expected {version}, got {actual}")
        if name in distributions:
            raise ValueError(f"Duplicate CUTLASS distribution: {name}")
        distributions[name] = wheel
    if set(distributions) != CUTLASS_DISTRIBUTIONS:
        missing = sorted(CUTLASS_DISTRIBUTIONS.difference(distributions))
        raise ValueError(f"Missing CUTLASS distributions: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"CUTLASS output directory is not empty: {output}")
    digests = {}
    for name, wheel in sorted(distributions.items()):
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member.filename
                    or not relative.parts
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe CUTLASS wheel member: {member.filename}")
                if member.is_dir():
                    continue
                target = output.joinpath(*relative.parts)
                digest = hashlib.sha256()
                with archive.open(member) as source:
                    if relative in digests:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                        if digest.digest() != digests[relative]:
                            raise ValueError(
                                f"Conflicting CUTLASS wheel payload: {relative} ({name})"
                            )
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as destination:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            destination.write(chunk)
                digests[relative] = digest.digest()
                target.chmod(0o755 if member.external_attr >> 16 & 0o111 else 0o644)

    package = output / "nvidia_cutlass_dsl" / "dsl_packages" / "cutlass"
    if not (package / "__init__.py").is_file() or not (package / "_mlir").is_dir():
        raise ValueError("Assembled CUTLASS installation is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    merge_wheels(args.wheels, args.output, args.version)


if __name__ == "__main__":
    main()
