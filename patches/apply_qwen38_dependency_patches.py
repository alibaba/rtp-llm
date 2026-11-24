"""Apply/verify the pinned Qwen3.8 ATOM and MI308X AITER install patches.

Run after installing the ROCm RTP, ATOM and AITER wheels, before importing
either package in the image. This is CPU-only so it also works in docker build.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_aiter_patch(root, directory, manifest, key, filename, verify_only):
    patch = directory / filename
    if sha256(patch.read_bytes()) != manifest[key + "_patch_sha256"]:
        raise RuntimeError(f"Unexpected patch digest: {filename}")
    before = manifest[key + "_source_files_sha256"]
    after = manifest[key + "_patched_files_sha256"]

    def file_hashes():
        return {path: sha256((root / path).read_bytes()) for path in before}

    current = file_hashes()
    if current == after:
        return
    if verify_only or current != before:
        raise RuntimeError(f"Unexpected installed sources for {filename}")
    subprocess.run(["git", f"--work-tree={root}", "apply", str(patch.resolve())], cwd=root, check=True)
    if file_hashes() != after:
        raise RuntimeError(f"Unexpected output after applying {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "aiter/qwen38_aiter_hipblaslt_70200.json")
    parser.add_argument("--atom-main-dir", type=Path, required=True)
    parser.add_argument("--flydsl-wheel", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    flydsl_wheel = args.flydsl_wheel or args.manifest.parent.parent / "wheels/flydsl-0.4.2-cp310-cp310-manylinux_2_27_x86_64.whl"
    if sha256(flydsl_wheel.read_bytes()) != manifest["flydsl_wheel_sha256"]:
        raise RuntimeError("FlyDSL wheel differs from the reviewed manifest")
    if importlib.metadata.version("flydsl") != manifest["flydsl_version"]:
        raise RuntimeError("FlyDSL version differs from the reviewed manifest")
    command = [sys.executable, str(args.atom_main_dir / "apply.py")]
    if args.verify_only:
        command.append("--verify-only")
    subprocess.run(command, check=True)
    package = importlib.metadata.distribution("amd-aiter")
    if package.version != manifest["aiter_version"]:
        raise RuntimeError(f"Unexpected amd-aiter version: {package.version}")
    library = Path("/opt/rocm/lib/libhipblaslt.so.1").resolve().name
    if library != manifest["hipblaslt_library"]:
        raise RuntimeError(f"Tuned solutions require {manifest['hipblaslt_library']}; found {library}")
    for key, filename in (
        ("aiter_bf16", "qwen38_aiter_hipblaslt_70200.patch"),
        ("aiter_gdn", "qwen38_flydsl_gdn_mtp.patch"),
        ("aiter_compat", "qwen38_python310_compat.patch"),
        ("aiter_gemm", "qwen38_gfx942_decode_gemm.patch"),
    ):
        apply_aiter_patch(Path(package.locate_file("")), args.manifest.parent, manifest, key, filename, args.verify_only)


if __name__ == "__main__":
    main()
