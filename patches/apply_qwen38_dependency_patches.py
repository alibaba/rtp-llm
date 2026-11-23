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
import tempfile
from pathlib import Path


AITER_CONFIG = Path("aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_gemm.csv")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_aiter(manifest_path: Path, verify_only: bool) -> None:
    manifest = json.loads(manifest_path.read_text())
    patch = manifest_path.with_suffix(".patch")
    if sha256(patch.read_bytes()) != manifest["aiter_patch_sha256"]:
        raise RuntimeError("AITER source patch differs from the reviewed manifest")
    version = importlib.metadata.version("amd-aiter")
    if version != manifest["aiter_version"]:
        raise RuntimeError(f"Unexpected amd-aiter version: {version}")
    library = Path("/opt/rocm/lib/libhipblaslt.so.1").resolve().name
    if library != manifest["hipblaslt_library"]:
        raise RuntimeError(f"Tuned solutions require {manifest['hipblaslt_library']}; found {library}")
    package = importlib.metadata.distribution("amd-aiter")
    destination = Path(package.locate_file(AITER_CONFIG))
    original = destination.read_bytes()
    digest = sha256(original)
    if digest == manifest["patched_sha256"]:
        print(f"AITER BF16 patch present: {destination}", flush=True)
        return
    if verify_only or digest != manifest["source_sha256"]:
        raise RuntimeError(f"AITER BF16 config has an unexpected SHA-256: {digest}")

    lines = original.decode("utf-8").splitlines(keepends=True)
    columns = {name: index for index, name in enumerate(lines[0].rstrip("\r\n").split(","))}
    solutions = manifest["solutions"]
    seen = set()
    patched = [lines[0]]
    for line in lines[1:]:
        ending = "\r\n" if line.endswith("\r\n") else "\n"
        fields = line.rstrip("\r\n").split(",")
        if (
            fields[columns["gfx"]] == "gfx942"
            and fields[columns["cu_num"]] == "80"
            and fields[columns["libtype"]] == "hipblaslt"
        ):
            key = ",".join(fields[columns[name]] for name in ("M", "N", "K"))
            if key not in solutions or key in seen:
                raise RuntimeError(f"Unexpected or duplicate AITER GEMM shape: {key}")
            fields[columns["solidx"]] = str(solutions[key])
            line = ",".join(fields) + ending
            seen.add(key)
        patched.append(line)
    if seen != set(solutions) or len(seen) != 74:
        raise RuntimeError(f"AITER BF16 patch covered {len(seen)} of 74 shapes")
    payload = "".join(patched).encode("utf-8")
    if sha256(payload) != manifest["patched_sha256"]:
        raise RuntimeError("AITER BF16 patch output hash differs from the reviewed artifact")
    # The manifest records each reviewed solution; the shipped unified diff is
    # what actually changes the installed AITER wheel.
    with tempfile.TemporaryDirectory(prefix="qwen38_aiter_patch_") as name:
        temporary = Path(name) / AITER_CONFIG
        temporary.parent.mkdir(parents=True)
        temporary.write_bytes(original)
        apply_git(patch, Path(name))
        if temporary.read_bytes() != payload:
            raise RuntimeError("AITER source patch and solution manifest disagree")
    apply_git(patch, Path(package.locate_file("")))
    if sha256(destination.read_bytes()) != manifest["patched_sha256"]:
        raise RuntimeError("Installed AITER source patch differs from reviewed output")
    print(f"Applied AITER BF16 patch: {destination} ({len(seen)} shapes)", flush=True)


def apply_git(patch: Path, directory: Path) -> None:
    command = ["git", "apply"]
    command.append(str(patch))
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")


def patch_aiter_gdn(manifest_path: Path, verify_only: bool,
                    key_prefix: str = "aiter_gdn",
                    patch_name: str = "qwen38_flydsl_gdn_mtp.patch") -> None:
    manifest = json.loads(manifest_path.read_text())
    patch = manifest_path.parent / patch_name
    if sha256(patch.read_bytes()) != manifest[key_prefix + "_patch_sha256"]:
        raise RuntimeError("AITER GDN patch differs from the reviewed manifest")
    if importlib.metadata.version("amd-aiter") != manifest["aiter_version"]:
        raise RuntimeError("Unexpected AITER version for GDN patch")
    root = Path(importlib.metadata.distribution("amd-aiter").locate_file(""))
    source = manifest[key_prefix + "_source_files_sha256"]
    expected = manifest[key_prefix + "_patched_files_sha256"]
    paths = []
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            fields = line.split()
            if len(fields) != 4 or not fields[2].startswith("a/aiter/") or fields[3] != "b/" + fields[2][2:]:
                raise RuntimeError("Unexpected AITER GDN patch path")
            paths.append(fields[2][2:])
    if len(paths) != len(set(paths)) or set(paths) != set(source) or set(paths) != set(expected):
        raise RuntimeError("AITER GDN patch file list differs from manifest")
    current = {name: sha256((root / name).read_bytes()) for name in paths}
    if current == expected:
        print(f"AITER patch present: {patch.name}", flush=True)
        return
    if current != source or verify_only:
        raise RuntimeError("AITER GDN source is unreviewed or patch is missing")
    apply_git(patch, root)
    if {name: sha256((root / name).read_bytes()) for name in paths} != expected:
        raise RuntimeError("Installed AITER GDN patch differs from reviewed output")
    print(f"Applied AITER patch: {patch.name}", flush=True)


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
    patch_aiter(args.manifest, args.verify_only)
    patch_aiter_gdn(args.manifest, args.verify_only)
    patch_aiter_gdn(args.manifest, args.verify_only, "aiter_compat", "qwen38_python310_compat.patch")
    patch_aiter_gdn(args.manifest, args.verify_only, "aiter_gemm", "qwen38_gfx942_decode_gemm.patch")


if __name__ == "__main__":
    main()
