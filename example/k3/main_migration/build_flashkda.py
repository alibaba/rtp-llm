#!/usr/bin/env python3
"""Build the pinned K3 FlashKDA backend from verified source archives.

Run inside the local build container with its matching PyTorch/CUDA environment.
The resulting directory is added to PYTHONPATH when selecting the K3 backend.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

FLASHKDA_REVISION = "b59532f1f464fbd536272780e30df5bf6a2ccc02"
CUTLASS_REVISION = "5c149f52a436782210263fb2f19b354443a61c6a"
ARCHIVES = {
    "flashkda": "973f6524b5f046886721f4468891850908b880df23ac40d5c9cd1576a53f511b",
    "cutlass": "ed237bc98c3028f1d7044c95d18c0ea09d10d6445b9f5f5b666d5765e08f39b3",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def unpack(archive, destination):
    with tarfile.open(archive) as reader:
        for member in reader.getmembers():
            resolved = (destination / member.name).resolve()
            if destination != resolved and destination not in resolved.parents:
                raise ValueError(f"Archive member escapes source root: {member.name}")
            if member.issym() or member.islnk():
                linked = (resolved.parent / member.linkname).resolve()
                if destination != linked and destination not in linked.parents:
                    raise ValueError(f"Archive link escapes source root: {member.name}")
        reader.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--flashkda-archive", type=Path, required=True)
    parser.add_argument("--cutlass-archive", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "patches/flashkda_fp32_recurrence.patch",
    )
    args = parser.parse_args()
    root = args.source_root.resolve()
    if root.exists():
        raise ValueError("Use a new source directory; existing builds are preserved")
    fs = subprocess.check_output(
        ["findmnt", "-T", str(root.parent), "-n", "-o", "FSTYPE"], text=True
    ).strip()
    if fs not in {"ext4", "xfs", "btrfs"}:
        raise ValueError(f"Build requires a local data filesystem, found {fs}")
    archives = {
        "flashkda": args.flashkda_archive.resolve(),
        "cutlass": args.cutlass_archive.resolve(),
    }
    for name, archive in archives.items():
        if digest(archive) != ARCHIVES[name]:
            raise ValueError(f"{name} source archive SHA256 mismatch")
    patch = args.patch.resolve()
    if not patch.is_file():
        raise FileNotFoundError(patch)
    root.mkdir()
    unpack(archives["flashkda"], root)
    (root / "cutlass").mkdir(exist_ok=True)
    unpack(archives["cutlass"], root / "cutlass")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        CUDA_HOME="/usr/local/cuda-13",
        FLASH_KDA_CUDA_ARCHS="103a",
        MAX_JOBS="2",
        NVCC_THREADS="4",
        TMPDIR=str(root / "build-tmp"),
    )
    Path(env["TMPDIR"]).mkdir()
    provenance = {
        "flashkda": FLASHKDA_REVISION,
        "cutlass": CUTLASS_REVISION,
        "source_archive_sha256": ARCHIVES,
        "patch_sha256": digest(patch),
        "python": sys.version,
        "source_root": str(root),
        "filesystem": fs,
        "cuda_arch": "103a",
        "cuda_home": env["CUDA_HOME"],
        "user": subprocess.check_output(["id", "-un"], text=True).strip(),
    }
    import torch

    provenance["torch"] = torch.__version__
    (root / "build-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import flash_kda, torch; assert torch.ops.flash_kda.supports_fp32_recurrence(); "
            'print("FlashKDA compiled FP32 recurrence capability verified")',
        ],
        cwd=root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
