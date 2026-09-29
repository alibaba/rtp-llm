"""Build the pinned K3 expert backend with the calling RTP Torch runtime.

Run inside the approved build container. Output is a private PYTHONPATH
package; the installed deep_gemm module and other models are not modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import urllib.request


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            for name in (member.name, member.linkname):
                if name and (Path(name).is_absolute() or ".." in Path(name).parts):
                    raise ValueError(f"Unsafe archive entry: {name}")
            if member.isdev() or member.isfifo():
                raise ValueError(f"Unexpected archive entry: {member.name}")
        roots = {Path(member.name).parts[0] for member in tar.getmembers()}
        if len(roots) != 1:
            raise ValueError("Expected one source root")
        tar.extractall(destination)
    return destination / roots.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--extra-include", action="append", default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    lock = json.loads((here / "UPSTREAM.json").read_text())
    patch = here / "graph_replay_epoch.patch"
    helper = here / "build_extension.py"
    if digest(patch) != lock["patch_sha256"] or digest(helper) != lock["build_helper_sha256"]:
        raise RuntimeError("Build input checksum mismatch")
    for path in (args.archives, args.work_dir, args.output_dir):
        resolved = path.resolve()
        if resolved == Path("/3fs-data/3fs/mtp_test") or Path("/3fs-data/3fs/mtp_test") in resolved.parents:
            raise ValueError("Protected destination")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = []
    for record in lock["archives"]:
        archive = args.archives / record["file"]
        if not archive.exists() and args.download:
            args.archives.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(record["url"], timeout=120) as response:
                archive.write_bytes(response.read())
        if digest(archive) != record["sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {archive}")
        sources.append(unpack(archive, args.work_dir))
    src = sources[0]
    for dependency, name in zip(sources[1:], ("cutlass", "deep_jit")):
        dest = src / "third-party" / name
        if dest.exists():
            dest.rmdir()
        dependency.rename(dest)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=src, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=src, check=True)
    package = args.output_dir / "k3_native_deep_gemm"
    shutil.copytree(src / "deep_gemm", package)
    shutil.copytree(src / "third-party/cutlass/include", package / "include", dirs_exist_ok=True)
    for license_file in src.glob("LICENSE*"):
        shutil.copy2(license_file, package / license_file.name)
    os.environ["CXX"] = args.cxx
    if args.extra_include:
        os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(args.extra_include + [os.environ.get("CPLUS_INCLUDE_PATH", "")])
    import torch
    sys.argv = [str(helper), str(src), str(package), sys.executable]
    runpy.run_path(str(helper), run_name="__main__")
    binary = next(package.glob("_C*.so"))
    runtime = subprocess.check_output([args.cxx, "-print-file-name=libstdc++.so.6"], text=True).strip()
    manifest = {**lock, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
                "binary_sha256": digest(binary), "python": sys.version,
                "runtime_libstdcxx": str(Path(runtime).resolve())}
    (package / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "launch_env.json").write_text(json.dumps({
        "PYTHONPATH": str(args.output_dir.resolve()),
        "LD_PRELOAD": str(Path(runtime).resolve()),
        "KIMI_K3_MOE_BACKEND": "vllm_native",
    }, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
