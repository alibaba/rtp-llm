#!/usr/bin/env python3
"""Build isolated GLM53 SM100 extensions from pinned public sources.

No GPU or model checkpoint is needed for compilation. The deploy target is
SM100 (verified with CUDA Runtime), irrespective of the build host's GPU.
Use --wheel to add the shared libraries, JIT headers, licenses and manifest
to a previously built RTP wheel, rebuilding its RECORD atomically.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "rtp_llm/libs/"
MANIFEST = PREFIX + "glm53_extensions.json"
SOURCE_FILES = [
    "csrc/api/glm53_only.cpp",
    "csrc/api/sparse_decode.cpp",
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def configure_compiler():
    sdk = Path("/opt/rh/gcc-toolset-12/root/usr/bin/g++")
    compiler = (
        os.environ.get("CXX")
        or (str(sdk) if sdk.is_file() else shutil.which("g++-12"))
        or shutil.which("g++")
    )
    if not compiler:
        raise RuntimeError("GCC 12 or newer is required")
    version = subprocess.check_output([compiler, "-dumpfullversion"], text=True).strip()
    if int(version.split(".")[0]) < 12:
        raise RuntimeError(f"GCC >=12 required for CUDA C++20 headers; found {version}")
    cc = Path(compiler).with_name(Path(compiler).name.replace("g++", "gcc"))
    if not cc.is_file():
        raise RuntimeError(f"Matching C compiler is required: {cc}")
    os.environ.update(CXX=compiler, CC=str(cc))
    private_lib = Path(compiler).parent.parent / "lib64"
    if private_lib.is_dir():
        os.environ["LD_LIBRARY_PATH"] = (
            str(private_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )
    return compiler


def unpack(archive, destination):
    """Reject paths/links that could escape the task's source directory."""
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            parts = Path(member.name).parts
            if not parts or Path(member.name).is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe source member: {member.name}")
            if member.issym() or member.islnk():
                target = Path(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(f"Unsafe source link: {member.name}")
        source.extractall(destination)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError(f"Expected one source root: {archive}")
    return roots[0]


def prepare(work):
    pins = json.loads((HERE / "sources.json").read_text())
    cache = work / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, info in pins.items():
        print("Preparing pinned source:", name, flush=True)
        archive = cache / (name + ".tar.gz")
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(info["url"], timeout=120) as response:
                        with temporary.open("wb") as output:
                            shutil.copyfileobj(response, output)
                    if sha256(temporary) != info["sha256"]:
                        raise ValueError(f"Source checksum mismatch: {name}")
                    temporary.replace(archive)
                    break
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
        if sha256(archive) != info["sha256"]:
            raise ValueError(f"Cached source checksum mismatch: {name}")
        sources[name] = unpack(archive, work / "sources" / name)
    for name, patch in [
        ("deep_gemm", "mega_mhc_graph_barriers.patch"),
        ("deep_jit", "deep_jit_portable_errors.patch"),
        ("flash_mla", "flash_mla_glm53.patch"),
    ]:
        # Git is already required by CI checkout; minimal images lack `patch`.
        run(
            [
                "git",
                "apply",
                "--unsafe-paths",
                "--unidiff-zero",
                "--directory=" + str(sources[name].resolve()),
                HERE / patch,
            ]
        )
    shutil.copyfile(
        HERE / "glm53_only.cpp", sources["flash_mla"] / "csrc/api/glm53_only.cpp"
    )
    cutlass = sources["flash_mla"] / "csrc/3rdparty/cutlass"
    if cutlass.exists():
        cutlass.rmdir()  # The upstream archive only contains an empty gitlink.
    cutlass.symlink_to(sources["cutlass_mla"], target_is_directory=True)
    return sources, pins


def build_mhc(sources, output, cuda):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    root = output / "mega_mhc_isolated"
    root.mkdir()
    shutil.copytree(sources["deep_gemm"] / "csrc", root / "csrc")
    includes = root / "include"
    includes.mkdir()
    for src, relative in [
        (sources["deep_gemm"], "deep_gemm/include/deep_gemm"),
        (sources["deep_jit"], "include/deep_jit"),
        (sources["cutlass_mhc"], "include/cute"),
        (sources["cutlass_mhc"], "include/cutlass"),
        (sources["fmt"], "include/fmt"),
    ]:
        shutil.copytree(src / relative, includes / Path(relative).name)
    (root / "compat").mkdir()
    shutil.copyfile(HERE / "format", root / "compat/format")
    shutil.copyfile(HERE / "binding.cpp", root / "binding.cpp")
    binary = root / ("mhc_isolated" + sysconfig.get_config_var("EXT_SUFFIX"))
    compiler = configure_compiler()
    args = [
        compiler,
        "-shared",
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-Wno-psabi",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi())),
        "-Ddeep_gemm=deep_gemm_mhc_isolated",
        "-Ddeep_jit=deep_jit_mhc_isolated",
    ]
    for path in [
        root / "compat",
        root,
        includes,
        cuda / "include",
        cuda / "include/cccl",
        sysconfig.get_paths()["include"],
        *include_paths(),
    ]:
        args.append("-I" + str(path))
    args.extend([root / "binding.cpp", "-o", binary])
    for path in [*library_paths(), str(cuda / "lib64")]:
        args.append("-L" + path)
    args.extend(
        [
            "-Wl,-Bsymbolic",
            "-Wl,-rpath,$ORIGIN/../../../torch/lib",
            "-ltorch",
            "-ltorch_python",
            "-ltorch_cuda",
            "-lc10",
            "-lc10_cuda",
            "-lcudart",
            "-lcublasLt",
        ]
    )
    run(args)


def build_sparse(sources, output, work, cuda):
    # Keep setuptools/ninja in a subprocess; its command parser must not consume
    # this builder's arguments. The explicit -gencode avoids GPU auto-detection.
    configure_compiler()
    script = work / "build_sparse.py"
    repo = sources["flash_mla"]
    common = [
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-DUSE_CUDA",
        "-DFLASH_MLA_IS_BUILD_ON_CUDA",
        "-DKERUTILS_IS_BUILD_ON_CUDA",
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
        "-Wno-deprecated-declarations",
        "-Dsm100=glm53_sm100_opt22",
        "-Dsmxx=glm53_smxx_opt22",
    ]
    nvcc = common + [
        "--threads",
        "2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-gencode",
        "arch=compute_100a,code=sm_100a",
        "-Xcompiler=-fvisibility=hidden",
    ]
    include = [
        repo / "csrc",
        repo / "csrc/3rdparty/cutlass/include",
        repo / "csrc/3rdparty/cutlass/tools/util/include",
        repo / "csrc/3rdparty/kerutils/include",
        cuda / "include/cccl",
    ]
    parameters = {
        "sources": [str(repo / name) for name in SOURCE_FILES],
        "include_dirs": [str(path) for path in include],
        "extra_compile_args": {"cxx": common + ["-fvisibility=hidden"], "nvcc": nvcc},
        "extra_link_args": [
            "-L" + str(cuda / "lib64/stubs"),
            "-lcuda",
            "-Wl,-Bsymbolic",
        ],
        "py_limited_api": True,
    }
    script.write_text(
        "from setuptools import setup\n"
        "from torch.utils.cpp_extension import CUDAExtension, BuildExtension\n"
        "setup(name='flash_mla_glm53_22',ext_modules=[CUDAExtension('flash_mla_glm53_22',**"
        + repr(parameters)
        + ")],cmdclass={'build_ext':BuildExtension})\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCH_CUDA_ARCH_LIST="10.0a")
    env.setdefault("MAX_JOBS", "4")
    run(
        [
            sys.executable,
            script,
            "build_ext",
            "--build-temp",
            work / "objects",
            "--build-lib",
            output,
        ],
        cwd=work,
        env=env,
    )


def stage_licenses(sources, output):
    root = output / "mega_mhc_isolated/licenses"
    root.mkdir()
    for name, source in sources.items():
        candidates = sorted(source.glob("LICENSE*"))
        if not candidates:
            raise ValueError(f"Missing upstream license: {name}")
        shutil.copyfile(candidates[0], root / (name + ".txt"))


def check_payload(files):
    """Check real ELF binaries, the complete recorded payload and required headers."""
    meta = json.loads(files[MANIFEST])
    expected = {PREFIX + name: info for name, info in meta["files"].items()}
    required = [
        PREFIX + "flash_mla_glm53_22.abi3.so",
        PREFIX + "mega_mhc_isolated/csrc/jit_kernels/impls/sm100_mega_mhc.hpp",
        PREFIX + "mega_mhc_isolated/include/deep_jit/utils/parser.hpp",
        PREFIX + "mega_mhc_isolated/include/cute/tensor.hpp",
        PREFIX + "mega_mhc_isolated/include/cutlass/cutlass.h",
    ]
    mhc = [
        name
        for name in expected
        if name.startswith(PREFIX + "mega_mhc_isolated/mhc_isolated")
        and name.endswith(".so")
    ]
    if len(mhc) != 1:
        raise ValueError("Expected exactly one isolated MegaMHC binary")
    for name in required + mhc:
        if name not in expected or name not in files:
            raise ValueError(f"Missing extension asset: {name}")
    for name, info in expected.items():
        if name not in files:
            raise ValueError(f"Missing extension asset: {name}")
        data = files[name]
        if (
            len(data) != info["bytes"]
            or hashlib.sha256(data).hexdigest() != info["sha256"]
        ):
            raise ValueError(f"Stale or corrupt extension asset: {name}")
        if name.endswith(".so") and not data.startswith(b"\x7fELF"):
            raise ValueError(f"Not an ELF binary: {name}")
    return meta


def payload_files(output):
    return {
        PREFIX + str(path.relative_to(output)): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


def extend_wheel(wheel, output):
    additions = payload_files(output)
    check_payload(additions)
    temporary = wheel.with_suffix(".glm53.tmp")
    try:
        with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Wheel has duplicate members")
            records = [name for name in names if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError("Expected one wheel RECORD")
            record = records[0]
            if record + ".jws" in names or record + ".p7s" in names:
                raise ValueError("Cannot modify a signed wheel")
            rows = []
            for info in source.infolist():
                name = info.filename
                if (
                    name == record
                    or name in additions
                    or name.startswith(PREFIX + "mega_mhc_isolated/")
                ):
                    continue
                digest = hashlib.sha256()
                size = 0
                with source.open(info) as old, target.open(
                    info, "w", force_zip64=True
                ) as new:
                    for block in iter(lambda: old.read(1024 * 1024), b""):
                        new.write(block)
                        digest.update(block)
                        size += len(block)
                rows.append(
                    (
                        name,
                        "sha256="
                        + base64.urlsafe_b64encode(digest.digest())
                        .rstrip(b"=")
                        .decode(),
                        str(size),
                    )
                )
            for name, data in sorted(additions.items()):
                target.writestr(name, data)
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    .rstrip(b"=")
                    .decode()
                )
                rows.append((name, "sha256=" + digest, str(len(data))))
            rows.append((record, "", ""))
            text = io.StringIO()
            csv.writer(text, lineterminator="\n").writerows(rows)
            target.writestr(record, text.getvalue())
        check_wheel(temporary)
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok=True)


def check_wheel(wheel):
    with zipfile.ZipFile(wheel) as source:
        meta = json.loads(source.read(MANIFEST))
        names = [MANIFEST] + [PREFIX + name for name in meta["files"]]
        files = {name: source.read(name) for name in names}
        check_payload(files)
        record = next(
            name for name in source.namelist() if name.endswith(".dist-info/RECORD")
        )
        rows = {
            row[0]: row[1:]
            for row in csv.reader(io.StringIO(source.read(record).decode()))
        }
        for name, data in files.items():
            expected = (
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            if rows.get(name) != [expected, str(len(data))]:
                raise ValueError(f"Invalid wheel RECORD for extension asset: {name}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--check-wheel", type=Path)
    args = parser.parse_args()
    if args.check_wheel:
        meta = check_wheel(args.check_wheel)
        print(
            "GLM53 wheel extension check passed",
            meta["cuda_architectures"],
            len(meta["files"]),
        )
        return
    if not args.work_dir or not args.wheel:
        parser.error("--work-dir and --wheel are required for compilation")
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if platform.machine() != "x86_64" or not (torch.version.cuda or "").startswith(
        "13."
    ):
        raise ValueError("This recipe is validated for CUDA 13 x86_64 only")
    if not CUDA_HOME:
        raise RuntimeError("CUDA toolkit is required")
    work = args.work_dir.resolve()
    wheel = args.wheel.resolve()
    sources, pins = prepare(work)
    output = work / "payload"
    output.mkdir()
    cuda = Path(CUDA_HOME)
    build_mhc(sources, output, cuda)
    build_sparse(sources, output, work, cuda)
    stage_licenses(sources, output)
    shutil.copyfile(HERE / "check.py", output / "glm53_extension_check.py")
    meta = {
        "schema": 1,
        "cuda_architectures": ["sm_100a"],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "sources": pins,
        "patches": {
            name: sha256(HERE / name)
            for name in [
                "mega_mhc_graph_barriers.patch",
                "deep_jit_portable_errors.patch",
                "flash_mla_glm53.patch",
            ]
        },
        "files": {
            str(path.relative_to(output)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "glm53_extensions.json").write_text(json.dumps(meta, indent=2) + "\n")
    extend_wheel(wheel, output)
    print("GLM53 isolated extensions built and wheel verified:", wheel)


if __name__ == "__main__":
    main()
