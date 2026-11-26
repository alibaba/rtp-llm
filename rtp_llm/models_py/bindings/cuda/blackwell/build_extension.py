"""Build one isolated native extension or record its declared Bazel outputs."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

SPARSE_SOURCES = (
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",
)


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


def build_mhc(
    deep_gemm,
    deep_gemm_license,
    bundle,
    root,
    binding,
    format_header,
    cuda,
):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    root.mkdir(exist_ok=True)
    shutil.copytree(deep_gemm / "csrc", root / "csrc")
    headers = [
        deep_gemm / "include/deep_gemm",
        deep_gemm / "include/cute",
        deep_gemm / "include/cutlass",
        bundle / "fmt/include/fmt",
        bundle / "deep_jit/include/deep_jit",
    ]
    include_dirs = [bundle, root, root / "include"]
    (root / "compat").mkdir()
    shutil.copyfile(format_header, root / "compat/format")
    include_dirs.insert(0, root / "compat")
    shutil.copytree(bundle / "licenses", root / "licenses")
    shutil.copyfile(deep_gemm_license, root / "licenses/deep_gemm.txt")
    for path in headers:
        shutil.copytree(path, root / "include" / path.name)
    # Upstream binding headers use quoted csrc paths. Both the compiler and
    # the installed JIT must see the same isolated source/header layout.
    binary = root / ("mhc_isolated" + sysconfig.get_config_var("EXT_SUFFIX"))
    include_dirs.extend(
        [cuda / "include", cuda / "include/cccl", sysconfig.get_paths()["include"]]
    )
    command = [
        configure_compiler(),
        "-shared",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-Wno-psabi",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch.compiled_with_cxx11_abi())),
        "-Ddeep_gemm=deep_gemm_mhc_isolated",
        "-Ddeep_jit=deep_jit_mhc_isolated",
        "-std=c++20",
        *["-I" + str(path) for path in include_dirs + include_paths()],
        str(binding),
        "-o",
        str(binary),
        *["-L" + str(path) for path in [*library_paths(), cuda / "lib64"]],
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
    # DeepJIT resolves libdw at runtime with dlopen. Only the build needs its
    # declarations; wheel users do not need elfutils development headers.
    with tempfile.TemporaryDirectory() as temp:
        elfutils = Path(temp) / "elfutils"
        elfutils.mkdir()
        for name in (
            "libdwfl/libdwfl.h",
            "libdw/libdw.h",
            "libelf/gelf.h",
            "libelf/libelf.h",
        ):
            header = bundle / "elfutils" / name
            include = elfutils if header.name.startswith("libdw") else Path(temp)
            shutil.copyfile(header, include / header.name)
        command.insert(1, "-I" + temp)
        subprocess.run(command, check=True)


def build_sparse(bundle, output, binding, cuda):
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    configure_compiler()
    repo = bundle / "flash_mla"
    cutlass = bundle / "cutlass_mla"
    common = [
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-DUSE_CUDA",
        "-DFLASH_MLA_IS_BUILD_ON_CUDA",
        "-DKERUTILS_IS_BUILD_ON_CUDA",
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
        "-Wno-deprecated-declarations",
        "-Dsm100=rtp_fp8_sparse_sm100",
        "-Dsmxx=rtp_fp8_sparse_smxx",
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
    extension = CUDAExtension(
        "rtp_fp8_sparse_decode",
        sources=[
            str(binding),
            str(bundle / "sparse_fp8_decode.cpp"),
            str(bundle / "sparse_fp8_kernel.cu"),
            *[str(repo / name) for name in SPARSE_SOURCES],
        ],
        include_dirs=[
            str(bundle),
            str(repo / "csrc"),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
            str(repo / "csrc/3rdparty/kerutils/include"),
            str(cuda / "include/cccl"),
        ],
        extra_compile_args={"cxx": common + ["-fvisibility=hidden"], "nvcc": nvcc},
        extra_link_args=["-L" + str(cuda / "lib64/stubs"), "-lcuda", "-Wl,-Bsymbolic"],
        py_limited_api=True,
    )
    # Explicit architecture and build directories also work on GPU-less CI workers.
    os.environ.update(CUDA_VISIBLE_DEVICES="", TORCH_CUDA_ARCH_LIST="10.0a")
    os.environ.setdefault("MAX_JOBS", "4")
    with tempfile.TemporaryDirectory() as temp:
        build = BuildExtension(Distribution({"ext_modules": [extension]}))
        build.build_temp = str(Path(temp) / "objects")
        build.build_lib = temp
        build.ensure_finalized()
        build.run()
        shutil.copyfile(build.get_ext_fullpath(extension.name), output)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output, assets, sources):
    import torch

    files = {}
    for asset in assets:
        paths = sorted(asset.rglob("*")) if asset.is_dir() else [asset]
        for path in paths:
            if path.is_file():
                files[str(path.relative_to(asset.parent))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
    metadata = {
        "schema": 1,
        "cuda_architectures": ["sm_100a"],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "sources": sources,
        "files": files,
    }
    output.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="part", required=True)
    for part in ("mhc", "sparse"):
        build = commands.add_parser(part)
        build.add_argument("--bundle", type=Path, required=True)
        build.add_argument("--binding", type=Path, required=True)
        build.add_argument("--output", type=Path, required=True)
        if part == "mhc":
            build.add_argument("--deep-gemm", type=Path, required=True)
            build.add_argument("--deep-gemm-license", type=Path, required=True)
            build.add_argument("--format-header", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--asset", type=Path, action="append", required=True)
    manifest.add_argument("--pins", type=json.loads, required=True)
    args = parser.parse_args()
    if args.part == "manifest":
        write_manifest(args.output, args.asset, args.pins)
        return

    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if platform.machine() != "x86_64" or not (torch.version.cuda or "").startswith(
        "13."
    ):
        raise RuntimeError("The Blackwell extension wheel requires CUDA 13 x86_64")
    if not CUDA_HOME:
        raise RuntimeError("CUDA toolkit is required")
    if args.part == "sparse":
        build_sparse(
            args.bundle.absolute(),
            args.output,
            args.binding.absolute(),
            Path(CUDA_HOME),
        )
    else:
        build_mhc(
            args.deep_gemm.absolute(),
            args.deep_gemm_license,
            args.bundle.absolute(),
            args.output.absolute(),
            args.binding.absolute(),
            args.format_header,
            Path(CUDA_HOME),
        )


if __name__ == "__main__":
    main()
