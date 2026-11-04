"""Development build loader. Importing this file does not build or initialize CUDA."""
import argparse
import os
from pathlib import Path


def load_native(build_directory=None, verbose=False):
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != "2.11.0+cu130":
        raise RuntimeError("this development loader requires torch 2.11.0+cu130")
    cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_root / "include/cuda.h").is_file():
        raise RuntimeError(f"CUDA Driver headers missing under {cuda_root}")
    # The stub is link-time only. Its SONAME is libcuda.so.1, resolved to the
    # container's real driver at runtime. Never add stubs to LD_LIBRARY_PATH/RPATH.
    driver_link_dir = Path(os.environ.get("CUDA_DRIVER_LINK_DIR", str(cuda_root / "lib64/stubs")))
    if not (driver_link_dir / "libcuda.so").exists():
        raise RuntimeError("set CUDA_DRIVER_LINK_DIR to a directory containing link-time libcuda.so")
    build = Path(build_directory or os.environ.get("ENGRAM_NATIVE_BUILD_DIR", str(Path.home() / ".cache/rtp_llm/engram_native")))
    build.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    return load(
        name="engram_native_dev_492ecf45",
        sources=[str(Path(__file__).with_name("engram_native.cc"))],
        extra_include_paths=[str(cuda_root / "include")],
        extra_cflags=["-O0", "-g0", "-std=c++17", "-Wall", "-Wextra"],
        extra_ldflags=[f"-L{driver_link_dir}", "-lcuda"],
        build_directory=str(build),
        with_cuda=False,  # Pure C++; CUDA Driver API only, no nvcc/kernel compilation.
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module = load_native(args.build_directory, args.verbose)
    print({"module": module.__name__, "counters": module.resource_counters()})
