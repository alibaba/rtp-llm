"""Resolve the CUDA13 JIT compiler declared in Bazel runfiles."""

import os
import shlex
from pathlib import Path
from typing import MutableMapping, Optional

_COMPILER_RUNFILE = "cuda13_jit_gcc/toolchain/bin/g++"


def _toolchain_root(env: MutableMapping[str, str]) -> Optional[Path]:
    for key in ("RUNFILES_DIR", "TEST_SRCDIR", "PYTHON_RUNFILES"):
        if env.get(key):
            compiler = Path(env[key]) / _COMPILER_RUNFILE
            if compiler.is_file():
                return compiler.resolve().parent.parent
    manifest = env.get("RUNFILES_MANIFEST_FILE")
    if manifest and Path(manifest).is_file():
        with open(manifest) as stream:
            for line in stream:
                name, _, target = line.rstrip("\n").partition(" ")
                if name == _COMPILER_RUNFILE and target:
                    return Path(target).resolve().parent.parent
    return None


def configure_bazel_jit_toolchain(
    env: Optional[MutableMapping[str, str]] = None,
) -> Optional[Path]:
    """Configure only a bundled toolchain; retain explicit compiler overrides."""
    env = os.environ if env is None else env
    root = _toolchain_root(env)
    if root is None:
        return None
    env.setdefault("CC", str(root / "bin/gcc"))
    env.setdefault("CXX", str(root / "bin/g++"))
    host_compiler = env.setdefault("CUDAHOSTCXX", env["CXX"])
    flags = env.get("NVCC_PREPEND_FLAGS", "")
    host_flags = shlex.split(flags) + shlex.split(env.get("NVCC_APPEND_FLAGS", ""))
    if not any(
        flag in ("-ccbin", "--compiler-bindir")
        or flag.startswith(("-ccbin=", "--compiler-bindir="))
        for flag in host_flags
    ):
        env["NVCC_PREPEND_FLAGS"] = (
            flags + " -ccbin=" + shlex.quote(host_compiler)
        ).strip()
    for key, directory in (("PATH", root / "bin"), ("LD_LIBRARY_PATH", root / "lib64")):
        paths = env.get(key, os.defpath if key == "PATH" else "").split(os.pathsep)
        path = str(directory)
        env[key] = os.pathsep.join([path] + [p for p in paths if p and p != path])
    env.setdefault("DG_JIT_CPP_STANDARD", "20")
    return root
