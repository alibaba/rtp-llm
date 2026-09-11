"""Verify the loaded optional native kernels before V4.1 dispatch."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
from functools import lru_cache
from pathlib import Path

import torch

PINS = {
    "deep-select": (
        "deep_select",
        "deep_select_cuda",
        "0f03b68748b304863fdf0181a11458d04ae533a9",
        "ae6bccf341fb4410241f696ba06873023d5ce4ed",
    ),
    "flash-mla": (
        "flash_mla",
        "cuda",
        "07a1089857b63e74e3133630c02b083b75e8d4b2",
        "147f5673d0c1c3dcf66f78d677fd647e4a020219",
    ),
}


@lru_cache(maxsize=None)
def native_identity(name):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("initialize V4.1 native kernels before Graph capture")
    package, extension, upstream, cutlass = PINS[name]
    module = importlib.import_module(package)
    native = importlib.import_module(package + "." + extension)
    root = Path(module.__file__).resolve().parent
    path = root / "rtp_build_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    build = manifest["build_inputs"]["build"]
    if (
        manifest["sources"] != {"upstream": upstream, "cutlass": cutlass}
        or manifest["kind"] != "rtp-aot-kernel"
        or manifest["package"] != name
        or manifest["version"] != importlib.metadata.version(name)
        or manifest["ptx_present"] is not False
        or build["host_arch"] != platform.machine()
        or build["torch_version"] != str(torch.__version__)
        or build["torch_cuda_version"] != torch.version.cuda
        or build["cxx11_abi"] != torch.compiled_with_cxx11_abi()
        or any(
            not manifest["native_cubins"].get(arch) for arch in ("sm_100a", "sm_103a")
        )
    ):
        raise RuntimeError("V4.1 native kernel source, ABI or cubin identity mismatch")
    libraries = {}
    for relative, expected in manifest["native_libraries"].items():
        candidate = (root.parent / relative).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise RuntimeError("V4.1 native library escapes its package")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("V4.1 native library checksum mismatch")
        libraries[str(candidate)] = actual
    if str(Path(native.__file__).resolve()) not in libraries:
        raise RuntimeError("loaded V4.1 extension is outside the verified inventory")
    return {
        "package": name,
        "version": manifest["version"],
        "build_identity": manifest["build_identity"],
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_libraries": libraries,
        "sources": manifest["sources"],
    }
