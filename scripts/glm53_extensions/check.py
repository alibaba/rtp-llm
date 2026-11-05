#!/usr/bin/env python3
"""Validate installed GLM53 assets and their CPU ABI without initializing CUDA."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path


def check(root, allow_cuda_stub=False):
    import torch

    meta = json.loads((root / "glm53_extensions.json").read_text())
    for name, info in meta["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or truncated GLM53 extension asset: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"GLM53 extension checksum mismatch: {name}")
    if (
        torch.__version__.split("+")[0].split(".")[:2]
        != meta["torch"].split("+")[0].split(".")[:2]
    ):
        raise ValueError("MegaMHC must use the PyTorch ABI it was built against")
    if allow_cuda_stub:
        # BuildKit has no GPU driver. Preload the toolkit's driver stub only
        # in this validation subprocess; never install it as a runtime driver.
        try:
            ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            stubs = sorted(Path("/usr/local").glob("cuda*/lib64/stubs/libcuda.so"))
            if not stubs:
                raise
            ctypes.CDLL(str(stubs[0]), mode=ctypes.RTLD_GLOBAL)
    binaries = list((root / "mega_mhc_isolated").glob("mhc_isolated*.so"))
    if len(binaries) != 1:
        raise ValueError("Expected exactly one MegaMHC binary")
    spec = importlib.util.spec_from_file_location("mhc_isolated", binaries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("init", "set_num_sms", "set_pdl", "mega_mhc"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Missing MegaMHC binding: {name}")
    torch.ops.load_library(str(root / "flash_mla_glm53_22.abi3.so"))
    if not callable(torch.ops.flash_mla_glm53_22.sparse_decode_fwd):
        raise ValueError("Missing native sparse MLA operator")
    print(
        "GLM53 MegaMHC and native sparse MLA assets/CPU ABI verified",
        len(meta["files"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-cuda-stub", action="store_true")
    args = parser.parse_args()
    check(args.root, args.allow_cuda_stub)
