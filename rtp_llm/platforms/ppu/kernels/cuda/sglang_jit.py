"""Load frozen SGLang kernels through RTP-owned, content-addressed JIT modules.

The source bundle retains its original FP4 Indexer directory for compatibility.
Model adapters select explicit symbols; this loader never imports SGLang Python.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _source_bundle():
    root = Path(__file__).with_name("dsv4_ppu") / "fp4_indexer"
    manifest = json.loads((root / "source-manifest.json").read_text())
    digest = hashlib.sha256()
    for name, expected in sorted(manifest["sources"].items()):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Modified frozen SGLang source: {name}")
        digest.update(name.encode())
        digest.update(data)
    return root, digest.hexdigest()[:16]


@lru_cache(maxsize=None)
def load_sglang_kernel(name, header, symbol, arch, extra_cuda_cflags=()):
    """Build one explicit entry point. Warm it up before graph capture."""
    from tvm_ffi.cpp import load_inline

    root, source_digest = _source_bundle()
    source = root / header
    flags = [
        f"-DSGL_CUDA_ARCH={arch[0] * 100 + arch[1] * 10}",
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        *extra_cuda_cflags,
    ]
    signature = hashlib.sha256(
        json.dumps([header, symbol, flags]).encode()
    ).hexdigest()[:12]
    return load_inline(
        f"rtp_sg_{name}_{arch[0]}{arch[1]}_{source_digest}_{signature}",
        cpp_sources=[],
        cuda_sources=[
            f'#include "{source}"',
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, ({symbol}));",
        ],
        extra_cflags=["-std=c++20", "-O3"],
        extra_cuda_cflags=flags,
        extra_include_paths=[str(root / "compat"), str(root / "include")],
    )
