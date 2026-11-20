"""Build the out-of-tree overlay against the exact default FlashInfer JIT recipe."""

import functools
import hashlib
from dataclasses import replace
from pathlib import Path

from flashinfer.jit import env
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module

from rtp_llm.utils.jit_cache_env import native_source_fingerprint

MODULE_NAME = "rtp_dsv4_sm120_mxfp8_mxfp4_separate_v1"
NATIVE = Path(__file__).absolute().parent / "native"
EXPECTED_BASE = {
    "flashinfer_cutlass_fused_moe_binding.cu": "4b433f0eff220b889744ba646cab3579f47aeae8d62c0b621d9dcab07aeed671",
    "cutlass_fused_moe_instantiation.cu": "db05873b50a03b8c96c48d2d339f032783b15152e4c11eae61ad5bf438980ac1",
    "cutlass_fused_moe_kernels.cuh": "e6c5fdaf6ae2738f3113a3d49af75fec2d77f76e897cf9083899c9f4bd9ca04c",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def spec():
    source_root = env.FLASHINFER_CSRC_DIR
    identity = hashlib.sha256(
        f"{NATIVE}:{source_root}:{native_source_fingerprint(NATIVE)}".encode()
    ).hexdigest()[:16]
    backend = source_root / "fused_moe/cutlass_backend"
    for name, expected in EXPECTED_BASE.items():
        actual = sha256(backend / name)
        if actual != expected:
            raise RuntimeError(f"Unexpected default source: {backend / name}: {actual}")
    baseline = gen_cutlass_fused_moe_sm120_module()
    replaced = {
        backend / name: NATIVE / name for name in EXPECTED_BASE if name.endswith(".cu")
    }
    assert set(replaced) <= set(baseline.sources)
    # Keep the default source closure except for the two local translation units.
    return replace(
        baseline,
        name=f"{MODULE_NAME}_{identity}",
        sources=[replaced.get(source, source) for source in baseline.sources],
        extra_include_dirs=[*baseline.extra_include_dirs, source_root],
        extra_cuda_cflags=[
            flag
            for flag in baseline.extra_cuda_cflags
            if not flag.startswith("-gencode")
        ]
        + ["-gencode=arch=compute_120a,code=sm_120a"],
        extra_ldflags=[*baseline.extra_ldflags, "-Wl,-Bsymbolic"],
    )


@functools.cache
def native_module():
    build = spec()
    module = build.build_and_load()
    module.set_deepgemm_jit_include_dirs(
        [str(env.FLASHINFER_CSRC_DIR / "nv_internal/tensorrt_llm")]
    )
    return module
