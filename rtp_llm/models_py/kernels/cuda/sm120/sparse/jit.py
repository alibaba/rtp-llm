import hashlib
from functools import cache
from pathlib import Path

from flashinfer.compilation_context import CompilationContext
from flashinfer.jit import env
from flashinfer.jit.core import gen_jit_spec

from rtp_llm.utils.jit_cache_env import native_source_fingerprint


@cache
def gen_sparse_mla_sm120_module():
    root = Path(__file__).absolute().parent
    identity = hashlib.sha256(
        f"{root}:{env.FLASHINFER_CSRC_DIR}:{native_source_fingerprint(root)}".encode()
    ).hexdigest()[:16]
    return gen_jit_spec(
        f"rtp_sparse_dsv4_399dff_sm120_{identity}",
        [
            root / "csrc" / name
            for name in (
                "sparse_mla_sm120.cu",
                "sparse_mla_sm120_decode_dsv4.cu",
                "sparse_mla_sm120_prefill.cu",
                "sparse_mla_sm120_jit_binding.cu",
            )
        ],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120a,code=sm_120a",
            *CompilationContext.COMMON_NVCC_FLAGS,
            "-Xfatbin=-compress-all",
        ],
        extra_include_paths=[root / "include"],
    )
