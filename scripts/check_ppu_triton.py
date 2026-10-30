"""Validate PPU Triton selection and compile DSV4 warmup without a GPU.

Run with the service's PATH, not just the SDK-enriched build environment.
The vendor wheel intentionally uses a CUDA GPUTarget for both backends;
ppu-smi discovery selects the compiler that emits ppu IR and hgbin.
"""

import shutil

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
    _v4_indexer_score_fwd,
)


def main():
    smi = shutil.which("ppu-smi")
    if smi is None:
        raise RuntimeError("ppu-smi is missing from PATH; vendor Triton selects NVIDIA")
    target = GPUTarget("cuda", 89, 32)
    backend = make_backend(target)
    if type(backend).__module__ != "triton.backends.ppu.compiler":
        raise RuntimeError(f"Expected PPU compiler, got {type(backend)}")
    print(
        f"Triton {triton.__version__}: {type(backend).__name__}, ppu-smi={smi}",
        flush=True,
    )

    constants = dict(
        q_s=8192, q_h=128, kv_t=128, w_s=64,
        H=64, D=128, BLOCK_S=16, BLOCK_T=256,
    )
    signature = {name: "i32" for name in _v4_indexer_score_fwd.arg_names}
    signature.update(q_ptr="*bf16", kv_ptr="*bf16", w_ptr="*fp32", out_ptr="*fp32")
    for masked in (False, True):
        constants.update(APPLY_MASK=masked, COMPRESS_RATIO=4 if masked else 1)
        signature.update({name: "constexpr" for name in constants})
        signature["q_pos_ptr"] = "*i32" if masked else "i32"
        compiled = triton.compile(
            ASTSource(_v4_indexer_score_fwd, signature, constexprs=constants),
            target=target,
            options={"num_warps": 4, "num_stages": 2},
        )
        assert "ppu:89" in compiled.asm["ttgir"], "Wrong Triton IR target"
        assert compiled.asm["hgbin"], "PPU native binary is empty"
        print(
            f"DSV4 indexer H=64 D=128 tile=16x256 mask={masked}: hgbin OK",
            flush=True,
        )


if __name__ == "__main__":
    main()
