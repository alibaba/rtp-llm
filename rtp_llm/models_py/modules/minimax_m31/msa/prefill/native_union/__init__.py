"""MiniMax-M3.1 SM103 native-layout NVFP4 sparse Prefill kernel."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    return load(
        name="minimax_m31_nvfp4_union_sm103a_v1",
        sources=[str(root / "binding.cpp"), str(root / "minimax_m31_nvfp4_union.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        verbose=False,
    )


def run(
    q8: torch.Tensor,
    pages: torch.Tensor,
    topk: torch.Tensor,
    cu_q: torch.Tensor,
    table: torch.Tensor,
    used_k: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _extension().run(
        q8.to(torch.bfloat16),
        pages,
        topk,
        cu_q,
        table,
        used_k,
        output,
        128**-0.5,
    )
