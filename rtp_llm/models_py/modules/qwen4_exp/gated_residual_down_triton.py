"""BF16 epilogue for a merged gated-residual down/injection projection."""

import torch
import triton
import triton.language as tl


@triton.jit
def _down_inject_epilogue_kernel(
    projection_ptr,
    mix_ptr,
    inject_ptr,
    RANK: tl.constexpr,
    BRANCHES: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    projected = tl.load(
        projection_ptr + row * WIDTH + column, mask=column < RANK + BRANCHES, other=0
    ).to(tl.float32)
    scaled = (projected / BRANCHES).to(tl.bfloat16).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-scaled))
    # PyTorch SiLU computes in float and rounds once to the BF16 output.
    tl.store(mix_ptr + row * RANK + column, scaled * sigmoid, mask=column < RANK)
    gate = sigmoid.to(tl.bfloat16).to(tl.float32) * 2.0
    tl.store(
        inject_ptr + row * BRANCHES + column - RANK,
        gate,
        mask=(column >= RANK) & (column < RANK + BRANCHES),
    )


def down_inject_epilogue(projection: torch.Tensor, rank: int, branches: int):
    if (
        not projection.is_cuda
        or torch.version.hip is not None
        or projection.dtype != torch.bfloat16
        or not projection.is_contiguous()
        or projection.dim() < 2
        or rank <= 0
        or branches != 4
        or projection.shape[-1] < rank + branches
    ):
        raise ValueError("unsupported gated-residual down/injection epilogue layout")
    mix = torch.empty(
        (*projection.shape[:-1], rank), device=projection.device, dtype=projection.dtype
    )
    inject = torch.empty(
        (*projection.shape[:-1], branches),
        device=projection.device,
        dtype=projection.dtype,
    )
    rows = projection.numel() // projection.shape[-1]
    if rows:
        _down_inject_epilogue_kernel[(rows,)](
            projection,
            mix,
            inject,
            RANK=rank,
            BRANCHES=branches,
            WIDTH=projection.shape[-1],
            BLOCK=triton.next_power_of_2(rank + branches),
            enable_fp_fusion=False,
        )
    return mix, inject
