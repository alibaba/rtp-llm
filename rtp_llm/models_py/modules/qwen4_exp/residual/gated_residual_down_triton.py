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


@triton.jit
def _separate_down_inject_epilogue(
    Down,
    Inject,
    Mix,
    Gate,
    RANK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    BLOCK: tl.constexpr,
    I_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    d = tl.load(Down + row * RANK + col, col < RANK, other=0).to(tl.float32)
    scaled = (d / 4.0).to(tl.bfloat16).to(tl.float32)
    tl.store(Mix + row * RANK + col, scaled * tl.sigmoid(scaled), col < RANK)
    if HAS_INJECT:
        i = tl.load(Inject + row * I_STRIDE + col, col < 4, other=0).to(tl.float32)
        z = (i / 4.0).to(tl.bfloat16).to(tl.float32)
        gate = tl.sigmoid(z).to(tl.bfloat16).to(tl.float32) * 2.0
        tl.store(Gate + row * 4 + col, gate, col < 4)


def separate_down_inject_epilogue(down, injection=None):
    """Fuse only elementwise work; retain independent FP8/BF16 GEMMs/scales."""
    if not (
        down.is_cuda
        and torch.version.hip is None
        and down.dtype == torch.bfloat16
        and down.is_contiguous()
        and down.ndim >= 2
        and down.shape[-1] == 320
        and (
            injection is None
            or (
                injection.device == down.device
                and injection.dtype == down.dtype
                and (
                    injection.is_contiguous()
                    or (
                        injection.ndim == 2
                        and injection.stride(1) == 1
                        and injection.stride(0) == 16
                    )
                )
                and injection.shape == (*down.shape[:-1], 4)
            )
        )
    ):
        return None
    mix = torch.empty_like(down)
    gate = (
        torch.empty(injection.shape, device=injection.device, dtype=injection.dtype)
        if injection is not None
        else None
    )
    rows = down.numel() // 320
    if rows:
        _separate_down_inject_epilogue[(rows,)](
            down,
            injection if injection is not None else down,
            mix,
            gate if gate is not None else mix,
            RANK=320,
            HAS_INJECT=injection is not None,
            BLOCK=512,
            I_STRIDE=injection.stride(-2) if injection is not None else 4,
            enable_fp_fusion=False,
        )
    return mix, gate
