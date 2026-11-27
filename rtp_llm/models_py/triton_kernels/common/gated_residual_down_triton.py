"""BF16 epilogue for a merged gated-residual down/injection projection."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.utils.device_capability import sm90_pdl_enabled


@triton.jit
def _down_inject_epilogue_kernel(
    projection_ptr,
    mix_ptr,
    inject_ptr,
    ROWS,
    RANK: tl.constexpr,
    BRANCHES: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PAD: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    valid_row = (row < ROWS) if HAS_PAD else True
    projected = tl.load(
        projection_ptr + row * WIDTH + column,
        mask=valid_row & (column < RANK + BRANCHES),
        other=0,
    ).to(tl.float32)
    scaled = (projected / BRANCHES).to(tl.bfloat16).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-scaled))
    gate = sigmoid.to(tl.bfloat16).to(tl.float32) * 2.0
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # PyTorch SiLU computes in float and rounds once to the BF16 output.
    mix_out = tl.where(valid_row, scaled * sigmoid, 0.0) if HAS_PAD else scaled * sigmoid
    tl.store(mix_ptr + row * RANK + column, mix_out, mask=column < RANK)
    tl.store(
        inject_ptr + row * BRANCHES + column - RANK,
        gate,
        mask=valid_row & (column >= RANK) & (column < RANK + BRANCHES),
    )


def down_inject_epilogue(
    projection: torch.Tensor, rank: int, branches: int, *, pad_rows: int = 0
):
    if (
        not projection.is_cuda
        or torch.version.hip is not None
        or projection.dtype != torch.bfloat16
        or not projection.is_contiguous()
        or projection.dim() < 2
        or rank <= 0
        or branches != 4
        or projection.shape[-1] < rank + branches
        or pad_rows < 0
        or (pad_rows > 0 and projection.dim() != 2)
    ):
        raise ValueError("unsupported gated-residual down/injection epilogue layout")
    rows = projection.numel() // projection.shape[-1]
    mix_shape = (
        (rows + pad_rows, rank) if pad_rows > 0 else (*projection.shape[:-1], rank)
    )
    mix = torch.empty(mix_shape, device=projection.device, dtype=projection.dtype)
    inject = torch.empty(
        (*projection.shape[:-1], branches),
        device=projection.device,
        dtype=projection.dtype,
    )
    if rows:
        pdl = sm90_pdl_enabled(projection.device)
        _down_inject_epilogue_kernel[(rows + pad_rows,)](
            projection,
            mix,
            inject,
            rows,
            RANK=rank,
            BRANCHES=branches,
            WIDTH=projection.shape[-1],
            BLOCK=triton.next_power_of_2(rank + branches),
            HAS_PAD=pad_rows > 0,
            PDL=pdl,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
    elif pad_rows > 0:
        mix.zero_()
    return mix, inject


@triton.jit
def _separate_down_inject_epilogue(
    Down,
    Inject,
    Mix,
    Gate,
    ROWS,
    RANK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    BLOCK: tl.constexpr,
    I_STRIDE: tl.constexpr,
    HAS_PAD: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    valid_row = (row < ROWS) if HAS_PAD else True
    d = tl.load(Down + row * RANK + col, valid_row & (col < RANK), other=0).to(
        tl.float32
    )
    scaled = (d / 4.0).to(tl.bfloat16).to(tl.float32)
    if HAS_INJECT:
        i = tl.load(
            Inject + row * I_STRIDE + col, valid_row & (col < 4), other=0
        ).to(tl.float32)
        z = (i / 4.0).to(tl.bfloat16).to(tl.float32)
        gate = tl.sigmoid(z).to(tl.bfloat16).to(tl.float32) * 2.0
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    mix_out = (
        tl.where(valid_row, scaled * tl.sigmoid(scaled), 0.0)
        if HAS_PAD
        else scaled * tl.sigmoid(scaled)
    )
    tl.store(Mix + row * RANK + col, mix_out, col < RANK)
    if HAS_INJECT:
        tl.store(Gate + row * 4 + col, gate, valid_row & (col < 4))


def separate_down_inject_epilogue(down, injection=None, *, pad_rows: int = 0):
    """Fuse only elementwise work; retain independent FP8/BF16 GEMMs/scales."""
    if not (
        down.is_cuda
        and torch.version.hip is None
        and down.dtype == torch.bfloat16
        and down.is_contiguous()
        and down.ndim >= 2
        and down.shape[-1] == 320
        and pad_rows >= 0
        and (pad_rows == 0 or down.ndim == 2)
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
    rows = down.numel() // 320
    mix = (
        torch.empty((rows + pad_rows, 320), device=down.device, dtype=down.dtype)
        if pad_rows > 0
        else torch.empty_like(down)
    )
    gate = (
        torch.empty(injection.shape, device=injection.device, dtype=injection.dtype)
        if injection is not None
        else None
    )
    if rows:
        pdl = sm90_pdl_enabled(down.device)
        _separate_down_inject_epilogue[(rows + pad_rows,)](
            down,
            injection if injection is not None else down,
            mix,
            gate if gate is not None else mix,
            rows,
            RANK=320,
            HAS_INJECT=injection is not None,
            BLOCK=512,
            I_STRIDE=injection.stride(-2) if injection is not None else 4,
            HAS_PAD=pad_rows > 0,
            PDL=pdl,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
    elif pad_rows > 0:
        mix.zero_()
    return mix, gate
