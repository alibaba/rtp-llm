"""NTC dilated PLE convolution with explicit BF16 consumer boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_conv(
    Gated,
    Normed,
    Initial,
    Weight,
    Output,
    Residual,
    ROWS: tl.constexpr,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    KERNEL: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    INITIAL_BATCH_STRIDE: tl.constexpr,
    INITIAL_ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // WIDTH, offset % WIDTH
    batch, token = row // LENGTH, row % LENGTH
    valid = row < ROWS
    state_len: tl.constexpr = (KERNEL - 1) * DILATION
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(KERNEL):
        position = token - state_len + k * DILATION
        value = tl.load(
            Normed + (batch * LENGTH + position) * WIDTH + col,
            valid & (position >= 0),
            other=0,
        ).to(tl.float32)
        if HAS_INITIAL:
            previous = tl.load(
                Initial
                + batch * INITIAL_BATCH_STRIDE
                + (position + state_len) * INITIAL_ROW_STRIDE
                + col,
                valid & (position < 0),
                other=0,
            ).to(tl.float32)
            value = tl.where(position < 0, previous, value)
        weight = tl.load(Weight + col * KERNEL + k, valid, other=0).to(tl.float32)
        acc = tl.fma(value, weight, acc)
    conv = acc.to(tl.bfloat16).to(tl.float32)
    activated = (conv * tl.sigmoid(conv)).to(tl.bfloat16).to(tl.float32)
    gated = tl.load(Gated + offset, valid, other=0).to(tl.float32)
    output = gated + activated
    if HAS_RESIDUAL:
        # Preserve the original BF16 PLE output before the outer residual add.
        output = output.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(Residual + offset, valid, other=0).to(tl.float32)
        output = residual + output
    tl.store(Output + offset, output, valid)


def try_ple_conv(gated, normed, initial, weight, dilation, *, residual=None):
    if not (
        gated.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(gated.device) == (12, 0)
        and gated.dtype == normed.dtype == weight.dtype == torch.bfloat16
        and gated.ndim == 3
        and gated.shape == normed.shape
        and gated.shape[-1] == 10240
        and gated.shape[1] > 0
        and gated.is_contiguous()
        and normed.is_contiguous()
        and weight.shape == (10240, 1, 4)
        and weight.is_contiguous()
        and gated.device == normed.device == weight.device
        and dilation == 3
        and (
            initial is None
            or (
                initial.shape == (gated.shape[0], 9, 10240)
                and initial.dtype == gated.dtype
                and initial.device == gated.device
                and initial.stride(-1) == 1
            )
        )
        and not any(x.requires_grad for x in (gated, normed, weight))
        and (
            residual is None
            or (
                residual.shape == gated.shape
                and residual.dtype == gated.dtype
                and residual.device == gated.device
                and residual.is_contiguous()
                and not residual.requires_grad
            )
        )
    ):
        return None
    output = torch.empty_like(gated)
    _ple_conv[(triton.cdiv(gated.numel(), 1024),)](
        gated,
        normed,
        initial if initial is not None else normed,
        weight,
        output,
        residual if residual is not None else gated,
        ROWS=gated.shape[0] * gated.shape[1],
        LENGTH=gated.shape[1],
        WIDTH=10240,
        KERNEL=4,
        DILATION=3,
        HAS_INITIAL=initial is not None,
        HAS_RESIDUAL=residual is not None,
        INITIAL_BATCH_STRIDE=initial.stride(0) if initial is not None else 0,
        INITIAL_ROW_STRIDE=initial.stride(1) if initial is not None else 0,
        BLOCK=1024,
        num_warps=4,
    )
    return output
