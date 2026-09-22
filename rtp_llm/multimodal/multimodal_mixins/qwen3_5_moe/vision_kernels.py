# Copyright (c) 2025, Tri Dao.
# As of 2025-04-23, we require triton >= 3.0

from typing import Optional, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_qk_rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    SEQLEN: tl.constexpr,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rotate packed Q/K together, sharing factors without padding half-heads."""
    # Packed QKV offsets exceed int32 for large video batches.
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    half_dim = HEAD_DIM // 2
    mask = index < SEQLEN * NHEADS * half_dim
    token = index // (NHEADS * half_dim)
    head = (index // half_dim) % NHEADS
    dim = index % half_dim
    input_offset = token * (3 * NHEADS * HEAD_DIM) + head * HEAD_DIM + dim
    factor_offset = token * half_dim + dim
    cos = tl.load(COS + factor_offset, mask, other=1).to(tl.float32)
    sin = tl.load(SIN + factor_offset, mask, other=0).to(tl.float32)
    q0 = tl.load(X + input_offset, mask, other=0).to(tl.float32)
    q1 = tl.load(X + input_offset + half_dim, mask, other=0).to(tl.float32)
    k0 = tl.load(X + input_offset + NHEADS * HEAD_DIM, mask, other=0).to(tl.float32)
    k1 = tl.load(X + input_offset + NHEADS * HEAD_DIM + half_dim, mask, other=0).to(
        tl.float32
    )
    output_offset = token * NHEADS * HEAD_DIM + head * HEAD_DIM + dim
    key_offset = output_offset + SEQLEN * NHEADS * HEAD_DIM
    tl.store(OUT + output_offset, q0 * cos - q1 * sin, mask)
    tl.store(OUT + output_offset + half_dim, q0 * sin + q1 * cos, mask)
    tl.store(OUT + key_offset, k0 * cos - k1 * sin, mask)
    tl.store(OUT + key_offset + half_dim, k0 * sin + k1 * cos, mask)


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
    # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    # Sequence tiles use grid.x, whose limit accommodates packed videos.
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)
    # RTP packs large video batches: Q/K batch/sequence offsets exceed int32.
    pid_batch = tl.program_id(axis=2).to(tl.int64)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            X + ROTARY_DIM_HALF * stride_x_headdim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    # Qwen3.5 packs Q/K as views of [tokens, 3, heads, head_dim].
    # Process both together to share rotary factors and avoid masked head padding.
    if (
        not is_varlen
        and not inplace
        and not interleaved
        and not conjugate
        and not isinstance(seqlen_offsets, torch.Tensor)
        and seqlen_offsets == 0
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and cos.dtype == sin.dtype == x.dtype
        and batch == 2
        and nheads == 16
        and headdim == rotary_dim == 72
        and x.stride() == (nheads * headdim, 3 * nheads * headdim, headdim, 1)
    ):
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        if seqlen:
            block = 512
            with torch.cuda.device(x.device.index):
                torch.library.wrap_triton(_packed_qk_rotary_kernel)[
                    (triton.cdiv(seqlen * nheads * (headdim // 2), block),)
                ](x, cos, sin, output, seqlen, nheads, headdim, block, num_warps=4)
        return output

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:headdim].copy_(x[..., rotary_dim:])

    grid = lambda META: (
        triton.cdiv(seqlen, META["BLOCK_M"]),
        triton.cdiv(nheads, META["BLOCK_H"]),
        batch,
    )  # noqa
    BLOCK_M = 8 if rotary_dim <= 128 else 4

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rotary_kernel)[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            seqlen_ro,
            (
                output.stride(0) if not is_varlen else 0
            ),  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return output


@triton.jit
def _nv12_rgb_kernel(
    video,
    output,
    count: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    cr: tl.constexpr,
    cu: tl.constexpr,
    cgu: tl.constexpr,
    cgv: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < count * height * width
    frame = i // (height * width)
    pixel = i % (height * width)
    y = pixel // width
    x = pixel % width
    base = frame * (height * 3 // 2) * width
    luma = tl.load(video + base + pixel, mask, other=16).to(tl.int32)
    chroma = base + height * width + (y // 2) * width + (x // 2) * 2
    u = tl.load(video + chroma, mask, other=128).to(tl.int32) - 128
    v = tl.load(video + chroma + 1, mask, other=128).to(tl.int32) - 128
    luma = ((luma - 16) * 9539) >> 13
    r = tl.minimum(255, tl.maximum(0, luma + ((v * cr) >> 13)))
    g = tl.minimum(255, tl.maximum(0, luma + ((u * cgu) >> 13) + ((v * cgv) >> 13)))
    b = tl.minimum(255, tl.maximum(0, luma + ((u * cu) >> 13)))
    dst = frame * 3 * height * width + pixel
    tl.store(output + dst, r, mask)
    tl.store(output + dst + height * width, g, mask)
    tl.store(output + dst + 2 * height * width, b, mask)


def nv12_rgb(video, height, color_space):
    count, _, width = video.shape
    output = torch.empty(
        (count, 3, height, width), device=video.device, dtype=torch.uint8
    )
    if output.numel():
        coefficients = (
            (14686, 17305, -1747, -4366)
            if color_space == 1
            else (13075, 16525, -3209, -6660)
        )
        with torch.cuda.device(video.device):
            _nv12_rgb_kernel[(triton.cdiv(count * height * width, 256),)](
                video,
                output,
                count,
                height,
                width,
                *coefficients,
                BLOCK=256,
            )
    return output
