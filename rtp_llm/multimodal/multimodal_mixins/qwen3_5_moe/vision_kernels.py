"""Fused inference kernels for Qwen3.5 vision preprocessing and rotary embedding."""

import torch
import triton
import triton.language as tl


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


@triton.jit
def _rope_kernel(
    q,
    k,
    cosine,
    sine,
    oq,
    ok,
    elements,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Packed QKV strides and the flattened output can exceed signed int32.
    # Widen before multiplying so both input and output offsets stay valid.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < elements
    token = i // (HEADS * DIM)
    head = (i // DIM) % HEADS
    col = i % DIM
    partner = (col + DIM // 2) % DIM
    sign = tl.where(col < DIM // 2, -1.0, 1.0)
    c = tl.load(cosine + token * C0 + col * C1, mask, other=0).to(tl.float32)
    s = tl.load(sine + token * S0 + col * S1, mask, other=0).to(tl.float32)
    qv = tl.load(q + token * Q0 + head * Q1 + col * Q2, mask, other=0).to(tl.float32)
    qr = tl.load(q + token * Q0 + head * Q1 + partner * Q2, mask, other=0).to(
        tl.float32
    )
    kv = tl.load(k + token * K0 + head * K1 + col * K2, mask, other=0).to(tl.float32)
    kr = tl.load(k + token * K0 + head * K1 + partner * K2, mask, other=0).to(
        tl.float32
    )
    tl.store(oq + i, qv * c + (qr * sign) * s, mask)
    tl.store(ok + i, kv * c + (kr * sign) * s, mask)


def rotary_embedding(q, k, cos, sin):
    # Materialize the FP32 sum before conversion. Folding the sum and BF16
    # store into one kernel can change halfway rounding on Blackwell.
    oq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    ok = torch.empty(k.shape, device=k.device, dtype=torch.float32)
    if q.numel():
        with torch.cuda.device(q.device):
            _rope_kernel[(triton.cdiv(q.numel(), 512),)](
                q,
                k,
                cos,
                sin,
                oq,
                ok,
                q.numel(),
                q.shape[1],
                q.shape[2],
                *q.stride(),
                *k.stride(),
                *cos.stride(),
                *sin.stride(),
                BLOCK=512,
                enable_fp_fusion=False,
            )
    return oq.to(q.dtype), ok.to(k.dtype)
