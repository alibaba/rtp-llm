"""SM120 Prefill packing, original BF16 L2 boundaries and CP gate preparation."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=["ROWS"])
def _prepare(
    X,
    A,
    B,
    Alog,
    Bias,
    Q,
    K,
    V,
    Alpha,
    Beta,
    ROWS,
    KH: tl.constexpr,
    VH: tl.constexpr,
    XSTRIDE: tl.constexpr,
    ABSTRIDE: tl.constexpr,
    BT: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.program_id(1) * BT + tl.arange(0, BT)
    cols = tl.arange(0, 128)
    mask = rows[:, None] < ROWS
    q = tl.load(
        X + rows[:, None] * XSTRIDE + head * 128 + cols[None, :], mask, other=0
    ).to(tl.float32)
    k = tl.load(
        X + rows[:, None] * XSTRIDE + (KH + head) * 128 + cols[None, :], mask, other=0
    ).to(tl.float32)
    qi = 1.0 / tl.sqrt(tl.sum(q * q, 1) + 1e-6)
    ki = 1.0 / tl.sqrt(tl.sum(k * k, 1) + 1e-6)
    tl.store(
        Q + (rows[:, None] * KH + head) * 128 + cols[None, :], q * qi[:, None], mask
    )
    tl.store(
        K + (rows[:, None] * KH + head) * 128 + cols[None, :], k * ki[:, None], mask
    )
    for part in tl.static_range(VH // KH):
        h = head * (VH // KH) + part
        value = tl.load(
            X + rows[:, None] * XSTRIDE + (2 * KH + h) * 128 + cols[None, :],
            mask,
            other=0,
        )
        tl.store(V + (rows[:, None] * VH + h) * 128 + cols[None, :], value, mask)
        aa = tl.load(A + rows * ABSTRIDE + h, rows < ROWS, other=0).to(tl.float32)
        bb = tl.load(B + rows * ABSTRIDE + h, rows < ROWS, other=0).to(tl.float32)
        bias = tl.load(Bias + h).to(tl.float32)
        alog = tl.load(Alog + h).to(tl.float32)
        z = aa + bias
        softplus = tl.where(z <= 20.0, tl.log(1.0 + tl.exp(z)), z)
        g = -tl.exp(alog) * softplus
        tl.store(Alpha + rows * VH + h, libdevice.exp(g), rows < ROWS)
        # Preserve the existing CUDA gating output dtype before CP promotion.
        beta = tl.sigmoid(bb).to(B.dtype.element_ty).to(tl.float32)
        tl.store(Beta + rows * VH + h, beta, rows < ROWS)


def is_supported(mixed_qkv, a, b, alog, bias, key_heads, value_heads):
    return (
        mixed_qkv.is_cuda
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype in (torch.bfloat16, torch.float32)
        and b.dtype in (torch.bfloat16, torch.float32)
        and mixed_qkv.ndim == 2
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) >= mixed_qkv.shape[1]
        and mixed_qkv.shape[1] == (2 * key_heads + value_heads) * 128
        and key_heads > 0
        and value_heads % key_heads == 0
        and value_heads // key_heads in (1, 2, 3, 4)
        and a.shape == b.shape == (mixed_qkv.shape[0], value_heads)
        and a.stride(0) == b.stride(0)
        and a.stride(1) == b.stride(1) == 1
        and alog.numel() == bias.numel() == value_heads
        and alog.is_contiguous()
        and bias.is_contiguous()
        and mixed_qkv.device == a.device == b.device == alog.device == bias.device
    )


def prepare_sm120_prefill(mixed_qkv, a, b, alog, bias, key_heads, value_heads):
    if not is_supported(mixed_qkv, a, b, alog, bias, key_heads, value_heads):
        metadata = [
            dict(shape=tuple(t.shape), stride=t.stride(), dtype=str(t.dtype))
            for t in (mixed_qkv, a, b, alog, bias)
        ]
        raise ValueError(
            f"unsupported SM120 GDN preparation kh={key_heads}, vh={value_heads}, tensors={metadata}"
        )
    rows = mixed_qkv.shape[0]
    q = mixed_qkv.new_empty((rows, key_heads, 128))
    k = torch.empty_like(q)
    v = mixed_qkv.new_empty((rows, value_heads, 128))
    alpha = torch.empty(
        (rows, value_heads), device=mixed_qkv.device, dtype=torch.float32
    )
    beta = torch.empty_like(alpha)
    _prepare[(key_heads, triton.cdiv(rows, 4))](
        mixed_qkv,
        a,
        b,
        alog,
        bias,
        q,
        k,
        v,
        alpha,
        beta,
        ROWS=rows,
        KH=key_heads,
        VH=value_heads,
        XSTRIDE=mixed_qkv.stride(0),
        ABSTRIDE=a.stride(0),
        BT=4,
        num_warps=16,
    )
    return q, k, v, alpha, beta


@triton.jit
def _checkpoint_starts(Cu, Starts, BATCH: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    begin = tl.load(Cu + b, b < BATCH, other=0)
    end = tl.load(Cu + b + 1, b < BATCH, other=0)
    pages = tl.where(b < BATCH, (end - begin) // 128, 0)
    inclusive = tl.cumsum(pages, 0).to(tl.int64)
    tl.store(Starts, 0)
    tl.store(Starts + b + 1, inclusive, b < BATCH)


def checkpoint_starts(cu_seqlens, batch):
    starts = torch.empty(batch + 1, device=cu_seqlens.device, dtype=torch.int64)
    _checkpoint_starts[(1,)](
        cu_seqlens,
        starts,
        BATCH=batch,
        BLOCK=triton.next_power_of_2(batch),
        num_warps=1,
    )
    return starts
