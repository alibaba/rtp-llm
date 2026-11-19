"""GLM four-tap arithmetic with K3's contiguous Q/K/V output planes."""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def glm_conv_decode_kernel(
    x,
    w,
    s,
    bm,
    lens,
    o,
    B: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    BS: tl.constexpr,
    PAGES: tl.constexpr,
    P: tl.constexpr,
    PAGE: tl.constexpr,
    N: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1).to(tl.int64) * N + tl.arange(0, N).to(tl.int64)
    mask = d < 3 * C
    seq = tl.load(lens + b).to(tl.int64)
    rp = (seq - 2) // PAGE
    wp = (seq - 1) // PAGE
    ri = tl.load(bm + b * BS + rp, mask=(seq > 1) & (rp < PAGES), other=0).to(tl.int64)
    wi = tl.load(bm + b * BS + wp, mask=(seq > 0) & (wp < PAGES), other=0).to(tl.int64)
    valid = (ri > 0) & (ri < P)
    h0 = tl.load(s + ri * SS0 + d * SS2, mask=mask & valid, other=0)
    h1 = tl.load(s + ri * SS0 + SS1 + d * SS2, mask=mask & valid, other=0)
    h2 = tl.load(s + ri * SS0 + 2 * SS1 + d * SS2, mask=mask & valid, other=0)
    cur = tl.load(x + b * XS + d, mask=mask, other=0).to(s.dtype.element_ty)
    w0 = tl.load(w + d * WS0, mask=mask, other=0)
    w1 = tl.load(w + d * WS0 + WS1, mask=mask, other=0)
    w2 = tl.load(w + d * WS0 + 2 * WS1, mask=mask, other=0)
    w3 = tl.load(w + d * WS0 + 3 * WS1, mask=mask, other=0)
    acc = tl.full((N,), 0, tl.float32)
    acc += h0 * w0
    acc += h1 * w1
    acc += h2 * w2
    acc += cur * w3
    acc = acc / (1 + tl.exp(-acc))
    tl.store(
        o + (d // C) * B * C + b * C + d % C, acc.to(s.dtype.element_ty), mask=mask
    )
    wm = mask & (wi > 0) & (wi < P)
    tl.store(s + wi * SS0 + d * SS2, h1, mask=wm)
    tl.store(s + wi * SS0 + SS1 + d * SS2, h2, mask=wm)
    tl.store(s + wi * SS0 + 2 * SS1 + d * SS2, cur, mask=wm)


def kda_short_conv_decode(x, w, s, bm, lens, page):
    if x.ndim != 2 or x.shape[1] % 3 or x.stride(1) != 1:
        raise ValueError(
            "GLM KDA convolution requires dense channels in [batch, 3*channels]"
        )
    if (
        x.dtype != torch.bfloat16
        or w.dtype not in (torch.bfloat16, torch.float32)
        or s.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "GLM KDA convolution requires BF16 input and BF16 or FP32 weights/history"
        )
    b, n = x.shape
    c = n // 3
    if c == 0 or s.ndim != 3 or s.shape[0] == 0:
        raise ValueError(
            "GLM KDA convolution requires nonempty channels and cache pages"
        )
    if w.shape != (n, 4) or s.ndim != 3 or tuple(s.shape[1:]) != (3, n):
        raise ValueError("GLM KDA convolution requires four taps and packed history")
    if bm.ndim != 2 or bm.shape[0] != b or lens.shape != (b,) or page <= 0:
        raise ValueError("GLM KDA convolution metadata does not match batch")
    if any(not t.is_cuda or t.device != x.device for t in (x, w, s, bm, lens)):
        raise ValueError("GLM KDA convolution tensors must share a CUDA device")
    if (
        bm.shape[1] == 0
        or bm.stride(1) != 1
        or lens.stride(0) != 1
        or bm.dtype not in (torch.int32, torch.int64)
        or lens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "GLM KDA convolution requires contiguous integer page/length metadata"
        )
    o = torch.empty((3, b, c), device=x.device, dtype=x.dtype)
    if b == 0:
        return o[0], o[1], o[2]
    glm_conv_decode_kernel[(b, triton.cdiv(n, BLOCK))](
        x,
        w,
        s,
        bm,
        lens,
        o,
        b,
        c,
        x.stride(0),
        *w.stride(),
        *s.stride(),
        bm.stride(0),
        bm.shape[1],
        s.shape[0],
        page,
        BLOCK
    )
    return o[0], o[1], o[2]


@triton.jit
def _kda_verify_conv_kernel(
    x,
    w,
    s,
    bm,
    lens,
    out,
    B: tl.int64,
    T: tl.int64,
    C: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    SS: tl.constexpr,
    BS: tl.int64,
    PAGES: tl.int64,
    POOL: tl.int64,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = d < 3 * C
    seq = tl.load(lens + n).to(tl.int64)
    rp = (seq - 2) // PAGE
    wp = (seq - 1) // PAGE
    ri = tl.load(bm + n * BS + rp, (seq > 1) & (rp >= 0) & (rp < PAGES), 0).to(tl.int64)
    valid = (ri > 0) & (ri < POOL)
    h0 = tl.load(s + ri * SS + d, mask & valid, 0)
    h1 = tl.load(s + ri * SS + 3 * C + d, mask & valid, 0)
    h2 = tl.load(s + ri * SS + 6 * C + d, mask & valid, 0)
    w0 = tl.load(w + d * WS, mask, 0)
    w1 = tl.load(w + d * WS + 1, mask, 0)
    w2 = tl.load(w + d * WS + 2, mask, 0)
    w3 = tl.load(w + d * WS + 3, mask, 0)
    for t in range(T):
        cur = tl.load(x + (n * T + t) * XS + d, mask, 0).to(s.dtype.element_ty)
        acc = tl.full((BLOCK,), 0, tl.float32)
        acc += h0 * w0
        acc += h1 * w1
        acc += h2 * w2
        acc += cur * w3
        acc = (acc / (1 + tl.exp(-acc))).to(out.dtype.element_ty)
        tl.store(out + (d // C) * B * T * C + (n * T + t) * C + d % C, acc, mask)
        wi = tl.load(bm + n * BS + wp + t, (wp >= 0) & (wp + t < PAGES), 0).to(tl.int64)
        wm = mask & (wi > 0) & (wi < POOL)
        tl.store(s + wi * SS + d, h1, wm)
        tl.store(s + wi * SS + 3 * C + d, h2, wm)
        tl.store(s + wi * SS + 6 * C + d, cur, wm)
        h0, h1, h2 = h1, h2, cur


def kda_short_conv_verify_supported(x, w, s, page, batch, tokens):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.float32
        and s.dtype == torch.bfloat16
        and x.ndim == 2
        and x.stride(1) == 1
        and x.shape[0] == batch * tokens
        and x.shape[1] % 3 == 0
        and tuple(w.shape) == (x.shape[1], 4)
        and w.stride(1) == 1
        and s.ndim == 3
        and tuple(s.shape[1:]) == (3, x.shape[1])
        and s.stride(2) == 1
        and batch > 0
        and 1 <= tokens <= 8
        and page >= tokens
    )


def kda_short_conv_verify(x, w, s, bm, lens, page, batch, tokens):
    """Four-tap verify convolution writing the per-token state tape."""
    if x.ndim != 2 or x.shape[0] != batch * tokens or x.shape[1] % 3:
        raise ValueError("KDA verify expects request-major [batch*tokens, 3*channels]")
    if (
        x.dtype != torch.bfloat16
        or w.dtype != torch.float32
        or s.dtype != torch.bfloat16
    ):
        raise ValueError(
            "GLM verify requires BF16 activations/history and FP32 convolution weights"
        )
    channels = x.shape[1] // 3
    if (
        w.shape != (3 * channels, 4)
        or w.stride(1) != 1
        or tuple(s.shape[1:]) != (3, 3 * channels)
        or s.stride(2) != 1
    ):
        raise ValueError("KDA verify requires four taps and packed history")
    if (
        batch <= 0
        or not 1 <= tokens <= 8
        or page < tokens
        or lens.shape != (batch,)
        or bm.ndim != 2
        or bm.shape[0] != batch
    ):
        raise ValueError("KDA verify metadata/width is unsupported")
    if (
        any(not t.is_cuda or t.device != x.device for t in (x, w, s, bm, lens))
        or bm.stride(1) != 1
    ):
        raise ValueError(
            "KDA verify tensors must share a CUDA device and dense page-table rows"
        )
    out = torch.empty((3, batch, tokens, channels), device=x.device, dtype=x.dtype)
    _kda_verify_conv_kernel[(batch, triton.cdiv(3 * channels, BLOCK))](
        x,
        w,
        s,
        bm,
        lens,
        out,
        batch,
        tokens,
        channels,
        x.stride(0),
        w.stride(0),
        s.stride(0),
        bm.stride(0),
        bm.shape[1],
        s.shape[0],
        page,
        BLOCK,
    )
    return out[0], out[1], out[2]
