"""Canonical CUDA arithmetic for the dense-v1 batch invariant profile.

No launch parameter or reduction partition depends on the number of rows.
Attention visits fixed logical key blocks, ending at the query's causal position;
packing, future tokens and physical KV page placement cannot change that order.
These deliberately conservative kernels are a correctness baseline, not a tuned
replacement for the default execution path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(
            X + m[:, None] * XS + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        b = tl.load(
            W + n[None, :] * W0 + kk[:, None] * W1,
            (n[None, :] < N) & (kk[:, None] < K),
            0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    if HAS_BIAS:
        acc += tl.load(B + n, n < N, 0)[None, :].to(tl.float32)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def linear(x, weight, bias=None, *, output_dtype=None):
    """F.linear layout: weight is [out_features, in_features]."""
    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("batch invariant linear requires CUDA BF16 or FP32")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError(
            "batch invariant linear requires matching input/weight dtype and device"
        )
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("invalid batch invariant linear dimensions")
    if bias is not None and (
        bias.shape != (weight.shape[0],) or bias.device != x.device
    ):
        raise ValueError("invalid batch invariant linear bias")
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=output_dtype or x.dtype, device=x.device)
    if m:
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            x,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x.stride(0),
            *weight.stride(),
            bias is not None,
            num_warps=4,
        )
    return out.view(*shape, n)


@triton.jit
def _rmsnorm(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    var = tl.sum(x * x, 0) / H
    y = (x * tl.rsqrt(var + EPS)) * w
    tl.store(Y + row * H + col, y, col < H)


def rmsnorm(x, weight, eps):
    x = x.contiguous()
    h = x.shape[-1]
    if weight.shape != (h,) or x.device != weight.device:
        raise ValueError("invalid batch invariant RMSNorm weight")
    out = torch.empty_like(x)
    if x.numel():
        _rmsnorm[(x.numel() // h,)](
            x, weight, out, h, eps, triton.next_power_of_2(h), num_warps=4
        )
    return out


@triton.jit
def _lse_parts(X, P, V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + i, i < V, -float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    total = tl.sum(tl.exp(x - maximum), 0)
    tl.store(P + row * PARTS * 2 + part * 2, maximum)
    tl.store(P + row * PARTS * 2 + part * 2 + 1, total)


@triton.jit
def _lse_merge(P, Y, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    maximum = tl.load(P + row * PARTS * 2 + i * 2, i < PARTS, -float("inf"))
    total = tl.load(P + row * PARTS * 2 + i * 2 + 1, i < PARTS, 0)
    m = tl.max(maximum, 0)
    s = tl.sum(total * tl.exp(maximum - m), 0)
    tl.store(Y + row, m + tl.log(s))


def logsumexp(logits):
    if logits.ndim != 2 or logits.shape[1] == 0 or not logits.is_cuda:
        raise ValueError(
            "batch invariant logsumexp requires nonempty CUDA vocabulary rows"
        )
    x = logits.contiguous()
    rows, vocab = x.shape
    parts = triton.cdiv(vocab, 1024)
    partials = torch.empty((rows, parts, 2), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        _lse_parts[(rows, parts)](x, partials, vocab, parts, 1024, num_warps=4)
        _lse_merge[(rows,)](
            partials, out, parts, triton.next_power_of_2(parts), num_warps=4
        )
    return out


def selected_logprobs(logits, token_ids):
    if token_ids.numel() != logits.shape[0]:
        raise ValueError("one target token is required per logits row")
    return logits.float().gather(1, token_ids.reshape(-1, 1).long()) - logsumexp(logits)


@triton.jit
def _paged_attention(
    Q,
    CACHE,
    TABLE,
    REQUEST,
    POSITION,
    OUT,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    CS3: tl.constexpr,
    CS4: tl.constexpr,
    TS: tl.constexpr,
    H: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BD: tl.constexpr,
    BLOCK: tl.constexpr = 32,
):
    row, head = tl.program_id(0), tl.program_id(1)
    req = tl.load(REQUEST + row)
    pos = tl.load(POSITION + row)
    d = tl.arange(0, BD)
    q = tl.load(Q + row * QS0 + head * QS1 + d, d < D, 0).to(tl.float32)
    kh = head // (H // KH)
    maximum = tl.full((), -float("inf"), tl.float32)
    total = tl.full((), 0, tl.float32)
    accum = tl.zeros((BD,), tl.float32)
    for block in range(tl.cdiv(pos + 1, BLOCK)):
        keypos = block * BLOCK + tl.arange(0, BLOCK)
        valid = keypos <= pos
        page = tl.load(TABLE + req * TS + keypos // PAGE, valid, 0)
        offsets = (
            page[:, None] * CS0
            + kh * CS2
            + (keypos % PAGE)[:, None] * CS3
            + d[None, :] * CS4
        )
        k = tl.load(CACHE + offsets, valid[:, None] & (d[None, :] < D), 0).to(
            tl.float32
        )
        v = tl.load(CACHE + CS1 + offsets, valid[:, None] & (d[None, :] < D), 0).to(
            tl.float32
        )
        scores = tl.sum(k * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 0))
        correction = tl.exp(maximum - next_max)
        probs = tl.exp(scores - next_max)
        accum = accum * correction + tl.sum(probs[:, None] * v, 0)
        total = total * correction + tl.sum(probs, 0)
        maximum = next_max
    tl.store(OUT + (row * H + head) * D + d, accum / total, d < D)


def paged_attention(q, cache, block_table, request_ids, positions):
    """Causal GQA over [pages, 2, kv_heads, page_size, head_dim] KV."""
    if q.dtype != torch.bfloat16 or cache.dtype != q.dtype:
        raise ValueError("dense-v1 attention requires BF16 Q and KV")
    if cache.ndim != 5 or cache.shape[1] != 2 or cache.shape[-1] != q.shape[-1]:
        raise ValueError("invalid batch invariant paged KV layout")
    h, d = q.shape[1:]
    kh, page = cache.shape[2:4]
    if h % kh or d not in (64, 128, 256) or q.stride(-1) != 1:
        raise ValueError("unsupported batch invariant GQA shape")
    if request_ids.numel() != q.shape[0] or positions.numel() != q.shape[0]:
        raise ValueError("one request and position are required per query")
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if q.shape[0]:
        _paged_attention[(q.shape[0], h)](
            q,
            cache,
            block_table,
            request_ids,
            positions,
            out,
            q.stride(0),
            q.stride(1),
            *cache.stride(),
            block_table.stride(0),
            h,
            kh,
            d,
            page,
            d**-0.5,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
