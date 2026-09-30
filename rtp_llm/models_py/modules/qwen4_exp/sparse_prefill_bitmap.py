"""Opt-in BF16 GQA prefill using a membership bitmap across query rows."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _membership(
    ids, mask, ends, S: tl.constexpr, T: tl.constexpr, K: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0)
    r = tl.arange(0, BK)
    idx = tl.load(ids + row * K + r, r < K, other=-1)
    valid = (r < K) & (idx >= 0) & (idx < T)
    tl.store(mask + row * T + idx, 1, valid)
    tl.store(ends + row, tl.max(tl.where(valid, idx, -1)) + 1)


@triton.jit
def _membership_words(
    ids,
    mask,
    byte_mask,
    ends,
    duplicates,
    T: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WORDS: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BK)
    idx = tl.load(ids + row * K + slots, slots < K, other=-1)
    valid = (slots < K) & (idx >= 0) & (idx < T)
    bit = tl.full((BK,), 1, tl.uint32) << (idx % 32)
    tl.store(byte_mask + row * T + idx, 1, valid)
    old = tl.atomic_or(
        mask + row * WORDS + idx // 32, bit, mask=valid, sem="relaxed"
    ).to(tl.uint32)
    repeated = valid & ((old & bit) != 0)
    invalid = (slots < K) & (idx >= T)
    tl.store(duplicates + row, tl.max((repeated | invalid).to(tl.int32), 0))
    tl.store(ends + row, tl.max(tl.where(valid, idx, -1)) + 1)


@triton.jit
def _bitmap_gqa(
    q,
    k,
    v,
    m,
    ends,
    o,
    S: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    QS: tl.constexpr,
    KT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    group = tl.program_id(2)
    # Five query positions use 15 of the tensor-core tile's 16 rows.
    r = tl.arange(0, 16)
    d = tl.arange(0, D)
    pos = group * QS + r // G
    live = (r < QS * G) & (pos < S)
    h = hk * G + r % G
    qv = tl.load(
        q + ((b * HQ + h[:, None]) * S + pos[:, None]) * D + d[None, :],
        live[:, None],
        other=0,
    )
    last = tl.max(tl.load(ends + b * S + pos, live, other=0))
    mx = tl.full((16,), float("-inf"), tl.float32)
    z = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, D), tl.float32)
    for tile in range(tl.cdiv(last, KT)):
        t = tile * KT + tl.arange(0, KT)
        valid = (
            tl.load(
                m + ((b * S + pos[:, None]) * T + t[None, :]),
                live[:, None] & (t[None, :] < T),
                other=0,
            )
            > 0
        )
        kval = tl.load(
            k + ((b * HK + hk) * T + t[:, None]) * D + d[None, :],
            t[:, None] < T,
            other=0,
        )
        logits = tl.dot(qv, tl.trans(kval)) * (D**-0.5)
        logits = tl.where(valid, logits, float("-inf"))
        nxt = tl.maximum(mx, tl.max(logits, 1))
        rescale = tl.where(nxt == float("-inf"), 0.0, tl.exp(mx - nxt))
        prob = tl.where(valid, tl.exp(logits - nxt[:, None]), 0.0)
        vv = tl.load(
            v + ((b * HK + hk) * T + t[:, None]) * D + d[None, :],
            t[:, None] < T,
            other=0,
        )
        acc = acc * rescale[:, None] + tl.dot(
            prob, vv.to(tl.float32), input_precision="tf32x3"
        )
        z = z * rescale + tl.sum(prob, 1)
        mx = nxt
    result = tl.where(z[:, None] > 0, acc / z[:, None], 0.0)
    tl.store(
        o + ((b * HQ + h[:, None]) * S + pos[:, None]) * D + d[None, :],
        result.to(o.dtype.element_ty),
        live[:, None],
    )


def try_bitmap_prefill(q, k, v, selected):
    """Return a BF16 result for supported unique-index prefill, else None.

    Called after sparse_prefill_attn validates metadata. A bitmap loses
    multiplicity, so repeated selected indices must use the existing path.
    """
    if not (
        q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and all(t.is_contiguous() for t in (q, k, v, selected))
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device)[0] >= 8
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    b, hq, s, d = q.shape
    hk, t, width = k.shape[1], k.shape[2], selected.shape[-1]
    if not (
        hq == 3 * hk
        and d == 256
        and 128 <= s <= 8192
        and t == s
        and 256 <= width <= 4096
        and b * s * t <= 64 * 1024 * 1024
    ):
        return None
    ends = torch.empty((b, s), device=q.device, dtype=torch.int32)
    membership = torch.zeros((b, s, t), device=q.device, dtype=torch.uint8)
    if os.environ.get("RTP_LLM_QWEN4_SPARSE_PREFILL_WORDS", "0") == "1":
        words = triton.cdiv(t, 32)
        # Keep the attention kernel and its accumulation order unchanged.
        # Packed words only replace the large byte-bitmap duplicate reduction.
        packed = torch.zeros((b, s, words), device=q.device, dtype=torch.int32)
        duplicates = torch.empty_like(ends)
        _membership_words[(b * s,)](
            selected,
            packed,
            membership,
            ends,
            duplicates,
            T=t,
            K=width,
            BK=triton.next_power_of_2(width),
            WORDS=words,
            num_warps=4,
        )
        if bool(torch.any(duplicates).item()):
            return None
    else:
        _membership[(b * s,)](
            selected,
            membership,
            ends,
            S=s,
            T=t,
            K=width,
            BK=triton.next_power_of_2(width),
            num_warps=4,
        )
        counts = (selected >= 0).sum(-1)
        if not bool(torch.all(membership.sum(-1) == counts).item()):
            return None
    out = torch.empty_like(q)
    _bitmap_gqa[(b, hk, triton.cdiv(s, 5))](
        q,
        k,
        v,
        membership,
        ends,
        out,
        S=s,
        T=t,
        HQ=hq,
        HK=hk,
        D=d,
        G=3,
        QS=5,
        KT=32,
        num_warps=4,
        num_stages=1,
    )
    return out
