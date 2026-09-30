"""BF16 prefill indexer scoring with FP32 logits and head reduction."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_score(
    q,
    k,
    o,
    S: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    QB: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KB: tl.constexpr,
    KN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    b = tl.program_id(0)
    group = tl.program_id(1)
    col = tl.program_id(2)
    row = group * (BM // H) + tl.arange(0, BM) // H
    head = tl.arange(0, BM) % H
    d = tl.arange(0, D)
    n = col * BN + tl.arange(0, BN)
    qv = tl.load(
        q + b * QB + row[:, None] * QS + head[:, None] * QH + d[None, :],
        row[:, None] < S,
        other=0,
    )
    kv = tl.load(k + b * KB + n[None, :] * KN + d[:, None], n[None, :] < N, other=0)
    logits = tl.dot(qv, kv)
    logits = tl.maximum(logits, 0.0)
    scores = tl.sum(tl.reshape(logits, (BM // H, H, BN)), 1) * (D**-0.5)
    pos = group * (BM // H) + tl.arange(0, BM // H)
    tl.store(
        o + (b * S + pos[:, None]) * N + n[None, :],
        scores,
        (pos[:, None] < S) & (n[None, :] < N),
    )


def try_prefill_score(q, block_keys):
    """Return FP32 block scores for supported prefill, else None.

    BF16 inputs use FP32 dot accumulation. Top-k and causal masking remain
    at the caller so the selection layout and invisible sentinel stay intact.
    """
    if os.environ.get("RTP_LLM_QWEN4_FUSED_PREFILL_SCORE", "0") != "1":
        return None
    if q.dim() != 4 or block_keys.dim() != 3:
        return None
    b, s, h, d = q.shape
    n = block_keys.shape[1]
    if not (
        q.is_cuda
        and q.device == block_keys.device
        and q.dtype == block_keys.dtype == torch.bfloat16
        and block_keys.shape[0] == b
        and block_keys.shape[2] == d
        and h == 4
        and d == 128
        and 128 <= s <= 8192
        and 32 <= n <= 2048
        and q.stride(-1) == block_keys.stride(-1) == 1
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device)[0] >= 8
    ):
        return None
    out = torch.empty((b, s, n), device=q.device, dtype=torch.float32)
    _prefill_score[(b, triton.cdiv(s, 16), triton.cdiv(n, 128))](
        q,
        block_keys,
        out,
        S=s,
        N=n,
        D=d,
        H=h,
        QB=q.stride(0),
        QS=q.stride(1),
        QH=q.stride(2),
        KB=block_keys.stride(0),
        KN=block_keys.stride(1),
        BM=64,
        BN=128,
        num_warps=4,
        num_stages=1,
    )
    return out
