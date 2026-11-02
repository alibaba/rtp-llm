"""TMA prefill for the causal, four-token-block QSA indexer layout.

The indexer emits up to 512 unique complete blocks followed by up to three tail
positions, with -1 padding. A packed membership mask lets adjacent queries
share contiguous KV tiles while retaining each query's selected set.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _build_membership(Ids, Mask, S: tl.constexpr, WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, 512)
    start = tl.load(Ids + row * 2051 + block * 4)
    bits = tl.full((512,), 15, tl.uint32) << (start % 32)
    tl.atomic_or(
        Mask + row * WORDS + start // 32,
        bits,
        (start >= 0) & (start + 3 < S),
        sem="relaxed",
    )
    tail = tl.arange(0, 4)
    index = tl.load(Ids + row * 2051 + 2048 + tail, tail < 3, other=-1)
    tail_bits = tl.full((4,), 1, tl.uint32) << (index % 32)
    tl.atomic_or(
        Mask + row * WORDS + index // 32,
        tail_bits,
        (tail < 3) & (index >= 0) & (index < S),
        sem="relaxed",
    )


@triton.jit
def _tiled_prefill(
    Q,
    K,
    V,
    Mask,
    O,
    S: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    WORDS: tl.constexpr,
    QM: tl.constexpr,
    KN: tl.constexpr,
    POSITIONS: tl.constexpr,
    QH_STRIDE: tl.constexpr,
    QS_STRIDE: tl.constexpr,
    OH_STRIDE: tl.constexpr,
    OS_STRIDE: tl.constexpr,
):
    tile = tl.cdiv(S, POSITIONS) - 1 - tl.program_id(0)
    group = tl.program_id(1)
    r = tl.arange(0, QM)
    d = tl.arange(0, D)
    c = tl.arange(0, KN)
    pos = tile * POSITIONS + r // G
    live = (r < POSITIONS * G) & (pos < S)
    h = group * G + r % G
    q = tl.load(
        Q + h[:, None] * QH_STRIDE + pos[:, None] * QS_STRIDE + d[None, :],
        live[:, None],
        other=0.0,
    )
    maximum = tl.full((QM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((QM,), tl.float32)
    acc = tl.zeros((QM, D), tl.float32)
    limit = tl.minimum(S, (tile + 1) * POSITIONS)
    for t in range(tl.cdiv(limit, KN)):
        column = t * KN + c
        bits = tl.load(
            Mask + pos[:, None] * WORDS + column[None, :] // 32,
            live[:, None] & (column[None, :] < S),
            other=0,
        ).to(tl.uint32)
        selected = ((bits >> (column[None, :] % 32)) & 1) != 0
        k = K.load([t * KN, 0])
        k = tl.where(column[:, None] < S, k, 0.0)
        scores = tl.dot(q, tl.trans(k)) * (D**-0.5 * 1.4426950408889634)
        v = V.load([t * KN, 0])
        v = tl.where(column[:, None] < S, v, 0.0)
        scores = tl.where(selected, scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(scores - safe_max[:, None])
        hi = probability.to(tl.bfloat16)
        lo = (probability - hi.to(tl.float32)).to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(hi, v, acc)
        acc = tl.dot(lo, v, acc)
        normalizer = normalizer * alpha + tl.sum(probability, 1)
        maximum = next_max
    output = tl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
    tl.store(
        O + h[:, None] * OH_STRIDE + pos[:, None] * OS_STRIDE + d[None, :],
        output.to(O.dtype.element_ty),
        live[:, None],
    )


def try_causal_qsa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Consume the model indexer's unique causal selection, or return None.

    The caller owns the selection provenance. Arbitrary or repeated injected
    indices must keep the general sparse kernel, which preserves multiplicity.
    Q/K/V may be views into a packed QKV projection; the returned view already
    has token-major storage so its consumer can flatten it without a copy.
    """
    if not (
        q.ndim == k.ndim == v.ndim == 4
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[0] == k.shape[0] == v.shape[0] == 1
        and k.shape == v.shape
        and k.shape[1] == 1
        and q.shape[1] in (6, 12)
        and q.shape[2] == k.shape[2]
        and 128 <= q.shape[2] <= 4096
        and q.shape[3] == k.shape[3] == 256
        and all(t.device == q.device for t in (k, v, selected))
        and all(t.stride(-1) == 1 for t in (q, k, v))
        and selected.dtype == torch.int32
        and selected.shape == (1, q.shape[2], 2051)
        and selected.is_contiguous()
        and all(t.stride(2) >= 256 and t.stride(2) % 8 == 0 for t in (k, v))
        and all(t.data_ptr() % 16 == 0 for t in (k, v))
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    ):
        return None
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        return None

    _, heads, tokens, dim = q.shape
    # Keep the accumulator and KV operands within the register/SMEM budget.
    # Larger head groups split across CTAs when sharing longer query tiles.
    if tokens <= 512:
        query_tile, group_heads = (16, 12) if heads == 12 else (32, 6)
    else:
        query_tile, group_heads = 32, 4 if heads == 12 else 6
    key_tile = 32
    words = triton.cdiv(tokens, 32)
    membership = torch.zeros((tokens, words), dtype=torch.int32, device=q.device)
    _build_membership[(tokens,)](selected, membership, tokens, words, num_warps=4)
    output = torch.empty((1, tokens, heads, dim), dtype=q.dtype, device=q.device)
    output_view = output.transpose(1, 2)
    key = TensorDescriptor(k, [tokens, dim], [k.stride(2), 1], [key_tile, dim])
    value = TensorDescriptor(v, [tokens, dim], [v.stride(2), 1], [key_tile, dim])
    positions = query_tile // group_heads
    _tiled_prefill[(triton.cdiv(tokens, positions), heads // group_heads)](
        q,
        key,
        value,
        membership,
        output_view,
        tokens,
        dim,
        group_heads,
        words,
        query_tile,
        key_tile,
        positions,
        q.stride(1),
        q.stride(2),
        output_view.stride(1),
        output_view.stride(2),
        num_warps=4,
        num_stages=1,
    )
    return output_view
