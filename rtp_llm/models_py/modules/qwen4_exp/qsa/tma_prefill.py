"""TMA prefill for the causal, four-token-block QSA indexer layout.

The indexer emits up to 512 unique complete blocks followed by up to three tail
positions, with -1 padding. A packed membership mask lets adjacent queries
share contiguous KV tiles while retaining each query's selected set.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S", "WORDS"])
def _build_membership(Ids, Mask, S, WORDS, MAX_WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    word = tl.arange(0, MAX_WORDS)
    tl.store(Mask + row * WORDS + word, 0, word < WORDS)
    # Every row belongs to one CTA; finish initialization before any atomic OR.
    tl.debug_barrier()
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
        and 1 <= q.shape[2] <= 4096
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
        from .tma_pipeline import run_prefill
    except ImportError:
        run_prefill = None
        try:
            from triton.tools.tensor_descriptor import TensorDescriptor
        except ImportError:
            return None

    _, heads, tokens, dim = q.shape
    words = triton.cdiv(tokens, 32)
    membership = torch.empty((tokens, words), dtype=torch.int32, device=q.device)
    _build_membership[(tokens,)](
        selected,
        membership,
        tokens,
        words,
        16 if tokens <= 512 else 128,
        num_warps=4,
    )
    if run_prefill is not None:
        return run_prefill(q, k, v, membership)
    # Keep the standard Triton path for installations without Gluon.
    if tokens <= 512:
        query_tile, group_heads = (16, 12) if heads == 12 else (32, 6)
    else:
        query_tile, group_heads = 32, 4 if heads == 12 else 6
    key_tile = 32
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


def try_ragged_causal_qsa_prefill(q, k, v, selected, lengths, cu_seqlens):
    """Consume packed views with the indexer's validated request partition.

    The caller owns both selection and CU provenance. Contents are not read
    back here; unrelated or injected metadata must keep the per-request path.
    """
    if not (
        2 <= len(lengths) <= 32
        and min(lengths) > 0
        and max(lengths) <= 1024
        and q.ndim == k.ndim == v.ndim == 2
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and k.shape == v.shape == (q.shape[0], 256)
        and q.shape[1] in (6 * 256, 12 * 256)
        and sum(lengths) == q.shape[0]
        and selected.shape == (q.shape[0], 2051)
        and selected.dtype == torch.int32
        and selected.is_contiguous()
        and isinstance(cu_seqlens, torch.Tensor)
        and cu_seqlens.shape == (len(lengths) + 1,)
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.is_contiguous()
        and all(t.device == q.device for t in (k, v, selected, cu_seqlens))
        and all(t.stride(1) == 1 for t in (q, k, v))
        and all(t.stride(0) >= 256 and t.stride(0) % 8 == 0 for t in (k, v))
        and all(t.data_ptr() % 16 == 0 for t in (k, v))
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    ):
        return None
    try:
        from .tma_ragged_pipeline import run_prefill
    except ImportError:
        return None
    tokens, width = q.shape
    q = q.view(1, tokens, width // 256, 256).transpose(1, 2)
    k = k.view(1, tokens, 1, 256).transpose(1, 2)
    v = v.view(1, tokens, 1, 256).transpose(1, 2)
    return run_prefill(q, k, v, selected, cu_seqlens, lengths)


def try_paged_causal_qsa_prefill(
    q, cache, table, selected, *, prefix, page_size, kv_head_num, kv_scale=None
):
    """Consume a prevalidated indexer selection and paged request, or return None.

    The caller validates visible indices and physical pages before writing KV.
    Prefix is its CPU plan snapshot; this helper never reads metadata back from
    the device. Arbitrary injected selections retain the general sparse path.
    """
    if not (
        isinstance(prefix, int)
        and prefix >= 0
        and q.ndim == 4
        and q.shape[0] == 1
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == torch.bfloat16
        and q.shape[1] in (6, 12)
        and q.shape[3] == 256
        and 128 <= q.shape[2]
        and prefix + q.shape[2] <= 4096
        and q.stride(-1) == 1
        and min(q.stride()) > 0
        and kv_head_num == 1
        and page_size == 128
        and cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and cache.ndim in (2, 5)
        and cache.shape[0] > 0
        and all(t.device == q.device for t in (cache, table, selected))
        and cache.data_ptr() % 16 == 0
        and cache.stride(0) % 256 == 0
        and cache.stride(0) >= 2 * page_size * 256
        and cache.shape[0] * (cache.stride(0) // 256) < 2**31
        and table.ndim == 2
        and table.shape[0] == 1
        and table.shape[1] >= triton.cdiv(prefix + q.shape[2], page_size)
        and table.dtype == torch.int32
        and table.stride(1) == 1
        and selected.shape == (1, q.shape[2], 2051)
        and selected.dtype == torch.int32
        and selected.is_contiguous()
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    ):
        return None
    if cache.ndim == 2:
        if cache.shape[1] < 2 * page_size * 256 or cache.stride(1) != 1:
            return None
    elif cache.shape[1:] != (2, 1, page_size, 256) or cache.stride()[1:] != (
        page_size * 256,
        page_size * 256,
        256,
        1,
    ):
        return None
    if cache.dtype == torch.float8_e4m3fn:
        if not (
            kv_scale is not None
            and kv_scale.device == q.device
            and kv_scale.dtype == torch.float32
            and kv_scale.shape == (cache.shape[0], 2, 1, page_size)
            and kv_scale.stride()[1:] == (page_size, page_size, 1)
            and kv_scale.stride(0) >= 2 * page_size
        ):
            return None
    else:
        kv_scale = None
    try:
        from .tma_paged_pipeline import run_prefill
    except ImportError:
        return None
    return run_prefill(q, cache, table, selected, prefix, page_size, kv_scale)
