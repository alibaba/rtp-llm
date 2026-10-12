"""SM120 BF16 sparse prefill for the upstream V4.1 attention owner.

Uses the qualified _v41_attention_triton online softmax and BF16 probability
contraction, with BF16 workspace gathers in place of paged FP8/FP4 decoding.
Fresh SWA values must not be requantized merely to select an SM120 backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_prefill(
    Q,
    KV,
    IDX,
    LENS,
    SINK,
    OUT,
    COUNT,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    DIRECT_SINK_DIV_RN: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    address = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    q = tl.load(Q + address, heads[:, None] < HEADS, other=0)
    limit = tl.load(LENS + row)
    acc = tl.full((16, 512), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    for tile in range(tl.cdiv(WIDTH, 32)):
        offset = tile * 32 + tl.arange(0, 32)
        slot = tl.load(IDX + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        live = (offset < WIDTH) & (offset < limit)
        tl.device_assert(
            ~live | ((slot >= -1) & (slot < COUNT)), "V4.1 invalid BF16 KV index"
        )
        valid = live & (slot >= 0) & (slot < COUNT)
        kv = tl.load(
            KV + slot[:, None] * 512 + columns[None, :], valid[:, None], other=0
        )
        scores = tl.dot(q, tl.trans(kv)) * (512**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - finite_max)
        probabilities = tl.exp(scores - finite_max[:, None])
        acc = tl.dot(probabilities.to(tl.bfloat16), kv, acc * alpha[:, None])
        total = total * alpha + tl.sum(probabilities, 1)
        maximum = next_max
    sink = tl.load(SINK + heads, heads < HEADS, other=0)
    if DIRECT_SINK_DIV_RN:
        # Test-only reference epilogue. Empty/masked rows have no key maximum
        # and must stay zero even when the sink's mass underflows.
        has_keys = total > 0
        finite_max = tl.where(has_keys, maximum, 0.0)
        denominator = total + tl.exp(sink - finite_max)
        result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    else:
        next_max = tl.maximum(maximum, sink)
        alpha = tl.exp(maximum - next_max)
        denominator = total * alpha + tl.exp(sink - next_max)
        result = acc * (alpha / denominator)[:, None]
    tl.store(OUT + address, result, heads[:, None] < HEADS)


def sparse_prefill_attention(query, kv, indices, lengths, sinks):
    """Consume upstream BF16 [K,1,512] and bounded [Q,1,TopK] index plans."""
    if not (
        query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (12, 0)
        and query.ndim == 3
        and query.shape[-1] == 512
        and 1 <= query.shape[1] <= 64
        and query.dtype == torch.bfloat16
        and query.is_contiguous()
        and kv.ndim == 3
        and kv.shape[1:] == (1, 512)
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and indices.ndim == 3
        and indices.shape[:2] == (query.shape[0], 1)
        and 0 < indices.shape[-1] <= 640
        and indices.dtype in (torch.int32, torch.int64)
        and indices.is_contiguous()
        and lengths.shape == (query.shape[0],)
        and lengths.dtype in (torch.int32, torch.int64)
        and lengths.is_contiguous()
        and sinks.shape == (query.shape[1],)
        and sinks.dtype == torch.float32
        and sinks.is_contiguous()
        and all(t.device == query.device for t in (kv, indices, lengths, sinks))
    ):
        raise ValueError("Unsupported SM120 V4.1 sparse BF16 prefill layout")
    output = torch.empty_like(query)
    if query.shape[0]:
        _sparse_prefill[(query.shape[0], triton.cdiv(query.shape[1], 16))](
            query,
            kv,
            indices,
            lengths,
            sinks,
            output,
            kv.shape[0],
            HEADS=query.shape[1],
            WIDTH=indices.shape[-1],
            num_warps=4,
            num_stages=1,
        )
    return output
