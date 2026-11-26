"""Fixed FP4 paged scores for the shared V4.1 index sources."""

import torch
import triton
import triton.language as tl

from ._v41_fp4_triton import _e2m1_to_float, _ue8m0_to_float


@triton.jit
def _score(
    query,
    q_scale,
    weights,
    pool,
    pages,
    lengths,
    requests,
    output,
    heads: tl.constexpr,
    width: tl.constexpr,
    table_stride: tl.constexpr,
    table_rows: tl.constexpr,
    pool_stride: tl.constexpr,
    pool_pages: tl.constexpr,
    logical_page: tl.constexpr,
    physical_page: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * block_n + tl.arange(0, block_n)
    h = tl.arange(0, block_h)
    d = tl.arange(0, 128)
    request = tl.load(requests + row).to(tl.int64)
    length = tl.load(lengths + row)
    valid = (n < width) & (n < length) & (request >= 0) & (request < table_rows)
    page = tl.load(
        pages + request * table_stride + n // logical_page,
        valid,
        other=0,
    )
    readable = valid & (page > 0) & (page < pool_pages)
    token = n % logical_page
    qrow = row * heads + h
    qp = tl.load(
        query + qrow[:, None] * 64 + d[None, :] // 2,
        h[:, None] < heads,
        other=0,
    ).to(tl.uint8)
    qs = tl.load(q_scale + qrow, h < heads, other=0)
    qcode = (qp >> ((d[None, :] & 1) * 4)) & 15
    qbyte = ((qs[:, None] >> (d[None, :] // 32 * 8)) & 255).to(tl.uint8)
    q = (_e2m1_to_float(qcode) * _ue8m0_to_float(qbyte)).to(tl.bfloat16)

    base = pool + page.to(tl.int64) * pool_stride
    kp = tl.load(
        base[None, :] + token[None, :] * 64 + d[:, None] // 2,
        readable[None, :],
        other=0,
    )
    kbyte = tl.load(
        base[None, :] + physical_page * 64 + token[None, :] * 4 + d[:, None] // 32,
        readable[None, :],
        other=127,
    )
    kcode = (kp >> ((d[:, None] & 1) * 4)) & 15
    k = (_e2m1_to_float(kcode) * _ue8m0_to_float(kbyte)).to(tl.bfloat16)
    weight = tl.load(weights + qrow, h < heads, other=0).to(tl.float32)
    dots = tl.maximum(tl.dot(q, k), 0.0)
    score = tl.sum(dots * weight[:, None], axis=0)
    tl.store(
        output + row * width + n, tl.where(readable, score, float("-inf")), n < width
    )


def score_paged(
    query: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    requests: torch.Tensor,
    logical_page: int,
) -> torch.Tensor:
    rows, heads, _ = query.shape
    width = pages.size(1) * logical_page
    output = torch.empty((rows, width), dtype=torch.float32, device=query.device)
    if rows and width:
        _score[(rows, triton.cdiv(width, 64))](
            query.contiguous(),
            q_scale.contiguous(),
            weights.contiguous(),
            pool,
            pages.contiguous(),
            lengths.contiguous(),
            requests.contiguous(),
            output,
            heads,
            width,
            pages.size(1),
            pages.size(0),
            pool.stride(0),
            pool.size(0),
            logical_page,
            pool.size(1),
            max(16, triton.next_power_of_2(heads)),
            64,
            num_warps=4,
        )
    return output
