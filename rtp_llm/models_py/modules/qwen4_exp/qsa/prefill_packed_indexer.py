"""Packed zero-prefix QSA selection, adapted from vLLM's prefill indexer.

The score grid follows vLLM 83f5de466 (Apache-2.0): request/query/key tiles,
causal tile pruning, and bounded logits workspace. The output keeps RTP's
2051-column ABI, including the fixed tail offset at 2048. Histogram selection
uses the vLLM register histogram; stable sorting and expansion share a
separate small block so the two stages retain their own occupancy.
"""

from bisect import bisect_left, bisect_right
from itertools import accumulate

import torch
import triton
import triton.language as tl

from .indexer_norm_rope import _norm_rope_row


@triton.jit(do_not_specialize=["Q_STRIDE", "MAX_LENGTH"])
def _packed_norm_rope(
    Q,
    Gamma,
    Cos,
    Sin,
    Cu,
    Out,
    Visible,
    Positions,
    Q_STRIDE,
    MAX_LENGTH,
    EPS: tl.constexpr,
):
    request, local_row, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(Cu + request)
    end = tl.load(Cu + request + 1)
    row = start + local_row
    if row < end:
        _norm_rope_row(
            Q,
            Gamma,
            Cos,
            Sin,
            Out,
            row,
            head,
            request * MAX_LENGTH + local_row,
            4,
            Q_STRIDE,
            EPS,
        )
        if head == 0:
            tl.store(Visible + row, (local_row + 1) // 4)
            tl.store(Positions + row, local_row)


@triton.jit(
    do_not_specialize=[
        "ROWS",
        "OFFSET",
        "COLUMNS",
        "KEY_BATCH_STRIDE",
        "REQUEST_OFFSET",
    ]
)
def _packed_causal_score(
    Q,
    Keys,
    Cu,
    Visible,
    Scores,
    ROWS,
    OFFSET,
    COLUMNS,
    KEY_BATCH_STRIDE,
    REQUEST_OFFSET,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    request = REQUEST_OFFSET + tl.program_id(0)
    request_start = tl.load(Cu + request)
    request_end = tl.minimum(tl.load(Cu + request + 1), OFFSET + ROWS)
    first = tl.maximum(request_start, OFFSET) + tl.program_id(1) * TILE_R
    if first < request_end:
        r = first + tl.arange(0, TILE_R)
        visible = tl.load(Visible + r, r < request_end, other=0)
        max_visible = tl.max(visible, 0)
        first_col = tl.program_id(2) * BLOCK_N
        if first_col < max_visible:
            m = tl.arange(0, TILE_R * 4)
            qr = first + m // 4
            head = m % 4
            dim = tl.arange(0, 128)
            query = tl.load(
                Q + qr[:, None] * 512 + head[:, None] * 128 + dim[None, :],
                qr[:, None] < request_end,
                other=0,
            )
            col = first_col + tl.arange(0, BLOCK_N)
            keys = tl.load(
                Keys + request * KEY_BATCH_STRIDE + col[None, :] * 128 + dim[:, None],
                col[None, :] < max_visible,
                other=0,
            )
            # Keep the original Q x K contraction/reduction order. Reordering
            # equivalent BF16 attention inputs can amplify through later MoE layers.
            dot = tl.dot(query, keys)
            score = tl.sum(tl.reshape(tl.maximum(dot, 0.0), (TILE_R, 4, BLOCK_N)), 1)
            score = score * (128.0**-0.5)
            tl.store(
                Scores + (r[:, None] - OFFSET) * COLUMNS + col[None, :],
                score,
                (r[:, None] < request_end) & (col[None, :] < visible[:, None]),
            )


def _is_supported(q, keys, cos, sin, cu, lengths, gamma):
    from .cub_topk import _TORCH_GIT

    # Short rows use a different native warp sort/tie order; retain their fallback.
    return (
        1 <= len(lengths) <= 64
        and min(lengths) >= 256
        and q.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and torch.cuda.get_device_capability(q.device) == (9, 0)
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and keys.ndim == 3
        and keys.shape[0] == len(lengths)
        and keys.shape[2] == 128
        and keys.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.shape == sin.shape == (len(lengths), max(lengths), 64)
        and cu.shape == (len(lengths) + 1,)
        and cu.dtype == torch.int32
        and cu.is_contiguous()
        and gamma.shape == (128,)
        and gamma.is_contiguous()
        and gamma.dtype in (torch.bfloat16, torch.float32)
        and q.dtype == keys.dtype == cos.dtype == sin.dtype == torch.bfloat16
        and all(
            t.device == q.device and not t.requires_grad
            for t in (q, keys, cos, sin, cu, gamma)
        )
        and max(lengths) <= 32768
        and sum(lengths) == q.shape[0]
        and keys.shape[1] >= max(lengths) // 4
        and not torch.cuda.is_current_stream_capturing()
    )


def try_packed_prefill(q, block_keys, cos, sin, cu, lengths, gamma, eps):
    """Return None outside the validated eager SM90 zero-prefix geometry.

    Cu and lengths are the partition already checked by the transactional
    cache writer. No device-to-host metadata read is introduced here.
    """
    if not isinstance(block_keys, torch.Tensor) or not _is_supported(
        q, block_keys, cos, sin, cu, lengths, gamma
    ):
        return None
    from .prefill_filtered_topk import _load_module

    selector = _load_module()
    if selector is None:
        return None

    rows = q.shape[0]
    columns = triton.cdiv(max(lengths) // 4, 64) * 64
    query = torch.empty((rows, 4, 128), device=q.device, dtype=q.dtype)
    visible = torch.empty((rows,), device=q.device, dtype=torch.int32)
    positions = torch.empty_like(visible)
    _packed_norm_rope[(len(lengths), max(lengths), 4)](
        q,
        gamma,
        cos,
        sin,
        cu,
        query,
        visible,
        positions,
        q.stride(0),
        max(lengths),
        EPS=eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    output = torch.empty((rows, 2051), device=q.device, dtype=torch.int32)
    # Bound temporary logits to 512 MiB, as in vLLM; never specialize on rows.
    chunk_rows = min(rows, (512 * 1024 * 1024) // (columns * 4))
    scores = torch.empty((chunk_rows, columns), device=q.device, dtype=torch.float32)
    boundaries = [0, *accumulate(lengths)]
    for offset in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - offset)
        logits = scores[:count]
        # Launch only requests intersecting this workspace chunk.
        first_request = bisect_right(boundaries, offset) - 1
        end_request = bisect_left(boundaries, offset + count)
        _packed_causal_score[
            (
                end_request - first_request,
                triton.cdiv(min(max(lengths), count), 32),
                triton.cdiv(columns, 64),
            )
        ](
            query,
            block_keys,
            cu,
            visible,
            logits,
            count,
            offset,
            columns,
            block_keys.stride(0),
            first_request,
            TILE_R=32,
            BLOCK_N=64,
            num_warps=4,
            num_stages=1,
        )
        selector.select(logits, visible, positions, output, offset)
    return output
