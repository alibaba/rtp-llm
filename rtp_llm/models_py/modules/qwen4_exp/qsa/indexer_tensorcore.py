# SPDX-License-Identifier: Apache-2.0
"""Tensor Core QSA paged scoring with RTP's weighted-head contract.

Query/head flattening follows SGLang's TileLang MQA implementation at
297e54fd105e974cdf401e5b4d20253a22c16f90. Unlike its weight-free interface,
this kernel retains each FP32 head weight after ReLU, including negative
weights, and reads RTP's paged BF16 compressed-key pool directly.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_mma(
    Q,
    W,
    POOL,
    TABLE,
    LENGTHS,
    OUT,
    QUERIES: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_CTX: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    POOL_BLOCKS: tl.constexpr,
    PAGE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    qr = tl.program_id(0) * BM + tl.arange(0, BM)
    nr = tl.program_id(1) * BN + tl.arange(0, BN)
    mh = tl.arange(0, BM * H)
    query = tl.program_id(0) * BM + mh // H
    head = mh % H
    d = tl.arange(0, D)
    qoff = (batch * QUERIES + query[:, None]) * H * D + head[:, None] * D + d[None, :]
    q = tl.load(Q + qoff, query[:, None] < QUERIES, 0)
    logical = nr // PAGE
    physical = tl.load(
        TABLE + batch * TABLE_WIDTH + logical,
        (nr < MAX_CTX) & (logical < TABLE_WIDTH),
        0,
    )
    valid_key = (
        (nr < MAX_CTX)
        & (logical < TABLE_WIDTH)
        & (physical > 0)
        & (physical < POOL_BLOCKS)
    )
    slot = physical.to(tl.int64) * PAGE + nr % PAGE
    key = tl.load(POOL + slot[:, None] * D + d[None, :], valid_key[:, None], 0)
    scores = tl.dot(q, tl.trans(key))
    weights = tl.load(W + (batch * QUERIES + query) * H + head, query < QUERIES, 0)
    weighted = tl.maximum(scores, 0.0) * weights[:, None]
    scores = tl.sum(tl.reshape(weighted, (BM, H, BN)), 1)
    visible = tl.load(LENGTHS + batch * QUERIES + qr, qr < QUERIES, 0)
    valid = valid_key[None, :] & (nr[None, :] < visible[:, None])
    tl.store(
        OUT + (batch * QUERIES + qr[:, None]) * MAX_CTX + nr[None, :],
        tl.where(valid, scores, -float("inf")),
        (qr < QUERIES)[:, None] & (nr < MAX_CTX)[None, :],
    )


def paged_indexer_mma(
    q, weights, pool, table, lengths, *, block_size, max_ctx_len, block_m=32, block_n=64
):
    """Private entry; the public scorer owns geometry and device validation."""
    batch, queries, heads, dim = q.shape
    output = torch.empty(
        (batch * queries, max_ctx_len), device=q.device, dtype=torch.float32
    )
    _paged_mqa_mma[
        (triton.cdiv(queries, block_m), triton.cdiv(max_ctx_len, block_n), batch)
    ](
        q,
        weights,
        pool,
        table,
        lengths,
        output,
        queries,
        heads,
        dim,
        max_ctx_len,
        table.shape[1],
        pool.shape[0] // block_size,
        block_size,
        block_m,
        block_n,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
