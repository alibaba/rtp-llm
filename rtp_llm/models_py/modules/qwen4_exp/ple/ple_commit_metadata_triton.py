"""Experimental GPU metadata and paired state copies for one PLE layer.

Inspired by the actual vLLM MambaHybridModelState GPU acceptance metadata and
fused state copies. RTP's destination validation, immutable undo, consensus,
and failure/rollback boundaries remain the responsibility of the coordinator.
"""

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.cache_views import PLECacheViews
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
    _write_pair_verdict,
)


@triton.jit
def _resolve_accepted_pair(
    ACCEPT,
    PREFIX,
    STATE_TABLE,
    CTX_TABLE,
    OUT,
    B: tl.constexpr,
    Q: tl.constexpr,
    PAGE: tl.constexpr,
    SC: tl.constexpr,
    SR: tl.constexpr,
    SS: tl.constexpr,
    CC: tl.constexpr,
    CR: tl.constexpr,
    CS: tl.constexpr,
    SP: tl.constexpr,
    CP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < B
    accepted = tl.load(ACCEPT + row, live, other=1).to(tl.int64)
    invalid = tl.sum((live & ((accepted < 1) | (accepted > Q))).to(tl.int32), 0) > 0
    tl.store(OUT, invalid)
    tl.store(OUT + 9 + 2 * B + row, accepted, live)
    position = tl.load(PREFIX + row, live, other=0).to(tl.int64) + accepted - 1
    page = tl.maximum(position, 0) // PAGE
    _write_pair_verdict(
        STATE_TABLE, OUT + 1, row, live, position, page, SC, SR, SS, SP, 0, 8
    )
    _write_pair_verdict(
        CTX_TABLE, OUT + 1, row, live, position, page, CC, CR, CS, CP, 4, 8 + B
    )


def resolve_accepted_pair(transaction, accept_len):
    """Return accepted int64 and validated pending writes, or None for fallback.

    No pool reads/writes occur before the original ordered CPU verdict checks.
    A single-layer gate keeps general multiple-layer alias/error handling on
    the existing path. Non-contiguous acceptance also keeps the old path.
    """
    if not (
        os.environ.get("QWEN4_PLE_COMMIT_METADATA_FUSED", "0") == "1"
        and torch.version.hip is None
        and accept_len.is_cuda
        and torch.cuda.get_device_capability(accept_len.device) == (12, 0)
        and accept_len.dtype == torch.int32
        and accept_len.ndim == 1
        and accept_len.is_contiguous()
        and 0 < accept_len.numel() <= 32
        and len(transaction.stages) == 1
        and 0 < transaction.query_len <= 32
    ):
        return None
    stage = next(iter(transaction.stages.values()))
    tags = (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG)
    try:
        tables = [
            PLECacheViews._resolve_block_table(inputs, tag)
            for inputs, tag in zip((stage.state_inputs, stage.ctx_inputs), tags)
        ]
    except RuntimeError:
        return None
    prefix = stage.prefixes
    if not (
        prefix.is_cuda
        and prefix.device == accept_len.device
        and prefix.dtype == torch.int64
        and prefix.ndim == 1
        and prefix.is_contiguous()
        and prefix.numel() == accept_len.numel()
        and all(
            t.is_cuda
            and t.device == accept_len.device
            and t.ndim == 2
            and t.dtype == torch.int32
            and t.shape[0] == accept_len.numel()
            and t.shape[1] > 0
            for t in tables
        )
        and stage.state_pool.device == stage.ctx_pool.device == accept_len.device
        and min(stage.state_pool.shape[0], stage.ctx_pool.shape[0], stage.page_size) > 0
    ):
        return None
    batch = accept_len.numel()
    payload = torch.empty(9 + 3 * batch, device=accept_len.device, dtype=torch.int64)
    _resolve_accepted_pair[(1,)](
        accept_len,
        prefix,
        tables[0],
        tables[1],
        payload,
        batch,
        transaction.query_len,
        stage.page_size,
        tables[0].shape[1],
        *tables[0].stride(),
        tables[1].shape[1],
        *tables[1].stride(),
        stage.state_pool.shape[0],
        stage.ctx_pool.shape[0],
        triton.next_power_of_2(batch),
        num_warps=4,
    )
    host = payload.tolist()
    if host[0]:
        raise RuntimeError(
            "qwen4_exp PLE target accept_len must be in "
            f"[1, {transaction.query_len}]"
        )
    state_host = host[9 : 9 + batch]
    ctx_host = host[9 + batch : 9 + 2 * batch]
    PLECacheViews._check_commit_verdict(
        tags[0], host[1:5], state_host, stage.state_pool.shape[0]
    )
    PLECacheViews._check_commit_verdict(
        tags[1], host[5:9], ctx_host, stage.ctx_pool.shape[0]
    )
    destinations = set()
    for pool, blocks, tag in (
        (stage.state_pool, state_host, tags[0]),
        (stage.ctx_pool, ctx_host, tags[1]),
    ):
        for block in blocks:
            row = (int(pool.data_ptr()), int(block))
            if row in destinations:
                raise RuntimeError(
                    f"qwen4_exp PLE cache {tag!r} has a duplicate target destination physical row"
                )
            destinations.add(row)
    return payload[9 + 2 * batch :], [
        (stage, payload[9 : 9 + batch], payload[9 + batch : 9 + 2 * batch])
    ]
