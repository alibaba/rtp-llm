"""Experimental GPU metadata and paired state copies for one PLE layer.

Inspired by the actual vLLM MambaHybridModelState GPU acceptance metadata and
fused state copies. RTP's destination validation, immutable undo, consensus,
and failure/rollback boundaries remain the responsibility of the coordinator.
"""

import threading

import torch
import triton

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit_metadata import (
    _resolve_accepted_pair,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews


class AcceptedPairResolution:
    """Deferred verdict for the fused PLE commit metadata resolution.

    begin_accepted_pair_resolution queues an 8-byte device-to-pinned verdict
    copy without blocking. verify() records after the dependent prepare
    kernels and waits once for both the verdict and the immutable undo values.

    The payload layout is [combined flag, accept-range flag, state verdict
    flags (4), ctx verdict flags (4), state blocks (B), ctx blocks (B),
    accepted (B)]. Block and accepted values are clamped in the payload so a
    speculative gather cannot read out of bounds before the verdict lands.
    """

    __slots__ = (
        "payload",
        "batch",
        "stage",
        "tags",
        "query_len",
        "pinned",
        "event",
        "stream",
    )

    def __init__(self, payload, batch, stage, tags, query_len, pinned, event):
        self.payload = payload
        self.batch = batch
        self.stage = stage
        self.tags = tags
        self.query_len = query_len
        self.pinned = pinned
        self.event = event
        self.stream = torch.cuda.current_stream(payload.device)

    @property
    def accepted(self):
        return self.payload[10 + 2 * self.batch :]

    @property
    def pending(self):
        return [
            (
                self.stage,
                self.payload[10 : 10 + self.batch],
                self.payload[10 + self.batch : 10 + 2 * self.batch],
            )
        ]

    def verify(self):
        # All prepare kernels have now been enqueued on this stream. One event
        # observes both their completion and the earlier verdict D2H copy.
        self.event.record(self.stream)
        self.event.synchronize()
        if int(self.pinned[0].item()):
            self._raise_verdict()

    def _raise_verdict(self):
        host = self.payload.tolist()
        batch = self.batch
        stage = self.stage
        tags = self.tags
        state_host = host[10 : 10 + batch]
        ctx_host = host[10 + batch : 10 + 2 * batch]
        if host[1]:
            raise RuntimeError(
                "qwen4_exp PLE target accept_len must be in " f"[1, {self.query_len}]"
            )
        PLECacheViews._check_commit_verdict(
            tags[0], host[2:6], state_host, stage.state_pool.shape[0]
        )
        PLECacheViews._check_commit_verdict(
            tags[1], host[6:10], ctx_host, stage.ctx_pool.shape[0]
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
                        f"qwen4_exp PLE cache {tag!r} has a duplicate "
                        "target destination physical row"
                    )
                destinations.add(row)
        raise RuntimeError("qwen4_exp PLE target commit verdict failed")


_VERDICT_CHANNEL_LOCK = threading.Lock()
_VERDICT_CHANNEL = {}


def _verdict_channel(device):
    """Per-device pinned flag buffer and event, reused across decode steps."""
    with _VERDICT_CHANNEL_LOCK:
        channel = _VERDICT_CHANNEL.get(device)
        if channel is None:
            channel = (
                torch.empty(1, dtype=torch.int64, pin_memory=True),
                torch.cuda.Event(),
            )
            _VERDICT_CHANNEL[device] = channel
        return channel


def begin_accepted_pair_resolution(transaction, accept_len):
    """Launch the fused resolve kernel and stage its verdict for a late read.

    Returns an AcceptedPairResolution whose accepted/pending tensors can feed
    prepare_commit_values immediately; call verify() before publishing the
    prepare plan. Returns None when the fused path does not apply; the caller
    falls back to the sequential host validation.
    """
    if not (
        torch.version.hip is None
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
    tables = [
        PLECacheViews._resolve_block_table(inputs, tag)
        for inputs, tag in zip((stage.state_inputs, stage.ctx_inputs), tags)
    ]
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
    payload = torch.empty(10 + 3 * batch, device=accept_len.device, dtype=torch.int64)
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
    pinned, event = _verdict_channel(accept_len.device)
    pinned.copy_(payload[:1], non_blocking=True)
    return AcceptedPairResolution(
        payload=payload,
        batch=batch,
        stage=stage,
        tags=tags,
        query_len=transaction.query_len,
        pinned=pinned,
        event=event,
    )
