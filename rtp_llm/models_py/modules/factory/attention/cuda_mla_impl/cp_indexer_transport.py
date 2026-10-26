from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather_current_cp,
    warmup_current_cp,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_transport import (
    PackedRows,
    PagedRows,
    is_supported,
    pack_rows,
    restore_indexer,
)

from .cp_packed_pipeline import PackedKvTransport, enabled
from .cp_packed_read import PackedKvRead

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
        IndexerCPChunkPlan,
    )


class IndexerTransportParams(Protocol):
    kv_cache_sharded: bool
    cp_size: int
    total_kv_len: int
    indexer_cp_plan: IndexerCPChunkPlan | None
    indexer_transport_rows: torch.Tensor | None
    indexer_transport_table: torch.Tensor
    packed_kv_transport: PackedKvTransport | PackedKvRead | None


@dataclass(frozen=True, slots=True)
class PendingIndexer:
    key: torch.Tensor
    scales: torch.Tensor
    ready: torch.cuda.Event

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream(self.key.device).wait_event(self.ready)
        return self.key, self.scales


def start_indexer_read(
    cache: torch.Tensor, params: IndexerTransportParams, head: int
) -> PendingIndexer | None:
    if not enabled() or not params.kv_cache_sharded or params.cp_size <= 1:
        return None
    plan, rows = params.indexer_cp_plan, params.indexer_transport_rows
    if plan is None or rows is None:
        return None
    source = PagedRows(
        cache.view(torch.uint8), params.indexer_transport_table, rows, head
    )
    if not is_supported(source):
        return None
    key = torch.empty(
        (params.total_kv_len, head), dtype=torch.float8_e4m3fn, device=cache.device
    )
    scales = torch.empty(
        (params.total_kv_len, cache.shape[-1] - head),
        dtype=torch.uint8,
        device=cache.device,
    )
    transport = params.packed_kv_transport
    if transport is not None and transport.defer_joint and transport.new_ready is None:
        ready = transport.launch(source, plan.restore_indices, (key, scales))
        return PendingIndexer(key, scales, ready)
    stream = warmup_current_cp(Group.TP)
    stream.wait_stream(torch.cuda.current_stream(cache.device))
    with torch.cuda.stream(stream):
        local = pack_rows(source)
        if local.numel():
            gathered = all_gather_current_cp(local, Group.TP)
            restore_indexer(
                PackedRows(gathered, rows.numel(), local.shape[1], local.numel()),
                plan.restore_indices,
                (key, scales),
            )
        ready = torch.cuda.Event()
        ready.record(stream)
    for tensor in (cache, source.table, rows, plan.restore_indices, key, scales):
        tensor.record_stream(stream)
    return PendingIndexer(key, scales, ready)
