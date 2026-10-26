import os
from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather_kv_overlap,
    warmup_current_cp,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_transport import (
    PackedRows,
    PagedRows,
    pack_rows,
    paged_row_map,
    restore_mla,
)
from .cp_packed_read import KvReadTarget, PackedKvRead

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import KVCache
    from .cp_kv_prefetch import CrossLayerKvPrefetch


def enabled() -> bool:
    return (
        os.environ.get("RTP_LLM_CP_PACKED_PIPELINE", "0") == "1"
        and torch.version.cuda is not None
    )


class PackedKvTransport:
    def __init__(self, pipeline: "CrossLayerKvPrefetch") -> None:
        self.pipeline = pipeline
        self.stream = warmup_current_cp(Group.TP)
        self.new_ready: torch.cuda.Event | None = None
        self.defer_joint = False
        self.current: PackedKvRead | None = None
        self.history_rows = paged_row_map(
            pipeline.plan["history_lengths"],
            pipeline.plan["history_padded_lengths"],
            pipeline.plan["history_padded"],
        )
        self.history_restore = pipeline.plan["history_restore"].clone()
        self.history_restore.index_fill_(0, pipeline.plan["new_dest"], -1)
        self.new_rows = torch.full(
            (pipeline.plan["new_max"],), -1, dtype=torch.int64, device=pipeline.device
        )
        count = pipeline.new_slots["hbm"].numel()
        self.new_rows[:count] = torch.arange(count, device=pipeline.device)
        op = pipeline.op
        self.full_rows = paged_row_map(
            op.sharded_actual_local_kv_lens,
            op.sharded_local_kv_lens,
            op.sharded_total_local_kv_len,
        )

    def prepare(
        self, layer: int, cache: "KVCache"
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        p = self.pipeline
        self.new_ready = None
        source = cache.kv_cache_base.view(torch.uint8)
        if source.ndim == 4:
            source = source.squeeze(2)
        if layer == 0:
            fresh = PagedRows(source, p.op.block_table, self.full_rows)
            target = KvReadTarget(p.op.sharded_kv_restore_indices, p.buffers[0])
        else:
            location = "host" if layer in p.cache.host_layers else "hbm"
            fresh = PagedRows(
                source.reshape(-1, 1, 656), p.new_slots[location], self.new_rows
            )
            target = KvReadTarget(
                p.plan["new_restore"], p.buffers[layer % 2], p.plan["new_dest"]
            )
        self.current = PackedKvRead(fresh, target, self.defer_joint)
        if not self.defer_joint:
            self.launch()
        if layer == 0:
            ready = self.new_ready
            if ready is None:
                ready = torch.cuda.Event()
                ready.record(p.compute)
            p.ready[0], p.owners[0] = ready, 0
        else:
            ready = p.ready[layer % 2]
        p.prefetch(layer + 1)
        return p.buffers[layer % 2], ready

    def launch(
        self,
        indexer: PagedRows | None = None,
        index_mapping: torch.Tensor | None = None,
        index_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.cuda.Event:
        assert self.current is not None
        self.new_ready = self.current.launch(indexer, index_mapping, index_outputs)
        return self.new_ready

    def consume(self) -> None:
        if self.new_ready is None:
            self.launch()
        self.pipeline.compute.wait_event(self.new_ready)

    def prefetch(self, layer: int) -> None:
        p = self.pipeline
        if layer == len(p.cache.backing):
            return
        source, table, ready = p.cache.prefetched_layer(layer, p.original_table)
        source = source.view(torch.uint8)
        if source.ndim == 4:
            source = source.squeeze(2)
        slot = layer % 2
        p.stream.wait_stream(p.compute)
        with torch.cuda.stream(p.stream):
            if p.free[slot] is not None:
                p.stream.wait_event(p.free[slot])
            if ready is not None:
                p.stream.wait_event(ready)
            pack_rows(PagedRows(source, table, self.history_rows), p.local)
            if p.local.numel():
                all_gather_kv_overlap(p.local, Group.TP, output=p.gathered)
                restore_mla(
                    PackedRows(p.gathered, p.local.shape[0], 656, p.local.numel()),
                    self.history_restore,
                    p.buffers[slot],
                )
            p._record_ready(layer)
        source.record_stream(p.stream)
        table.record_stream(p.stream)

    def close(self) -> None:
        self.pipeline.compute.wait_stream(self.stream)
