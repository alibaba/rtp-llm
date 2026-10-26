from dataclasses import dataclass

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather_current_cp,
    warmup_current_cp,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_transport import (
    PackedRows,
    PagedRows,
    pack_rows,
    restore_indexer,
    restore_mla,
)


@dataclass(frozen=True, slots=True)
class KvReadTarget:
    rows: torch.Tensor
    output: torch.Tensor
    destinations: torch.Tensor | None = None


class PackedKvRead:
    def __init__(
        self, source: PagedRows, target: KvReadTarget, defer_joint: bool = False
    ) -> None:
        self.source, self.target = source, target
        self.defer_joint = defer_joint
        self.new_ready: torch.cuda.Event | None = None
        self.stream = warmup_current_cp(Group.TP)

    def launch(
        self,
        indexer: PagedRows | None = None,
        index_mapping: torch.Tensor | None = None,
        index_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.cuda.Event:
        source, target = self.source, self.target
        rows, device = source.rows.numel(), source.cache.device
        fresh_bytes = rows * 656
        index_bytes = (
            0 if indexer is None else indexer.rows.numel() * indexer.cache.shape[-1]
        )
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            packet = torch.empty(
                (1, fresh_bytes + index_bytes), dtype=torch.uint8, device=device
            )
            if rows:
                pack_rows(source, packet.view(-1)[:fresh_bytes].view(rows, 656))
            if indexer is not None:
                pack_rows(
                    indexer,
                    packet.view(-1)[fresh_bytes:].view(-1, indexer.cache.shape[-1]),
                )
            if packet.numel():
                gathered = all_gather_current_cp(packet, Group.TP)
                restore_mla(
                    PackedRows(gathered, rows, 656, packet.numel()),
                    target.rows,
                    target.output,
                    target.destinations,
                )
                if indexer is not None:
                    assert index_mapping is not None and index_outputs is not None
                    restore_indexer(
                        PackedRows(
                            gathered,
                            indexer.rows.numel(),
                            indexer.cache.shape[-1],
                            packet.numel(),
                            fresh_bytes,
                        ),
                        index_mapping,
                        index_outputs,
                    )
            self.new_ready = torch.cuda.Event()
            self.new_ready.record(self.stream)
        for tensor in (
            source.cache,
            source.table,
            source.rows,
            target.rows,
            target.output,
        ):
            tensor.record_stream(self.stream)
        if target.destinations is not None:
            target.destinations.record_stream(self.stream)
        if indexer is not None:
            assert index_mapping is not None and index_outputs is not None
            for tensor in (
                indexer.cache,
                indexer.table,
                indexer.rows,
                index_mapping,
                *index_outputs,
            ):
                tensor.record_stream(self.stream)
        return self.new_ready

    def consume(self) -> None:
        if self.new_ready is None:
            self.launch()
        torch.cuda.current_stream(self.source.cache.device).wait_event(self.new_ready)

    def close(self) -> None:
        torch.cuda.current_stream(self.source.cache.device).wait_stream(self.stream)
