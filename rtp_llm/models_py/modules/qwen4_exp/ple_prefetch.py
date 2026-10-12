"""Side-stream UVA lookup with stable CUDA Graph input buffers.

The start/join placement follows vLLM's pinned-host PLE embedding. RTP keeps
the consumer in its complete graph by using an external event wait node.
"""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple_gather_triton import gather_local


@dataclass
class PLEPrefetch:
    rows: torch.Tensor
    ready: torch.cuda.Event
    owner: "PLEPrefetcher"

    def wait(self) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.rows.device)
        stream.wait_event(self.ready)
        if not torch.cuda.is_current_stream_capturing():
            self.rows.record_stream(stream)
        return self.rows


class PLEPrefetcher:
    """Own pinned tables, a side stream, and bounded reusable lookup storage.

    Graph callers provide their graph-owned output and protect its reuse with
    the runner's forward event. Eager callers call finish() after queuing all
    consumers, including projections that read the looked-up rows.
    The budget bounds retained eager storage per embedding. Stream-deferred
    reclamation during growth and graph-owned storage are separate allocations.
    """

    def __init__(
        self,
        shards,
        shard_indices,
        total_shards,
        pointer_table,
        *,
        max_eager_bytes=256 * 1024 * 1024
    ):
        if max_eager_bytes <= 0:
            raise ValueError("PLE prefetch memory budget must be positive")
        self.shards = shards
        self.shard_indices = shard_indices
        self.total_shards = total_shards
        self.pointer_table = pointer_table
        self.device = pointer_table.device
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        self._eager_slots = {}
        self._graph_events = {}
        self._active_slots = set()
        self._max_eager_bytes = max_eager_bytes
        self._closed = False

    def start(self, ids, *, slot=0, out=None) -> PLEPrefetch | None:
        if self._closed:
            raise RuntimeError("PLE prefetcher is closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE prefetch must start outside CUDA Graph capture")
        if ids.device != self.device or ids.dtype != torch.int64:
            raise ValueError("PLE prefetch requires int64 ids on its CUDA device")
        if out is None:
            if not isinstance(slot, int) or not 0 <= slot < 128:
                raise ValueError("PLE eager prefetch slot must be in [0, 128)")
            if slot in self._active_slots:
                raise RuntimeError("PLE prefetch slot is already active")
            capacity = max(1, 1 << (max(1, ids.numel()) - 1).bit_length())
            entry = self._eager_slots.get(slot)
            if entry is None or entry[0].shape[0] < capacity:
                needed = (
                    capacity * self.shards[0].shape[1] * self.shards[0].element_size()
                )
                if needed > self._max_eager_bytes:
                    return None
                previous = entry[0].nbytes if entry is not None else 0
                while (
                    sum(x[0].nbytes for x in self._eager_slots.values())
                    - previous
                    + needed
                    > self._max_eager_bytes
                ):
                    victim = next(
                        (
                            key
                            for key, value in self._eager_slots.items()
                            if key != slot
                            and key not in self._active_slots
                            and value[3]
                            and value[2].query()
                        ),
                        None,
                    )
                    if victim is None:
                        return None
                    del self._eager_slots[victim]
                entry = (
                    torch.empty(
                        capacity,
                        self.shards[0].shape[1],
                        device=self.device,
                        dtype=self.shards[0].dtype,
                    ),
                    torch.cuda.Event(external=True),
                    torch.cuda.Event(),
                    False,
                )
                self._eager_slots[slot] = entry
            storage, ready, consumed, recorded = entry
            out = storage[: ids.numel()].view(*ids.shape, self.shards[0].shape[1])
            if recorded:
                self.stream.wait_event(consumed)
            self._active_slots.add(slot)
        else:
            key = (out.data_ptr(), tuple(out.shape))
            ready = self._graph_events.get(key)
            if ready is None:
                ready = torch.cuda.Event(external=True)
                self._graph_events[key] = ready
        producer = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(producer)
        ids.record_stream(self.stream)
        out.record_stream(self.stream)
        self.pointer_table.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            gather_local(
                ids,
                self.shards,
                self.shard_indices,
                self.total_shards,
                self.pointer_table,
                out=out,
            )
            ready.record(self.stream)
        return PLEPrefetch(out, ready, self)

    def finish(self) -> None:
        """Protect eager storage until every queued consumer is finished."""
        stream = torch.cuda.current_stream(self.device)
        for slot in self._active_slots:
            storage, ready, consumed, _ = self._eager_slots[slot]
            stream.wait_event(ready)
            consumed.record(stream)
            self._eager_slots[slot] = (storage, ready, consumed, True)
        self._active_slots.clear()

    def close(self) -> None:
        """Drain UVA reads before releasing their pinned CPU owners."""
        if not self._closed:
            self.stream.synchronize()
            self._closed = True

    def __del__(self):
        if hasattr(self, "stream"):
            try:
                self.close()
            except Exception:
                # CUDA may already have been torn down during interpreter exit.
                pass
