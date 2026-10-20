"""One-layer lookahead for immutable CP prefix KV, with two consumer buffers."""

import os

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather,
    all_gather_kv_overlap,
    registered_kv_overlap_buffers,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def prefix_plan(prefixes, lengths, size, rank, owner):
    """CPU-only RR plan; new rows follow the cache writer's request-major order."""
    if (
        not prefixes
        or len(prefixes) != len(lengths)
        or size < 1
        or not 0 <= rank < size
    ):
        raise ValueError("invalid CP prefix plan")
    if owner < 1 or any(p < 0 or q < 0 for p, q in zip(prefixes, lengths)):
        raise ValueError("prefix/new lengths must be nonnegative")
    options = dict(dtype=torch.int64, device="cpu")
    padded = [((p + size * owner - 1) // (size * owner)) * owner for p in prefixes]
    actual = [
        p // (size * owner) * owner
        + min(owner, max(0, p % (size * owner) - rank * owner))
        for p in prefixes
    ]
    restore = torch.zeros(sum(prefixes) + sum(lengths), **options)
    scatter, new_dest, new_owner = [], [], []
    padded_start = full_start = 0
    for p, q, pad, count in zip(prefixes, lengths, padded, actual):
        positions = torch.arange(p, **options)
        restore[full_start : full_start + p] = (
            (positions // owner % size) * sum(padded)
            + padded_start
            + positions // (size * owner) * owner
            + positions % owner
        )
        scatter.append(torch.arange(count, **options) + padded_start)
        positions = torch.arange(p, p + q, **options)
        new_dest.append(positions + full_start)
        new_owner.append(positions // owner % size)
        padded_start += pad
        full_start += p + q
    owners = torch.cat(new_owner)
    rows_by_rank = [torch.where(owners == r)[0] for r in range(size)]
    new_max = max(map(len, rows_by_rank))
    new_restore = torch.empty(sum(lengths), **options)
    for r, rows in enumerate(rows_by_rank):
        new_restore[rows] = r * new_max + torch.arange(len(rows), **options)
    actual_starts = [sum(actual[:i]) for i in range(len(actual))]
    return {
        "history_lengths": torch.tensor(actual, dtype=torch.int32, device="cpu"),
        "history_starts": torch.tensor(actual_starts, dtype=torch.int32, device="cpu"),
        "history_scatter": torch.cat(scatter),
        "history_restore": restore,
        "new_local_rows": rows_by_rank[rank],
        "new_restore": new_restore,
        "new_dest": torch.cat(new_dest),
        "history_padded": sum(padded),
        "history_actual": sum(actual),
        "new_max": new_max,
    }


class CrossLayerKvPrefetch:
    def __init__(self, fmha, cache):
        self.fmha = fmha
        self.op = fmha.fmha_impl
        self.cache = cache
        self.stream = fmha._kv_overlap_stream
        self.device = self.op.block_table.device
        self.compute = torch.cuda.current_stream(self.device)
        self.original_table = self.op.block_table
        prefixes = self.op.attn_inputs.prefix_lengths.reshape(-1).tolist()
        lengths = self.op.cp_info.prefill_actual_input_lengths_cpu.reshape(-1).tolist()
        plan = prefix_plan(
            prefixes,
            lengths,
            fmha._cp_size,
            fmha._cp_rank,
            fmha._kv_owner_tokens_per_block,
        )
        self.plan = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in plan.items()
        }
        self.buffers = [self.op._allocate_fused_kv() for _ in range(2)]
        width = self.buffers[0].size(1)
        options = dict(dtype=torch.bfloat16, device=self.device)
        pair = registered_kv_overlap_buffers(
            plan["history_padded"], width, torch.bfloat16, self.device, Group.TP
        )
        self.registered = pair is not None
        if pair is None:
            self.local = torch.empty((plan["history_padded"], width), **options)
            self.gathered = torch.empty(
                (fmha._cp_size * plan["history_padded"], width), **options
            )
        else:
            self.local, self.gathered = pair
        # Each request's actual count is <= its padded count. Equal totals
        # therefore prove a dense identity layout without a GPU-side check.
        self.history_is_dense = (
            os.environ.get("RTP_LLM_CP_KV_INDEX_OPT", "1") == "1"
            and plan["history_actual"] == plan["history_padded"]
        )
        self.actual = (
            self.local
            if self.history_is_dense
            else torch.empty((plan["history_actual"], width), **options)
        )
        self.new_slots = {}
        for location, slots in (("hbm", cache.original_slots), ("host", cache.slots)):
            self.new_slots[location] = (
                slots.index_select(0, self.plan["new_local_rows"])
                .to(torch.int32)
                .view(1, -1)
            )
        self.new_lengths = torch.tensor(
            [len(plan["new_local_rows"])], dtype=torch.int32, device=self.device
        )
        self.new_starts = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.ready = [None, None]
        self.free = [None, None]
        self.owners = [None, None]
        self.pending_new = None
        self.first_buffers = None
        self.latest_ready = None
        # All long-lived buffers/metadata are allocated on compute, once per
        # forward. Subsequent submissions wait only on their own slot events.
        self.stream.wait_stream(self.compute)

    def prepare(self, layer, kv_cache):
        if layer == 0:
            # First layer follows the existing full-KV path, after its KV write.
            local = self.op._pack_sharded_kv_cache(kv_cache, self.buffers[0])
            gathered = torch.empty(
                (self.fmha._cp_size * local.size(0), local.size(1)),
                dtype=local.dtype,
                device=self.device,
            )
            self.first_buffers = (local, gathered)
            self.stream.wait_stream(self.compute)
            with torch.cuda.stream(self.stream):
                all_gather_kv_overlap(local, Group.TP, output=gathered)
                torch.index_select(
                    gathered, 0, self.op.sharded_kv_restore_indices, out=self.buffers[0]
                )
                self._record_ready(0)
        else:
            if self.owners[layer % 2] != layer:
                raise RuntimeError("next-layer history was not prefetched")
            # Read newly written bytes through the SAME FP8 upconverter as the
            # full gather. Never substitute unquantized QKV projection values.
            source = kv_cache.kv_cache_base.view(torch.uint8)
            source = source.reshape(-1, 1, source.size(-1))
            count = self.new_slots["hbm"].size(1)
            local = torch.zeros(
                (self.plan["new_max"], self.buffers[0].size(1)),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if count:
                location = "host" if layer in self.cache.host_layers else "hbm"
                rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                    source,
                    local,
                    self.new_slots[location],
                    self.new_lengths,
                    self.new_starts,
                    1,
                    count,
                )
            # This uses the ordinary communicator, independent of the history
            # producer. Every rank participates, including zero-owned-Q ranks.
            gathered = all_gather(local, group=Group.TP, role="mla_new_rows")
            self.pending_new = gathered.index_select(0, self.plan["new_restore"])
        # Submit while current Indexer/Q work is still ahead. Waiting until
        # consume lets Attention occupy the SMs before the prefix pack starts,
        # leaving NCCL queued behind it despite independent CUDA streams.
        self.prefetch(layer + 1)
        return self.buffers[layer % 2], self.ready[layer % 2]

    def _record_ready(self, layer):
        slot = layer % 2
        self.ready[slot] = torch.cuda.Event()
        self.ready[slot].record(self.stream)
        self.owners[slot] = layer
        self.latest_ready = self.ready[slot]

    def consume(self, layer):
        """Called after the consumer waits on this layer's ready event."""
        if self.owners[layer % 2] != layer:
            raise RuntimeError("history buffer owner mismatch")
        if layer:
            self.buffers[layer % 2].index_copy_(
                0, self.plan["new_dest"], self.pending_new
            )
            self.pending_new = None
        else:
            # The current-layer event has been joined; these main-stream
            # allocations can now return to the allocator before MoE.
            self.first_buffers = None

    def prefetch(self, layer):
        if layer == len(self.cache.backing):
            return
        source, table, ready = self.cache.prefetched_layer(layer, self.original_table)
        slot = layer % 2
        # The CPU can enqueue several layers ahead of the GPU. Open this
        # window only after current KV is ready (and preceding MoE is done),
        # otherwise the producer could race ahead into that preceding MoE.
        self.stream.wait_stream(self.compute)
        with torch.cuda.stream(self.stream):
            if self.free[slot] is not None:
                self.stream.wait_event(self.free[slot])
            if ready is not None:
                self.stream.wait_event(ready)
            if not self.history_is_dense:
                self.local.zero_()
            if self.plan["history_actual"]:
                source = source.view(torch.uint8)
                if source.ndim == 4:
                    source = source.squeeze(2)
                rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                    source,
                    self.actual,
                    table,
                    self.plan["history_lengths"],
                    self.plan["history_starts"],
                    len(self.plan["history_lengths"]),
                    self.plan["history_actual"],
                )
                if not self.history_is_dense:
                    self.local.index_copy_(0, self.plan["history_scatter"], self.actual)
            all_gather_kv_overlap(self.local, Group.TP, output=self.gathered)
            # New-token holes map to row zero and are overwritten at consume.
            # No uninitialized cache row is read during this history prefetch.
            torch.index_select(
                self.gathered, 0, self.plan["history_restore"], out=self.buffers[slot]
            )
            self._record_ready(layer)

    def release(self, layer):
        self.free[layer % 2] = torch.cuda.Event()
        self.free[layer % 2].record(self.compute)

    def wait_before_moe(self):
        # AllGather and Mega-MoE share NVLink. Use Attention/V BMM/O as the
        # overlap window, then drain its tail before launching Mega-MoE.
        # This is an event dependency, never a CPU/device-wide synchronize.
        if self.latest_ready is not None:
            self.compute.wait_event(self.latest_ready)

    def close(self):
        # Join all outstanding work on forward cleanup, including exceptions.
        # Normal layer exits leave prefetch running until the MoE fence.
        self.compute.wait_stream(self.stream)
        self.fmha._cross_layer_prefetch = None
