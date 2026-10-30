"""One-layer lookahead for immutable CP prefix KV, with two consumer buffers."""

import os

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import (
    all_gather_kv_overlap,
    registered_kv_overlap_buffers,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_kv_patch import restore_patch_kv
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
        # RR ownership is constant within an owner-sized block. Compute the
        # expensive division/modulo once per block, then expand contiguous
        # offsets; preserve the partial last block and zero-length requests.
        blocks = torch.arange((p + owner - 1) // owner, **options)
        bases = (blocks % size) * sum(padded) + (blocks // size) * owner + padded_start
        offsets = torch.arange(min(owner, p), **options)
        restore[full_start : full_start + p] = (bases[:, None] + offsets).reshape(-1)[
            :p
        ]
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
        "history_padded_lengths": torch.tensor(padded, dtype=torch.int32, device="cpu"),
        "history_padded_starts": torch.tensor(
            [sum(padded[:i]) for i in range(len(padded))],
            dtype=torch.int32,
            device="cpu",
        ),
        "new_max": new_max,
    }


def _upload_immutable_plan(plan, device):
    """One owned host snapshot/transfer per dtype; consumers use tensor views.

    Call on the producer stream before the auxiliary stream's wait_stream.
    Keep returned snapshots with the forward owner. PyTorch's pinned allocator
    also records nonblocking copy use, protecting early exception destruction.
    """
    device = torch.device(device)
    groups = {}
    result = dict(plan)
    for key, value in plan.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("CP upload plan must contain CPU tensors")
            groups.setdefault(value.dtype, []).append((key, value))
    snapshots = []
    for dtype, entries in groups.items():
        host = torch.empty(
            sum(value.numel() for _, value in entries),
            dtype=dtype,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        offset = 0
        for _, value in entries:
            host[offset : offset + value.numel()].copy_(value.reshape(-1))
            offset += value.numel()
        uploaded = host.to(device, non_blocking=True)
        snapshots.append(host)
        offset = 0
        for key, value in entries:
            result[key] = uploaded[offset : offset + value.numel()].view(value.shape)
            offset += value.numel()
    return result, snapshots


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
        # Keep registered windows on their existing allocator/lifetime path.
        # Unregistered receive storage can be the consumer's double buffer.
        self.logical_rows = sum(prefixes) + sum(lengths)
        self.history_rows = fmha._cp_size * plan["history_padded"]
        physical_rows = self.history_rows + fmha._cp_size * plan["new_max"]
        capacity = max(self.logical_rows, physical_rows)
        self.rank_major = (
            os.environ.get("RTP_LLM_CP_KV_RANK_MAJOR", "0") == "1"
            and os.environ.get("RTP_LLM_CP_KV_REGISTERED", "1") == "0"
            # Padding must not exceed the old two consumers + receive scratch.
            and 2 * capacity <= 2 * self.logical_rows + self.history_rows
        )
        if self.rank_major:
            mapping = plan.pop("history_restore")
            mapping[plan.pop("new_dest")] = self.history_rows + plan.pop("new_restore")
            plan["rank_major_restore"] = mapping
        batch_upload = os.environ.get("RTP_LLM_CP_BATCH_PLAN_UPLOAD", "0") == "1"
        self._plan_host_snapshots = []
        if batch_upload:
            plan["new_lengths"] = torch.tensor(
                [len(plan["new_local_rows"])], dtype=torch.int32, device="cpu"
            )
            plan["new_starts"] = torch.zeros(1, dtype=torch.int32, device="cpu")
            plan["new_padded_lengths"] = torch.tensor(
                [plan["new_max"]], dtype=torch.int32, device="cpu"
            )
            self.plan, self._plan_host_snapshots = _upload_immutable_plan(
                plan, self.device
            )
        else:
            self.plan = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in plan.items()
            }
        width = self.op.kv_lora_rank + self.op.qk_rope_head_dim
        if self.rank_major:
            self.buffers = [
                torch.empty((capacity, width), dtype=torch.bfloat16, device=self.device)
                for _ in range(2)
            ]
        else:
            self.buffers = [self.op._allocate_fused_kv() for _ in range(2)]
        options = dict(dtype=torch.bfloat16, device=self.device)
        pair = registered_kv_overlap_buffers(
            plan["history_padded"], width, torch.bfloat16, self.device, Group.TP
        )
        self.registered = pair is not None
        if pair is None:
            self.local = torch.empty((plan["history_padded"], width), **options)
            self.gathered = (
                None
                if self.rank_major
                else torch.empty(
                    (fmha._cp_size * plan["history_padded"], width), **options
                )
            )
        else:
            self.local, self.gathered = pair
        # Each request's actual count is <= its padded count. Equal totals
        # therefore prove a dense identity layout without a GPU-side check.
        self.history_is_dense = (
            os.environ.get("RTP_LLM_CP_KV_INDEX_OPT", "1") == "1"
            and plan["history_actual"] == plan["history_padded"]
        )
        self.actual = self.local if self.history_is_dense else None
        self.new_slots = {}
        for location, slots in (("hbm", cache.original_slots), ("host", cache.slots)):
            self.new_slots[location] = (
                slots.index_select(0, self.plan["new_local_rows"])
                .to(torch.int32)
                .view(1, -1)
            )
        if batch_upload:
            self.new_lengths = self.plan["new_lengths"]
            self.new_starts = self.plan["new_starts"]
            self.new_padded_lengths = self.plan["new_padded_lengths"]
        else:
            self.new_lengths = torch.tensor(
                [len(plan["new_local_rows"])], dtype=torch.int32, device=self.device
            )
            self.new_starts = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.new_padded_lengths = torch.tensor(
                [plan["new_max"]], dtype=torch.int32, device=self.device
            )
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
            local = self.op._pack_sharded_kv_cache(
                kv_cache, self.buffers[0][: self.logical_rows]
            )
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
                    gathered,
                    0,
                    self.op.sharded_kv_restore_indices,
                    out=self.buffers[0][: self.logical_rows],
                )
                self._record_ready(0)
        else:
            if self.owners[layer % 2] != layer:
                raise RuntimeError("next-layer history was not prefetched")
            # Read newly written bytes through the SAME FP8 upconverter as the
            # full gather. Never substitute unquantized QKV projection values.
            # Newly written cache bytes depend on the compute stream. Keep
            # packing, communication and patching together on the history
            # stream, while compute submits the independent Indexer work.
            self.stream.wait_stream(self.compute)
            with torch.cuda.stream(self.stream):
                source = kv_cache.kv_cache_base.view(torch.uint8)
                source = source.reshape(-1, 1, source.size(-1))
                count = self.new_slots["hbm"].size(1)
                local = torch.empty(
                    (self.plan["new_max"], self.buffers[0].size(1)),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                location = "host" if layer in self.cache.host_layers else "hbm"
                # The existing padded upconverter defines every row in one
                # launch, including empty-rank padding, without a separate fill.
                rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                    source,
                    local,
                    self.new_slots[location],
                    self.new_lengths,
                    self.new_padded_lengths,
                    self.new_starts,
                )
                if self.rank_major:
                    # Append rank-major fresh rows directly after immutable
                    # history. The same ready/free events protect both regions.
                    gathered = self.buffers[layer % 2][
                        self.history_rows : self.history_rows
                        + self.fmha._cp_size * local.size(0)
                    ]
                else:
                    gathered = torch.empty(
                        (self.fmha._cp_size * local.size(0), local.size(1)),
                        dtype=local.dtype,
                        device=self.device,
                    )
                all_gather_kv_overlap(local, Group.TP, output=gathered)
                if not self.rank_major:
                    restore_patch_kv(
                        gathered,
                        self.plan["new_restore"],
                        self.plan["new_dest"],
                        self.buffers[layer % 2],
                    )
                # The consumer must join both prefetched history and this
                # patch, not the older history-only event.
                self._record_ready(layer)
            self.pending_new = (local, gathered)
        # Submit while current Indexer/Q work is still ahead. Waiting until
        # consume lets Attention occupy the SMs before the prefix pack starts,
        # leaving NCCL queued behind it despite independent CUDA streams.
        self.prefetch(layer + 1)
        values = self.buffers[layer % 2]
        if self.rank_major and layer:
            # Local import avoids the parent module's pipeline import cycle.
            from .flashmla_sparse_cp_impl import _RankMajorKv

            values = _RankMajorKv(values, self.plan["rank_major_restore"])
        else:
            values = values[: self.logical_rows]
        return values, self.ready[layer % 2]

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
            # The patch was submitted on the producer before its ready event.
            # Drop temporary storage only after the consumer has joined it.
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
            source = source.view(torch.uint8)
            if source.ndim == 4:
                source = source.squeeze(2)
            if self.history_is_dense:
                if self.plan["history_actual"]:
                    rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                        source,
                        self.local,
                        table,
                        self.plan["history_lengths"],
                        self.plan["history_starts"],
                        len(self.plan["history_lengths"]),
                        self.plan["history_actual"],
                    )
            else:
                rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_padded(
                    source,
                    self.local,
                    table,
                    self.plan["history_lengths"],
                    self.plan["history_padded_lengths"],
                    self.plan["history_padded_starts"],
                )
            gathered = (
                self.buffers[slot][: self.history_rows]
                if self.rank_major
                else self.gathered
            )
            all_gather_kv_overlap(self.local, Group.TP, output=gathered)
            if not self.rank_major:
                # New-token holes are patched before the final ready event.
                torch.index_select(
                    gathered, 0, self.plan["history_restore"], out=self.buffers[slot]
                )
            # The direct receive slot is protected by the same free event as
            # restored storage. TopK remapping never selects padding; fresh
            # rows are appended and patched before the consumer-ready event.
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
