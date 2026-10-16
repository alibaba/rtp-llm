"""Bounded prefill staging for a single logical HBM/host MLA block pool.

ContiguousPrefillCache stages a contiguous host suffix in layer order on one
copy stream, with an optional bounded-window policy for contention experiments.
It reuses bounded HBM staging slots across forwards and write back only pages
containing newly computed tokens.
"""

import json
import math
import os
from collections import deque
from contextlib import contextmanager
from typing import Sequence

import torch


def contiguous_runs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Coalesce (source page, destination page) pairs without copying gaps."""
    runs: list[tuple[int, int, int]] = []
    for source, dest in pairs:
        if (
            runs
            and source == runs[-1][0] + runs[-1][2]
            and dest == runs[-1][1] + runs[-1][2]
        ):
            first, target, count = runs[-1]
            runs[-1] = (first, target, count + 1)
        else:
            runs.append((source, dest, 1))
    return runs


class _PrefillStaging:
    def __init__(
        self,
        backing,
        mapping: Sequence[int],
        staging_tokens: int,
        *,
        hbm_layers,
        staging_layers,
    ):
        if not backing or len(backing) != len(mapping):
            raise ValueError(
                "one cache and shared-index mapping entry required per layer"
            )
        self.backing = tuple(backing)
        first = backing[0]
        self.device = first.device
        if not first.is_cuda or first.ndim != 3:
            raise ValueError("layer zero must own a paged HBM MLA cache")
        self.page_size = first.shape[1]
        if staging_tokens <= 0 or staging_tokens % self.page_size:
            raise ValueError("staging capacity must contain whole kernel pages")
        self.capacity = staging_tokens // self.page_size
        for layer, tensor in enumerate(backing):
            if (
                tensor.shape != first.shape
                or tensor.dtype != first.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError("all layers must have the same contiguous MLA layout")
            if layer >= hbm_layers:
                if tensor.is_cuda or not tensor.is_pinned():
                    raise ValueError("host suffix must have pinned CPU backing")
            elif not tensor.is_cuda or tensor.device != self.device:
                raise ValueError("HBM prefix must stay on the same CUDA device")
        self.staging = tuple(
            torch.empty(
                (self.capacity, *first.shape[1:]), dtype=first.dtype, device=self.device
            )
            for _ in range(staging_layers)
        )
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.writeback_stream = torch.cuda.Stream(device=self.device)
        self.ready = tuple(torch.cuda.Event() for _ in range(staging_layers))
        self.active = False

    def begin_forward(self, fmha) -> None:
        if self.active:
            raise RuntimeError("layered prefill forwards must be serialized")
        if not fmha.attn_inputs.is_prefill or torch.cuda.is_current_stream_capturing():
            raise ValueError("layered MLA staging requires eager prefill")
        op = fmha.fmha_impl
        table = getattr(fmha.attn_inputs, "kv_cache_kernel_block_id_host", None)
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            table = fmha.attn_inputs.kv_cache_block_id_host
        if table.is_cuda or table.shape != op.block_table.shape:
            raise ValueError(
                "prefill staging requires the matching CPU kernel block table"
            )
        self.original_slots = (
            op.sharded_slot_mapping
            if getattr(op, "kv_cache_sharded", False)
            else fmha.rope_params.slot_mapping.clone()
        )
        # Plan once per batch. No per-layer index D2H or allocator-generation
        # cache: every forward reloads the live pages, including reused IDs.
        slots_cpu = self.original_slots.detach().cpu().to(torch.int64)
        table_cpu = table.to(torch.int64)
        pages = torch.unique(table_cpu[table_cpu >= 0], sorted=True)
        if pages.numel() > self.capacity:
            raise ValueError(
                f"prefill batch needs {pages.numel()} staging pages; capacity={self.capacity}; "
                "batch exceeds staging derived from the scheduler KV budget"
            )
        if pages.numel() and int(pages[-1]) >= self.backing[0].shape[0]:
            raise ValueError("prefill block table exceeds logical cache capacity")
        mapping = {page: slot for slot, page in enumerate(pages.tolist())}
        dirty_pages = torch.unique(
            slots_cpu[slots_cpu >= 0] // self.page_size, sorted=True
        ).tolist()
        if any(page not in mapping for page in dirty_pages):
            raise ValueError("KV write slots must belong to the current block table")
        safe_table = table_cpu.clamp_min(0)
        compact_table = torch.searchsorted(pages, safe_table).to(torch.int32)
        compact_table[table_cpu < 0] = -1
        self.block_table = compact_table.to(self.device)
        compact_slots = torch.searchsorted(
            pages, slots_cpu.clamp_min(0) // self.page_size
        )
        compact_slots = compact_slots * self.page_size + slots_cpu.remainder(
            self.page_size
        )
        compact_slots[slots_cpu < 0] = -1
        self.slots = compact_slots.to(
            device=self.device, dtype=self.original_slots.dtype
        )
        self.fetch_runs = self._fetch_plan(mapping, slots_cpu)
        self.write_runs = contiguous_runs(
            [(mapping[page], page) for page in dirty_pages]
        )
        self.active = True
        self.copy_stats = {
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "h2d_calls": 0,
            "d2h_calls": 0,
        }
        self.page_bytes = self.backing[0][0].numel() * self.backing[0].element_size()

    @staticmethod
    def _copy_runs(source, dest, runs) -> None:
        from rtp_llm.ops.compute_ops import copy_paged_kv_cache_async

        copy_paged_kv_cache_async(source, dest, runs)

    def _wait_ready(self, slot, compute):
        compute.wait_event(self.ready[slot])

    @contextmanager
    def attention(self, layer: int, fmha, kv_cache):
        slot = self._acquire(layer)
        if slot is None:
            yield kv_cache
            return
        from rtp_llm.ops.compute_ops import LayerKVCache

        compute = torch.cuda.current_stream(self.device)
        self._wait_ready(slot, compute)
        op = fmha.fmha_impl
        original_table = op.block_table
        original_store = fmha.write_cache_store_impl
        has_inner_store = hasattr(op, "write_cache_store_impl")
        original_inner_store = op.write_cache_store_impl if has_inner_store else None
        sharded = getattr(op, "kv_cache_sharded", False)
        target = LayerKVCache()
        target.kv_cache_base = self.staging[slot]
        op.block_table = self.block_table
        if sharded:
            op.sharded_slot_mapping = self.slots
        else:
            fmha.rope_params.slot_mapping.copy_(self.slots)
        # Publish backing storage only after D2H, never the temporary slot.
        fmha.write_cache_store_impl = None
        if has_inner_store:
            op.write_cache_store_impl = None
        succeeded = False
        try:
            yield target
            succeeded = True
        finally:
            op.block_table = original_table
            if sharded:
                op.sharded_slot_mapping = self.original_slots
            else:
                fmha.rope_params.slot_mapping.copy_(self.original_slots)
            fmha.write_cache_store_impl = original_store
            if has_inner_store:
                op.write_cache_store_impl = original_inner_store
            if succeeded:
                self._writeback(layer, slot, compute, fmha, kv_cache, original_store)

    def end_forward(self, failed: bool = False) -> None:
        # Model output readiness must include backing writes before request
        # completion can release/reassign the logical block IDs.
        compute = torch.cuda.current_stream(self.device)
        compute.wait_stream(self.prefetch_stream)
        compute.wait_stream(self.writeback_stream)
        if failed:
            compute.synchronize()
        self.active = False


def chunk_runs(runs, max_pages):
    """Divide a fetch plan into quota units without changing sparse destinations."""
    if max_pages < 1:
        raise ValueError("a copy chunk must hold at least one page")
    chunks, chunk, used = [], [], 0
    for source, target, count in runs:
        while count:
            take = min(count, max_pages - used)
            chunk.append((source, target, take))
            used += take
            source, target, count = source + take, target + take, count - take
            if used == max_pages:
                chunks.append((chunk, used))
                chunk, used = [], 0
    if chunk:
        chunks.append((chunk, used))
    return chunks


class ContiguousPrefillCache(_PrefillStaging):
    """Deadline-ordered lookahead across a contiguous host suffix.

    Copies never depend on Indexer selections. At forward start every free
    staging slot is prefetched on a single FIFO stream. Each attention release
    immediately queues the next layer behind that slot's D2H completion event.
    Attention waits only for its own layer.
    The optional window policy retains bounded, MoE-excluding submissions for
    A/B contention measurements. Neither policy synchronizes the CPU.
    """

    def __init__(self, backing, mapping, staging_tokens, hbm_layers, staging_layers):
        if not 1 <= hbm_layers < len(backing):
            raise ValueError(
                "contiguous placement requires a nonempty HBM prefix and host suffix"
            )
        if not 1 <= staging_layers <= len(backing) - hbm_layers:
            raise ValueError("staging slots must fit the host suffix")
        super().__init__(
            backing,
            mapping,
            staging_tokens,
            hbm_layers=hbm_layers,
            staging_layers=staging_layers,
        )
        self.hbm_layers = hbm_layers
        self.copy_schedule = os.environ.get(
            "RTP_LLM_DSA_PREFILL_COPY_SCHEDULE", "layer"
        )
        if self.copy_schedule not in ("layer", "window"):
            raise ValueError("prefill copy schedule must be layer or window")
        self.layerwise = self.copy_schedule == "layer"
        self.indexer_weight = float(
            os.environ.get("RTP_LLM_DSA_PREFILL_INDEXER_WEIGHT", "3")
        )
        self.chunk_bytes = (
            int(os.environ.get("RTP_LLM_DSA_PREFILL_COPY_CHUNK_MB", "8")) * 1024 * 1024
        )
        self.window_bytes = (
            int(os.environ.get("RTP_LLM_DSA_PREFILL_COPY_WINDOW_MB", "256"))
            * 1024
            * 1024
        )
        if (
            not math.isfinite(self.indexer_weight)
            or self.indexer_weight < 1
            or self.chunk_bytes <= 0
            or self.window_bytes < self.chunk_bytes
        ):
            raise ValueError(
                "invalid prefill copy window, chunk size or Indexer weight"
            )
        self.weights = [
            self.indexer_weight if i == 0 or mapping[i] != mapping[i - 1] else 1
            for i in range(len(backing))
        ]
        self.mean_host_weight = sum(self.weights[hbm_layers:]) / (
            len(backing) - hbm_layers
        )
        self.host_layers = {
            layer: (layer - hbm_layers) % staging_layers
            for layer in range(hbm_layers, len(backing))
        }

    def _fetch_plan(self, mapping, slots_cpu):
        # Only full pages overwritten by this forward may skip the old bytes.
        # Deduplicate write slots before counting; repeated slots and padding
        # cannot turn a partially written page into a fully overwritten page.
        written = torch.unique(slots_cpu[slots_cpu >= 0]) // self.page_size
        pages, counts = torch.unique(written, return_counts=True)
        full = set(pages[counts == self.page_size].tolist())
        self.skipped_pages = len(full)
        return contiguous_runs(
            [(page, slot) for page, slot in mapping.items() if page not in full]
        )

    def prefetch_before_indexer(self, layer):
        # A projection has already been launched. No top-k result is needed,
        # so use the Indexer's entire compute window for independent prefetch.
        self.prefetch_owner(layer)

    def begin_forward(self, fmha):
        self.flush_timings()
        self.timing_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
        self.capture_timing = bool(self.timing_root) and os.path.exists(
            os.path.join(self.timing_root, "prefill_copy_timing.enable")
        )
        self.timing_rows = []
        self.kv_prefetch = None
        super().begin_forward(fmha)
        self.jobs = {}
        self.pending_writes = []
        self.next_fetch = self.hbm_layers
        self.slot_owners = [None] * len(self.staging)
        self.slot_free = {}
        self.ready_layers = set()
        self.current_layer = -1
        self.window_open = False
        self.window_submitted = False
        self.layer_pages = sum(n for _, _, n in self.fetch_runs)
        self.fetch_chunks = chunk_runs(
            self.fetch_runs,
            max(
                1,
                min(
                    self.chunk_bytes // self.page_bytes,
                    math.ceil(self.layer_pages / 16),
                ),
            ),
        )
        if self.layerwise:
            self.fetch_chunks = (
                [(self.fetch_runs, self.layer_pages)] if self.fetch_runs else []
            )
            # One forward-wide dependency; do not fence a later copy behind
            # preceding MoE kernels. Slot-specific writeback events suffice.
            self.prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.window_credit = 0
        self.copy_stats.update(
            {
                "placement": "contiguous",
                "copy_schedule": self.copy_schedule,
                "hbm_layers": self.hbm_layers,
                "host_layers": len(self.host_layers),
                "staging_layers": len(self.staging),
                "indexer_weight": self.indexer_weight,
                "fully_overwritten_pages_skipped": self.skipped_pages,
                "layers_submitted_ahead": 0,
                "demand_h2d_bytes": 0,
                "indexer_window_h2d_bytes": 0,
                "shared_window_h2d_bytes": 0,
                "max_window_h2d_bytes": 0,
                "initial_h2d_bytes": 0,
                "initial_h2d_calls": 0,
                "copy_windows": [],
            }
        )
        if self.layerwise:
            self._refill_slots()
            self.copy_stats["initial_h2d_bytes"] = self.copy_stats["h2d_bytes"]
            self.copy_stats["initial_h2d_calls"] = self.copy_stats["h2d_calls"]
        begin_kv = getattr(fmha, "begin_cross_layer_prefetch", None)
        if begin_kv is not None:
            self.kv_prefetch = begin_kv(self)

    def prefetched_layer(self, layer, original_table):
        """Borrow a future layer's already-reserved slot without changing owners.

        Attention for this layer is the only operation that releases the slot;
        its history-ready event is joined before that can happen.
        """
        if layer not in self.host_layers:
            return self.backing[layer], original_table, None
        slot = self.host_layers[layer]
        if (
            not self.layerwise
            or self.slot_owners[slot] != layer
            or layer not in self.ready_layers
        ):
            raise RuntimeError("future KV cache layer has no ready staging slot")
        return self.staging[slot], self.block_table, self.ready[slot]

    def _flush_writes(self):
        for layer, slot, ready, fmha, kv_cache, store in self.pending_writes:
            self.writeback_stream.wait_event(ready)
            if not self.layerwise:
                # The opening fence was recorded after the preceding MoE.
                self.writeback_stream.wait_event(self.window_start)
            with torch.cuda.stream(self.writeback_stream):
                self._copy_runs(
                    self.staging[slot], self.backing[layer], self.write_runs
                )
                self.copy_stats["d2h_bytes"] += (
                    sum(n for _, _, n in self.write_runs) * self.page_bytes
                )
                self.copy_stats["d2h_calls"] += 1
                if fmha.attn_inputs.cache_store_inputs and store is not None:
                    from rtp_llm.models_py.modules.factory.attention import common

                    common.apply_write_cache_store(store, fmha.attn_inputs, kv_cache)
                done = torch.cuda.Event()
                done.record(self.writeback_stream)
            # Other staging slots may fetch while this slot writes back.
            # Only the first future use of this particular slot needs the fence.
            self.slot_free[slot] = done
            self.slot_owners[slot] = None
        self.pending_writes.clear()

    def before_layer(self, layer):
        if not self.active or layer != self.current_layer + 1:
            raise RuntimeError("contiguous prefill layers must execute once in order")
        self.current_layer = layer
        self.window_start = torch.cuda.Event(enable_timing=self.capture_timing)
        self.window_start.record(torch.cuda.current_stream(self.device))
        if self.capture_timing:
            self.timing_rows.append(
                {
                    "layer": layer,
                    "weight": self.weights[layer],
                    "start": self.window_start,
                }
            )
        if not self.layerwise:
            self.prefetch_stream.wait_event(self.window_start)
        self.window_open = True
        self.window_submitted = False
        self._flush_writes()
        self._refill_slots()

    def _refill_slots(self):
        # Allocate in attention order. With whole-layer copies, enqueue every
        # available slot immediately; compute-layer hooks impose no throttle.
        copied = 0
        while self.next_fetch < len(self.backing):
            future = self.next_fetch
            slot = self.host_layers[future]
            if self.slot_owners[slot] is not None:
                break
            self.slot_owners[slot] = future
            self.jobs[future] = deque(self.fetch_chunks)
            self.next_fetch += 1
            if self.layerwise:
                self.jobs[future].clear()
                amount = self._submit(future, (self.fetch_runs, self.layer_pages))
                copied += amount
                self.copy_stats["copy_windows"].append(
                    {
                        "layer": self.current_layer,
                        "target_layer": future,
                        "trigger": (
                            "forward_start"
                            if self.current_layer < 0
                            else "slot_release"
                        ),
                        "weight": (
                            self.weights[self.current_layer]
                            if self.current_layer >= 0
                            else 0
                        ),
                        "quota_bytes": self.layer_pages * self.page_bytes,
                        "h2d_bytes": amount,
                    }
                )
        if self.layerwise:
            self.copy_stats["max_window_h2d_bytes"] = max(
                self.copy_stats["max_window_h2d_bytes"], copied
            )
            if self.current_layer >= 0:
                key = (
                    "indexer_window_h2d_bytes"
                    if self.weights[self.current_layer] > 1
                    else "shared_window_h2d_bytes"
                )
                self.copy_stats[key] += copied

    def _submit(self, layer, chunk):
        runs, pages = chunk
        slot = self.host_layers[layer]
        with torch.cuda.stream(self.prefetch_stream):
            free = self.slot_free.pop(slot, None)
            if free is not None:
                self.prefetch_stream.wait_event(free)
            if pages:
                self._copy_runs(self.backing[layer], self.staging[slot], runs)
            # Also record empty-layer readiness; waiting on an unrecorded event
            # must never be used as evidence that this slot contains this layer.
            if not self.jobs[layer]:
                self.ready[slot].record(self.prefetch_stream)
                self.ready_layers.add(layer)
        amount = pages * self.page_bytes
        self.copy_stats["h2d_bytes"] += amount
        self.copy_stats["h2d_calls"] += bool(pages)
        return amount

    def prefetch_owner(self, layer):
        if not self.active or not self.window_open or layer != self.current_layer:
            raise RuntimeError("before_layer must open the prefetch window")
        if self.window_submitted:
            return
        self.window_submitted = True
        if self.layerwise:
            if layer in self.host_layers:
                if layer not in self.ready_layers:
                    raise RuntimeError("host attention has no submitted layer copy")
                self.copy_stats["layers_submitted_ahead"] += 1
            return
        # Normalize over the host suffix so its steady-state windows can move
        # one layer on average. Earlier HBM windows build the lookahead cushion.
        quota = min(
            self.window_bytes,
            self.window_credit
            + math.ceil(
                self.layer_pages
                * self.page_bytes
                * self.weights[layer]
                / self.mean_host_weight
            ),
        )
        if layer in self.ready_layers:
            self.copy_stats["layers_submitted_ahead"] += 1
        copied = 0
        for future, pending in self.jobs.items():
            if future < layer:
                continue
            runs, pages = [], 0
            while (
                pending and copied + (pages + pending[0][1]) * self.page_bytes <= quota
            ):
                chunk, count = pending.popleft()
                runs.extend(chunk)
                pages += count
            if runs or (not pending and future not in self.ready_layers):
                # One native batch per layer/window. Chunk boundaries choose
                # the bounded quota, but do not require one CUDA API each.
                copied += self._submit(future, (runs, pages))
            if pending:
                break  # Earliest attention deadline always wins.
        # Only a real deadline miss may exceed a window budget; record it rather
        # than silently treating a synchronous demand copy as successful overlap.
        if layer in self.host_layers:
            if layer not in self.jobs:
                raise RuntimeError("host attention has no staging slot")
            pending = self.jobs[layer]
            if pending:
                runs = [run for chunk, _ in pending for run in chunk]
                pages = sum(count for _, count in pending)
                pending.clear()
                amount = self._submit(layer, (runs, pages))
                copied += amount
                self.copy_stats["demand_h2d_bytes"] += amount
        # Carry only rounding residue, not unused windows while every slot is
        # full. This prevents an accumulated quota from flooding a later MoE.
        self.window_credit = min(max(0, quota - copied), self.chunk_bytes - 1)
        key = (
            "indexer_window_h2d_bytes"
            if self.weights[layer] > 1
            else "shared_window_h2d_bytes"
        )
        self.copy_stats[key] += copied
        self.copy_stats["max_window_h2d_bytes"] = max(
            self.copy_stats["max_window_h2d_bytes"], copied
        )
        self.copy_stats["copy_windows"].append(
            {
                "layer": layer,
                "weight": self.weights[layer],
                "quota_bytes": quota,
                "h2d_bytes": copied,
            }
        )

    def _timing_event(self, name):
        if self.capture_timing:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.timing_rows[-1][name] = event

    def _wait_ready(self, slot, compute):
        self._timing_event("kv_wait_begin")
        super()._wait_ready(slot, compute)
        self._timing_event("kv_wait_end")

    def flush_timings(self):
        pending = getattr(self, "pending_timings", [])
        remaining = []
        for done, rows, root in pending:
            if not done.query():
                remaining.append((done, rows, root))
                continue
            result = []
            for row in rows:
                record = {k: row[k] for k in ("layer", "weight")}
                for label, first, last in (
                    ("projection_indexer_ms", "start", "projections_end"),
                    ("attention_kv_wait_ms", "kv_wait_begin", "kv_wait_end"),
                    ("moe_fence_wait_ms", "moe_wait_begin", "moe_wait_end"),
                    ("non_moe_window_ms", "start", "moe_wait_begin"),
                ):
                    record[label] = (
                        row[first].elapsed_time(row[last])
                        if first in row and last in row
                        else 0
                    )
                result.append(record)
            rank = os.environ.get("WORLD_RANK", "0")
            with open(
                os.path.join(root, f"prefill_copy_timing_rank{rank}.jsonl"), "a"
            ) as out:
                out.write(json.dumps({"layers": result}) + "\n")
        self.pending_timings = remaining

    def _acquire(self, layer):
        self._timing_event("projections_end")
        self.prefetch_owner(layer)
        return self.host_layers.get(layer)

    def wait_prefetch_complete(self):
        if not self.active:
            raise RuntimeError("begin_forward must precede wait_prefetch_complete")
        if not self.window_submitted:
            raise RuntimeError(
                "attention must submit the current copy window before MoE"
            )
        compute = torch.cuda.current_stream(self.device)
        self._timing_event("moe_wait_begin")
        pipeline = getattr(self, "kv_prefetch", None)
        if pipeline is not None:
            pipeline.wait_before_moe()
        if not self.layerwise:
            compute.wait_stream(self.prefetch_stream)
            compute.wait_stream(self.writeback_stream)
        self._timing_event("moe_wait_end")
        self.window_open = False

    def _writeback(self, layer, slot, compute, fmha, kv_cache, original_store):
        ready = torch.cuda.Event()
        ready.record(compute)
        self.pending_writes.append((layer, slot, ready, fmha, kv_cache, original_store))
        del self.jobs[layer]
        if self.layerwise:
            # This layer's attention has released its staging slot. Write back
            # now, allowing both DMA directions to overlap subsequent compute.
            self._flush_writes()
            # The D2H event is already recorded, so its dependent H2D can be
            # submitted now without waiting on the CPU or a later layer hook.
            self._refill_slots()

    def end_forward(self, failed=False):
        pipeline = getattr(self, "kv_prefetch", None)
        if pipeline is not None:
            pipeline.close()
            self.kv_prefetch = None
        # The final layer's dirty pages have no following attention window.
        # Flush after its MoE, then join backing readiness before block release.
        self.window_start = torch.cuda.Event()
        self.window_start.record(torch.cuda.current_stream(self.device))
        self._flush_writes()
        super().end_forward(failed)
        if self.capture_timing:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self.device))
            self.pending_timings.append((done, self.timing_rows, self.timing_root))
            self.timing_rows = []
            self.flush_timings()
