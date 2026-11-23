"""Layer-ordered staging for a tagged MLA cache with a pinned host suffix."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch


def contiguous_runs(
    pairs: Sequence[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Coalesce adjacent source and destination pages in submission order."""
    runs: list[tuple[int, int, int]] = []
    for source, target in pairs:
        if runs:
            first_source, first_target, size = runs[-1]
            if source == first_source + size and target == first_target + size:
                runs[-1] = (first_source, first_target, size + 1)
                continue
        runs.append((source, target, 1))
    return runs


@dataclass(frozen=True)
class _PagePlan:
    table: torch.Tensor
    slots: torch.Tensor
    fetch: list[tuple[int, int, int]]
    write: list[tuple[int, int, int]]


def _local_lengths(lengths, owner_tokens, cp_size, cp_rank):
    cycle = owner_tokens * cp_size
    return (lengths // cycle) * owner_tokens + (
        lengths.remainder(cycle) - cp_rank * owner_tokens
    ).clamp(0, owner_tokens)


def _plan_pages(table, lengths, slots, page_size, capacity, logical_pages):
    live_columns = (lengths + page_size - 1) // page_size
    if bool((live_columns > table.shape[1]).any()):
        raise ValueError("prefill KV length exceeds the cache block table")
    columns = torch.arange(table.shape[1], device=table.device)
    live = columns < live_columns[:, None]
    if bool((table[live] < 0).any()):
        raise ValueError("prefill cache is missing a live kernel page")
    table = table.to(torch.int64).masked_fill(~live, -1)
    pages = torch.unique(table[table >= 0], sorted=True)
    if pages.numel() > capacity:
        raise ValueError(
            f"prefill batch needs {pages.numel()} kernel pages; "
            f"staging capacity is {capacity}"
        )
    if pages.numel() and int(pages[-1]) >= logical_pages:
        raise ValueError("prefill block table exceeds logical cache capacity")
    mapping = {page: slot for slot, page in enumerate(pages.tolist())}

    slots = slots.to(torch.int64)
    valid_slots = torch.unique(slots[slots >= 0])
    dirty_pages, write_counts = torch.unique(
        valid_slots // page_size, return_counts=True
    )
    dirty = dirty_pages.tolist()
    if any(page not in mapping for page in dirty):
        raise ValueError("KV write slots must belong to live prefill cache pages")
    full = set(dirty_pages[write_counts == page_size].tolist())
    compact_table = torch.searchsorted(pages, table.clamp_min(0))
    compact_table.masked_fill_(table < 0, -1)
    compact_slots = torch.searchsorted(
        pages, slots.clamp_min(0) // page_size
    ) * page_size + slots.remainder(page_size)
    compact_slots.masked_fill_(slots < 0, -1)
    return _PagePlan(
        compact_table,
        compact_slots,
        contiguous_runs(
            [(page, slot) for page, slot in mapping.items() if page not in full]
        ),
        contiguous_runs([(mapping[page], page) for page in dirty]),
    )


class ContiguousPrefillCache:
    """Stage live pages on one H2D stream and write dirty pages back on another.

    A slot is reusable only after its consumer finishes and its D2H completes.
    Backing tensors and their logical block IDs remain stable across chunks.
    """

    def __init__(self, backing, staging_pages, staging_layers, device):
        first = backing[0]
        if first.ndim != 3 or staging_pages <= 0 or staging_layers <= 0:
            raise ValueError("layered prefill requires paged MLA backing and staging")
        self.backing = tuple(backing)
        self.device = torch.device(device)
        self.page_size = first.shape[1]
        self.capacity = staging_pages
        self.hbm_layers = next(
            i for i, tensor in enumerate(backing) if not tensor.is_cuda
        )
        for layer, tensor in enumerate(backing):
            if (
                tensor.shape != first.shape
                or tensor.dtype != first.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError("all default cache layers must share one paged layout")
            if layer < self.hbm_layers:
                if tensor.device != self.device:
                    raise ValueError(
                        "prefill HBM layers must use the active CUDA device"
                    )
            elif tensor.is_cuda or not tensor.is_pinned():
                raise ValueError("prefill host layers must form a pinned CPU suffix")

        self.staging = tuple(
            torch.empty(
                (staging_pages, *first.shape[1:]), dtype=first.dtype, device=self.device
            )
            for _ in range(min(staging_layers, len(backing) - self.hbm_layers))
        )
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.writeback_stream = torch.cuda.Stream(device=self.device)
        self.active = False

    @classmethod
    def from_kv_cache(cls, kv_cache, layer_num):
        if kv_cache is None or layer_num == 0:
            return None
        get_layers = getattr(kv_cache, "get_prefill_staging_layers", None)
        if get_layers is None or not (staging_layers := get_layers("default")):
            return None
        caches = [kv_cache.get_layer_cache(i, "default") for i in range(layer_num)]
        if any(
            cache.kv_scale_base is not None and cache.kv_scale_base.numel() > 0
            for cache in caches
        ):
            raise ValueError(
                "layered prefill expects MLA cache scales stored in each page"
            )
        physical = kv_cache.get_seq_size_per_block("default")
        kernel = kv_cache.get_kernel_seq_size_per_block("default")
        if kernel <= 0 or physical <= 0 or physical % kernel:
            raise ValueError("invalid default cache physical to kernel page ratio")
        pages = kv_cache.get_prefill_staging_blocks("default") * (physical // kernel)
        device = torch.device("cuda", torch.cuda.current_device())
        return cls(
            [cache.kv_cache_base for cache in caches], pages, staging_layers, device
        )

    @staticmethod
    def _eligible(fmha):
        inputs = fmha.attn_inputs
        return (
            not getattr(inputs, "is_target_verify", False)
            and not getattr(inputs, "is_draft_extend", False)
            and not getattr(inputs, "is_cuda_graph", False)
        )

    def _prepare(self, fmha):
        op = fmha.fmha_impl
        self.sharded = bool(getattr(op, "kv_cache_sharded", False))
        table = getattr(fmha.attn_inputs, "kv_cache_kernel_block_id", None)
        if not isinstance(table, torch.Tensor) or table.shape != op.block_table.shape:
            table = op.block_table
        # Device-first builders may publish only a CUDA kernel table. A
        # physical block table may use different IDs when a block has many pages.
        table = table.detach().cpu()

        if fmha.attn_inputs.is_prefill:
            indptr = fmha.fmha_params.prefill_ragged_kv_len_indptr_h[
                : table.shape[0] + 1
            ]
            if indptr.numel() != table.shape[0] + 1:
                raise ValueError("missing prefill KV token lengths")
            lengths = indptr[1:].to(torch.int64) - indptr[:-1]
        else:
            # Decode adds one query token per row to the ragged indptr.
            # kvlen_h includes the entire history and the current token.
            lengths = fmha.fmha_params.kvlen_h[: table.shape[0]].to(torch.int64)
            if lengths.numel() != table.shape[0]:
                raise ValueError("missing decode KV token lengths")
        if self.sharded:
            lengths = _local_lengths(
                lengths,
                op.kv_owner_tokens_per_block,
                op.prefill_cp_size,
                op.prefill_cp_rank,
            )
        slots = (
            op.sharded_slot_mapping if self.sharded else fmha.rope_params.slot_mapping
        )
        self.original_slots = slots if self.sharded else slots.clone()
        plan = _plan_pages(
            table,
            lengths,
            self.original_slots.detach().cpu(),
            self.page_size,
            self.capacity,
            self.backing[0].shape[0],
        )
        self.block_table = plan.table.to(device=self.device, dtype=op.block_table.dtype)
        self.slots = plan.slots.to(device=self.device, dtype=slots.dtype)
        self.fetch_runs, self.write_runs = plan.fetch, plan.write

    @staticmethod
    def _copy(source, target, runs):
        if runs:
            from rtp_llm.ops.compute_ops import copy_paged_kv_cache_async

            copy_paged_kv_cache_async(source, target, runs)

    def _submit(self, layer, slot, reusable=None):
        if layer >= len(self.backing):
            return
        with torch.cuda.stream(self.prefetch_stream):
            if reusable is not None:
                self.prefetch_stream.wait_event(reusable)
            self._copy(self.backing[layer], self.staging[slot], self.fetch_runs)
            ready = torch.cuda.Event()
            ready.record(self.prefetch_stream)
        self.pending[slot] = (layer, ready)

    @contextmanager
    def forward(self, fmha):
        if self.active:
            raise RuntimeError("layered prefill forwards must be serialized")
        if not self._eligible(fmha) or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "layered MLA cache requires eager prefill or ordinary decode"
            )
        self._prepare(fmha)
        self.pending = {}
        self.next_layer = self.hbm_layers
        compute = torch.cuda.current_stream(self.device)
        self.active = True
        succeeded = False
        try:
            fmha.layered_prefill_cache = self
            # A prior forward can have joined D2H on a different compute stream.
            self.prefetch_stream.wait_stream(self.writeback_stream)
            self.prefetch_stream.wait_stream(compute)
            for slot in range(len(self.staging)):
                self._submit(self.hbm_layers + slot, slot)
            begin_kv = getattr(fmha, "begin_cross_layer_prefetch", None)
            if begin_kv is not None:
                begin_kv(self)
            yield
            if self.next_layer != len(self.backing):
                raise RuntimeError(
                    "layered prefill did not consume all host cache layers"
                )
            succeeded = True
        finally:
            close_kv = getattr(fmha, "close_cross_layer_prefetch", None)
            if close_kv is not None:
                close_kv()
            fmha.layered_prefill_cache = None
            compute.wait_stream(self.prefetch_stream)
            compute.wait_stream(self.writeback_stream)
            self.pending.clear()
            self.active = False
            if not succeeded:
                compute.synchronize()

    def borrow_next_layer(self, layer, original_table):
        if layer < self.hbm_layers:
            return self.backing[layer], original_table, None
        slot = (layer - self.hbm_layers) % len(self.staging)
        owner, ready = self.pending[slot]
        if owner != layer:
            raise RuntimeError("future MLA layer doesn't own its staging slot")
        return self.staging[slot], self.block_table, ready

    @contextmanager
    def attention(self, layer, fmha, kv_cache, *, ready_stream=None):
        if layer < self.hbm_layers:
            yield kv_cache
            return
        if layer != self.next_layer:
            raise RuntimeError("layered prefill must consume host layers in order")
        from rtp_llm.models_py.modules.factory.attention import common
        from rtp_llm.ops.compute_ops import LayerKVCache

        slot = (layer - self.hbm_layers) % len(self.staging)
        owner, ready = self.pending.pop(slot)
        if owner != layer:
            raise RuntimeError("invalid staging slot owner")
        compute = torch.cuda.current_stream(self.device)
        (ready_stream if ready_stream is not None else compute).wait_event(ready)
        op = fmha.fmha_impl
        original_table = op.block_table
        # CP publishes inside its sparse op, while ordinary prefill publishes
        # in the wrapper. Preserve the matching plan and tagged cache handle.
        writer_owner = op if hasattr(op, "write_cache_store_impl") else fmha
        writer = writer_owner.write_cache_store_impl
        target_cache = LayerKVCache(
            self.staging[slot],
            kv_cache.seq_size_per_block,
            kv_cache.layer_id,
            kv_cache.tag,
        )

        try:
            op.block_table = self.block_table
            if self.sharded:
                op.sharded_slot_mapping = self.slots
            else:
                fmha.rope_params.slot_mapping.copy_(self.slots)
            writer_owner.write_cache_store_impl = None
            yield target_cache
        finally:
            writer_owner.write_cache_store_impl = writer
            op.block_table = original_table
            if self.sharded:
                op.sharded_slot_mapping = self.original_slots
            else:
                fmha.rope_params.slot_mapping.copy_(self.original_slots)

        consumed = torch.cuda.Event()
        consumed.record(compute)
        with torch.cuda.stream(self.writeback_stream):
            self.writeback_stream.wait_event(consumed)
            self._copy(self.staging[slot], self.backing[layer], self.write_runs)
            common.apply_write_cache_store(writer, writer_owner.attn_inputs, kv_cache)
            reusable = torch.cuda.Event()
            reusable.record(self.writeback_stream)
        self._submit(layer + len(self.staging), slot, reusable)
        self.next_layer += 1
