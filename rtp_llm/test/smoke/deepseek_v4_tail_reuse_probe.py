"""Physical cache and direct-state probe for four-layer tail reuse smoke.

Loaded by the existing hot-hook runtime in both P/D backends. The prefill
callback snapshots tagged physical source rows after each model forward and
compares a direct reference using separate Python-owned pools. The decode
callback records actual tagged block tables after P→D handoff.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    DSV4_KERNEL_ROW_TAGS,
    as_attention_inputs_by_tag,
)
from rtp_llm.ops.compute_ops import LayerKVCache

_source_ids: dict[tuple[int, str], torch.Tensor] = {}
_source_rows: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
_sequence = 0
_decode_sequence = 0
_private_fork_cache = None
_source_production_cache = None
_fault_forward = None
_fault_f800_slots = None
_fault_f800_writer_returned = False


# Opt-in test instrumentation. Production and private-oracle forwards are
# distinguished at the outer forward boundary; no cache values are modified.
_cost_active = False
_cost_endpoint = None
_cost_samples = []
_cost_pending = []
_cost_layouts = []
_cost_memory_before = None


def before_forward_layers_cost(ctx):
    global _cost_active, _cost_samples, _cost_pending, _cost_layouts, _cost_memory_before
    _cost_active = False
    if os.environ.get("T10_F800_COST_PROBE") != "1":
        return
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    if (resume, _end_position(ctx)) != (768, 2000):
        return
    if _cp_mode(ctx):
        raise AssertionError("F800 cost probe currently supports non-CP only")
    _cost_samples, _cost_pending, _cost_layouts = [], [], []
    cache = ctx.args[1]
    for layer_id in range(int(cache.layer_count)):
        for layer in cache.get_layer_cache_groups(layer_id):
            tag = str(layer.tag)
            pages = cache.get_seq_size_per_block(
                tag
            ) // cache.get_kernel_seq_size_per_block(tag)
            base, scale = layer.kv_cache_base, layer.kv_scale_base
            _cost_layouts.append(
                {
                    "layer": layer_id,
                    "tag": tag,
                    "pointer": base.data_ptr(),
                    "pages_per_physical_block": pages,
                    "kv_block_stride_bytes": pages
                    * base.stride(0)
                    * base.element_size(),
                    "scale_block_stride_bytes": (
                        pages * scale.stride(0) * scale.element_size()
                        if scale is not None and scale.numel()
                        else 0
                    ),
                    "kv_tensor_storage_bytes": pages
                    * base[0].numel()
                    * base.element_size(),
                    "scale_tensor_storage_bytes": (
                        pages * scale[0].numel() * scale.element_size()
                        if scale is not None and scale.numel()
                        else 0
                    ),
                }
            )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _cost_memory_before = torch.cuda.memory_allocated()
    _cost_active = True


def replace_sparse_save_slots_for_cost(ctx):
    original = ctx.runtime._patched[ctx.target].original_callable
    iterator = original(*ctx.args, **ctx.kwargs)
    if not _cost_active:
        return iterator

    def selected_slots():
        global _cost_endpoint
        for endpoint, slots in iterator:
            _cost_endpoint = endpoint
            try:
                yield endpoint, slots
            finally:
                _cost_endpoint = None

    return selected_slots()


def before_selected_writer_for_cost(ctx):
    if not _cost_active:
        return
    state = ctx.target.endswith("run_save_partial_states")
    pool = ctx.args[4 if state else 1]
    slots = ctx.args[5 if state else 2]
    matches = [
        layout for layout in _cost_layouts if layout["pointer"] == pool.data_ptr()
    ]
    if len(matches) != 1:
        raise AssertionError(f"unidentified cost writer pool: {ctx.target}")
    # Events surround the existing call on its actual stream. Slot decoding and
    # event synchronization happen only after the entire production forward.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    _cost_pending.append(
        {
            "layout": matches[0],
            "endpoint": _cost_endpoint,
            "writer": ctx.target,
            "start": start,
            "end": end,
            "slots": slots,
            "bytes_per_written_row": (
                2 * ctx.args[0].shape[-1] * pool.element_size() if state else 584
            ),
        }
    )
    start.record()


def after_selected_writer_for_cost(ctx):
    if not _cost_active:
        return
    sample = _cost_pending.pop()
    if sample["writer"] != ctx.target:
        raise AssertionError("cost writer before/after pairing mismatch")
    sample["end"].record()
    _cost_samples.append(sample)


def _finish_forward_cost():
    global _cost_active
    if not _cost_active:
        return
    _cost_active = False
    if _cost_pending:
        raise AssertionError("unfinished cost writer event")
    torch.cuda.synchronize()
    memory = {
        "torch_allocated_before_bytes": _cost_memory_before,
        "torch_allocated_after_bytes": torch.cuda.memory_allocated(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": "production forward only; excludes later physical snapshots/private oracle; C++ preallocated pools are not PyTorch allocator peak",
    }
    rows = []
    for sample in _cost_samples:
        layout = sample["layout"]
        slots = sample["slots"].detach().cpu()
        active = slots[slots >= 0]
        rows.append(
            {
                "layer": layout["layer"],
                "tag": layout["tag"],
                "endpoint": sample["endpoint"],
                "purpose": (
                    "runtime"
                    if sample["endpoint"] is None
                    else (
                        "internal_f"
                        if sample["endpoint"] == 800
                        else "tail_n" if sample["endpoint"] == 2000 else "original"
                    )
                ),
                "writer": sample["writer"],
                "cuda_event_ms": sample["start"].elapsed_time(sample["end"]),
                "selected_rows": active.numel(),
                "unique_destination_slots": torch.unique(active).numel(),
                "executed_slot_mapping": slots.tolist(),
                "bytes_per_written_row": sample["bytes_per_written_row"],
                "logical_written_bytes": active.numel()
                * sample["bytes_per_written_row"],
            }
        )
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "f800_cost.json").write_text(
        json.dumps(
            {
                "resume": 768,
                "end_position": 2000,
                "fork": 800,
                "rank": _rank(),
                "samples": rows,
                "physical_layouts": [
                    {k: v for k, v in row.items() if k != "pointer"}
                    for row in _cost_layouts
                ],
                "memory": memory,
                "limits": "Single instrumented B forward, no warmup/repetition claim; CUDA event intervals include launch gaps/JIT on first use and are not isolated kernel latency. Logical writes are not DRAM traffic. Copy timings and F-only allocator peaks are not measured.",
            },
            indent=2,
        )
        + "\n"
    )


def _private_fork_geometry(cp_mode):
    return (
        int(os.environ.get("T10_PRIVATE_SOURCE_END", "1300")),
        int(os.environ.get("T10_PRIVATE_FORK_AT", "800")),
        int(os.environ.get("T10_PRIVATE_B_RESUME", "512" if cp_mode else "768")),
    )


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _cp_mode(ctx) -> bool:
    return int(getattr(ctx.args[0], "_cp_size", 1)) > 1


def _end_position(ctx) -> int:
    if _cp_mode(ctx):
        cp_info = ctx.args[0]._cp_info
        lengths = cp_info.prefill_actual_input_lengths_cpu.reshape(-1)
        prefix_input = getattr(cp_info, "prefill_prefix_lengths_cpu", None)
        if prefix_input is None:
            prefix_input = ctx.kwargs["attn_inputs"].prefix_lengths
        prefixes = prefix_input.reshape(-1)
        if lengths.numel() != 1 or prefixes.numel() != 1:
            raise AssertionError("T07 CP probe requires one request per forward")
        return int(lengths[0]) + int(prefixes[0])
    return int(ctx.args[3][-1].item()) + 1


def before_forward_layers_fault_scope(ctx):
    """Identify the exact B request before its selected-state writers run."""
    global _fault_forward
    if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    _fault_forward = (resume, _end_position(ctx))


def after_f800_selected_state_writer(ctx):
    """Record that the unchanged real writer returned for the yielded F slot."""
    global _fault_f800_writer_returned
    if len(ctx.args) > 5 and ctx.args[5] is _fault_f800_slots:
        _fault_f800_writer_returned = True


def _source_bytes_unchanged_after_fault():
    if _source_production_cache is None or not _source_rows:
        raise AssertionError("missing A source snapshot for F800 writer fault")
    checks = {}
    for (layer_id, tag), source in _source_rows.items():
        layer = next(
            group
            for group in _source_production_cache.get_layer_cache_groups(layer_id)
            if str(group.tag) == tag
        )
        current = layer.kv_cache_base.index_select(0, source["ids"])
        checks[f"{layer_id}:{tag}:values"] = bool(
            torch.equal(current, source["values"])
        )
        if "scales" in source:
            current_scales = layer.kv_scale_base.index_select(0, source["ids"])
            checks[f"{layer_id}:{tag}:scales"] = bool(
                torch.equal(current_scales, source["scales"])
            )
    return checks


def replace_sparse_save_slots_for_f800_fault(ctx):
    """Resume after the real F800 writer, then fail the one armed B forward."""
    original = ctx.runtime._patched[ctx.target].original_callable
    iterator = original(*ctx.args, **ctx.kwargs)
    arm_file = os.environ.get("T10_F800_WRITER_FAULT_ARM_FILE")
    if (
        not arm_file
        or os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill"
        or _fault_forward != (768, 2000)
        or "cp_size" not in ctx.kwargs
        or "cp_rank" not in ctx.kwargs
        or not Path(arm_file).is_file()
    ):
        return iterator

    def fail_after_selected_write():
        global _fault_f800_slots, _fault_f800_writer_returned
        for endpoint, slots in iterator:
            selected_f800 = endpoint == 800
            if selected_f800:
                _fault_f800_slots = slots
                _fault_f800_writer_returned = False
            yield endpoint, slots  # The compressor calls the real CUDA writer here.
            if not selected_f800:
                continue
            if not _fault_f800_writer_returned:
                raise AssertionError("real F800 selected-state writer did not return")
            # Synchronize only the armed failure path, after the real writer.
            torch.cuda.synchronize()
            marker = Path(arm_file).with_suffix(".json")
            marker.write_text(
                json.dumps(
                    {
                        "resume": 768,
                        "end_position": 2000,
                        "fork": 800,
                        "writer": "compressor.run_save_partial_states",
                        "real_writer_returned": True,
                        "cuda_synchronized": True,
                        "source_unchanged": _source_bytes_unchanged_after_fault(),
                        "rank": _rank(),
                        "captured_unix_ns": time.time_ns(),
                    },
                    indent=2,
                )
                + "\n"
            )
            Path(arm_file).unlink()
            raise RuntimeError("injected F800 selected-state writer post-CUDA failure")

    return fail_after_selected_write()


class _ReferenceCache:
    """Small Python-owned pools with the same native tagged layer views."""

    def __init__(
        self,
        production,
        tables,
        resume_endpoint,
        cp_mode=False,
        cp_size=1,
        independent=False,
    ):
        self.group_tags = list(production.group_tags)
        self.layer_count = int(production.layer_count)
        self._production = production
        self._resume_endpoint = resume_endpoint
        self._cp_mode = cp_mode
        self._cp_size = cp_size
        self._independent = independent
        self._private_tables = (
            {tag: table.detach().clone() for tag, table in tables.items()}
            if independent
            else None
        )
        if cp_mode and cp_size <= 1:
            raise AssertionError("CP reference requires the producer CP size")
        self._layers = {}
        for layer_id in range(self.layer_count):
            for layer in production.get_layer_cache_groups(layer_id):
                tag = str(layer.tag)
                key = (layer_id, tag)
                source = None if independent else _source_rows[key]
                table = tables[tag]
                valid = table[table >= 0]
                max_id = (
                    max(127, int(valid.max()))
                    if independent
                    else max(int(source["ids"].max()), int(valid.max()))
                )
                base = layer.kv_cache_base
                clone = torch.zeros(
                    (max_id + 1, *base.shape[1:]), dtype=base.dtype, device=base.device
                )
                if resume_endpoint == 1300 and not cp_mode:
                    clone.index_copy_(0, source["ids"], source["values"])
                scale = layer.kv_scale_base
                clone_scale = None
                if scale is not None and scale.numel():
                    clone_scale = torch.zeros(
                        (max_id + 1, *scale.shape[1:]),
                        dtype=scale.dtype,
                        device=scale.device,
                    )
                    if resume_endpoint == 1300 and not cp_mode:
                        clone_scale.index_copy_(0, source["ids"], source["scales"])
                self._layers[key] = LayerKVCache(
                    clone,
                    layer.seq_size_per_block,
                    layer_id=layer_id,
                    tag=tag,
                    kv_scale_base=clone_scale,
                )
                if independent:
                    continue
                if cp_mode:
                    self._seed_cp_prefix(key, table)
                elif resume_endpoint == 1300:
                    self._seed_verified_1300(key, table)
                else:
                    self._seed_prefix(key, table)

    def seed_private_continuation(self, tables, resume_endpoint):
        """Move only independently computed F state to C's private block IDs."""
        if not self._independent or self._private_tables is None:
            raise AssertionError("private F reference was not initialized")
        for tag in self.group_tags:
            source = self._private_tables[tag]
            destination = tables[tag]
            if (
                source.ndim != 2
                or destination.ndim != 2
                or source.shape[0] != 1
                or destination.shape[0] != 1
            ):
                raise AssertionError(f"private F reference table shape: {tag}")
            mappings = {}
            physical = self.get_seq_size_per_block(tag)
            pages = physical // self.get_kernel_seq_size_per_block(tag)
            last_slot = (resume_endpoint - 1) // physical
            if self._cp_mode and tag in DSV4_KERNEL_ROW_TAGS:
                last_slot //= self._cp_size
            selected = getattr(self, "_private_selected", {}).get(tag)
            for column in range(min(source.shape[1], destination.shape[1])):
                if column // pages > last_slot:
                    continue
                source_id = int(source[0, column])
                if selected is not None and column // pages == last_slot:
                    source_id = selected * pages + column % pages
                elif (
                    resume_endpoint == getattr(self, "_private_fork_at", 800)
                    and tag in DSV4_KERNEL_ROW_TAGS
                    and column // pages == last_slot
                ):
                    # FULL remains on A's immutable canonical block; B's
                    # runtime block contains tokens beyond this fork.
                    source_id = int(self._private_a_tables[tag][0, column])
                destination_id = int(destination[0, column])
                if source_id > 0 and destination_id > 0:
                    mappings[destination_id] = source_id
            if not mappings:
                if tag == "hca_state" and resume_endpoint % 128 == 0:
                    # HCA has no pending state at this compression boundary.
                    # Its new private runtime block starts from zero.
                    for layer_id in range(self.layer_count):
                        layer = self._layers.get((layer_id, tag))
                        if layer is None:
                            continue
                        ids = torch.unique(destination[destination > 0]).to(
                            device=layer.kv_cache_base.device, dtype=torch.long
                        )
                        if ids.numel():
                            layer.kv_cache_base.index_fill_(0, ids, 0)
                    continue
                raise AssertionError(
                    f"private F reference has no shared table columns: {tag}"
                )
            for layer_id in range(self.layer_count):
                layer = self._layers.get((layer_id, tag))
                if layer is None:
                    continue
                source_ids = torch.tensor(
                    list(mappings.values()), device=layer.kv_cache_base.device
                )
                destination_ids = torch.tensor(
                    list(mappings), device=layer.kv_cache_base.device
                )
                if int(destination_ids.max()) >= layer.kv_cache_base.shape[0]:
                    raise AssertionError(
                        f"private F reference pool is too small: {tag}"
                    )
                values = layer.kv_cache_base.index_select(0, source_ids)
                layer.kv_cache_base.index_copy_(0, destination_ids, values)
                if layer.kv_scale_base is not None and layer.kv_scale_base.numel():
                    scales = layer.kv_scale_base.index_select(0, source_ids)
                    layer.kv_scale_base.index_copy_(0, destination_ids, scales)

    def _seed_cp_prefix(self, key, table):
        source = _source_rows[key]
        tag = key[1]
        physical = self.get_seq_size_per_block(tag)
        pages = physical // self.get_kernel_seq_size_per_block(tag)
        endpoint = self._resume_endpoint
        global_last_slot = (endpoint - 1) // physical
        # FULL pages are round-robin owned by CP ranks; each rank's table
        # contains only its local pages. Fixed STATE/SWA tables retain global
        # slots even though their physical payload is sliced across ranks.
        last_slot = (
            global_last_slot // self._cp_size
            if tag in DSV4_KERNEL_ROW_TAGS
            else global_last_slot
        )
        saved_id = None
        if tag not in DSV4_KERNEL_ROW_TAGS:
            points = source["save_positions"].reshape(-1)
            blocks = source["save_block_ids"].reshape(-1)
            matched = blocks[points == endpoint]
            if matched.numel() != 1 or int(matched[0]) <= 0:
                raise AssertionError(
                    f"missing A short state save for CP reference: {key} G={endpoint}"
                )
            saved_id = int(matched[0])
        for slot in range(last_slot + 1):
            for offset in range(pages):
                column = slot * pages + offset
                destination_id = int(table[0, column])
                if destination_id <= 0:
                    continue
                source_id = (
                    saved_id * pages + offset
                    if slot == last_slot and saved_id is not None
                    else int(source["runtime_table"][0, column])
                )
                if source_id <= 0:
                    raise AssertionError(
                        f"missing A CP prefix column {key} column={column}"
                    )
                self._copy_source_row(key, source_id, destination_id)

    def _seed_verified_1300(self, key, table):
        source = _source_rows[key]
        tag = key[1]
        physical = self.get_seq_size_per_block(tag)
        kernel = self.get_kernel_seq_size_per_block(tag)
        pages = physical // kernel
        points = source["save_positions"].reshape(-1)
        blocks = source["save_block_ids"].reshape(-1)
        matched = blocks[points == 1300]
        if matched.numel() and int(matched[0]) >= 0:
            source_ids = [int(matched[0]) * pages + offset for offset in range(pages)]
        else:
            first = ((1300 - 1) // physical) * pages
            source_ids = [
                int(
                    source["runtime_table"][
                        0, (first + offset) % source["runtime_table"].shape[1]
                    ]
                )
                for offset in range(pages)
            ]
        first = ((1300 - 1) // physical) * pages
        destinations = [
            int(table[0, (first + offset) % table.shape[1]]) for offset in range(pages)
        ]
        for source_id, destination_id in zip(source_ids, destinations):
            self._copy_source_row(key, source_id, destination_id)

    def _copy_source_row(self, key, source_id, destination_id):
        source = _source_rows[key]
        source_row = torch.nonzero(source["ids"] == source_id).reshape(-1)
        if source_row.numel() != 1:
            raise RuntimeError(f"missing A1300 source row {key} id={source_id}")
        layer = self._layers[key]
        layer.kv_cache_base[destination_id].copy_(source["values"][source_row[0]])
        if layer.kv_scale_base is not None and layer.kv_scale_base.numel():
            layer.kv_scale_base[destination_id].copy_(source["scales"][source_row[0]])

    def _seed_prefix(self, key, table):
        source = _source_rows[key]
        tag = key[1]
        physical = self.get_seq_size_per_block(tag)
        kernel = self.get_kernel_seq_size_per_block(tag)
        pages = physical // kernel
        endpoint = self._resume_endpoint
        if endpoint % physical:
            raise AssertionError(
                f"fork reference requires a complete block: {key} G={endpoint}"
            )
        last_slot = (endpoint - 1) // physical
        # The B table keeps absolute logical columns even after evicting older
        # fixed/ring blocks. Seed only the columns B actually retained at G.
        # HCA_STATE has no live prefix column at these complete boundaries.
        for slot in range(last_slot + 1):
            for offset in range(pages):
                column = slot * pages + offset
                destination_id = int(table[0, column])
                if destination_id <= 0:
                    continue
                source_id = int(source["runtime_table"][0, column])
                if source_id <= 0:
                    raise AssertionError(
                        f"missing A1300 full-prefix column {key} column={column}"
                    )
                self._copy_source_row(key, source_id, destination_id)

    def get_layer_cache(self, layer_id, tag):
        return self._layers[(layer_id, tag)]

    def get_layer_cache_groups(self, layer_id):
        return [
            self._layers[(layer_id, tag)]
            for tag in self.group_tags
            if (layer_id, tag) in self._layers
        ]

    def get_seq_size_per_block(self, tag):
        return self._production.get_seq_size_per_block(tag)

    def get_kernel_seq_size_per_block(self, tag):
        return self._production.get_kernel_seq_size_per_block(tag)

    def get_entries_per_block(self, tag):
        return self._production.get_entries_per_block(tag)


class _WithoutCacheStore:
    def __init__(self, original, keep_selected=False):
        self._original = original
        self._keep_selected = keep_selected

    def __getattr__(self, name):
        if name == "cache_store_inputs" or (
            not self._keep_selected
            and name in ("cache_save_positions", "cache_save_block_ids")
        ):
            return None
        return getattr(self._original, name)


class _CaptureLayers:
    enabled = False

    def __init__(self, v4=None):
        self.layers = {}
        self._v4 = v4

    def capture_layer(self, layer_id, hidden):
        if self._v4 is None:
            selected = hidden[-1:]
        else:
            cp_ctx = self._v4.layers[int(layer_id)].attn._cp_ctx
            start = max(cp_ctx.seq_len_total - 128, 0)
            mask = (
                (cp_ctx.global_positions >= start)
                & (cp_ctx.global_positions < cp_ctx.seq_len_total)
                & cp_ctx.local_is_real
            )
            selected = hidden[mask]
        self.layers[int(layer_id)] = selected.detach().cpu()


def _run_direct_reference(ctx, directory, end_position):
    from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder

    production = ctx.args[1]
    tables = ctx.args[5]
    resume_endpoint = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    cp_mode = _cp_mode(ctx)
    cp_size = int(getattr(ctx.args[0], "_cp_size", 1)) if cp_mode else 1
    reference = _ReferenceCache(
        production,
        tables,
        resume_endpoint,
        cp_mode=cp_mode,
        cp_size=cp_size,
        independent=resume_endpoint == 0,
    )
    args = list(ctx.args)
    args[1] = reference
    kwargs = dict(ctx.kwargs)
    kwargs["attn_inputs"] = _WithoutCacheStore(kwargs["attn_inputs"])
    kwargs["attention_inputs"] = None
    v4 = ctx.args[0]
    capture = _CaptureLayers(v4 if cp_mode else None)
    kwargs["capture_context"] = capture
    # Call the unwrapped layer stack to avoid re-entering the probe. The
    # reference uses the same checkpoint weights and Python kernels, but its
    # cache is built directly from A's saved physical rows, bypassing C++
    # restore/copy/stream resource plumbing and all cache-store writers.
    original = ctx.runtime._patched[ctx.target].original_callable
    was_recording = recorder.ENABLED
    recorder.ENABLED = False
    try:
        with torch.inference_mode():
            final = original(*args, **kwargs)
            if cp_mode:
                cp_ctx = v4.layers[0].attn._cp_ctx
                last_mask = (
                    cp_ctx.global_positions == cp_ctx.seq_len_total - 1
                ) & cp_ctx.local_is_real
                final_last = final[last_mask].contiguous()
            else:
                final_last = final[-1:].contiguous()
            logits = torch.mm(
                final_last.to(v4.head_weight.dtype), v4.head_weight.t()
            ).float()
    finally:
        recorder.ENABLED = was_recording
    torch.save(
        {
            "layers": capture.layers,
            "final_last": final_last.detach().cpu(),
            "logits_last": logits.detach().cpu(),
        },
        directory
        / (
            f"direct_reference_rank{_rank()}_{end_position}_g{resume_endpoint}.pt"
            if cp_mode
            else f"direct_reference_{end_position}_g{resume_endpoint}.pt"
        ),
    )


def _run_private_fork_reference(ctx, directory, end_position, resume_endpoint):
    """Replay A and B in private pools, then continue C from B's exact F."""
    global _private_fork_cache

    from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder

    production = ctx.args[1]
    tables = ctx.args[5]
    cp_mode = _cp_mode(ctx)
    cp_size = int(getattr(ctx.args[0], "_cp_size", 1)) if cp_mode else 1
    source_end, fork_at, b_resume = _private_fork_geometry(cp_mode)
    if (resume_endpoint, end_position) == (0, source_end):
        if _private_fork_cache is not None:
            raise AssertionError(f"duplicate private A{source_end} source")
        _private_fork_cache = _ReferenceCache(
            production,
            tables,
            0,
            cp_mode=cp_mode,
            cp_size=cp_size,
            independent=True,
        )
        _private_fork_cache._private_endpoint = 0
        _private_fork_cache._private_fork_at = fork_at
        _private_fork_cache._private_stage = "source"
        _private_fork_cache._skip_first_b = int(
            os.environ.get("T10_PRIVATE_SKIP_FIRST_B", "0")
        )
    elif (resume_endpoint, end_position) == (b_resume, 2000):
        if (
            _private_fork_cache is None
            or _private_fork_cache._private_stage != "source"
        ):
            raise AssertionError(f"missing independently computed A{source_end} state")
        if _private_fork_cache._skip_first_b:
            _private_fork_cache._skip_first_b -= 1
            return
        _private_fork_cache.seed_private_continuation(tables, resume_endpoint)
    elif (resume_endpoint, end_position) == (fork_at, 2000):
        if (
            _private_fork_cache is None
            or _private_fork_cache._private_stage != "producer"
        ):
            raise AssertionError(
                f"missing independently computed B2000 F{fork_at} state"
            )
        _private_fork_cache.seed_private_continuation(tables, resume_endpoint)
    else:
        return
    args = list(ctx.args)
    args[1] = _private_fork_cache
    kwargs = dict(ctx.kwargs)
    private_b = (resume_endpoint, end_position) == (b_resume, 2000)
    private_a_selected = cp_mode and (resume_endpoint, end_position) == (0, source_end)
    keep_selected = private_a_selected or private_b
    kwargs["attn_inputs"] = _WithoutCacheStore(
        kwargs["attn_inputs"], keep_selected=keep_selected
    )
    kwargs["attention_inputs"] = (
        {
            tag: _WithoutCacheStore(group, keep_selected=True)
            for tag, group in as_attention_inputs_by_tag(
                ctx.kwargs.get("attention_inputs"), production
            ).items()
        }
        if keep_selected
        else None
    )
    v4 = ctx.args[0]
    capture = _CaptureLayers(v4 if cp_mode else None)
    kwargs["capture_context"] = capture
    original = ctx.runtime._patched[ctx.target].original_callable
    was_recording = recorder.ENABLED
    recorder.ENABLED = False
    try:
        with torch.inference_mode():
            final = original(*args, **kwargs)
            if cp_mode:
                cp_ctx = v4.layers[0].attn._cp_ctx
                last_mask = (
                    cp_ctx.global_positions == cp_ctx.seq_len_total - 1
                ) & cp_ctx.local_is_real
                final_last = final[last_mask].contiguous()
            else:
                final_last = final[-1:].contiguous()
            logits = torch.mm(
                final_last.to(v4.head_weight.dtype), v4.head_weight.t()
            ).float()
    finally:
        recorder.ENABLED = was_recording
    reference_name = (
        f"direct_reference_{end_position}_g{resume_endpoint}.pt"
        if resume_endpoint == fork_at
        else f"private_reference_{end_position}_g{resume_endpoint}.pt"
    )
    if cp_mode:
        reference_name = reference_name.replace(
            "reference_", f"reference_rank{_rank()}_", 1
        )
    torch.save(
        {
            "layers": capture.layers,
            "final_last": final_last.detach().cpu(),
            "logits_last": logits.detach().cpu(),
            "provenance": f"private_no_store_A0_to_{source_end}_B{b_resume}_to_2000_then_C{fork_at}_to_2000",
            "production_cache_bytes_copied": 0,
            "selected_save_slots": int(keep_selected),
        },
        directory / reference_name,
    )
    _private_fork_cache._private_endpoint = end_position
    if resume_endpoint == 0:
        _private_fork_cache._private_stage = "source"
        _private_fork_cache._private_a_tables = {
            tag: table.detach().clone() for tag, table in tables.items()
        }
    elif private_b:
        _private_fork_cache._private_stage = "producer"
    if keep_selected:
        _private_fork_cache._private_tables = {
            tag: table.detach().clone() for tag, table in tables.items()
        }
        tagged = as_attention_inputs_by_tag(
            ctx.kwargs.get("attention_inputs"), production
        )
        selected = {}
        selected_endpoint = fork_at if private_b else b_resume
        for tag, inputs in tagged.items():
            positions = getattr(inputs, "cache_save_positions", None)
            blocks = getattr(inputs, "cache_save_block_ids", None)
            if positions is None or blocks is None:
                continue
            matched = blocks.reshape(-1)[positions.reshape(-1) == selected_endpoint]
            if matched.numel() == 1 and int(matched[0]) > 0:
                selected[tag] = int(matched[0])
        _private_fork_cache._private_selected = selected
    if private_b:
        private_rows = {}
        for key, layer in _private_fork_cache._layers.items():
            tag = key[1]
            physical = _private_fork_cache.get_seq_size_per_block(tag)
            pages = physical // _private_fork_cache.get_kernel_seq_size_per_block(tag)
            slot = (fork_at - 1) // physical
            if cp_mode and tag in DSV4_KERNEL_ROW_TAGS:
                slot //= cp_size
            table = _private_fork_cache._private_tables[tag]
            if tag in selected:
                ids = selected[tag] * pages + torch.arange(
                    pages, device=layer.kv_cache_base.device
                )
            else:
                columns = (
                    slot * pages + torch.arange(pages, device=table.device)
                ) % table.shape[1]
                ids = table[0, columns].to(
                    device=layer.kv_cache_base.device, dtype=torch.long
                )
            if (ids < 0).any():
                raise AssertionError(f"private F{fork_at} missing runtime state: {key}")
            private_rows[key] = {
                "ids": ids.cpu(),
                "values": layer.kv_cache_base.index_select(0, ids).detach().cpu(),
            }
            if layer.kv_scale_base is not None and layer.kv_scale_base.numel():
                private_rows[key]["scales"] = (
                    layer.kv_scale_base.index_select(0, ids).detach().cpu()
                )
        torch.save(
            {"rows": private_rows, "production_cache_bytes_copied": 0},
            directory
            / (
                f"private_f{fork_at}_runtime_rank{_rank()}.pt"
                if cp_mode
                else f"private_f{fork_at}_runtime.pt"
            ),
        )
    if resume_endpoint == fork_at:
        _private_fork_cache = None


def after_forward_layers(ctx):
    global _sequence, _source_production_cache

    _finish_forward_cost()

    if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return

    kv_cache = ctx.args[1]
    positions = ctx.args[3]
    if kv_cache is None or positions.numel() == 0:
        return
    cp_mode = _cp_mode(ctx)
    end_position = _end_position(ctx)
    source_end, fork_at, b_resume = _private_fork_geometry(cp_mode)
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    source_capture = end_position in (1025, 1300) or (
        os.environ.get("T10_PRIVATE_FORK_REFERENCE") == "1"
        and (resume, end_position) == (0, source_end)
    )
    if end_position not in (
        768,
        800,
        1025,
        1026,
        1100,
        1300,
        1301,
        1320,
        1537,
        1544,
        1793,
        2000,
    ):
        return

    tagged_inputs = as_attention_inputs_by_tag(
        ctx.kwargs.get("attention_inputs"), kv_cache
    )
    if not tagged_inputs:
        return
    if source_capture:
        _source_ids.clear()
        _source_rows.clear()
        _source_production_cache = kv_cache
    rows = {}
    for layer_id in range(int(kv_cache.layer_count)):
        for layer_cache in kv_cache.get_layer_cache_groups(layer_id):
            tag = str(layer_cache.tag)
            group_inputs = tagged_inputs.get(tag)
            if group_inputs is None:
                continue
            table = group_inputs.kv_cache_kernel_block_id_device
            if table is None or table.numel() == 0:
                continue
            ids = torch.unique(table.reshape(-1).to(torch.long), sorted=True)
            ids = ids[(ids >= 0) & (ids < layer_cache.kv_cache_base.shape[0])]
            if ids.numel() == 0:
                continue
            key = (layer_id, tag)
            source_ids = _source_ids.get(key)
            if source_capture:
                # Optional short-point destinations are separate physical
                # blocks and do not appear in the runtime table. Capture them
                # too, or the immutability check would miss the short leaf.
                save_ids = getattr(group_inputs, "cache_save_block_ids", None)
                if save_ids is not None and save_ids.numel():
                    physical = save_ids.reshape(-1).to(
                        device=ids.device, dtype=torch.long
                    )
                    physical = physical[physical >= 0]
                    pages = kv_cache.get_seq_size_per_block(
                        tag
                    ) // kv_cache.get_kernel_seq_size_per_block(tag)
                    if physical.numel():
                        offsets = torch.arange(pages, device=ids.device)
                        sparse_ids = (physical[:, None] * pages + offsets).reshape(-1)
                        sparse_ids = sparse_ids[
                            sparse_ids < layer_cache.kv_cache_base.shape[0]
                        ]
                        ids_for_source = torch.unique(
                            torch.cat((ids, sparse_ids)), sorted=True
                        )
                    else:
                        ids_for_source = ids
                else:
                    ids_for_source = ids
                _source_ids[key] = ids_for_source.detach().clone()
                source_ids = _source_ids[key]
            base = layer_cache.kv_cache_base
            scales = layer_cache.kv_scale_base
            if source_capture:
                save_positions = getattr(group_inputs, "cache_save_positions", None)
                save_block_ids = getattr(group_inputs, "cache_save_block_ids", None)
                _source_rows[key] = {
                    "ids": source_ids,
                    "values": base.index_select(0, source_ids).detach().clone(),
                    "runtime_table": table.detach().clone(),
                    "save_positions": (
                        save_positions.detach().cpu().clone()
                        if save_positions is not None
                        else torch.empty(0, dtype=torch.int64)
                    ),
                    "save_block_ids": (
                        save_block_ids.detach().cpu().clone()
                        if save_block_ids is not None
                        else torch.empty(0, dtype=torch.int32)
                    ),
                }
                if scales is not None and scales.numel():
                    _source_rows[key]["scales"] = (
                        scales.index_select(0, source_ids).detach().clone()
                    )
            entry = {
                "table_ids": ids.cpu(),
                "table_values": base.index_select(0, ids).detach().cpu(),
                "runtime_table": table.detach().cpu().clone(),
                "seq_size_per_block": kv_cache.get_seq_size_per_block(tag),
                "kernel_seq_size_per_block": kv_cache.get_kernel_seq_size_per_block(
                    tag
                ),
            }
            current_positions = getattr(group_inputs, "cache_save_positions", None)
            current_block_ids = getattr(group_inputs, "cache_save_block_ids", None)
            if current_positions is not None and current_block_ids is not None:
                entry["current_save_positions"] = (
                    current_positions.detach().cpu().clone()
                )
                entry["current_save_block_ids"] = (
                    current_block_ids.detach().cpu().clone()
                )
                if end_position == 2000:
                    positions = current_positions.reshape(-1)
                    destinations = current_block_ids.reshape(-1)
                    selected = destinations[positions == fork_at]
                    if selected.numel() == 1 and int(selected[0]) > 0:
                        pages = kv_cache.get_seq_size_per_block(
                            tag
                        ) // kv_cache.get_kernel_seq_size_per_block(tag)
                        selected_ids = selected[0].to(
                            device=base.device, dtype=torch.long
                        ) * pages + torch.arange(pages, device=base.device)
                        entry[f"selected_f{fork_at}_ids"] = selected_ids.cpu()
                        entry[f"selected_f{fork_at}_values"] = (
                            base.index_select(0, selected_ids).detach().cpu()
                        )
                        if scales is not None and scales.numel():
                            entry[f"selected_f{fork_at}_scales"] = (
                                scales.index_select(0, selected_ids).detach().cpu()
                            )
            if scales is not None and scales.numel():
                entry["table_scales"] = scales.index_select(0, ids).detach().cpu()
            if source_ids is not None:
                entry["source_ids"] = source_ids.cpu()
                entry["source_values"] = base.index_select(0, source_ids).detach().cpu()
                source = _source_rows[key]
                entry["save_positions"] = source["save_positions"]
                entry["save_block_ids"] = source["save_block_ids"]
                if scales is not None and scales.numel():
                    entry["source_scales"] = (
                        scales.index_select(0, source_ids).detach().cpu()
                    )
            rows[key] = entry

    if not rows:
        return
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"prefill_rank{_rank()}_{_sequence:03d}_end{end_position}.pt"
        if cp_mode
        else f"prefill_{_sequence:03d}_end{end_position}.pt"
    )
    file = directory / name
    torch.save(
        {
            "end_position": end_position,
            "rank": _rank(),
            "captured_unix_ns": time.time_ns(),
            "rows": rows,
        },
        file,
    )
    _sequence += 1
    if os.environ.get("T10_PRIVATE_FORK_REFERENCE") == "1" and (
        resume,
        end_position,
    ) in (
        (0, source_end),
        (b_resume, 2000),
        (fork_at, 2000),
    ):
        try:
            _run_private_fork_reference(ctx, directory, end_position, resume)
        except Exception:
            (directory / "private_fork_reference_error.txt").write_text(
                traceback.format_exc()
            )
            raise
    if (
        not (
            os.environ.get("T10_PRIVATE_FORK_REFERENCE") == "1"
            and (resume, end_position) == (fork_at, 2000)
        )
        and end_position in (1026, 1301, 1320, 1544, 2000)
        and resume
        in (
            768,
            1024,
            1025,
            1300,
        )
    ):
        try:
            _run_direct_reference(ctx, directory, end_position)
        except Exception:
            error_name = (
                f"direct_reference_error_rank{_rank()}.txt"
                if cp_mode
                else "direct_reference_error.txt"
            )
            (directory / error_name).write_text(traceback.format_exc())
            raise
    if os.environ.get("T10_F800_FATAL_COLD_REFERENCE") == "1" and (
        resume,
        end_position,
    ) == (0, 2000):
        _run_direct_reference(ctx, directory, end_position)


def after_decode_forward(ctx):
    """Record the decode side's physical and kernel block tables for P→D."""
    global _decode_sequence

    if os.environ.get("T07_CACHE_PROBE_ROLE") != "decode":
        return
    kv_cache = ctx.args[1]
    inputs = ctx.args[3]
    if kv_cache is None or inputs.input_ids.numel() == 0:
        return
    tagged = as_attention_inputs_by_tag(inputs.attention_inputs, kv_cache)
    tables = {}
    for tag, group_inputs in tagged.items():
        physical = group_inputs.kv_cache_block_id_device
        kernel = group_inputs.kv_cache_kernel_block_id_device
        if physical is None or kernel is None:
            continue
        tables[tag] = {
            "physical_ids": physical[physical >= 0].detach().cpu().tolist(),
            "kernel_ids": kernel[kernel >= 0].detach().cpu().tolist(),
        }
    if not tables:
        return
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"decode_rank{_rank()}_{_decode_sequence:03d}.json"
        if int(os.environ.get("T07_CACHE_PROBE_CP_SIZE", "1")) > 1
        else f"decode_{_decode_sequence:03d}.json"
    )
    (directory / name).write_text(
        json.dumps(
            {
                "rank": _rank(),
                "input_len": int(inputs.input_ids.numel()),
                "tables": tables,
            },
            indent=2,
        )
        + "\n"
    )
    _decode_sequence += 1
