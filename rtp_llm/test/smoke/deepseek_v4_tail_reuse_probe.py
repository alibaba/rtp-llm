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


class _ReferenceCache:
    """Small Python-owned pools with the same native tagged layer views."""

    def __init__(self, production, tables, resume_endpoint, cp_mode=False, cp_size=1):
        self.group_tags = list(production.group_tags)
        self.layer_count = int(production.layer_count)
        self._production = production
        self._resume_endpoint = resume_endpoint
        self._cp_mode = cp_mode
        self._cp_size = cp_size
        if cp_mode and cp_size <= 1:
            raise AssertionError("CP reference requires the producer CP size")
        self._layers = {}
        for layer_id in range(self.layer_count):
            for layer in production.get_layer_cache_groups(layer_id):
                tag = str(layer.tag)
                key = (layer_id, tag)
                source = _source_rows[key]
                table = tables[tag]
                valid = table[table >= 0]
                max_id = max(int(source["ids"].max()), int(valid.max()))
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
                if cp_mode:
                    self._seed_cp_prefix(key, table)
                elif resume_endpoint == 1300:
                    self._seed_verified_1300(key, table)
                else:
                    self._seed_prefix(key, table)

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
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        if name in (
            "cache_store_inputs",
            "cache_save_positions",
            "cache_save_block_ids",
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
        production, tables, resume_endpoint, cp_mode=cp_mode, cp_size=cp_size
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


def after_forward_layers(ctx):
    global _sequence

    if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return

    kv_cache = ctx.args[1]
    positions = ctx.args[3]
    if kv_cache is None or positions.numel() == 0:
        return
    cp_mode = _cp_mode(ctx)
    end_position = _end_position(ctx)
    if end_position not in (1025, 1026, 1300, 1301, 1320, 1537, 1544, 1793, 2000):
        return

    tagged_inputs = as_attention_inputs_by_tag(
        ctx.kwargs.get("attention_inputs"), kv_cache
    )
    if not tagged_inputs:
        return
    if end_position in (1025, 1300):
        _source_ids.clear()
        _source_rows.clear()
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
            if end_position in (1025, 1300):
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
            if end_position in (1025, 1300):
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
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    if end_position in (1026, 1301, 1320, 1544, 2000) and resume in (
        768,
        1024,
        1025,
        1300,
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
