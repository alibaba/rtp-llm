"""Read-only rank-local physical LINEAR cache snapshots for the CP smoke."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids

_source_ids: dict[tuple[int, str], torch.Tensor] = {}
_sequence = 0


def _rank() -> int:
    return (
        dist.get_rank()
        if dist.is_available() and dist.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )


def _capture(ctx, phase: str) -> None:
    global _sequence
    if os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill":
        return
    model, inputs = ctx.args[:2]
    cache = model.kv_cache
    if cache is None:
        return
    linear_inputs = select_attention_inputs_for_layer(inputs, cache, 0)
    if isinstance(linear_inputs, list) or not linear_inputs.is_prefill:
        return
    cp_info = linear_inputs.context_parallel_info
    prefixes = linear_inputs.prefix_lengths.reshape(-1)
    lengths = (
        cp_info.prefill_actual_input_lengths_cpu
        if cp_info is not None
        else linear_inputs.input_lengths
    ).reshape(-1)
    if prefixes.numel() != 1 or lengths.numel() != 1:
        return
    prefix = int(prefixes[0])
    endpoint = prefix + int(lengths[0])
    if endpoint not in (1300, 2000) or (phase == "before" and prefix == 0):
        return

    rows = {}
    for layer_id in range(
        min(int(os.environ.get("T09_CACHE_PROBE_LAYERS", "3")), int(cache.layer_count))
    ):
        group_inputs = select_attention_inputs_for_layer(inputs, cache, layer_id)
        if isinstance(group_inputs, list):
            raise AssertionError(f"layer {layer_id}: expected one cache group")
        table = group_inputs.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            continue
        for layer_cache in cache.get_layer_cache_groups(layer_id):
            tag = str(layer_cache.tag)
            key = (layer_id, tag)
            pages = cache.get_seq_size_per_block(
                tag
            ) // cache.get_kernel_seq_size_per_block(tag)
            ids = table.reshape(-1).to(torch.long)
            ids = ids[ids >= 0]
            save_positions = group_inputs.cache_save_positions
            save_blocks = group_inputs.cache_save_block_ids
            if save_blocks is not None and save_blocks.numel():
                physical = save_blocks.reshape(-1).to(
                    device=ids.device, dtype=torch.long
                )
                physical = physical[physical > 0]
                if physical.numel():
                    offsets = torch.arange(pages, device=ids.device)
                    ids = torch.cat(
                        (ids, (physical[:, None] * pages + offsets).reshape(-1))
                    )
            if phase == "after" and prefix == 0:
                _source_ids[key] = torch.unique(ids, sorted=True)
            if key in _source_ids:
                ids = torch.cat((ids, _source_ids[key]))
            ids = torch.unique(ids, sorted=True)
            ids = ids[ids < layer_cache.kv_cache_base.shape[0]]
            rows[key] = {
                "ids": ids.detach().cpu(),
                "values": layer_cache.kv_cache_base.index_select(0, ids)
                .detach()
                .cpu()
                .clone(),
                "runtime_table": table.detach().cpu().clone(),
                "save_positions": (
                    save_positions.detach().cpu().clone()
                    if save_positions is not None
                    else None
                ),
                "save_block_ids": (
                    save_blocks.detach().cpu().clone()
                    if save_blocks is not None
                    else None
                ),
                "seq_size_per_block": cache.get_seq_size_per_block(tag),
                "kernel_seq_size_per_block": cache.get_kernel_seq_size_per_block(tag),
            }
    if not rows:
        return
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"prefill_rank{_rank()}_{phase}_{_sequence:03d}_end{endpoint}.pt"
    hidden = None
    positions = None
    if phase == "after" and os.environ.get("T09_CACHE_PROBE_LAYERS") == "4":
        hidden = ctx.result.hidden_states.detach().cpu().clone()
        positions = select_prefill_position_ids(linear_inputs).detach().cpu().clone()
    torch.save(
        {
            "rank": _rank(),
            "phase": phase,
            "prefix": prefix,
            "end_position": endpoint,
            "captured_unix_ns": time.time_ns(),
            "rows": rows,
            "hidden_states": hidden,
            "position_ids": positions,
        },
        file,
    )
    _sequence += 1


def before_model_forward(ctx) -> None:
    _begin_numeric(ctx)
    _capture(ctx, "before")


def after_model_forward(ctx) -> None:
    _capture(ctx, "after")
    _finish_numeric(ctx)
    _run_private_reference(ctx)


_numeric = None
_numeric_sequence = 0


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    return value


def _begin_numeric(ctx):
    global _numeric
    _numeric = None
    if (
        os.environ.get("T10_GDN_NUMERIC_PROBE") != "1"
        or os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill"
    ):
        return
    model, inputs = ctx.args[:2]
    if model.kv_cache is None:
        return
    attention = select_attention_inputs_for_layer(inputs, model.kv_cache, 0)
    if not attention.is_prefill:
        return
    prefix = attention.prefix_lengths.reshape(-1)
    cp = attention.context_parallel_info
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    ).reshape(-1)
    if prefix.numel() != 1 or lengths.numel() != 1:
        return
    end = int(prefix[0]) + int(lengths[0])
    if end not in (1300, 2000):
        return
    _numeric = {
        "prefix": int(prefix[0]),
        "end_position": end,
        "rank": _rank(),
        "position_ids": _snapshot(select_prefill_position_ids(attention)),
        "layers": {},
        "current_layer": None,
    }


def _finish_numeric(ctx):
    global _numeric, _numeric_sequence
    if _numeric is None:
        return
    _numeric["final_hidden"] = _snapshot(ctx.result.hidden_states)
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"]).parent / "gdn_numeric_probe"
    directory.mkdir(exist_ok=True)
    torch.save(
        _to_cpu(_numeric),
        directory
        / f"rank{_rank()}_req{_numeric_sequence}_G{_numeric['prefix']}_N{_numeric['end_position']}.pt",
    )
    _numeric_sequence += 1
    _numeric = None


def before_decoder(ctx):
    if _numeric is None:
        return
    layer = int(ctx.args[0].layer_idx)
    _numeric["current_layer"] = layer
    _numeric["layers"][layer] = {
        "input_hidden": _snapshot(ctx.args[1]),
        "input_residual": _snapshot(ctx.args[2]),
    }


def after_decoder(ctx):
    if _numeric is None:
        return
    _numeric["layers"][int(ctx.args[0].layer_idx)]["output"] = _snapshot(ctx.result)


def after_input_project(ctx):
    if _numeric is not None:
        _numeric["layers"][_numeric["current_layer"]]["projection"] = _snapshot(
            ctx.result
        )


def before_conv(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["conv_before"] = {
            key: _snapshot(ctx.kwargs.get(key))
            for key in (
                "x",
                "initial_states",
                "prefix_lengths",
                "query_start_loc",
                "block_map",
            )
        }


def after_conv(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["conv_output"] = _snapshot(
            ctx.result
        )


def before_gdn_chunk(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["fla_before"] = {
            "qkvgb": _snapshot(ctx.args[:5]),
            **{
                key: _snapshot(ctx.kwargs.get(key))
                for key in ("initial_state", "cu_seqlens", "state_positions")
            },
        }


def after_gdn_chunk(ctx):
    if _numeric is not None and _rank() == 0:
        _numeric["layers"][_numeric["current_layer"]]["fla_after"] = {
            "output": _snapshot(ctx.result[0]),
            "final_state": _snapshot(ctx.result[2]),
            "selected_states": _snapshot(ctx.kwargs.get("selected_states")),
        }


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_cpu(item) for item in value]
    return value


class _PrivateReferenceCache:
    """Small, independent real KV pools; no production cache bytes are copied."""

    def __init__(self, production):
        from rtp_llm.ops.compute_ops import LayerKVCache

        self._production = production
        self.group_tags = list(production.group_tags)
        self.layer_count = int(production.layer_count)
        self._layers = {}
        self.store_isolation = []
        for layer_id in range(self.layer_count):
            for layer in production.get_layer_cache_groups(layer_id):
                tag = str(layer.tag)
                base = production.get_seq_size_per_block(tag)
                pages = base // production.get_kernel_seq_size_per_block(tag)
                # Reserve the actual 2000-token logical sequence, plus block 0.
                # FULL CP ownership uses fewer slots; keeping the same maximum
                # here avoids any assumption about LINEAR vs FULL tag names.
                count = ((2000 + base - 1) // base + 1) * pages
                values = layer.kv_cache_base
                private = torch.zeros(
                    (count, *values.shape[1:]),
                    dtype=values.dtype,
                    device=values.device,
                )
                scale = layer.kv_scale_base
                private_scale = None
                if scale is not None and scale.numel():
                    private_scale = torch.zeros(
                        (count, *scale.shape[1:]),
                        dtype=scale.dtype,
                        device=scale.device,
                    )
                self._layers[(layer_id, tag)] = LayerKVCache(
                    private,
                    layer.seq_size_per_block,
                    layer_id=layer_id,
                    tag=tag,
                    kv_scale_base=private_scale,
                )

    def get_layer_cache(self, layer_id):
        groups = self.get_layer_cache_groups(layer_id)
        if len(groups) != 1:
            raise AssertionError("Qwen reference expects one cache group per layer")
        return groups[0]

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

    def attention_inputs(self, original, tag):
        original_writer = original.cache_store_writer
        had_store_inputs = original.cache_store_inputs is not None
        result = original.clone_without_cache_store()
        if (
            result.cache_store_inputs is not None
            or result.cache_store_writer is not None
        ):
            raise AssertionError(
                "private reference retained external publication handles"
            )
        if (
            original.cache_store_writer is not original_writer
            or (original.cache_store_inputs is not None) != had_store_inputs
        ):
            raise AssertionError(
                "reference cloning modified production publication handles"
            )
        for name in ("prefix_lengths_device", "input_lengths_device"):
            actual, cloned = getattr(original, name), getattr(result, name)
            if actual is None:
                if cloned is not None:
                    raise AssertionError(f"reference changed undefined mirror: {name}")
            elif cloned is None or actual.data_ptr() != cloned.data_ptr():
                raise AssertionError(f"reference lost the real device mirror: {name}")
        self.store_isolation.append(
            {
                "tag": tag,
                "production_had_store_inputs": had_store_inputs,
                "production_had_store_writer": original_writer is not None,
                "reference_store_inputs": False,
                "reference_store_writer": False,
            }
        )
        pages = self.get_seq_size_per_block(tag) // self.get_kernel_seq_size_per_block(
            tag
        )
        for field, subdivision in (
            ("kv_cache_block_id", 1),
            ("kv_cache_block_id_device", 1),
            ("kv_cache_kernel_block_id", pages),
            ("kv_cache_kernel_block_id_device", pages),
        ):
            table = getattr(original, field)
            if table is None or not table.numel():
                continue
            if table.shape[0] != 1:
                raise AssertionError("reference requires one real request")
            # Logical column j always maps to private physical block j+1;
            # kernel columns retain contiguous subpages within that block.
            columns = torch.arange(
                table.shape[1], device=table.device, dtype=table.dtype
            )
            remapped = columns + subdivision
            base = self.get_seq_size_per_block(tag)
            limit = ((2000 + base - 1) // base) * subdivision
            remapped = torch.where(
                columns < limit, remapped, torch.zeros_like(remapped)
            )
            setattr(result, field, remapped.reshape(1, -1).contiguous())
        device = original.kv_cache_kernel_block_id_device.device
        result.cache_save_positions = torch.empty(
            (1, 0), dtype=torch.int32, device=device
        )
        result.cache_save_block_ids = torch.empty(
            (1, 0), dtype=torch.int32, device=device
        )
        return result


_private_reference = None
_private_reference_endpoint = 0


def _run_private_reference(ctx):
    """Two complete model forwards at 1300 and 700, with no selected writer."""
    global _private_reference, _private_reference_endpoint
    if (
        os.environ.get("T09_CACHE_PROBE_ROLE") != "prefill"
        or os.environ.get("T09_CACHE_PROBE_LAYERS") != "4"
    ):
        return
    from rtp_llm.ops.compute_ops import PyModelInputs

    model, inputs = ctx.args[:2]
    production = model.kv_cache
    if production is None:
        return
    attention = select_attention_inputs_for_layer(inputs, production, 0)
    if not attention.is_prefill or attention.prefix_lengths.numel() != 1:
        return
    cp = attention.context_parallel_info
    prefix = int(attention.prefix_lengths.reshape(-1)[0])
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    )
    endpoint = prefix + int(lengths.reshape(-1)[0])
    if (prefix, endpoint) not in ((0, 1300), (1300, 2000)):
        return
    if prefix == 0:
        if _private_reference is not None:
            raise AssertionError("duplicate private reference source")
        _private_reference = _PrivateReferenceCache(production)
    elif _private_reference is None or _private_reference_endpoint != prefix:
        raise AssertionError(
            "private reference must retain its own A1300 runtime state"
        )
    groups = inputs.attention_inputs
    if not isinstance(groups, dict):
        raise AssertionError("FULL+LINEAR reference requires real tagged inputs")
    private_inputs = PyModelInputs(
        input_ids=inputs.input_ids,
        input_hiddens=(
            inputs.input_hiddens if inputs.input_hiddens is not None else torch.empty(0)
        ),
        combo_position_ids=inputs.combo_position_ids,
        embedding_inputs=inputs.embedding_inputs,
        multimodal_inputs=inputs.multimodal_inputs,
        attention_inputs={
            tag: _private_reference.attention_inputs(group, tag)
            for tag, group in groups.items()
        },
        bert_embedding_inputs=inputs.bert_embedding_inputs,
    )
    private_inputs.capture_hidden_states = inputs.capture_hidden_states
    private_inputs.need_all_hidden_states = inputs.need_all_hidden_states
    private_inputs.need_all_logits = inputs.need_all_logits
    original = ctx.runtime._patched[ctx.target].original_callable
    production_hidden = ctx.result.hidden_states.detach().clone()
    try:
        model.kv_cache = _private_reference
        with torch.inference_mode():
            # Re-plan all real attention backends against the independent table.
            result = original(model, private_inputs, fmha_impl=None)
            hidden = result.hidden_states.detach().clone()
    finally:
        model.kv_cache = production
        ctx.result.hidden_states = production_hidden
    _private_reference_endpoint = endpoint
    directory = Path(os.environ["T09_CACHE_PROBE_DIR"]).parent / "private_reference"
    directory.mkdir(exist_ok=True)
    torch.save(
        {
            "rank": _rank(),
            "prefix": prefix,
            "end_position": endpoint,
            "hidden_states": hidden.cpu(),
            "position_ids": select_prefill_position_ids(attention)
            .detach()
            .cpu()
            .clone(),
            "provenance": "whole_model_no_save_private_runtime_1300_then_700",
            "production_cache_bytes_copied": 0,
            "selected_save_slots": 0,
            "cache_store_isolation": _private_reference.store_isolation,
        },
        directory / f"rank{_rank()}_G{prefix}_N{endpoint}.pt",
    )
    if endpoint == 2000:
        _private_reference = None
