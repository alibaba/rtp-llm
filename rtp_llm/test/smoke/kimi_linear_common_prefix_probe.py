"""Real Kimi Linear F800 physical state and independent whole-model oracle."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from rtp_llm.test.smoke.qwen35_tail_reuse_cp_probe import (
    _private_model_forward,
    _PrivateReferenceCache,
    _resolved_head_projection_comparison,
)

_private_cache = None
_source_ids: dict[tuple[int, str], torch.Tensor] = {}
_sequence = 0
_layer_capture = None


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _point(ctx):
    if os.environ.get("T10_KIMI_PROBE_ROLE") != "prefill":
        return None
    model, inputs = ctx.args[:2]
    cache = model.kv_cache
    if cache is None or int(cache.layer_count) != 27:
        return None
    attention = select_attention_inputs_for_layer(inputs, cache, 0)
    if isinstance(attention, list) or not attention.is_prefill:
        return None
    prefixes = attention.prefix_lengths.reshape(-1)
    lengths = attention.input_lengths.reshape(-1)
    if prefixes.numel() != 1 or lengths.numel() != 1:
        return None
    start = int(prefixes[0])
    end = start + int(lengths[0])
    if (start, end) not in ((0, 1300), (768, 2000), (800, 2000)):
        return None
    return model, inputs, cache, attention, start, end


def _positions(attention, start, end, hidden_rows):
    """Kimi non-CP omits explicit positions; a single unpadded request is ordered."""
    if hidden_rows != end - start:
        raise AssertionError(f"hidden rows {hidden_rows} do not match [{start},{end})")
    explicit = select_prefill_position_ids(attention)
    if explicit is None or not explicit.numel():
        if attention.context_parallel_info is not None:
            raise AssertionError("CP reference requires explicit absolute positions")
        return (
            torch.arange(start, end),
            "sequential_single_request_from_prefix_and_input_lengths",
        )
    if explicit.numel() != hidden_rows:
        raise AssertionError("explicit position ids do not match hidden rows")
    return explicit.detach().cpu().clone(), "model_attention_inputs"


def _begin_layer_capture(ctx):
    global _layer_capture
    if os.environ.get("T10_KIMI_NUMERICAL_EXTENSION") != "1" or _point(ctx) is None:
        return
    if _layer_capture is not None:
        raise AssertionError("previous Kimi layer capture was not removed")
    model = ctx.args[0]
    if len(model.layers) != 27:
        raise AssertionError("Kimi numerical extension requires all 27 real layers")
    capture = {"mode": "production", "source": {}, "comparisons": {}, "handles": []}

    def capture_layer(layer_id, output):
        hidden, residual = output.hidden_states, output.residual
        if not isinstance(hidden, torch.Tensor) or not isinstance(
            residual, torch.Tensor
        ):
            raise AssertionError(f"layer {layer_id}: missing hidden/residual tensors")
        if hidden.shape != residual.shape or hidden.ndim != 2:
            raise AssertionError(f"layer {layer_id}: invalid hidden/residual shapes")
        canonical = (hidden + residual).detach().cpu().clone()
        if capture["mode"] == "production":
            if layer_id in capture["source"]:
                raise AssertionError(f"layer {layer_id}: duplicate production output")
            capture["source"][layer_id] = canonical
        elif capture["mode"] == "private":
            produced = capture["source"].pop(layer_id, None)
            if produced is None or layer_id in capture["comparisons"]:
                raise AssertionError(
                    f"layer {layer_id}: missing/duplicate production output"
                )
            if produced.shape != canonical.shape or produced.dtype != canonical.dtype:
                raise AssertionError(
                    f"layer {layer_id}: canonical output shape/dtype mismatch"
                )
            delta = (produced.float() - canonical.float()).abs()
            unequal = produced != canonical
            capture["comparisons"][layer_id] = {
                "source_components": ["hidden_states", "residual"],
                "canonical_method": "KimiLinearModel._capture_canonical_layer: hidden_states + residual",
                "shape": list(canonical.shape),
                "dtype": str(canonical.dtype),
                "compared_element_count": canonical.numel(),
                "atol": 0,
                "rtol": 0,
                "maxabs": float(delta.nan_to_num(nan=float("inf")).max()),
                "outside_tolerance_count": int(unequal.sum()),
                "passes_original_tolerance": not bool(unequal.any()),
            }
        else:
            raise AssertionError(
                f"unexpected Kimi layer capture mode {capture['mode']}"
            )

    try:
        for layer_id, layer in enumerate(model.layers):
            capture["handles"].append(
                layer.register_forward_hook(
                    lambda _module, _args, output, idx=layer_id: capture_layer(
                        idx, output
                    )
                )
            )
    except BaseException:
        for handle in capture["handles"]:
            handle.remove()
        raise
    _layer_capture = capture


def _end_layer_capture():
    global _layer_capture
    if _layer_capture is not None:
        for handle in _layer_capture["handles"]:
            handle.remove()
        _layer_capture = None


def _selected_ids(group, cache, tag, endpoint):
    positions = group.cache_save_positions
    blocks = group.cache_save_block_ids
    device = group.kv_cache_kernel_block_id_device.device
    if positions is None or blocks is None or not positions.numel():
        return torch.empty(0, dtype=torch.long, device=device)
    matches = blocks[positions == endpoint].reshape(-1)
    if matches.numel() != 1 or int(matches[0]) <= 0:
        raise AssertionError(f"{tag} selected endpoint {endpoint} absent")
    pages = cache.get_seq_size_per_block(tag) // cache.get_kernel_seq_size_per_block(
        tag
    )
    return int(matches[0]) * pages + torch.arange(
        pages, dtype=torch.long, device=device
    )


def _ready_source_ids(group, cache, tag, endpoint):
    """An original aligned point may be a stable runtime alias, not a sparse save."""
    positions = group.cache_save_positions
    blocks = group.cache_save_block_ids
    if positions is not None and blocks is not None and positions.numel():
        matches = blocks[positions == endpoint].reshape(-1)
        if matches.numel() == 1 and int(matches[0]) > 0:
            return _selected_ids(group, cache, tag, endpoint)
    pages = cache.get_seq_size_per_block(tag) // cache.get_kernel_seq_size_per_block(
        tag
    )
    slot = (endpoint - 1) // cache.get_seq_size_per_block(tag)
    ids = group.kv_cache_kernel_block_id_device[
        0, slot * pages : (slot + 1) * pages
    ].to(torch.long)
    if ids.numel() != pages or not bool((ids > 0).all()):
        raise AssertionError(f"{tag} runtime source at {endpoint} absent")
    return ids


def _capture(ctx, phase):
    global _sequence
    point = _point(ctx)
    if point is None:
        return
    model, inputs, cache, attention, start, end = point
    if start == 0 and phase == "before":
        return
    rows = {}
    for layer_id in range(27):
        groups = cache.get_layer_cache_groups(layer_id)
        if len(groups) != 1:
            raise AssertionError(f"layer {layer_id}: expected one cache group")
        layer = groups[0]
        tag = str(layer.tag)
        group = select_attention_inputs_for_layer(inputs, cache, layer_id)
        if isinstance(group, list):
            raise AssertionError(f"layer {layer_id}: ambiguous attention inputs")
        key = (layer_id, tag)
        table = group.kv_cache_kernel_block_id_device
        if table is None or table.numel() == 0:
            raise AssertionError(f"layer {layer_id}: missing runtime table")
        ids = torch.empty(0, dtype=torch.long, device=table.device)
        if tag == "linear":
            if phase == "after" and start == 0:
                _source_ids[key] = _ready_source_ids(group, cache, tag, 768)
            if (
                phase == "after"
                and start == 768
                and os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") != "1"
            ):
                _source_ids[key] = torch.unique(
                    torch.cat(
                        (_source_ids[key], _selected_ids(group, cache, tag, 800))
                    ),
                    sorted=True,
                )
            if (
                phase == "after"
                and start == 768
                and os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") != "1"
            ):
                ids = torch.cat((ids, _selected_ids(group, cache, tag, 800)))
            if phase == "before" and start == 800:
                pages = cache.get_seq_size_per_block(
                    tag
                ) // cache.get_kernel_seq_size_per_block(tag)
                slot = (800 - 1) // cache.get_seq_size_per_block(tag)
                ids = torch.cat(
                    (ids, table[0, slot * pages : (slot + 1) * pages].to(torch.long))
                )
        elif tag == "full" and phase == "after" and start == 0:
            kernel_tokens = cache.get_kernel_seq_size_per_block(tag)
            used_pages = (1300 + kernel_tokens - 1) // kernel_tokens
            _source_ids[key] = table[0, :used_pages].to(torch.long).detach().clone()
        if key in _source_ids:
            ids = torch.cat((ids, _source_ids[key]))
        ids = torch.unique(ids, sorted=True)
        ids = ids[(ids > 0) & (ids < layer.kv_cache_base.shape[0])]
        if ids.numel() == 0:
            raise AssertionError(f"layer {layer_id}: no physical cache rows")
        rows[key] = {
            "ids": ids.detach().cpu(),
            "values": layer.kv_cache_base.index_select(0, ids).detach().cpu().clone(),
            "runtime_table": table.detach().cpu().clone(),
            "save_positions": (
                group.cache_save_positions.detach().cpu().clone()
                if group.cache_save_positions is not None
                else None
            ),
            "save_block_ids": (
                group.cache_save_block_ids.detach().cpu().clone()
                if group.cache_save_block_ids is not None
                else None
            ),
            "seq_size_per_block": cache.get_seq_size_per_block(tag),
            "kernel_seq_size_per_block": cache.get_kernel_seq_size_per_block(tag),
        }
    directory = Path(os.environ["T10_KIMI_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    hidden = (
        ctx.result.hidden_states.detach().cpu().clone() if phase == "after" else None
    )
    position_ids, position_source = (
        _positions(attention, start, end, hidden.shape[0])
        if hidden is not None
        else (None, None)
    )
    torch.save(
        {
            "rank": _rank(),
            "phase": phase,
            "prefix": start,
            "end_position": end,
            "captured_unix_ns": time.time_ns(),
            "rows": rows,
            "hidden_states": hidden,
            "position_ids": position_ids,
            "position_ids_source": position_source,
        },
        directory / f"prefill_rank{_rank()}_{phase}_{_sequence:03d}_end{end}.pt",
    )
    _sequence += 1


def _private_reference(ctx):
    global _private_cache
    point = _point(ctx)
    if point is None:
        return
    model, inputs, production, attention, start, end = point
    stages = {(0, 1300): (0, 768), (768, 2000): (1, 800), (800, 2000): (2, None)}
    stage, selected_endpoint = stages[(start, end)]
    if stage == 0:
        if _private_cache is not None:
            raise AssertionError("duplicate Kimi private source")
        _private_cache = _PrivateReferenceCache(production, fork_reference=True)
    elif _private_cache is None or _private_cache.fork_stage != stage - 1:
        raise AssertionError(f"Kimi private stage {stage} lacks predecessor")
    groups = inputs.attention_inputs
    if not isinstance(groups, dict) or set(groups) != {"linear", "full"}:
        raise AssertionError(f"Kimi requires real LINEAR/FULL groups: {type(groups)}")
    private_groups = {
        tag: _private_cache.attention_inputs(
            group,
            tag,
            stage=stage,
            # A's canonical G768 state is already held by the runtime slot.
            # It needs no new selected save in this independent model pool.
            save_endpoint=selected_endpoint if stage == 1 else None,
        )
        for tag, group in groups.items()
    }
    if stage:
        _private_cache.seed_fork_continuation(
            private_groups, stage=stage, resume_endpoint=start
        )
    numerical = os.environ.get("T10_KIMI_NUMERICAL_EXTENSION") == "1"
    if numerical and _layer_capture is None:
        raise AssertionError("Kimi 27-layer production capture absent")
    if numerical:
        _layer_capture["mode"] = "private"
    try:
        hidden = _private_model_forward(ctx, _private_cache, private_groups)
    finally:
        if numerical:
            _layer_capture["mode"] = "complete"
    position_ids, position_source = _positions(attention, start, end, hidden.shape[0])
    projection = None
    layer_comparisons = None
    if numerical:
        if _layer_capture["source"] or set(_layer_capture["comparisons"]) != set(
            range(27)
        ):
            raise AssertionError(
                "Kimi private reference did not compare every canonical decoder output"
            )
        layer_comparisons = dict(_layer_capture["comparisons"])
        projection = _resolved_head_projection_comparison(
            model,
            ctx.result.hidden_states,
            hidden,
            position_ids,
            valid_start=start,
            valid_end=end,
            atol=0,
            rtol=0,
        )
    if selected_endpoint is not None:
        _private_cache.remember_fork_inputs(
            private_groups, stage=stage, save_endpoint=selected_endpoint
        )
    state_rows = {}
    if stage in (0, 1):
        selected = _private_cache.fork_selected.get("linear") if stage == 1 else None
        if stage == 1 and selected is None:
            raise AssertionError("private KDA F800 selected state absent")
        pages = production.get_seq_size_per_block(
            "linear"
        ) // production.get_kernel_seq_size_per_block("linear")
        runtime_ids = private_groups["linear"].kv_cache_kernel_block_id_device
        source_slot = (768 - 1) // production.get_seq_size_per_block("linear")
        for layer_id in range(27):
            layer = _private_cache._layers.get((layer_id, "linear"))
            if layer is None:
                continue
            ids = (
                selected * pages
                + torch.arange(
                    pages, dtype=torch.long, device=layer.kv_cache_base.device
                )
                if selected is not None
                else runtime_ids[0, source_slot * pages : (source_slot + 1) * pages].to(
                    torch.long
                )
            )
            state_rows[(layer_id, "linear")] = (
                layer.kv_cache_base.index_select(0, ids).detach().cpu().clone()
            )
        if len(state_rows) != 20:
            raise AssertionError(
                f"expected 20 real KDA state rows, got {len(state_rows)}"
            )
    directory = Path(os.environ["T10_KIMI_PROBE_DIR"]).parent / "private_reference"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": _rank(),
            "prefix": start,
            "end_position": end,
            "hidden_states": hidden,
            "resolved_head_projection": projection,
            "canonical_layer_comparisons": layer_comparisons,
            "position_ids": position_ids,
            "position_ids_source": position_source,
            "state_rows": state_rows,
            "private_stage": stage,
            "selected_block_ids": dict(_private_cache.fork_selected),
            "production_cache_bytes_copied": 0,
            "cache_store_isolation": list(_private_cache.store_isolation),
            "provenance": "whole_27_layer_Kimi_private_no_store_A0_1300_B768_2000_C800_2000",
        },
        directory / f"rank{_rank()}_G{start}_N{end}.pt",
    )
    if stage == 2:
        _private_cache = None


def before_model_forward(ctx):
    _begin_layer_capture(ctx)
    try:
        _capture(ctx, "before")
    except BaseException:
        _end_layer_capture()
        raise


def after_model_forward(ctx):
    try:
        _capture(ctx, "after")
        if os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") != "1":
            _private_reference(ctx)
    finally:
        _end_layer_capture()
