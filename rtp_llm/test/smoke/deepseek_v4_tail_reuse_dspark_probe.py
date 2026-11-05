"""Physical and independent numeric oracles for real Flash-DSpARK P/D."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    SWA_KV,
    as_attention_inputs_by_tag,
    primary_attention_inputs,
)
from rtp_llm.test.smoke import deepseek_v4_tail_reuse_4layer_mtp_probe as mtp_probe
from rtp_llm.test.smoke import deepseek_v4_tail_reuse_probe as target_probe

_pending = threading.local()
_commit_source = {}
_target_aux = None
_commit_sequence = 0
_decode_sequence = 0
_SELECTED_LAYERS = frozenset((0, 40, 41, 42))


class _SelectedCaptureLayers(target_probe._CaptureLayers):
    def capture_layer(self, layer_id, hidden):
        if int(layer_id) in _SELECTED_LAYERS:
            super().capture_layer(layer_id, hidden)


class _NoCacheStore:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        if name == "cache_store_inputs":
            return None
        return getattr(self._original, name)


def _reference_inputs(inputs, pre_hidden, cache):
    groups = as_attention_inputs_by_tag(inputs.attention_inputs, cache)
    return SimpleNamespace(
        attention_inputs={tag: _NoCacheStore(value) for tag, value in groups.items()},
        input_hiddens=pre_hidden,
    )


def _save_path(name):
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _endpoints(inputs, cache):
    attn = primary_attention_inputs(inputs.attention_inputs, cache)
    prefix = int(attn.prefix_lengths.reshape(-1)[0])
    cp_info = getattr(attn, "context_parallel_info", None)
    lengths = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
    if lengths is None or lengths.numel() == 0:
        lengths = attn.input_lengths
    if lengths.numel() != 1:
        raise AssertionError("DSpARK acceptance expects one prefill request")
    return prefix, prefix + int(lengths.reshape(-1)[0])


def _swa_rows(cache, inputs, include_save=False, endpoint=None):
    group = as_attention_inputs_by_tag(inputs.attention_inputs, cache)[SWA_KV]
    table = group.kv_cache_kernel_block_id_device
    if table is None or table.numel() == 0:
        raise AssertionError("DSpARK draft SWA table missing")
    pages = cache.get_seq_size_per_block(SWA_KV) // cache.get_kernel_seq_size_per_block(
        SWA_KV
    )
    save_positions = getattr(group, "cache_save_positions", None)
    save_block_ids = getattr(group, "cache_save_block_ids", None)
    rows = {}
    for layer_id in range(int(cache.layer_count)):
        layer = cache.get_layer_cache(layer_id, SWA_KV)
        ids = torch.unique(table.reshape(-1).long(), sorted=True)
        ids = ids[(ids >= 0) & (ids < layer.kv_cache_base.size(0))]
        if include_save and save_block_ids is not None and save_block_ids.numel():
            if endpoint is None or save_positions is None:
                raise AssertionError(
                    "DSpARK source snapshot needs an exact save endpoint"
                )
            selected = save_positions.reshape(-1).to(save_block_ids.device) == endpoint
            logical = save_block_ids.reshape(-1)[selected]
            logical = logical.to(device=ids.device, dtype=torch.long)
            logical = logical[logical >= 0]
            offsets = torch.arange(pages, device=ids.device)
            sparse_ids = (logical[:, None] * pages + offsets).reshape(-1)
            sparse_ids = sparse_ids[sparse_ids < layer.kv_cache_base.size(0)]
            ids = torch.unique(torch.cat((ids, sparse_ids)), sorted=True)
        row = {
            "ids": ids.detach().clone(),
            "values": layer.kv_cache_base.index_select(0, ids).detach().clone(),
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
            "seq_size_per_block": cache.get_seq_size_per_block(SWA_KV),
            "kernel_seq_size_per_block": cache.get_kernel_seq_size_per_block(SWA_KV),
        }
        scale = layer.kv_scale_base
        if scale is not None and scale.numel():
            row["scales"] = scale.index_select(0, ids).detach().clone()
        rows[(layer_id, SWA_KV)] = row
    return rows


def _cpu_rows(rows):
    return {
        key: {
            name: (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in row.items()
        }
        for key, row in rows.items()
    }


def before_forward_layers(ctx):
    if os.environ.get("T07_CACHE_PROBE_ROLE") == "prefill":
        recorder._DBG_CASE = (
            "dspark_target_cp2"
            if int(getattr(ctx.args[0], "_cp_size", 1)) > 1
            else "dspark_target"
        )


def after_forward_layers(ctx):
    global _target_aux
    if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return
    v4 = ctx.args[0]
    product = mtp_probe._remember_runtime(v4)
    count = int(getattr(v4, "_mtp_hidden_valid_tokens", 0))
    if count <= 0 or "_mtp_hidden_buffer" not in product[0]:
        raise AssertionError("DSpARK target did not capture aux feature rows")
    _target_aux = product[0]["_mtp_hidden_buffer"][:count].detach().clone()
    capture_type = target_probe._CaptureLayers
    target_probe._CaptureLayers = _SelectedCaptureLayers
    try:
        target_probe.after_forward_layers(ctx)
        resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
        if resume > 0:
            reference = v4._mtp_hidden_buffer[:count].detach().clone()
            torch.testing.assert_close(_target_aux, reference, rtol=0.002, atol=0.002)
            diff = (_target_aux.float() - reference.float()).abs()
            rank = target_probe._rank()
            torch.save(
                {
                    "rows": count,
                    "maxabs": float(diff.max()),
                    "capture_layers": (40, 41, 42),
                },
                _save_path(f"dspark_aux_rank{rank}.pt"),
            )
    finally:
        target_probe._CaptureLayers = capture_type
        mtp_probe._restore_runtime(v4, product)


def before_forward_commit(ctx):
    model = ctx.args[0]
    inputs = ctx.args[1]
    cache = getattr(model, "kv_cache", None)
    _pending.commit = None
    if (
        cache is None
        or inputs.input_hiddens is None
        or inputs.input_hiddens.numel() == 0
    ):
        return
    role = os.environ.get("T07_CACHE_PROBE_ROLE")
    pre_hidden = inputs.input_hiddens.detach().clone()
    if role == "prefill":
        resume, end = _endpoints(inputs, cache)
        if end not in (1025, 1026, 1300, 1320, 2000):
            return
        if _target_aux is None or pre_hidden.numel() != _target_aux.numel():
            raise AssertionError(
                "DSpARK commit features do not match target aux capture shape"
            )
        torch.testing.assert_close(
            pre_hidden.reshape(-1).float(),
            _target_aux.reshape(-1).float(),
            rtol=0.002,
            atol=0.002,
        )
        before = (
            _swa_rows(cache, inputs, include_save=True, endpoint=end)
            if resume > 0
            else None
        )
        _pending.commit = (role, resume, end, pre_hidden, before)
    elif role == "decode":
        source = _swa_rows(cache, inputs)
        _pending.commit = (role, None, None, pre_hidden, source)


def _compare_rows(model, reference, ids_by_layer):
    equal = {}
    for key, ids in ids_by_layer.items():
        actual = model.kv_cache.get_layer_cache(*key)
        direct = reference.get_layer_cache(*key)
        same = torch.equal(
            actual.kv_cache_base.index_select(0, ids),
            direct.kv_cache_base.index_select(0, ids),
        )
        if actual.kv_scale_base is not None and actual.kv_scale_base.numel():
            same &= torch.equal(
                actual.kv_scale_base.index_select(0, ids),
                direct.kv_scale_base.index_select(0, ids),
            )
        equal[str(key)] = bool(same)
    return equal


def _replay(ctx, reference, pre_hidden):
    model = ctx.args[0]
    production = model.kv_cache
    args = list(ctx.args)
    args[1] = _reference_inputs(ctx.args[1], pre_hidden, production)
    original = ctx.runtime._patched[ctx.target].original_callable
    try:
        model.kv_cache = reference
        with torch.inference_mode():
            return original(*args, **ctx.kwargs)
    finally:
        model.kv_cache = production


def after_forward_commit(ctx):
    global _commit_sequence
    pending = getattr(_pending, "commit", None)
    if pending is None:
        return
    role, resume, end, pre_hidden, before = pending
    model = ctx.args[0]
    cache = model.kv_cache
    if role == "decode":
        reference = mtp_probe._DraftSnapshotCache(cache, before)
        direct = _replay(ctx, reference, pre_hidden)
        ids_by_layer = {key: row["ids"] for key, row in before.items()}
        physical = _compare_rows(model, reference, ids_by_layer)
        hidden = ctx.result.hidden_states.detach().float()
        expected = direct.hidden_states.detach().float()
        torch.testing.assert_close(hidden, expected, rtol=0.002, atol=0.002)
        record = {
            "role": role,
            "rank": target_probe._rank(),
            "hidden_maxabs": (
                float((hidden - expected).abs().max()) if hidden.numel() else 0.0
            ),
            "physical_equal": physical,
            "captured_unix_ns": time.time_ns(),
        }
        torch.save(
            record, _save_path(f"dspark_decode_commit_{_commit_sequence:03d}.pt")
        )
        _commit_sequence += 1
        if not all(physical.values()):
            raise AssertionError(f"DSpARK decode commit physical mismatch: {physical}")
        return

    current = _swa_rows(cache, ctx.args[1], include_save=True, endpoint=end)
    if resume == 0:
        _commit_source.clear()
        _commit_source.update(current)
    else:
        for key, row in current.items():
            old = _commit_source[key]
            layer = cache.get_layer_cache(*key)
            row["source_ids"] = old["ids"].detach().clone()
            row["source_values"] = (
                layer.kv_cache_base.index_select(0, old["ids"]).detach().clone()
            )
            if "scales" in old:
                row["source_scales"] = (
                    layer.kv_scale_base.index_select(0, old["ids"]).detach().clone()
                )
    result = {
        "role": role,
        "rank": target_probe._rank(),
        "resume": resume,
        "end_position": end,
        "captured_unix_ns": time.time_ns(),
        "rows": _cpu_rows(current),
    }
    if resume > 0:
        restored = {}
        for key, row in before.items():
            old = _commit_source[key]
            physical = int(old["seq_size_per_block"])
            pages = physical // int(old["kernel_seq_size_per_block"])
            last_slot = (resume - 1) // physical
            old_save = old["save_block_ids"].reshape(-1)[
                old["save_positions"].reshape(-1) == resume
            ]
            if old_save.numel() != 1 or int(old_save[0]) <= 0:
                raise AssertionError(f"DSpARK short source missing: {key} G={resume}")
            old_ids = int(old_save[0]) * pages + torch.arange(
                pages, device=old["ids"].device
            )
            new_ids = row["runtime_table"].reshape(-1)[
                last_slot * pages : (last_slot + 1) * pages
            ]
            if new_ids.numel() != pages or bool((new_ids <= 0).any()):
                raise AssertionError(f"DSpARK restored runtime block absent: {key}")
            old_index = torch.searchsorted(old["ids"], old_ids)
            new_index = torch.searchsorted(row["ids"], new_ids)
            if bool((old_index >= old["ids"].numel()).any()) or bool(
                (new_index >= row["ids"].numel()).any()
            ):
                raise AssertionError(f"DSpARK short source/copy row missing: {key}")
            if not torch.equal(old["ids"][old_index], old_ids) or not torch.equal(
                row["ids"][new_index], new_ids
            ):
                raise AssertionError(f"DSpARK short source/copy row missing: {key}")
            same = torch.equal(old["values"][old_index], row["values"][new_index])
            if "scales" in old:
                same &= torch.equal(old["scales"][old_index], row["scales"][new_index])
            if (end - 1) // physical == last_slot:
                new_save = row["save_block_ids"].reshape(-1)[
                    row["save_positions"].reshape(-1) == end
                ]
                if new_save.numel() != 1 or int(new_save[0]) <= 0:
                    raise AssertionError(f"DSpARK new short destination absent: {key}")
                save_ids = int(new_save[0]) * pages + torch.arange(
                    pages, device=row["ids"].device
                )
                save_index = torch.searchsorted(row["ids"], save_ids)
                if bool((save_index >= row["ids"].numel()).any()) or not torch.equal(
                    row["ids"][save_index], save_ids
                ):
                    raise AssertionError(
                        f"DSpARK new short destination row absent: {key}"
                    )
                same &= torch.equal(old["values"][old_index], row["values"][save_index])
                if "scales" in old:
                    same &= torch.equal(
                        old["scales"][old_index], row["scales"][save_index]
                    )
            restored[str(key)] = bool(same)
        reference = mtp_probe._DraftSnapshotCache(cache, before)
        direct = _replay(ctx, reference, pre_hidden)
        hidden = ctx.result.hidden_states.detach().float()
        expected = direct.hidden_states.detach().float()
        torch.testing.assert_close(hidden, expected, rtol=0.002, atol=0.002)
        immutable = {}
        for key, row in current.items():
            old = _commit_source[key]
            layer = cache.get_layer_cache(*key)
            same = torch.equal(
                layer.kv_cache_base.index_select(0, old["ids"]), old["values"]
            )
            if "scales" in old:
                same &= torch.equal(
                    layer.kv_scale_base.index_select(0, old["ids"]), old["scales"]
                )
            immutable[str(key)] = bool(same)
        physical_equal = _compare_rows(
            model, reference, {key: row["ids"] for key, row in before.items()}
        )
        result.update(
            hidden_maxabs=(
                float((hidden - expected).abs().max()) if hidden.numel() else 0.0
            ),
            physical_equal=physical_equal,
            source_immutable=immutable,
            restored_copy_equal=restored,
        )
        if (
            not all(physical_equal.values())
            or not all(immutable.values())
            or not all(restored.values())
        ):
            raise AssertionError(f"DSpARK commit source/copy mismatch: {result}")
    name = f"dspark_commit_prefill_rank{target_probe._rank()}_{_commit_sequence:03d}_end{end}.pt"
    torch.save(result, _save_path(name))
    _commit_sequence += 1
    _pending.commit = None


def before_forward_propose(ctx):
    _pending.propose = None
    if os.environ.get("T07_CACHE_PROBE_ROLE") != "decode":
        return
    model = ctx.args[0]
    cache = getattr(model, "kv_cache", None)
    if cache is None or ctx.args[1].input_ids.numel() == 0:
        return
    _pending.propose = _swa_rows(cache, ctx.args[1])


def after_forward_propose(ctx):
    global _decode_sequence
    source = getattr(_pending, "propose", None)
    if not source:
        return
    model = ctx.args[0]
    reference = mtp_probe._DraftSnapshotCache(model.kv_cache, source)
    production = model.kv_cache
    original = ctx.runtime._patched[ctx.target].original_callable
    try:
        model.kv_cache = reference
        with torch.inference_mode():
            direct = original(*ctx.args, **ctx.kwargs)
    finally:
        model.kv_cache = production
    hidden = ctx.result.hidden_states.detach().float()
    expected = direct.hidden_states.detach().float()
    logits = torch.mm(
        hidden.to(model.v4.head_weight.dtype), model.v4.head_weight.t()
    ).float()
    expected_logits = torch.mm(
        expected.to(model.v4.head_weight.dtype), model.v4.head_weight.t()
    ).float()
    torch.testing.assert_close(hidden, expected, rtol=0.002, atol=0.002)
    torch.testing.assert_close(logits, expected_logits, rtol=0.002, atol=0.002)
    physical = _compare_rows(
        model, reference, {key: row["ids"] for key, row in source.items()}
    )
    record = {
        "role": "decode",
        "rank": target_probe._rank(),
        "hidden_maxabs": float((hidden - expected).abs().max()),
        "logits_maxabs": float((logits - expected_logits).abs().max()),
        "physical_equal": physical,
        "draft_swa_table": source[(0, SWA_KV)]["runtime_table"].detach().cpu().clone(),
        "seq_size_per_block": source[(0, SWA_KV)]["seq_size_per_block"],
        "kernel_seq_size_per_block": source[(0, SWA_KV)]["kernel_seq_size_per_block"],
        "captured_unix_ns": time.time_ns(),
    }
    torch.save(record, _save_path(f"dspark_decode_propose_{_decode_sequence:03d}.pt"))
    _decode_sequence += 1
    _pending.propose = None
    if not all(physical.values()):
        raise AssertionError(f"DSpARK decode proposal physical mismatch: {physical}")


def after_forward_decode(ctx):
    if os.environ.get("T07_CACHE_PROBE_ROLE") == "decode":
        target_probe.after_decode_forward(ctx)
