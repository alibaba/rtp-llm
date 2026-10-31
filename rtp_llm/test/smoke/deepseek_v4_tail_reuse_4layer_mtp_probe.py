"""Separate target-prefill and draft-decode evidence for the four-layer MTP smoke."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import as_attention_inputs_by_tag
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.test.smoke import deepseek_v4_tail_reuse_probe as target_probe

_pending = threading.local()
_draft_sequence = 0
_draft_source_ids = {}
_draft_source_rows = {}


def _mode(ctx) -> str | None:
    cache = ctx.args[1]
    if cache is None:
        return None
    count = int(cache.layer_count)
    if count == 4:
        return "target"
    if count == 1:
        return "draft"
    raise AssertionError(f"unexpected MTP cache layer count: {count}")


def _remember_runtime(v4):
    fields = ("_mtp_hidden_buffer", "_mtp_last_hidden_buffer")
    values = {
        name: getattr(v4, name).detach().clone()
        for name in fields
        if getattr(v4, name, None) is not None
    }
    counts = {
        name: getattr(v4, name)
        for name in ("_mtp_hidden_valid_tokens", "_mtp_last_hidden_valid_tokens")
        if hasattr(v4, name)
    }
    return values, counts


def _restore_runtime(v4, saved):
    values, counts = saved
    for name, value in values.items():
        getattr(v4, name).copy_(value)
    for name, count in counts.items():
        setattr(v4, name, count)


def before_forward_layers(ctx):
    mode = _mode(ctx)
    if mode is not None:
        recorder._DBG_CASE = "four_layer" if mode == "target" else "mtp_draft_prefill"
    if mode != "draft" or os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
        return
    resume = int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0])
    _pending.draft_prefill_input = None
    if resume <= 0:
        return
    prepare = ctx.kwargs["prepare_hidden_fn"]
    model = prepare.__self__
    input_hiddens = model._cur_inputs.input_hiddens
    rows = int(ctx.args[2].numel())
    flat = input_hiddens.reshape(-1, input_hiddens.size(-1))
    if flat.size(0) < rows:
        raise AssertionError(
            "MTP draft pre-HC input has fewer rows than the continuation"
        )
    _pending.draft_prefill_input = (flat, flat[:rows].detach().clone(), rows)
    _pending.draft_prefill_runtime = _remember_runtime(ctx.args[0])


def after_forward_layers(ctx):
    global _draft_sequence
    mode = _mode(ctx)
    if mode is None:
        return
    if mode == "draft":
        if os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill":
            return
        # The one-layer draft uses the same forward helper as the four-layer
        # target. Give the existing physical/direct oracle separate source
        # state and filenames; its A1300 rows must never replace target A1300.
        target_ids, target_rows = target_probe._source_ids, target_probe._source_rows
        target_sequence = target_probe._sequence
        output_root = os.environ["T07_CACHE_PROBE_DIR"]
        target_probe._source_ids = _draft_source_ids
        target_probe._source_rows = _draft_source_rows
        target_probe._sequence = _draft_sequence
        os.environ["T07_CACHE_PROBE_DIR"] = str(Path(output_root) / "mtp_draft_prefill")
        v4 = ctx.args[0]
        saved = _remember_runtime(v4)
        pending_input = getattr(_pending, "draft_prefill_input", None)
        if pending_input is not None:
            flat, pre_input, rows = pending_input
            post_input = flat[:rows].detach().clone()
            changed = float((pre_input.float() - post_input.float()).abs().max())
            (
                Path(os.environ["T07_CACHE_PROBE_DIR"])
                / "input_replay_diagnostics.json"
            ).write_text(
                json.dumps(
                    {
                        "pre_vs_post_input_maxabs": changed,
                        "input_aliases_shared_buffer": bool(
                            v4._mtp_hidden_buffer is not None
                            and flat.data_ptr() == v4._mtp_hidden_buffer.data_ptr()
                        ),
                        "rows": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
        private_fork_reference = os.environ.get("T10_PRIVATE_FORK_REFERENCE")
        try:
            if pending_input is not None:
                _restore_runtime(v4, _pending.draft_prefill_runtime)
                flat[:rows].copy_(pre_input)
            os.environ["T10_PRIVATE_FORK_REFERENCE"] = "0"
            target_probe.after_forward_layers(ctx)
            _draft_sequence = target_probe._sequence
        finally:
            if private_fork_reference is None:
                os.environ.pop("T10_PRIVATE_FORK_REFERENCE", None)
            else:
                os.environ["T10_PRIVATE_FORK_REFERENCE"] = private_fork_reference
            _restore_runtime(v4, saved)
            if pending_input is not None:
                flat[:rows].copy_(post_input)
                _pending.draft_prefill_input = None
            os.environ["T07_CACHE_PROBE_DIR"] = output_root
            target_probe._source_ids = target_ids
            target_probe._source_rows = target_rows
            target_probe._sequence = target_sequence
        return
    v4 = ctx.args[0]
    saved = _remember_runtime(v4)
    try:
        target_probe.after_forward_layers(ctx)
    finally:
        # The target reference replays the layer stack. MTP's subsequent
        # draft consumes this shared pre-HC buffer from the product forward.
        _restore_runtime(v4, saved)


class _DraftSnapshotCache:
    def __init__(self, production, source):
        self.group_tags = list(production.group_tags)
        self.layer_count = int(production.layer_count)
        self._production = production
        self._layers = {}
        for layer_id in range(self.layer_count):
            for layer in production.get_layer_cache_groups(layer_id):
                key = (layer_id, str(layer.tag))
                row = source[key]
                ids = row["ids"]
                size = max(int(ids.max()) + 1, 1)
                base = layer.kv_cache_base
                clone = torch.zeros(
                    (size, *base.shape[1:]), dtype=base.dtype, device=base.device
                )
                clone.index_copy_(0, ids, row["values"])
                scale = layer.kv_scale_base
                clone_scale = None
                if scale is not None and scale.numel():
                    clone_scale = torch.zeros(
                        (size, *scale.shape[1:]), dtype=scale.dtype, device=scale.device
                    )
                    clone_scale.index_copy_(0, ids, row["scales"])
                self._layers[key] = LayerKVCache(
                    clone,
                    layer.seq_size_per_block,
                    layer_id=layer_id,
                    tag=str(layer.tag),
                    kv_scale_base=clone_scale,
                )

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


def before_forward_decode(ctx):
    mode = _mode(ctx)
    if mode is None:
        return
    recorder._DBG_CASE = "four_layer_decode" if mode == "target" else "mtp_draft_decode"
    if mode != "draft" or os.environ.get("T07_CACHE_PROBE_ROLE") != "decode":
        return
    cache = ctx.args[1]
    inputs = ctx.args[3]
    if cache is None or inputs.input_ids.numel() == 0:
        _pending.source = None
        return
    tagged = as_attention_inputs_by_tag(inputs.attention_inputs, cache)
    source = {}
    for layer_id in range(int(cache.layer_count)):
        for layer in cache.get_layer_cache_groups(layer_id):
            key = (layer_id, str(layer.tag))
            group = tagged.get(key[1])
            if group is None:
                continue
            table = group.kv_cache_kernel_block_id_device
            if table is None or table.numel() == 0:
                continue
            ids = torch.unique(table.reshape(-1).long(), sorted=True)
            ids = ids[(ids >= 0) & (ids < layer.kv_cache_base.size(0))]
            if ids.numel() == 0:
                continue
            row = {
                "ids": ids,
                "values": layer.kv_cache_base.index_select(0, ids).detach().clone(),
                "table": table.detach().cpu().clone(),
            }
            if layer.kv_scale_base is not None and layer.kv_scale_base.numel():
                row["scales"] = (
                    layer.kv_scale_base.index_select(0, ids).detach().clone()
                )
            source[key] = row
    _pending.source = source or None
    _pending.runtime = _remember_runtime(ctx.args[0])


def after_forward_decode(ctx):
    global _draft_sequence
    if os.environ.get("T07_CACHE_PROBE_ROLE") != "decode":
        return
    mode = _mode(ctx)
    if mode is None:
        return
    if mode == "target":
        target_probe.after_decode_forward(ctx)
        return
    source = getattr(_pending, "source", None)
    if not source:
        return
    directory = Path(os.environ["T07_CACHE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    cache = ctx.args[1]
    v4 = ctx.args[0]
    product_runtime = _remember_runtime(v4)
    reference_cache = _DraftSnapshotCache(cache, source)
    args = list(ctx.args)
    args[1] = reference_cache
    original = ctx.runtime._patched[ctx.target].original_callable
    was_recording = recorder.ENABLED
    recorder.ENABLED = False
    try:
        _restore_runtime(v4, _pending.runtime)
        with torch.inference_mode():
            direct = original(*args, **ctx.kwargs)
    finally:
        recorder.ENABLED = was_recording
        _restore_runtime(v4, product_runtime)
        _pending.source = None

    product = ctx.result.hidden_states.detach().float()
    expected = direct.hidden_states.detach().float()
    product_logits = torch.mm(
        product.to(v4.head_weight.dtype), v4.head_weight.t()
    ).float()
    expected_logits = torch.mm(
        expected.to(v4.head_weight.dtype), v4.head_weight.t()
    ).float()
    compared = {
        "draft_hidden_maxabs": float((product - expected).abs().max()),
        "draft_logits_maxabs": float((product_logits - expected_logits).abs().max()),
    }
    physical_equal = {}
    for key, row in source.items():
        layer = cache.get_layer_cache(*key)
        independent = reference_cache.get_layer_cache(*key)
        ids = row["ids"]
        physical_equal[str(key)] = bool(
            torch.equal(
                layer.kv_cache_base.index_select(0, ids),
                independent.kv_cache_base.index_select(0, ids),
            )
        )
        if layer.kv_scale_base is not None and layer.kv_scale_base.numel():
            physical_equal[str(key)] &= bool(
                torch.equal(
                    layer.kv_scale_base.index_select(0, ids),
                    independent.kv_scale_base.index_select(0, ids),
                )
            )
    file = directory / f"mtp_draft_decode_{_draft_sequence:03d}.pt"
    torch.save(
        {
            "captured_unix_ns": time.time_ns(),
            "input_ids": ctx.args[3].input_ids.detach().cpu().clone(),
            "source": {
                key: {name: value.cpu() for name, value in row.items()}
                for key, row in source.items()
            },
            "compared": compared,
            "physical_equal": physical_equal,
        },
        file,
    )
    (directory / f"mtp_draft_decode_{_draft_sequence:03d}.json").write_text(
        json.dumps({"compared": compared, "physical_equal": physical_equal}, indent=2)
        + "\n"
    )
    _draft_sequence += 1
    torch.testing.assert_close(product, expected, rtol=0.002, atol=0.002)
    torch.testing.assert_close(product_logits, expected_logits, rtol=0.002, atol=0.002)
    if not all(physical_equal.values()):
        raise AssertionError(f"MTP draft physical cache mismatch: {physical_equal}")
