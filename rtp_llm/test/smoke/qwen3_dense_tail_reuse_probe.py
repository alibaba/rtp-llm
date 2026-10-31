"""Read-only real Qwen3 FULL cache/hidden evidence; never changes model inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from rtp_llm.test.smoke.qwen35_tail_reuse_cp_probe import _PrivateReferenceCache


class _DenseReferenceCache(_PrivateReferenceCache):
    """Reuse the verified 2000-token private pool; fork only its computed data."""

    def fork(self):
        from rtp_llm.ops.compute_ops import LayerKVCache

        result = object.__new__(type(self))
        result._production = self._production
        result.group_tags = list(self.group_tags)
        result.layer_count = self.layer_count
        result.store_isolation = []
        result._layers = {}
        for key, layer in self._layers.items():
            scale = layer.kv_scale_base
            result._layers[key] = LayerKVCache(
                layer.kv_cache_base.clone(),
                layer.seq_size_per_block,
                layer_id=layer.layer_id,
                tag=layer.tag,
                kv_scale_base=scale.clone() if scale is not None else None,
            )
            if (
                result._layers[key].kv_cache_base.data_ptr()
                == layer.kv_cache_base.data_ptr()
            ):
                raise AssertionError(
                    "private branch aliases its independent source pool"
                )
        return result


_seen_ids = {}
_sequence = 0
_active_request = None
_reference_running = False
_reference_source = None


def _capture(ctx, phase):
    global _sequence
    model, inputs = ctx.args[:2]
    cache = model.kv_cache
    attention = select_attention_inputs_for_layer(inputs, cache, 0)
    if not attention.is_prefill:
        return
    cp = attention.context_parallel_info
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    ).reshape(-1)
    prefixes = attention.prefix_lengths.reshape(-1)
    if lengths.numel() != 1 or prefixes.numel() != 1:
        raise AssertionError("dense tail fixture requires one prefill request")
    prefix = int(prefixes[0])
    endpoint = prefix + int(lengths[0])
    positions = select_prefill_position_ids(attention)
    position_source = "explicit_attention_positions"
    if positions is None:
        # Ordinary Qwen3 RoPE derives positions from the one-request prefix
        # and input lengths. CP must retain its actual zigzag position tensor.
        if cp is not None or inputs.input_ids.numel() != int(lengths[0]):
            raise AssertionError("missing or ambiguous actual prefill positions")
        positions = torch.arange(prefix, endpoint, dtype=torch.long)
        position_source = "implicit_noncp_single_request_prefix_plus_token_offset"
    rows = {}
    for layer in range(cache.layer_count):
        attn = select_attention_inputs_for_layer(inputs, cache, layer)
        groups = cache.get_layer_cache_groups(layer)
        if len(groups) != 1:
            raise AssertionError("Qwen3 must have exactly one FULL group per layer")
        physical = groups[0]
        tag = str(physical.tag)
        table = attn.kv_cache_kernel_block_id_device
        ids = table.reshape(-1).to(torch.long)
        ids = ids[ids >= 0]
        key = (layer, tag)
        if key in _seen_ids:
            ids = torch.cat((ids, _seen_ids[key]))
        ids = torch.unique(ids, sorted=True)
        if phase == "after":
            _seen_ids[key] = ids
        rows[key] = {
            "ids": ids.cpu(),
            "values": physical.kv_cache_base.index_select(0, ids)
            .detach()
            .cpu()
            .clone(),
            "runtime_table": table.cpu().clone(),
            "save_positions": (
                attn.cache_save_positions.cpu().clone()
                if attn.cache_save_positions is not None
                else None
            ),
            "save_block_ids": (
                attn.cache_save_block_ids.cpu().clone()
                if attn.cache_save_block_ids is not None
                else None
            ),
            "seq_size_per_block": cache.get_seq_size_per_block(tag),
            "kernel_seq_size_per_block": cache.get_kernel_seq_size_per_block(tag),
        }
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(os.environ["T10_DENSE_PROBE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": rank,
            "phase": phase,
            "prefix": prefix,
            "end_position": endpoint,
            "model_class": type(model).__name__,
            "supports_sparse_cache_save": bool(
                getattr(model, "supports_sparse_cache_save", False)
            ),
            "rows": rows,
            "hidden_states": (
                ctx.result.hidden_states.detach().cpu().clone()
                if phase == "after"
                else None
            ),
            "position_ids": positions.cpu().clone(),
            "position_source": position_source,
        },
        directory / f"prefill_rank{rank}_{phase}_{_sequence:03d}_end{endpoint}.pt",
    )
    _sequence += 1


def before_model_forward(ctx):
    global _active_request
    model, inputs = ctx.args[:2]
    attention = select_attention_inputs_for_layer(inputs, model.kv_cache, 0)
    _active_request = None
    if attention.is_prefill and attention.context_parallel_info is not None:
        cp = attention.context_parallel_info
        prefix = int(attention.prefix_lengths.reshape(-1)[0])
        length = int(cp.prefill_actual_input_lengths_cpu.reshape(-1)[0])
        _active_request = {
            "prefix": prefix,
            "end_position": prefix + length,
            "positions": select_prefill_position_ids(attention).detach().cpu().clone(),
            "sequence": _sequence,
            "attention_records": [],
        }
    _capture(ctx, "before")


def after_model_forward(ctx):
    _capture(ctx, "after")
    _flush_attention_records()
    _run_private_reference(ctx)


def after_cp_attention(ctx):
    """Optional actual attention boundary evidence; no replay or output changes."""
    if _active_request is None or _reference_running:
        return
    op = ctx.args[0]
    qkv = ctx.args[1] if len(ctx.args) > 1 else ctx.kwargs["qkv"]
    cache = ctx.args[2] if len(ctx.args) > 2 else ctx.kwargs["kv_cache"]
    if cache.layer_id not in (0, 1):
        return
    # Clone on device at the boundary; flush only after the whole model.
    _active_request["attention_records"].append(
        {
            "layer": int(cache.layer_id),
            "qkv": qkv.detach().clone(),
            "output": ctx.result.detach().clone(),
            "global_staged_kv": cache.kv_cache_base.detach().clone(),
            "heads_q": int(op.num_qo_heads),
            "heads_kv": int(op.num_kv_heads),
            "head_dim": int(op.head_dim),
            "kernel_tokens": int(op.seq_size_per_block),
            "has_prefix": bool(op.has_prefix),
        }
    )


def _flush_attention_records():
    if _active_request is None or not _active_request["attention_records"]:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(os.environ["T10_DENSE_PROBE_DIR"]).parent / "attention_numeric"
    directory.mkdir(parents=True, exist_ok=True)
    for tensors in _active_request["attention_records"]:
        record = {
            key: value
            for key, value in _active_request.items()
            if key != "attention_records"
        }
        record["rank"] = rank
        record.update(
            {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            }
        )
        torch.save(
            record,
            directory
            / (
                f"rank{rank}_seq{record['sequence']}_G{record['prefix']}_N{record['end_position']}_layer{record['layer']}.pt"
            ),
        )
    _active_request["attention_records"].clear()


def _run_private_reference(ctx):
    """Compute A700 independently, then fork its dense pool for each B suffix."""
    global _reference_source, _reference_running
    from rtp_llm.ops.compute_ops import PyModelInputs

    model, inputs = ctx.args[:2]
    production = model.kv_cache
    attention = select_attention_inputs_for_layer(inputs, production, 0)
    if not attention.is_prefill or attention.context_parallel_info is None:
        return
    cp = attention.context_parallel_info
    prefix = int(attention.prefix_lengths.reshape(-1)[0])
    lengths = (
        cp.prefill_actual_input_lengths_cpu
        if cp is not None
        else attention.input_lengths
    )
    endpoint = prefix + int(lengths.reshape(-1)[0])
    if (prefix, endpoint) == (0, 700):
        if _reference_source is not None:
            raise AssertionError("duplicate independent source")
        private = _DenseReferenceCache(production)
    elif endpoint == 900 and prefix in (256, 650, 700):
        if _reference_source is None:
            raise AssertionError("independent A700 must run before B")
        private = _reference_source.fork()
    else:
        return
    groups = inputs.attention_inputs
    if isinstance(groups, dict):
        private_attention = {
            tag: private.attention_inputs(value, tag) for tag, value in groups.items()
        }
    else:
        if len(private.group_tags) != 1:
            raise AssertionError("single attention input requires one private group")
        private_attention = private.attention_inputs(groups, private.group_tags[0])
    private_inputs = PyModelInputs(
        input_ids=inputs.input_ids,
        embedding_inputs=inputs.embedding_inputs,
        multimodal_inputs=inputs.multimodal_inputs,
        attention_inputs=private_attention,
        bert_embedding_inputs=inputs.bert_embedding_inputs,
    )
    for name in ("input_hiddens", "combo_position_ids", "engram_token_windows"):
        value = getattr(inputs, name)
        if value is not None:
            setattr(private_inputs, name, value)
    for name in ("capture_hidden_states", "need_all_hidden_states", "need_all_logits"):
        setattr(private_inputs, name, getattr(inputs, name))
    original = ctx.runtime._patched[ctx.target].original_callable
    production_hidden = ctx.result.hidden_states.detach().clone()
    try:
        _reference_running = True
        model.kv_cache = private
        with torch.inference_mode():
            output = original(model, private_inputs, fmha_impl=None)
            if output is ctx.result:
                raise AssertionError("reference reused the actual model result object")
            hidden = output.hidden_states.detach().clone()
    finally:
        model.kv_cache = production
        ctx.result.hidden_states = production_hidden
        _reference_running = False
    if endpoint == 700:
        _reference_source = private
    positions = select_prefill_position_ids(attention)
    if positions is None:
        if cp is not None:
            raise AssertionError("CP reference lost actual positions")
        positions = torch.arange(prefix, endpoint)
    if positions.numel() != hidden.shape[0]:
        raise AssertionError("reference hidden rows do not match actual positions")
    if model.kv_cache is not production:
        raise AssertionError("reference failed to restore production cache")
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(os.environ["T10_DENSE_PROBE_DIR"]).parent / "private_reference"
    directory.mkdir(exist_ok=True)
    manifest = json.loads(
        (directory.parent / "checkpoint_config_manifest.json").read_text()
    )
    if int(manifest["config"]["num_hidden_layers"]) != private.layer_count:
        raise AssertionError("reference layer count differs from bound checkpoint")
    torch.save(
        {
            "rank": rank,
            "prefix": prefix,
            "end_position": endpoint,
            "hidden_states": hidden.cpu(),
            "position_ids": positions.cpu().clone(),
            "provenance": "independent whole-model no-store A700 then forked dense prefix",
            "source_endpoint": 700,
            "production_cache_bytes_copied": 0,
            "checkpoint_config_sha256": manifest["config_sha256"],
            "layer_count": private.layer_count,
            "selected_save_slots": 0,
            "cache_store_isolation": private.store_isolation,
        },
        directory / f"rank{rank}_G{prefix}_N{endpoint}.pt",
    )
