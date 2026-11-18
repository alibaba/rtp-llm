"""RTP-only adapter for cache-manager-owned GDN state and replay records.

Installed as atom.plugin.rtpllm.utils.linear_double_buffer by atom.patch.
The public helper owns the transaction and kernels; this file only connects
ATOM's cache/context/runtime hooks to that contract.
"""

from types import MethodType

import torch
import triton
from atom.config import KVCacheTensor

from rtp_llm.models_py.modules.qwen4_exp.linear_double_buffer import (
    LinearDoubleBufferStage,
    LinearDoubleBufferTransaction,
    resolve_pool_indices,
    split_pool_views,
)


def enabled(runtime):
    cache = runtime.kv_cache
    cached = getattr(runtime, "_linear_db_layout_cache", None)
    if cached is not None and cached[0] is cache:
        return cached[1]
    # MTP compatibility topologies include unbound target groups. Only actual
    # GDN layers may enable this adapter; the full-attention-only draft has none.
    # Published topology is immutable, so recheck only when the cache is rebound.
    layers = runtime._rtp_layer_maps[0]
    if not layers:
        runtime._linear_db_layout_cache = (cache, False)
        return False
    tags = set(getattr(cache, "group_tags", ()))
    present = tags.intersection(("linear_ssm", "linear_replay"))
    if present and present != {"linear_ssm", "linear_replay"}:
        raise ValueError("Incomplete linear SSM/replay split cache layout")
    if present:
        required = {"linear", "linear_ssm", "linear_replay"}
        for layer in layers:
            bound = {str(view.tag) for view in cache.get_layer_cache_groups(layer)}
            if not required.issubset(bound):
                raise ValueError(
                    f"Incomplete linear SSM/replay split bindings for layer {layer}"
                )
        # Graph tables can be padded to a common width. Read the immutable
        # published topology instead of inferring the number of state slots.
        layout = cache.get_linear_state_layout("linear_ssm")
        slots = cache.get_request_state_slots("linear_ssm")
        if (layout, slots) not in ((3, 2), (5, 1)):
            raise ValueError("Unsupported linear SSM layout/slot contract")
        if cache.get_request_state_slots("linear_replay") != 1:
            raise ValueError("Linear replay requires one request-owned record row")
        runtime._linear_db_state_slots = slots
    runtime._linear_db_layout_cache = (cache, bool(present))
    return bool(present)


def initialize_runtime(runtime):
    # Native checks irreversible-commit capability before the first target
    # forward. Graph warmup must not be required to install this contract.
    if enabled(runtime):
        _install(runtime)


def pool_signature(runtime, layers):
    if not enabled(runtime):
        return ()
    return tuple(
        (layer, tag, raw.data_ptr(), raw.shape, raw.stride(), raw.dtype)
        for layer in sorted(layers)
        for tag in ("linear_ssm", "linear_replay")
        for raw in (runtime.kv_cache.get_layer_cache(layer, tag).kv_cache_base,)
    )


def _clear(self):
    self._linear_db_transaction.clear()


def _save(self, key):
    self._linear_db_transaction.save(key)


def _activate(self, key):
    self._linear_db_transaction.activate(key)


def _prepare(self, accept):
    self._linear_db_transaction.prepare(accept)


def _finish(self, commit):
    self._linear_db_transaction.finish(commit)


def _finalize(self):
    self._linear_db_transaction.finalize()


def _requires_fail_stop(self):
    return self._linear_db_transaction.in_place


def _install(runtime):
    in_place = runtime._linear_db_state_slots == 1
    if hasattr(runtime, "_linear_db_transaction"):
        if runtime._linear_db_transaction.in_place is not in_place:
            raise RuntimeError("Cannot change an initialized split-state commit mode")
        return
    config = runtime._rtp_model_config
    role = str(getattr(runtime.parallelism_config, "role_type", "")).rsplit(".", 1)[-1]
    fusion = role == "PDFUSION"
    if (
        role not in ("DECODE", "PDFUSION")
        or not getattr(config, "atom_qwen4_exp_mtp_target", False)
        or getattr(config, "is_mtp", False)
        or (not fusion and getattr(config, "atom_qwen4_exp_prefix_cache_enabled", False))
        or not torch.version.hip
        or not getattr(runtime, "_hybrid_enabled", False)
        or getattr(runtime, "_hybrid_reducer", None) is None
    ):
        raise ValueError(
            "Split GDN state requires fusion or D without reuse, chain MTP, and FP32 hybrid/TP consensus"
        )
    runtime._linear_db_fusion = fusion
    runtime._hybrid_enabled = False
    runtime._linear_db_transaction = LinearDoubleBufferTransaction(in_place=in_place)
    runtime._linear_db_indices = {}
    for name, function in (
        ("clear_speculative_target_graph_capture", _clear),
        ("save_speculative_target_graph_capture", _save),
        ("activate_speculative_target_graph_replay", _activate),
        ("prepare_speculative_target_commit", _prepare),
        ("finish_speculative_target_commit", _finish),
        ("finalize_speculative_target_commit", _finalize),
        ("requires_speculative_target_commit_fail_stop", _requires_fail_stop),
    ):
        setattr(runtime, name, MethodType(function, runtime))


def build_cache_views(runtime, layer_maps, base_builder):
    _install(runtime)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Split cache views and pointer tables must be warmed before capture"
        )
    transaction = runtime._linear_db_transaction
    if transaction.pending is not None or transaction.graphs:
        raise RuntimeError(
            "Cannot rebind SSM pools with a pending transaction or captured graph"
        )
    # The parent's LINEAR parser assumes combined conv+SSM rows. Exclude GDN
    # layers from it, while retaining the unmodified full-attention bridge.
    gdn, full, mla = layer_maps
    caches = base_builder(runtime, ({}, full, mla))
    query = int(runtime._rtp_model_config.gen_num_per_cycle) + 1
    states, records, checkpoints, keys = [], [], [], tuple(sorted(gdn))
    for number in keys:
        layer = gdn[number]
        linear = runtime.kv_cache.get_layer_cache(number, "linear").kv_cache_base
        if runtime._linear_db_fusion:
            heads, kd, vd = int(layer.dt_bias.numel()), layer.head_k_dim, layer.head_v_dim
            width = heads * kd * vd
            linear = linear.reshape(linear.shape[0], -1)
            checkpoint = linear.view(torch.float32)[:, :width]
            checkpoints.append(checkpoint.view(-1, heads, vd, kd).transpose(-1, -2))
            linear = linear[:, 2 * width:]
        conv, state, record = split_pool_views(
            linear,
            runtime.kv_cache.get_layer_cache(number, "linear_ssm").kv_cache_base,
            runtime.kv_cache.get_layer_cache(number, "linear_replay").kv_cache_base,
            heads=int(layer.dt_bias.numel()),
            key_dim=int(layer.head_k_dim),
            value_dim=int(layer.head_v_dim),
            conv_dim=int(layer.conv1d.weight.size(0)),
            conv_length=int(layer.conv1d.weight.size(2)) - 1,
            query=query,
            combined_linear=runtime._linear_db_fusion,
        )
        if states and (
            state.shape != states[0].shape or record.shape != records[0].shape
        ):
            raise ValueError(
                "Split hybrid layers require identical SSM/replay geometry"
            )
        caches[f"layer_{number}"] = KVCacheTensor(
            layer_num=number,
            k_cache=conv,
            v_cache=state,
            k_scale=None,
            v_scale=None,
            per_request_state=True,
        )
        states.append(state)
        records.append(record)
    if not keys:
        raise ValueError("Split hybrid layout has no GDN layers")
    # Only small pointer/index tables are runtime allocations. All large
    # FP32 state and record storage above is a view of manager-owned pools.
    runtime._linear_db_pools = (
        tuple(states),
        tuple(records),
        torch.tensor(
            [state.data_ptr() for state in states],
            dtype=torch.int64,
            device=states[0].device,
        ),
        torch.tensor(
            [record.data_ptr() for record in records],
            dtype=torch.int64,
            device=states[0].device,
        ),
    )
    runtime._linear_db_records = dict(zip(keys, records))
    if checkpoints:
        runtime._linear_db_checkpoints = dict(zip(keys, checkpoints))
        runtime._linear_db_checkpoint_ptrs = torch.tensor(
            [state.data_ptr() for state in checkpoints], dtype=torch.int64, device=states[0].device
        )
    return caches


def prepare_metadata(runtime, metadata, inputs):
    if metadata is None:
        raise ValueError("Split SSM layout requires GDN metadata")
    if metadata.num_prefills:
        if not runtime._linear_db_fusion or metadata.num_decodes or metadata.num_spec_decodes:
            raise ValueError("Split SSM prefill requires a pure fusion prefill batch")
        state_inputs = inputs.attention_inputs_by_tag["linear_ssm"]
        slots = state_inputs.kv_cache_block_id_device[:, 0].to(torch.int64)
        metadata.rtp_linear_prefill_slots = slots
        attn = inputs.attention_inputs
        prefixes, lengths = attn.prefix_lengths, attn.input_lengths
        page_size = int(runtime.kv_cache.get_seq_size_per_block("linear"))
        # Partial pages cannot be matched or used to resume a prefill chunk.
        # If the batch has any complete tails, use the existing batched copy
        # to avoid per-row gather and scatter launches in every GDN layer.
        metadata.rtp_linear_prefill_save_tail = (
            prefixes.device.type != "cpu"
            or lengths.device.type != "cpu"
            or any((prefix + length) % page_size == 0
                   for prefix, length in zip(prefixes.tolist(), lengths.tolist()))
        )
        metadata.rtp_linear_prefill_checkpoints = runtime._linear_db_checkpoints
        runtime._linear_db_transaction.clear()
        return
    spec = metadata.num_spec_decodes > 0
    if spec and (
        metadata.num_decodes or not getattr(metadata, "rtp_flydsl_mtp", False)
    ):
        raise ValueError("Split SSM layout requires pure FlyDSL chain verification")
    if spec and getattr(metadata, "rtp_target_verify_block_map", None) is None:
        raise ValueError("Split SSM layout requires RTP chain target-verify metadata")
    count = metadata.num_spec_decodes if spec else metadata.num_decodes
    query = metadata.spec_state_indices_tensor.shape[1] if spec else 1
    if count <= 0 or metadata.num_actual_tokens != count * query:
        raise ValueError("Split SSM metadata batch/token shape is inconsistent")
    states, records, state_ptrs, record_ptrs = runtime._linear_db_pools
    if spec and query != records[0].shape[1]:
        raise ValueError("Verify width differs from the allocated replay-row width")
    tagged = getattr(inputs, "attention_inputs_by_tag", None)
    if not tagged or not {"linear_ssm", "linear_replay"}.issubset(tagged):
        raise ValueError("Split SSM/replay device block tables are missing")
    tables = [
        getattr(tagged[tag], "kv_cache_block_id_device", None)
        for tag in ("linear_ssm", "linear_replay")
    ]
    if any(table is None for table in tables):
        raise ValueError("Split SSM/replay requires refreshed device block tables")
    columns_by_group = (runtime._linear_db_state_slots, 1)
    if any(
        table.ndim != 2 or table.shape[1] < columns
        for table, columns in zip(tables, columns_by_group)
    ):
        raise ValueError(
            "Split SSM/replay block table is narrower than its fixed slots"
        )
    # Framework graph buffers may pad every group to a common page width.
    # These are zero-copy views, retaining the original per-row stride.
    tables = [table[:, :columns] for table, columns in zip(tables, columns_by_group)]
    primary = inputs.attention_inputs
    # Plain decode deliberately has no prefix_lengths tensor. Its padded
    # sequence length is zero, hence sequence_lengths_plus_1 == 1. Target
    # verify instead uses prefix == 0 as its padding marker.
    length_field = "prefix_lengths_device" if spec else "sequence_lengths_plus_1_device"
    prefix = getattr(primary, length_field, None)
    if prefix is None:
        raise ValueError(f"Split SSM requires {length_field} to identify graph padding")
    key = (count, query)
    if key not in runtime._linear_db_indices:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Double-buffer index metadata was not warmed for this graph bucket"
            )
        device = states[0].device
        runtime._linear_db_indices[key] = (
            torch.empty(count, device=device, dtype=torch.int32),
            torch.empty((count, query), device=device, dtype=torch.int32),
            torch.empty(count, device=device, dtype=torch.int32),
            torch.empty(triton.cdiv(count, 128), device=device, dtype=torch.int32),
        )
    buffers = runtime._linear_db_indices[key]
    resolve_pool_indices(
        *tables,
        prefix,
        buffers,
        states[0].shape[0],
        records[0].shape[0],
        minimum_length=0 if spec else 1,
        state_slots=runtime._linear_db_state_slots,
    )
    reads, writes, record_ids, _ = buffers
    if spec:
        metadata.rtp_flydsl_read_indices = reads
        metadata.rtp_flydsl_ssm_write_indices = writes
        metadata.rtp_hybrid_records = runtime._linear_db_records
        metadata.rtp_hybrid_record_indices = record_ids
        metadata.rtp_hybrid_record_only = runtime._linear_db_transaction.in_place
        checkpoint = None
        if runtime._linear_db_fusion and runtime._rtp_model_config.atom_qwen4_exp_prefix_cache_enabled:
            # Decode retains the most recent LINEAR pages for prefix reuse.
            # Every accepted page boundary needs a recurrent state even when
            # it is not a periodic checkpoint retained for the full request.
            paged = next(iter(runtime._linear_db_checkpoints.values()))
            checkpoint = (
                runtime._linear_db_checkpoint_ptrs, metadata.spec_state_indices_tensor,
                metadata.rtp_target_verify_sequence_lengths,
                metadata.rtp_seq_size_per_block,
                paged.stride(0),
            )
        runtime._linear_db_transaction.bind(
            LinearDoubleBufferStage(
                states,
                records,
                state_ptrs,
                record_ptrs,
                reads,
                writes,
                record_ids,
                in_place=runtime._linear_db_transaction.in_place,
                checkpoint=checkpoint,
            )
        )
    else:
        # A non-speculative single-token decode is immediately accepted. Use
        # committed in-place after conv has consumed its independent indices.
        # Never use this route for speculative candidates or prefill.
        metadata.rtp_ssm_non_spec_indices = reads
        runtime._linear_db_transaction.clear()
