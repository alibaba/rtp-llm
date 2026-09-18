"""Prepare K3 model inputs and per-layer contexts before single-round computation."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from types import SimpleNamespace

import torch
from rtp_llm.models_py.distributed.sequence_parallel import (
    SequenceParallelLayout,
    sequence_parallel_layout_from_attention_inputs,
    local_physical_token_view,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
    kda_materialized_block_maps,
    kda_round_state_mapping,
)
from rtp_llm.models_py.modules.kimi_k3.kda import KDAExecutionMode
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
    KimiKDACurrentStateRegistry,
    KimiKDAPrefillMetadata,
    prepare_kimi_kda_prefill_metadata,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext
from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode
from rtp_llm.models_py.modules.kimi_k3.moe import (
    KimiK3LatentMoE,
    validate_mega_moe_topology,
)
from rtp_llm.models_py.modules.kimi_k3.utils import (
    mask_multimodal_token_ids,
    prefill_chunk_tokens,
    resolve_cu_seqlens,
    sequence_offsets,
)


@dataclass(frozen=True)
class KimiK3DecoderMetadata:
    """Request-scoped execution state shared by every decoder layer."""

    cu_seqlens: torch.Tensor
    mode: KDAExecutionMode
    sp_layout: SequenceParallelLayout
    kda_prefill_metadata: Optional[KimiKDAPrefillMetadata] = None
    kda_current_state_registry: Optional[KimiKDACurrentStateRegistry] = None
    valid_token_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class KimiK3ExecutionSpec:
    tp_size: int
    tp_rank: int
    chunk_tokens: int
    sp_type: str
    aux_layers: tuple[int, ...]
    aux_layer_set: frozenset[int]
    kda_layers: tuple[int, ...]

    @classmethod
    def from_model(cls, model):
        sp_type = os.environ.get("SP_TYPE", "").lower()
        raw = os.environ.get("KIMI_K3_EAGLE3_AUX_LAYER_IDS")
        aux = (
            tuple(map(int, raw.split(",")))
            if raw and sp_type == "eagle3"
            else (0, max(0, model.layer_num // 2), model.layer_num - 1)
        )
        if sp_type == "eagle3" and (
            len(aux) != 3 or any(i < 0 or i >= model.layer_num for i in aux)
        ):
            raise ValueError("Eagle3 requires three valid auxiliary layer ids")
        return cls(
            int(model.parallelism_config.get_attn_tp_size()),
            int(model.parallelism_config.get_attn_tp_rank()),
            prefill_chunk_tokens(),
            sp_type,
            aux,
            frozenset(aux),
            tuple(i for i, layer in enumerate(model.layers) if layer.is_kda),
        )


@dataclass(frozen=True)
class KimiK3PreparedRound:
    inputs: Any
    fmha_impl: Any
    embedding_ids: torch.Tensor
    embedding_injections: tuple
    residual_bank: torch.Tensor
    attn_meta: Any
    prefill_sp: bool
    eagle3_enabled: bool
    aux_layers: tuple[int, ...]
    aux_layer_set: frozenset[int]
    writer: Any
    publish_context: Any
    layer_inputs: tuple[Any, ...]
    layer_caches: tuple[Any, ...]
    attention_contexts: tuple[Any, ...]
    moe_contexts: tuple[Any, ...]


def prepare_embedding_ids(input_ids, multimodal_inputs):
    if multimodal_inputs is None or not multimodal_inputs.multimodal_features:
        return input_ids
    features = multimodal_inputs.multimodal_features
    locs = multimodal_inputs.mm_features_locs_host
    if locs is None or locs.numel() != len(features):
        raise ValueError("multimodal feature locations must match feature count")
    if locs.device.type != "cpu":
        raise ValueError("multimodal feature locations require a host mirror")
    return mask_multimodal_token_ids(input_ids, features, locs)


def prepare_embedding_injections(input_ids, multimodal_inputs, hidden_size, dtype):
    if multimodal_inputs is None or not multimodal_inputs.multimodal_features:
        return ()
    injections = []
    locations = multimodal_inputs.mm_features_locs_host.tolist()
    for idx, (feature, loc) in enumerate(
        zip(multimodal_inputs.multimodal_features, locations)
    ):
        if feature is None or feature.numel() == 0:
            continue
        if feature.ndim != 2 or feature.shape[1] != hidden_size:
            raise ValueError(f"feature[{idx}] must have shape [N, {hidden_size}]")
        if feature.dtype != dtype:
            raise TypeError(f"feature[{idx}] dtype does not match embedding dtype")
        if loc < 0:
            feature, loc = feature[-loc:], 0
        if loc + feature.shape[0] > input_ids.numel():
            raise IndexError("multimodal feature extends beyond this token batch")
        if feature.shape[0]:
            injections.append((loc, feature.to(input_ids.device).contiguous()))
    return tuple(injections)


def prepare_round(
    model,
    inputs,
    fmha_impl=None,
    *,
    kda_current_state_registry=None,
    round_plan=None,
    chunk_publish_context=None,
):
    attention_inputs = inputs.attention_inputs
    if attention_inputs is None:
        raise ValueError("Kimi K3 requires PyAttentionInputs")
    if not attention_inputs.is_prefill and model.kv_cache is None:
        raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
    input_ids = inputs.input_ids.reshape(-1)
    tp_size = model.execution_spec.tp_size
    tp_rank = model.execution_spec.tp_rank
    layout = sequence_parallel_layout_from_attention_inputs(
        attention_inputs, physical_tokens=int(input_ids.numel()),
        world_size=tp_size, rank=tp_rank,
    )
    prefill_sp = layout.mode == "prefill" and tp_size > 1
    mode: KDAExecutionMode = "prefill" if layout.mode == "prefill" else "decode"
    if not attention_inputs.is_prefill and not getattr(model, "_decode_sp_startup_logged", False):
        logging.info(
            "[K3_PARALLEL_MODE] rank=%d mode=%s tokens=%d tp=%d ktp=%d ep=%d",
            tp_rank, model.parallel_mode.value, input_ids.numel(), tp_size,
            int(getattr(model.parallelism_config, "ktp_size", 1)),
            int(model.parallelism_config.ep_size),
        )
        model._decode_sp_startup_logged = True
    cu_seqlens = resolve_cu_seqlens(attention_inputs, input_ids)
    if model.parallel_mode is KimiK3ParallelMode.TP_SP and tp_size > 1:
        # The token layout above belongs to this request's TP group. Expert
        # dispatch spans EP, which may contain several independent DP groups.
        parallelism = model.parallelism_config
        validate_mega_moe_topology(
            attention_tp_size=tp_size,
            dp_size=int(parallelism.dp_size),
            ktp_size=int(getattr(parallelism, "ktp_size", 1)),
            ep_size=int(parallelism.ep_size),
            world_size=int(parallelism.world_size),
            label="Kimi K3 Sequence Parallel",
        )
    if model._layer_group_ids is None:
        layer_map_host = getattr(attention_inputs, "kv_cache_layer_to_group_host", None)
        if layer_map_host is not None and layer_map_host.numel():
            model._layer_group_ids = tuple(
                int(value) for value in layer_map_host.tolist()
            )
    kda_prefill_metadata: Optional[KimiKDAPrefillMetadata] = None
    if mode == "prefill" and model.kv_cache is not None:
        cu_host = getattr(attention_inputs, "cu_seqlens_host", None)
        lengths_host = getattr(attention_inputs, "input_lengths_host", None)
        prefixes_host = getattr(attention_inputs, "prefix_lengths_host", None)
        if (
            cu_host is None
            or not cu_host.numel()
            or lengths_host is None
            or not lengths_host.numel()
            or prefixes_host is None
            or not prefixes_host.numel()
        ):
            raise RuntimeError(
                "cache-backed K3 Prefill requires host sequence metadata"
            )
        checkpoint_tokens = model._kda_checkpoint_tokens
        if checkpoint_tokens is None:
            raise RuntimeError("Kimi K3 cache geometry is not initialized")
        materialized_maps = kda_materialized_block_maps(
            attention_inputs,
            layer_group_ids=model._layer_group_ids,
            kda_layer_indices=model.execution_spec.kda_layers,
        )
        sequence_count = int(lengths_host.numel())
        real_count = len(round_plan.slices) if round_plan is not None else sequence_count
        padding_requests = sequence_count - real_count
        if padding_requests not in (0, 1):
            raise RuntimeError("K3 chunk supports at most one TP dummy request")
        padding_index = (
            kda_current_state_registry.original_batch_size - 1
            if padding_requests and kda_current_state_registry is not None else None
        )
        active_indices, continuation_mask = kda_round_state_mapping(
            round_plan, padding_original_batch_idx=padding_index,
        )
        kda_prefill_metadata = prepare_kimi_kda_prefill_metadata(
            cu_host,
            lengths_host,
            prefixes_host,
            checkpoint_tokens=checkpoint_tokens,
            local_heads=model._kda_local_heads,
            head_dim=model._kda_head_dim,
            device=input_ids.device,
            active_original_batch_indices=active_indices,
            continuation_mask=continuation_mask,
            materialized_block_maps_host=materialized_maps,
        )
    valid_token_mask = getattr(inputs, "ktp_valid_row_mask", None)
    if valid_token_mask is None or not valid_token_mask.numel():
        valid_token_mask = None
    else:
        valid_token_mask = local_physical_token_view(valid_token_mask, layout)
    attn_meta = KimiK3DecoderMetadata(
        cu_seqlens=cu_seqlens,
        mode=mode,
        sp_layout=layout,
        kda_prefill_metadata=kda_prefill_metadata,
        kda_current_state_registry=kda_current_state_registry,
        valid_token_mask=valid_token_mask,
    )
    write_cache_store_impl = create_write_cache_store_impl(
        attention_inputs, model.kv_cache
    )
    spec = model.execution_spec
    eagle3_enabled = spec.sp_type == "eagle3" and not inputs.force_disable_sp_run
    aux_layers = spec.aux_layers if eagle3_enabled else ()
    aux_layer_set = spec.aux_layer_set if eagle3_enabled else frozenset()
    moe_tokens = layout.tokens.local_tokens
    valid_tokens = (
        layout.tokens.local_valid_tokens
        if layout.tokens.local_valid_tokens < moe_tokens else None
    )
    moe_contexts = []
    moe_context_by_topology = {}
    for layer in model.layers:
        mlp = layer.mlp
        context = None
        if isinstance(mlp, KimiK3LatentMoE):
            key = (mlp.attn_tp_size, mlp.attn_tp_rank)
            if key not in moe_context_by_topology:
                moe_context_by_topology[key] = mlp.prepare_context(
                    moe_tokens, valid_tokens
                )
            context = moe_context_by_topology[key]
        moe_contexts.append(context)
    groups = {}
    layer_inputs, layer_caches, attention_contexts = [], [], []
    for idx, layer in enumerate(model.layers):
        gid = model._layer_group_ids[idx] if model._layer_group_ids is not None else 0
        if gid not in groups:
            view = copy.copy(attention_inputs)
            select_block_map_for_layer(view, idx, gid)
            groups[gid] = view
        view = groups[gid]
        bound_caches = getattr(model, "_bound_layer_caches", None)
        cache = (
            bound_caches[idx]
            if bound_caches is not None
            else (
                model.kv_cache.get_layer_cache(idx)
                if model.kv_cache is not None
                else None
            )
        )
        layer_inputs.append(view)
        layer_caches.append(cache)
        if layer.is_kda:
            context = (
                layer.self_attn.decode_executor.prepare_context(
                    input_ids.numel(), input_ids.device, cache, view
                )
                if mode == "decode" and cache is not None
                else None
            )
            if cache is not None and bound_caches is None:
                layer.prepare_kda_cache_store(cache)
        else:
            context = KimiK3MLAContext(layout)
        attention_contexts.append(context)
    rows = layout.tokens.local_tokens
    hidden_size = int(model.config.hidden_size)
    dtype = model.embedding_weight.dtype
    residual_bank = (
        model._ensure_prefill_static_attn_res_bank(
            rows=rows, hidden_size=hidden_size, device=input_ids.device, dtype=dtype
        )
        if prefill_sp
        else torch.empty(
            (rows, model.num_attn_res_blocks, hidden_size),
            device=input_ids.device,
            dtype=dtype,
        )
    )
    embedding_ids = prepare_embedding_ids(input_ids, inputs.multimodal_inputs)
    return KimiK3PreparedRound(
        inputs=inputs,
        fmha_impl=fmha_impl,
        embedding_ids=embedding_ids,
        embedding_injections=prepare_embedding_injections(
            input_ids, inputs.multimodal_inputs, hidden_size, dtype
        ),
        residual_bank=residual_bank,
        attn_meta=attn_meta,
        prefill_sp=prefill_sp,
        eagle3_enabled=eagle3_enabled,
        aux_layers=aux_layers,
        aux_layer_set=aux_layer_set,
        writer=write_cache_store_impl,
        publish_context=chunk_publish_context,
        layer_inputs=tuple(layer_inputs),
        layer_caches=tuple(layer_caches),
        attention_contexts=tuple(attention_contexts),
        moe_contexts=tuple(moe_contexts),
    )


@dataclass(frozen=True)
class KimiK3PreparedDraft:
    inputs: Any
    fmha_impl: Any
    embedding_ids: torch.Tensor
    embedding_injections: tuple
    layer_cache: Any


def prepare_draft_round(model, inputs, fmha_impl):
    """Prepare draft-owned metadata without retaining target request state."""
    input_ids = inputs.input_ids
    multimodal = inputs.multimodal_inputs
    embedding_ids, injections = input_ids, ()
    if multimodal is not None and multimodal.multimodal_features:
        locations = multimodal.mm_features_locs_host
        features = multimodal.multimodal_features
        if (
            locations is None
            or locations.device.type != "cpu"
            or locations.numel() != len(features)
        ):
            raise ValueError(
                "Eagle3 multimodal locations require matching host metadata"
            )
        ranges = sequence_offsets(
            inputs.attention_inputs.cu_seqlens,
            input_ids.numel(),
            cu_seqlens_host=inputs.attention_inputs.cu_seqlens_host,
        )
        shifted_features, shifted_locations = [], []
        for feature, location in zip(features, locations.tolist()):
            start = max(s for s, _ in ranges if s <= location + feature.size(0) - 1)
            dropped = max(0, start - location + 1)
            shifted_features.append(feature[dropped:])
            shifted_locations.append(max(location - 1, start))
        shifted = SimpleNamespace(
            multimodal_features=shifted_features,
            mm_features_locs_host=torch.tensor(shifted_locations, dtype=torch.int32),
        )
        embedding_ids = prepare_embedding_ids(input_ids, shifted)
        injections = prepare_embedding_injections(
            input_ids, shifted, model.hidden_size, model.embedding_dtype
        )
    if inputs.attention_inputs is not None:
        select_block_map_for_layer(
            inputs.attention_inputs, 0, getattr(model, "_draft_cache_group", None)
        )
    return KimiK3PreparedDraft(
        inputs,
        fmha_impl,
        embedding_ids,
        injections,
        getattr(model, "_draft_layer_cache", None),
    )
