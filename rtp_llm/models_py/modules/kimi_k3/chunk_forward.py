"""Eager whole-model chunk execution with completion-only cache publication."""

import logging
from math import lcm

import torch

from rtp_llm.models_py.model_desc.block_map import (
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
)
from rtp_llm.models_py.modules.kimi_k3.chunk_inputs import build_chunk_inputs
from rtp_llm.models_py.modules.kimi_k3.chunk_plan import plan_kimi_k3_chunk_rounds
from rtp_llm.ops.compute_ops import PyModelOutputs


def forward_prefill_chunks(model, inputs):
    primary = get_primary_attention_inputs(inputs, model.kv_cache)
    if model.kv_cache is None:
        raise ValueError("K3 chunk prefill requires paged cache state")
    if primary.is_cuda_graph or primary.is_target_verify or primary.is_mtp_draft_update:
        raise ValueError("K3 chunk prefill requires eager context-only input")
    if primary.context_parallel_info is not None:
        raise ValueError("K3 chunk prefill does not support CP")
    multimedia = inputs.multimodal_inputs
    if multimedia is not None and (
        multimedia.multimodal_features or multimedia.mm_extra_input
    ):
        raise ValueError("K3 chunk migration supports text only")
    count = primary.logical_request_count or primary.input_lengths.numel()
    lengths = primary.input_lengths[:count].tolist()
    prefixes = primary.prefix_lengths[:count].tolist()
    # Use the actual ordinary cache groups. TP affects only physical padding.
    alignment = lcm(
        *(
            model.kv_cache.get_layer_cache(i).seq_size_per_block
            for i in range(model.layer_num)
        )
    )
    plans = plan_kimi_k3_chunk_rounds(
        lengths,
        prefixes,
        chunk_budget=model.chunk_prefill_budget,
        alignment_tokens=alignment,
    )
    logits_input, recurrent = None, None
    physical = inputs.input_ids.shape[0]
    for index, plan in enumerate(plans):
        chunk_inputs = build_chunk_inputs(inputs, plan, model.tp_size)
        output = model._forward_single(chunk_inputs, None)
        features = output.mtp_target_hidden_states
        if features is None or features.shape[0] != chunk_inputs.input_ids.shape[0]:
            raise RuntimeError(
                "K3 chunk requires explicit per-token recurrent features"
            )
        if logits_input is None:
            logits_input = output.hidden_states.new_zeros(
                (physical, *output.hidden_states.shape[1:])
            )
            recurrent = features.new_zeros((physical, *features.shape[1:]))
        offset = 0
        for part in plan.slices:
            logits_input.narrow(0, part.source_start, part.new_length).copy_(
                output.hidden_states.narrow(0, offset, part.new_length)
            )
            recurrent.narrow(0, part.source_start, part.new_length).copy_(
                features.narrow(0, offset, part.new_length)
            )
            offset += part.new_length
        logging.info(
            "K3 chunk role=%s round=%d/%d logical_tokens=%d physical_tokens=%d requests=%d alignment=%d",
            type(model).__name__,
            index + 1,
            len(plans),
            plan.token_count,
            chunk_inputs.input_ids.numel(),
            len(plan.slices),
            alignment,
        )
        del output, features, chunk_inputs
    # Round metadata has no writer. Publish the original group descriptors only
    # after all model layers and rounds succeeded, using main's normal completion.
    for index in range(model.layer_num):
        attention = select_attention_inputs_for_layer(inputs, model.kv_cache, index)
        if (
            attention.cache_store_inputs is not None
            and attention.cache_store_writer is not None
        ):
            attention.cache_store_writer.write(
                attention.cache_store_inputs, model.kv_cache.get_layer_cache(index)
            )
    return PyModelOutputs(logits_input, recurrent)
