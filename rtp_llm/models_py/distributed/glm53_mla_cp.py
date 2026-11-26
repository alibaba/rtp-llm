"""Dedicated MLA CP metadata shared by the GLM53 main and MTP models."""

import copy

from rtp_llm.models_py.distributed.sequence_parallel import token_shard_layout
from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.utils.model_weight import W


def prepare_glm53_fmha(model, inputs, is_cuda_graph=False):
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_sparse_mla import (
        SparseMlaImpl,
    )

    groups = get_attention_inputs_value(inputs)
    attn_inputs = select_attention_inputs_for_tag(groups, "full")
    parallelism = model.parallelism_config
    if model.prefill_mla_cp:
        return _prepare_cp_fmha(model, inputs, attn_inputs, groups, is_cuda_graph)
    attn_inputs.is_cuda_graph = is_cuda_graph
    fmha = SparseMlaImpl(
        model.config.getAttentionConfigs(parallelism.get_attn_tp_size()),
        attn_inputs,
        model.weight.weights,
        cos_sin_cache=model.weight.get_global_weight_or_none(W.rope_cos_sin_cache),
        fmha_config=model.fmha_config,
        quant_config=model.config.quant_config,
        max_seq_len=model.config.max_seq_len,
        is_cuda_graph=is_cuda_graph,
        parallelism_config=parallelism,
    )
    fmha.glm53_group_inputs = groups
    return fmha


def _prepare_cp_fmha(model, inputs, attn_inputs, groups, is_cuda_graph):
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_sparse_mla_cp import (
        SparseMlaCpImpl,
    )

    if (
        not attn_inputs.is_prefill
        or attn_inputs.is_target_verify
        or getattr(attn_inputs, "is_draft_extend", False)
        or is_cuda_graph
    ):
        raise ValueError("GLM53 dedicated MLA CP supports ordinary Prefill only")
    q_lens = attn_inputs.input_lengths.cpu().tolist()
    if sum(q_lens) != inputs.input_ids.shape[0]:
        raise ValueError("GLM53 MLA CP input token count disagrees with Q lengths")
    layout = token_shard_layout(
        sum(q_lens), model.parallelism_config.tp_size, model.parallelism_config.tp_rank
    )
    cp_layout = ZigzagTokenLayout(
        q_lens,
        layout,
        model.parallelism_config.tp_size,
        model.parallelism_config.tp_rank,
        inputs.input_ids.device,
    )
    # The C++ executor still supplies complete, canonical token/hidden rows.
    # Only attention sees CP metadata; preserve the original grouped block maps
    # for the rest of the model and typed KV/KPool cache-store writer.
    cp_inputs = copy.copy(attn_inputs)
    cp_inputs.context_parallel_info = cp_layout.context_parallel_info()
    parallelism = model.mla_parallelism
    fmha = SparseMlaCpImpl(
        model.config.getAttentionConfigs(parallelism.get_attn_tp_size()),
        cp_inputs,
        model.weight.weights,
        cos_sin_cache=model.weight.get_global_weight_or_none(W.rope_cos_sin_cache),
        fmha_config=model.fmha_config,
        quant_config=model.config.quant_config,
        max_seq_len=model.config.max_seq_len,
        parallelism_config=parallelism,
    )
    fmha.glm53_cp_layout = cp_layout
    fmha.glm53_group_inputs = groups
    return fmha
