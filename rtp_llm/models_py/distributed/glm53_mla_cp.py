"""Dedicated MLA CP metadata shared by the GLM53 main and MTP models."""

import copy

from rtp_llm.models_py.distributed.sequence_parallel import token_shard_layout
from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout


def prepare_mla_cp_fmha(model, inputs, is_cuda_graph=False):
    from rtp_llm.models_py.modules import AttnImplFactory

    attn_inputs = inputs.attention_inputs
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
    fmha = AttnImplFactory.get_fmha_impl(
        model.config,
        model.mla_parallelism,
        model.weight,
        cp_inputs,
        model.fmha_config,
        is_cuda_graph,
        pinned_mla=bool(model.pinned_mla_groups),
    )
    fmha.glm53_cp_layout = cp_layout
    return fmha
