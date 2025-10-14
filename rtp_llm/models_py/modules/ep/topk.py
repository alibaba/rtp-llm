# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py
# Licensed under the Apache License, Version 2.0

import logging
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

import rtp_llm.models_py.modules.utils as utils
from rtp_llm.models_py.modules.ep.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)

if utils.is_cuda():
    from librtp_compute_ops.rtp_llm_ops import moe_topk_softmax
else:
    logging.warning("can't import from rtp_llm_ops, only support cuda!")


def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    moe_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    if torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    elif custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk"
        # Qwen3MOE uses fused_topk
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )

    return topk_weights, topk_ids
