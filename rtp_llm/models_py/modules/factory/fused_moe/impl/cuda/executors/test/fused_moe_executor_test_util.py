import random
from typing import Dict, Tuple

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.utils.model_weight import W


def generate_payload_and_weights(
    config: MoEConfigAdapter,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor]]:
    M = (
        (config.max_generate_batch_size + config.tp_size - 1)
        // config.tp_size
        * config.ep_size
    )
    N = config.model_config.moe_inter_size * 2
    K = config.hidden_size
    top_k = 8
    torch_dtype = torch.bfloat16
    # generate payload
    num_local_experts = config.expert_num // config.ep_size
    expert_x = torch.zeros((num_local_experts, M, K), device="cuda", dtype=torch_dtype)
    expert_topk_ids = (
        torch.ones((num_local_experts, M, top_k), device="cuda", dtype=torch.int) * -1
    )
    recv_topk_weights = torch.ones(
        (num_local_experts, M, top_k), device="cuda", dtype=torch.float32
    )
    expert_num_tokens = torch.zeros(
        (num_local_experts,), device="cuda", dtype=torch.int32
    )
    expert_x_scale = None
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = max(
            min(int(config.expert_num * config.dp_size * random.uniform(0.7, 1.3)), M),
            1,
        )
        expert_topk_ids[local_expert_id, :num_actual_tokens, 0] = local_expert_id
        expert_x[local_expert_id, :num_actual_tokens, :] = (
            torch.rand((num_actual_tokens, K), device="cuda", dtype=torch.float32).to(
                torch_dtype
            )
            * 0.1
            - 0.05
        )
        expert_num_tokens[local_expert_id] = num_actual_tokens
    payload = ExpertForwardPayload(
        expert_x=expert_x,
        expert_x_scale=expert_x_scale,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=expert_topk_ids,
        expert_topk_weights=recv_topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.tolist(),
        ),
    )
    # generate weights
    w1_scale = None
    w2_scale = None
    weights = {
        W.moe_w1: torch.rand(
            (num_local_experts, N, K), device="cuda", dtype=torch.float32
        ).to(torch_dtype)
        * 2
        - 1,
        W.moe_w2: torch.rand(
            (num_local_experts, K, N // 2), device="cuda", dtype=torch.float32
        ).to(torch_dtype)
        * 2
        - 1,
        W.moe_s1: w1_scale,
        W.moe_s2: w2_scale,
    }
    return payload, weights


def generate_ref_output(
    config: MoEConfigAdapter,
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    M = (
        (config.max_generate_batch_size + config.tp_size - 1)
        // config.tp_size
        * config.ep_size
    )
    N = config.model_config.moe_inter_size * 2
    K = config.hidden_size
    num_local_experts = config.expert_num // config.ep_size
    expert_x = payload.expert_x
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    w1 = weights[W.moe_w1]
    w2 = weights[W.moe_w2]
    ref_output = torch.zeros(
        (num_local_experts, M, K), device="cuda", dtype=torch.bfloat16
    )
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = expert_num_tokens[local_expert_id].item()
        expert_x_local = expert_x[local_expert_id, :num_actual_tokens, :]
        w1_local = w1[local_expert_id]
        w2_local = w2[local_expert_id]
        workspace1 = expert_x_local @ w1_local.transpose(0, 1)
        gate = workspace1[..., N // 2 :].to(torch.float32)
        value = workspace1[..., : N // 2].to(torch.float32)
        gate = gate * (1.0 / (1.0 + torch.exp(-gate)))
        workspace2 = (gate * value).to(torch.bfloat16)
        ref_output[local_expert_id, :num_actual_tokens, :] = (
            workspace2 @ w2_local.transpose(0, 1)
        )
    return ref_output
