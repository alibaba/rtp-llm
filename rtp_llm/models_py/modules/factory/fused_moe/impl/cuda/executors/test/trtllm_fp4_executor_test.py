import random
from typing import Dict, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.utils.model_weight import W

from flashinfer import (
    fp4_quantize,
    e2m1_and_ufp8sf_scale_to_float,
)
from flashinfer.fused_moe import (
    RoutingMethodType,
    GatedActType,
    trtllm_fp4_block_scale_moe,
)
from flashinfer.utils import (
    device_support_pdl,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
    TrtllmFp4Executor,
)

DP_SIZE = 1
TP_SIZE = 1
EP_SIZE = 1
NUM_EXPERTS = 128
SEQ_LEN = 908
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOP_K = 8

NVFP4_BLOCK_SIZE = 16

def routing_reference(expertLogits, topK, padding):
    """Reference routing implementation for permutation calculation."""
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }

def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores

def _generate_config() -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.expert_num = NUM_EXPERTS
    model_config.hidden_size = HIDDEN_SIZE
    model_config.moe_inter_size = MOE_INTERMEDIATE_SIZE
    model_config.moe_k = TOP_K
    parallelism_config = ParallelismConfig()
    parallelism_config.dp_size = DP_SIZE
    parallelism_config.tp_size = TP_SIZE
    parallelism_config.ep_size = EP_SIZE
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
    )

def _generate_payload_and_weights(
    config: MoEConfigAdapter,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    hidden_states = torch.empty(
        (SEQ_LEN, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device='cuda:0',
    ).normal_(-0.003, 0.15).clamp_(-2.9, 2.2)
    routing_logits = torch.empty(
        (SEQ_LEN, config.expert_num),
        dtype=torch.bfloat16,
        device='cuda:0',
    ).normal_(-4.8, 0.86).clamp_(-9.6, -1.3)
    w13_input_scale = torch.empty(
        (config.expert_num,),
        dtype=torch.float32,
        device='cuda:0',
    ).fill_(0.0014)
    w13 = torch.empty(
        (config.expert_num, config.model_config.moe_inter_size * 2, config.hidden_size // 2),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(130, 72).clamp_(0, 255).round().to(torch.uint8)
    w13_scale = torch.empty(
        (config.expert_num, config.model_config.moe_inter_size * 2, config.hidden_size // NVFP4_BLOCK_SIZE),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(76.1, 36.3).clamp_(0.1, 448.0).to(torch.float8_e4m3fn)
    w13_scale_2 = torch.empty(
        (config.expert_num,),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(9e-05, 3.6e-05).clamp_(4.8e-05, 0.0002)
    w2_input_scale = torch.empty(
        (config.expert_num,),
        dtype=torch.float32,
        device='cuda:0',
    ).fill_(0.0028)
    w2 = torch.empty(
        (config.expert_num, config.hidden_size, config.model_config.moe_inter_size // 2),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(130, 72).clamp_(0, 255).round().to(torch.uint8)
    w2_scale = torch.empty(
        (config.expert_num, config.hidden_size, config.model_config.moe_inter_size // NVFP4_BLOCK_SIZE),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(53.2, 20.7).clamp_(4.5, 448.0).to(torch.float8_e4m3fn)
    w2_scale_2 = torch.empty(
        (config.expert_num,),
        dtype=torch.float32,
        device='cuda:0',
    ).normal_(0.0001, 4.2e-05).clamp_(8.5e-05, 0.0003)

    permute_info, topk_weights = routing_reference_renormalize(
        routing_logits, config.moe_k, config.expert_num, 8
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    topk_weights = topk_weights.view(SEQ_LEN, config.expert_num)[
        torch.arange(SEQ_LEN).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    payload = ExpertForwardPayload(
        expert_x=hidden_states,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
    )
    weights = {
        W.moe_w1: w13,
        W.moe_w2: w2,
        W.moe_s1: w13_scale,
        W.moe_s2: w2_scale,
        "w13_input_scale": w13_input_scale,
        "w13_weight_scale_2": w13_scale_2,
        "w2_input_scale": w2_input_scale,
        "w2_weight_scale_2": w2_scale_2,
    }
    extra_kwargs = {
        "routing_logits": routing_logits,
    }
    return payload, weights, extra_kwargs

def _generate_ref_output(
    config: MoEConfigAdapter,
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
    extra_kwargs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    # g1_alphas = weights["w13_input_scale"] * weights["w13_weight_scale_2"]
    # g2_alphas = weights["w2_input_scale"] * weights["w2_weight_scale_2"]
    # g1_scale_c = g1_alphas / weights["w2_input_scale"]
    # hidden_states, hidden_states_scale = fp4_quantize(
    #     payload.expert_x, 1 / weights["w13_input_scale"], is_sf_swizzled_layout=False)
    # ref_output = trtllm_fp4_block_scale_moe(
    #     extra_kwargs["routing_logits"],
    #     None,  # routing_bias
    #     hidden_states,
    #     hidden_states_scale.view(torch.float8_e4m3fn),
    #     weights[W.moe_w1],
    #     weights[W.moe_s1],
    #     None,  # w13_bias
    #     None,  # gemm1_alpha
    #     None,  # gemm1_beta
    #     None,  # gemm1_clamp_limit
    #     weights[W.moe_w2],
    #     weights[W.moe_s2],
    #     None,  # w2_bias
    #     g1_scale_c,
    #     g1_alphas,
    #     g2_alphas,
    #     config.expert_num,
    #     config.moe_k,
    #     None,  # n_group
    #     None,  # topk_group
    #     config.model_config.moe_inter_size,
    #     0,  # local_expert_offset
    #     config.expert_num,
    #     None,  # routed_scaling_factor
    #     None,  # tile_tokens_dim
    #     RoutingMethodType.Renormalize.value,
    #     True,  # do_finalize
    #     device_support_pdl(payload.expert_x.device),
    #     GatedActType.SwiGlu.value,  # gated_act_type
    #     None,
    # )[0]

    # return ref_output

    hidden_states = payload.expert_x
    topk_ids = payload.expert_topk_ids
    topk_weights = payload.expert_topk_weights

    device = hidden_states.device
    dtype = hidden_states.dtype

    w13_global_scale = weights["w13_weight_scale_2"]
    w13_float_list = []
    for expert_id in range(config.expert_num):
        expert_w13 = weights[W.moe_w1][expert_id]
        expert_w13_scale = weights[W.moe_s1][expert_id]
        expert_global_scale = w13_global_scale[expert_id]

        expert_w13_float = e2m1_and_ufp8sf_scale_to_float(
            expert_w13.view(torch.uint8),
            expert_w13_scale.view(torch.uint8),
            expert_global_scale,
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
            is_sf_swizzled_layout=True,
        )
        w13_float_list.append(expert_w13_float)
    w13_float = torch.stack(w13_float_list, dim=0)

    w2_global_scale = weights["w2_weight_scale_2"]
    w2_float_list = []
    for expert_id in range(config.expert_num):
        expert_w2 = weights[W.moe_w2][expert_id]
        expert_w2_scale = weights[W.moe_s2][expert_id]
        expert_global_scale = w2_global_scale[expert_id]

        expert_w2_float = e2m1_and_ufp8sf_scale_to_float(
            expert_w2.view(torch.uint8),
            expert_w2_scale.view(torch.uint8),
            expert_global_scale,
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
            is_sf_swizzled_layout=True,
        )
        w2_float_list.append(expert_w2_float)
    w2_float = torch.stack(w2_float_list, dim=0)

    w13_float = w13_float.to(device).to(dtype)
    w2_float = w2_float.to(device).to(dtype)

    ref_output = torch.zeros((SEQ_LEN, config.hidden_size), dtype=dtype, device=device)

    for token_idx in range(SEQ_LEN):
        token_hidden = hidden_states[token_idx:token_idx+1]
        token_output = torch.zeros((1, config.hidden_size), dtype=dtype, device=device)

        for k in range(config.moe_k):
            expert_id = topk_ids[token_idx, k].item()
            expert_weight = topk_weights[token_idx, k]
            w13_expert = w13_float[expert_id]
            w2_expert = w2_float[expert_id]
            workspace1 = torch.matmul(token_hidden, w13_expert.transpose(0, 1))
            N = workspace1.shape[-1]
            gate = workspace1[..., N // 2:].to(torch.float32)
            value = workspace1[..., :N // 2].to(torch.float32)
            gate = gate * torch.sigmoid(gate)
            workspace2 = (gate * value).to(dtype)
            expert_output = torch.matmul(workspace2, w2_expert.transpose(0, 1))
            token_output += expert_output / expert_weight
        ref_output[token_idx] = token_output[0]

    return ref_output

def test_trtllm_fp4_executor():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    config = _generate_config()
    payload, weights, extra_kwargs = _generate_payload_and_weights(config)
    ref_output = _generate_ref_output(config, payload, weights, extra_kwargs)

    executor = TrtllmFp4Executor(config, weights, FusedMoEQuantConfig())

    output = executor.execute(payload, "silu", None, None, False, None)

    print(output)
    print(ref_output)
    mask = torch.isclose(output, ref_output, rtol=1e-3, atol=1e-3)
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 6, f"Mismatch percentage is {mismatch_pct:.2f}"

if __name__ == "__main__":
    test_trtllm_fp4_executor()

