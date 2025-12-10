import random
from typing import Dict, Tuple
from pathlib import Path

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
from rtp_llm.utils.model_weight import W

DP_SIZE = 1
TP_SIZE = 1
EP_SIZE = 1
NUM_EXPERTS = 128
SEQ_LEN = 908
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOP_K = 8

NVFP4_BLOCK_SIZE = 16

REAL_DATA_DIR = Path("/home/xiebaijie.xbj/fp4/dump_908")

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
def _generate_config() -> GptInitModelParameters:
    config = GptInitModelParameters(
        head_num=2,
        size_per_head=128,
        layer_num=2,
        max_seq_len=2048,
        vocab_size=500000,
    )
    config.world_size = DP_SIZE * EP_SIZE
    config.dp_size = DP_SIZE
    config.tp_size = TP_SIZE
    config.ep_size = EP_SIZE
    config.dp_rank = 0
    config.tp_rank = 0
    config.ep_rank = 0
    config.expert_num = NUM_EXPERTS
    config.hidden_size = HIDDEN_SIZE
    config.moe_inter_padding_size = MOE_INTERMEDIATE_SIZE
    return config

def load_pt(filename, shape, dtype):
    f = REAL_DATA_DIR / filename
    assert f.is_file()
    t = torch.load(f)
    assert t.shape == shape, f"Shape mismatch: expected {shape}, got {t.shape}"
    assert t.dtype == dtype, f"Dtype mismatch: expected {dtype}, got {t.dtype}"
    return t

def _generate_payload_and_weights(
    config: GptInitModelParameters,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if REAL_DATA_DIR.is_dir():
        hidden_states = load_pt("x.pt", (SEQ_LEN, HIDDEN_SIZE), torch.bfloat16)
        routing_logits = load_pt("router_logits.pt", (SEQ_LEN, NUM_EXPERTS), torch.bfloat16)
        # Refer to weights/scales after vllm::ModelOptNvFp4FusedMoE::process_weights_after_loading
        #   https://github.com/vllm-project/vllm/blob/releases/v0.12.0/vllm/model_executor/layers/quantization/modelopt.py#L1316
        w13_input_scale = load_pt("process_weights_after_loading_w13_input_scale.pt", (NUM_EXPERTS,), torch.float32)
        w13 = load_pt("process_weights_after_loading_gemm1_weights_fp4_shuffled.pt", (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE * 2, HIDDEN_SIZE / 2), torch.uint8)
        w13_scale = load_pt("process_weights_after_loading_gemm1_scales_fp4_shuffled.pt", (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE * 2, HIDDEN_SIZE / NVFP4_BLOCK_SIZE), torch.float8_e4m3fn)
        w13_scale_2 = load_pt("process_weights_after_loading_w13_weight_scale_2.pt", (NUM_EXPERTS,), torch.float32)
        w2_input_scale = load_pt("process_weights_after_loading_w2_input_scale.pt", (NUM_EXPERTS,), torch.float32)
        w2 = load_pt("process_weights_after_loading_gemm2_weights_fp4_shuffled.pt", (NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE / 2), torch.uint8)
        w2_scale = load_pt("process_weights_after_loading_gemm2_scales_fp4_shuffled.pt", (NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE / NVFP4_BLOCK_SIZE), torch.float8_e4m3fn)
        w2_scale_2 = load_pt("process_weights_after_loading_w2_weight_scale_2.pt", (NUM_EXPERTS,), torch.float32)
        rmute_info, topk_weights = routing_reference_renormalize(
            routing_logits, TOP_K, NUM_EXPERTS, 8
        )
        topk_ids = permute_info["topKIndices"].to(torch.int32)
        check_meta(topk_ids, (SEQ_LEN, TOP_K), torch.int32)
        topk_weights = topk_weights.view(SEQ_LEN, NUM_EXPERTS)[
            torch.arange(SEQ_LEN).unsqueeze(1), topk_ids
        ].to(torch.bfloat16)
        check_meta(topk_weights, (SEQ_LEN, TOP_K), torch.bfloat16)
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
            "w13_scale_2": w13_scale_2,
            "w2_input_scale": w2_input_scale,
            "w2_scale_2": w2_scale_2,
        }
        return payload, weights

def _generate_ref_output(
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if REAL_DATA_DIR.is_dir():
        output = load_pt("output.pt", (SEQ_LEN, HIDDEN_SIZE), torch.bfloat16)
        return output

        w2_local_float32 = e2m1_and_ufp8sf_scale_to_float(
            weights[W.moe_w2],
            weights[W.moe_s2],
            w2_global_scale,?
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
        )
        # Compute MoE forward pass
        workspace1 = expert_x_local @ w1_local.transpose(0, 1)
        gate = workspace1[..., N // 2 :].to(torch.float32)
        value = workspace1[..., : N // 2].to(torch.float32)
        gate = gate * (1.0 / (1.0 + torch.exp(-gate)))  # SiLU
        workspace2 = (gate * value).to(torch.bfloat16)
        ref_output[local_expert_id, :num_actual_tokens, :] = (
            workspace2 @ w2_local.transpose(0, 1)
        )
    
    return ref_output

def test_trtllm_fp4_executor():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    config = _generate_config()
    payload, weights = _generate_payload_and_weights(config)
    ref_output = _generate_ref_output(payload, weights)

    from rtp_llm.models_py.modules.moe.executors.trtllm_fp4_executor import (
        TrtllmFp4Executor,
    )
    
    executor = TrtllmFp4Executor(
        config,
        weights,
        FusedMoEQuantConfig(
            quant_dtype=torch.uint8,  # NVFP4 uses uint8
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE],
        ),
    )
    
    output = executor.execute(payload, "silu", NUM_EXPERTS, None, None, False, None)

    print(output)
    print(ref_output)
    torch.testing.assert_close(output, ref_output)

if __name__ == "__main__":
    test_trtllm_fp4_executor()

