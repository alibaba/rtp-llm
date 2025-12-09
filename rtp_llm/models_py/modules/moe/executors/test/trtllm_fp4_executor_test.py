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
SEQ_LEN = 17
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOP_K = 8

REAL_DATA_DIR = Path("/home/xiebaijie.xbj/fp4/moe_params_prefill")

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

def load_pt(filename):
    f = REAL_DATA_DIR / filename
    assert f.is_file()
    return torch.load(f)
def check_meta(tensor, shape, dtype):
    assert tensor.shape == shape, f"Shape mismatch: expected {shape}, got {tensor.shape}"
    assert tensor.dtype == dtype, f"Dtype mismatch: expected {dtype}, got {tensor.dtype}"

def _generate_payload_and_weights(
    config: GptInitModelParameters,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if REAL_DATA_DIR.is_dir():
        hidden_states = load_pt("x.pt")
        check_meta(hidden_states, (SEQ_LEN, HIDDEN_SIZE), torch.bfloat16)
        hidden_states_scale = load_pt("layer_a1_gscale.pt")
        check_meta(hidden_states_scale, (NUM_EXPERTS,), torch.float32)
        routing_logits = load_pt("router_logits.pt")
        check_meta(routing_logits, (SEQ_LEN, NUM_EXPERTS), torch.bfloat16)
        w13 = load_pt("layer_gemm1_weights.pt")
        check_meta(w13, (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE * 2, HIDDEN_SIZE / 2), torch.bfloat16)
        w13_scale = load_pt("layer_gemm1_weights_scale.pt")
        check_meta(w13_scale, (NUM_EXPERTS, MOE_INTERMEDIATE_SIZE * 2, 1), torch.float32)
        w2 = load_pt("layer_gemm2_weights.pt")
        check_meta(w2, (NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE), torch.bfloat16)
        w2_scale = load_pt("layer_gemm2_weights_scale.pt")
        check_meta(w2_scale, (NUM_EXPERTS, HIDDEN_SIZE, 1), torch.float32)
        output1_scale_scalar = load_pt("layer_g1_scale_c.pt")
        check_meta(output1_scale_scalar, (1,), torch.float32)
        output1_scale_gate_scalar = load_pt("layer_g1_alphas.pt")
        check_meta(output1_scale_gate_scalar, (1,), torch.float32)
        output2_scale_scalar = load_pt("layer_g2_alphas.pt")
        check_meta(output2_scale_scalar, (1,), torch.float32)
        if routing_method_type == RoutingMethodType.Renormalize:
            permute_info, topk_weights = routing_reference_renormalize(
                routing_logits, TOP_K, NUM_EXPERTS, 8
            )
        else:
            assert 0, f"Routing method {routing_method_type} not supported"
        topk_ids = permute_info["topKIndices"].to(torch.int32)
        check_meta(topk_ids, (SEQ_LEN, TOP_K), torch.int32)
        topk_weights = topk_weights.view(SEQ_LEN, NUM_EXPERTS)[
            torch.arange(SEQ_LEN).unsqueeze(1), topk_ids
        ].to(torch.bfloat16)
        check_meta(topk_weights, (SEQ_LEN, NUM_EXPERTS), torch.bfloat16)
    payload = ExpertForwardPayload(
        expert_x=hidden_states,
        expert_x_origin_dtype=torch.bfloat16,
        expert_x_scale=hidden_states_scale,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
    )
    weights = {
        W.moe_w1: w13,
        W.moe_w2: w2,
        W.moe_s1: w13_scale,
        W.moe_s2: w2_scale,
        "output1_scale_scalar": output1_scale_scalar,
        "output1_scale_gate_scalar": output1_scale_gate_scalar,
        "output2_scale_scalar": output2_scale_scalar,
    }
    return payload, weights

def _generate_ref_output(
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Generate reference output by dequantizing and computing MoE forward pass.
    
    Args:
        payload: Expert forward payload
        weights: Weight dictionary
        global_scales: Dictionary containing global scale factors
        use_nvfp4: Whether using NVFP4 (should be True)
    
    Returns:
        Reference output tensor
    """
    if REAL_DATA_DIR.is_dir():
        output = load_pt("output.pt")
        check_meta(output, (SEQ_LEN, HIDDEN_SIZE), torch.bfloat16)
        return output
    
    num_local_experts = NUM_EXPERTS // EP_SIZE
    expert_x = payload.expert_x
    expert_x_scale = payload.expert_x_scale
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    w1 = weights[W.moe_w1]
    w1_scale = weights[W.moe_s1]
    w2 = weights[W.moe_w2]
    w2_scale = weights[W.moe_s2]
    
    # Get global scales from the quantization process
    input_global_scale = global_scales["input_global_scale"]
    w1_global_scale = global_scales["w1_global_scale"]
    w2_global_scale = global_scales["w2_global_scale"]
    
    ref_output = torch.zeros(
        (num_local_experts, M, K), device="cuda", dtype=torch.bfloat16
    )
    
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = expert_num_tokens[local_expert_id].item()
        if num_actual_tokens == 0:
            continue
        
        # Dequantize input
        expert_x_local_q = expert_x[local_expert_id, :num_actual_tokens, :]
        expert_x_local_sf = expert_x_scale[local_expert_id, :num_actual_tokens, :]
        expert_x_local_float32 = e2m1_and_ufp8sf_scale_to_float(
            expert_x_local_q,
            expert_x_local_sf,
            input_global_scale,
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        )
        expert_x_local = expert_x_local_float32.to(device=expert_x_local_q.device).to(torch.bfloat16)
        
        # Note: We can't verify expert_x dequantization here because we don't have the original tensor
        # The verification is done in _generate_payload_and_weights during quantization
        
        # Dequantize w1
        w1_local_q = w1[local_expert_id]
        w1_local_sf = w1_scale[local_expert_id]
        w1_local_float32 = e2m1_and_ufp8sf_scale_to_float(
            w1_local_q,
            w1_local_sf,
            w1_global_scale,
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
        )
        w1_local = w1_local_float32.to(device=w1_local_q.device).to(torch.bfloat16)
        
        # Verify w1 dequantization matches original
        if "w1_bf16" in global_scales:
            w1_original = global_scales["w1_bf16"][local_expert_id]
            w1_local_reshaped = w1_local.reshape(N, K)
            w1_diff = (w1_original.cpu() - w1_local_reshaped.cpu()).abs()
            w1_max_diff = w1_diff.max().item()
            w1_mean_diff = w1_diff.mean().item()
            print(f"[DEBUG _generate_ref_output] Expert {local_expert_id} w1 dequantization check:")
            print(f"  Original shape: {w1_original.shape}, Dequantized shape: {w1_local_reshaped.shape}")
            print(f"  Max diff: {w1_max_diff:.6f}, Mean diff: {w1_mean_diff:.6f}")
            if w1_max_diff > 0.1:
                print(f"w1_original: {w1_original}")
                print(f"w1_local_reshaped: {w1_local_reshaped}")
                assert 0, f"  ⚠ WARNING: Large w1 dequantization difference! Max diff: {w1_max_diff:.6f}"
            else:
                print(f"  ✓ w1 dequantization check passed")
        
        # Dequantize w2
        w2_local_q = w2[local_expert_id]
        w2_local_sf = w2_scale[local_expert_id]
        w2_local_float32 = e2m1_and_ufp8sf_scale_to_float(
            w2_local_q,
            w2_local_sf,
            w2_global_scale,
            sf_vec_size=NVFP4_BLOCK_SIZE,
            ufp8_type=1,
        )
        w2_local = w2_local_float32.to(device=w2_local_q.device).to(torch.bfloat16)
        
        # Verify w2 dequantization matches original
        if "w2_bf16" in global_scales:
            w2_original = global_scales["w2_bf16"][local_expert_id]
            w2_local_reshaped = w2_local.reshape(K, N // 2)
            w2_diff = (w2_original.cpu() - w2_local_reshaped.cpu()).abs()
            w2_max_diff = w2_diff.max().item()
            w2_mean_diff = w2_diff.mean().item()
            print(f"[DEBUG _generate_ref_output] Expert {local_expert_id} w2 dequantization check:")
            print(f"  Original shape: {w2_original.shape}, Dequantized shape: {w2_local_reshaped.shape}")
            print(f"  Max diff: {w2_max_diff:.6f}, Mean diff: {w2_mean_diff:.6f}")
            if w2_max_diff > 0.1:
                print(f"w2_original: {w2_original}")
                print(f"w2_local_reshaped: {w2_local_reshaped}")
                assert 0, f"  ⚠ WARNING: Large w2 dequantization difference! Max diff: {w2_max_diff:.6f}"
            else:
                print(f"  ✓ w2 dequantization check passed")
        
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


def _extract_valid_tokens_2d(
    tensor_3d: torch.Tensor,
    expert_num_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Extract valid tokens from 3D tensor (n_experts, max_num_token_per_expert, hidden_size)
    and convert to 2D tensor (total_valid_tokens, hidden_size).
    
    Args:
        tensor_3d: 3D tensor of shape [n_experts, max_num_token_per_expert, hidden_size]
        expert_num_tokens: 1D tensor of shape [n_experts] indicating valid tokens per expert
    
    Returns:
        2D tensor of shape [total_valid_tokens, hidden_size]
    """
    num_experts = tensor_3d.shape[0]
    valid_tokens_list = []
    
    for expert_id in range(num_experts):
        num_valid_tokens = expert_num_tokens[expert_id].item()
        if num_valid_tokens > 0:
            valid_tokens = tensor_3d[expert_id, :num_valid_tokens, :]
            valid_tokens_list.append(valid_tokens)
    
    if len(valid_tokens_list) == 0:
        return torch.empty((0, tensor_3d.shape[2]), device=tensor_3d.device, dtype=tensor_3d.dtype)
    
    return torch.cat(valid_tokens_list, dim=0)


def test_nvfp4_masked_executor():
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

    torch.testing.assert_close(output, ref_output)

if __name__ == "__main__":
    test_nvfp4_masked_executor()

