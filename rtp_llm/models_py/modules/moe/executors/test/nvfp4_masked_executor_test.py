import random
from typing import Dict, Tuple, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
# from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
from rtp_llm.utils.model_weight import W

try:
    from flashinfer import fp4_quantize, e2m1_and_ufp8sf_scale_to_float
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    fp4_quantize = None
    e2m1_and_ufp8sf_scale_to_float = None

# Constants - matching bench_fp4_moe.py default parameters
DP_SIZE = 1
TP_SIZE = 1
EP_SIZE = 1
NUM_EXPERTS = 32
BATCH_SIZE = 1  # Single request in prefill
SEQ_LEN = 512  # Sequence length for prefill
MAX_GENERATE_BATCH_SIZE = 128  # For decode phase (not used in prefill test)
HIDDEN_SIZE = 4096  # Changed from 2048 to match bench_fp4_moe.py
MOE_INTERMEDIATE_SIZE = 1536  # Changed from 768 to match bench_fp4_moe.py
TOP_K = 8  # Changed from 2 to match bench_fp4_moe.py

# M represents max tokens per expert in prefill scenario
# For prefill: total tokens = batch_size * seqlen = 1 * 128 = 128
# These tokens are distributed across experts based on routing
# In worst case, all 128 tokens could go to one expert, so M = SEQ_LEN = 128
M = SEQ_LEN  # Max tokens per expert (in worst case, all tokens go to one expert)
K = HIDDEN_SIZE
N = MOE_INTERMEDIATE_SIZE * 2

# Test scenario verification:
# - batch_size = 1 (single request)
# - seqlen = 128 (sequence length for prefill)
# - Single GPU: DP_SIZE=1, TP_SIZE=1, EP_SIZE=1
# - Prefill phase: all 128 tokens are processed in parallel

# NVFP4 constants
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16


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
    config.max_generate_batch_size = MAX_GENERATE_BATCH_SIZE
    config.moe_inter_padding_size = MOE_INTERMEDIATE_SIZE
    return config


def _generate_payload_and_weights(
    config: GptInitModelParameters,
    use_nvfp4: bool,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate payload and weights for NVFP4 MoE executor test.
    
    Args:
        config: Model configuration
        use_nvfp4: Whether to use NVFP4 quantization
    
    Returns:
        Tuple of (payload, weights)
    """
    if not use_nvfp4:
        raise ValueError("This test is specifically for NVFP4 executor")
    
    if not FLASHINFER_AVAILABLE:
        raise ImportError("flashinfer is required for NVFP4 quantization")
    
    num_local_experts = config.expert_num // config.ep_size
    
    # Generate input data in bfloat16
    expert_x_bf16 = torch.zeros(
        (num_local_experts, M, K), device="cuda", dtype=torch.bfloat16
    )
    expert_num_tokens = torch.zeros(
        (num_local_experts,), device="cuda", dtype=torch.int32
    )
    
    # Quantize input to NVFP4
    expert_x = torch.zeros(
        (num_local_experts, M, K // 2), device="cuda", dtype=torch.uint8
    )
    expert_x_scale = torch.zeros(
        (num_local_experts, M, K // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    
    # Generate global scale for input quantization
    # NOTE: This is quantization global scale (used for quantization)
    # For dequantization, we need the inverse (1.0 / quantization_global_scale)
    input_quantization_global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX], device="cuda", dtype=torch.float32
    )
    # For dequantization and executor, we need the inverse
    input_global_scale = 1.0 / input_quantization_global_scale
    
    # For prefill scenario: total tokens = batch_size * seqlen
    # Distribute these tokens across experts more evenly
    total_tokens = BATCH_SIZE * SEQ_LEN
    tokens_per_expert = [0] * num_local_experts
    
    # Calculate average tokens per expert for more balanced distribution
    avg_tokens_per_expert = total_tokens / num_local_experts
    
    # Distribute tokens across experts (simulate routing)
    # Each token can go to TOP_K experts, but for simplicity, we assign tokens to experts
    # with some randomness around the average to ensure balanced distribution
    remaining_tokens = total_tokens
    assigned_tokens = 0
    
    # First pass: assign tokens to each expert with small random variation around average
    # Use a tighter range (0.7x to 1.3x of average) to ensure more balanced distribution
    for local_expert_id in range(num_local_experts - 1):  # Leave last expert for remainder
        if remaining_tokens <= 0:
            break
        
        # Calculate target tokens for this expert (around average with small variation)
        # Use smaller variation range for more balanced distribution
        min_tokens = max(1, int(avg_tokens_per_expert * 0.7))
        max_tokens = min(M, remaining_tokens, int(avg_tokens_per_expert * 1.3))
        
        # Ensure we don't leave too few tokens for remaining experts
        min_tokens_for_remaining = (num_local_experts - local_expert_id - 1) * 1
        max_tokens_for_this_expert = remaining_tokens - min_tokens_for_remaining
        
        if max_tokens_for_this_expert > 0:
            tokens_for_this_expert = random.randint(
                min(min_tokens, max_tokens_for_this_expert),
                min(max_tokens, max_tokens_for_this_expert)
            )
        else:
            tokens_for_this_expert = 0
        
        tokens_per_expert[local_expert_id] = tokens_for_this_expert
        remaining_tokens -= tokens_for_this_expert
        assigned_tokens += tokens_for_this_expert
    
    # Last expert gets all remaining tokens to ensure total = batch_size * seqlen
    tokens_per_expert[num_local_experts - 1] = remaining_tokens
    assigned_tokens += remaining_tokens
    
    print(f"DEBUG tokens_per_expert: {tokens_per_expert}")
    print(f"DEBUG total assigned: {assigned_tokens}, expected: {total_tokens}, avg per expert: {avg_tokens_per_expert:.2f}")
    
    # Final verification: total tokens must equal batch_size * seqlen
    assert assigned_tokens == total_tokens, (
        f"Token assignment error: assigned {assigned_tokens} tokens, expected {total_tokens}"
    )
    
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = tokens_per_expert[local_expert_id]
        if num_actual_tokens == 0:
            continue
        
        expert_x_bf16_local = torch.randn(
            (num_actual_tokens, K), device="cuda", dtype=torch.bfloat16
        ) * 0.1
        
        # Quantize to NVFP4
        # Use is_sf_swizzled_layout=False to get linear layout that can be directly reshaped
        # NOTE: fp4_quantize expects quantization global_scale (not dequantization scale)
        expert_x_q, expert_x_sf = fp4_quantize(
            expert_x_bf16_local,
            input_quantization_global_scale,  # Use quantization scale for quantization
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )

        # 验证量化/反量化：反量化量化后的 tensor，与原始 tensor 比较
        print(f"expert_x_q dtype: {expert_x_q.dtype}")
        print(f"expert_x_sf dtype: {expert_x_sf.dtype}")
        print(f"input_global_scale dtype: {input_global_scale.dtype}")
        expert_x_dequantized = e2m1_and_ufp8sf_scale_to_float(
            expert_x_q,
            expert_x_sf,
            input_global_scale,
            sf_vec_size=16,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        )
        
        # Reshape scale factors (linear layout can be directly reshaped)
        # Convert to float8_e4m3fn view and take the first num_actual_tokens rows
        expert_x_sf_viewed = expert_x_sf.view(torch.float8_e4m3fn)
        expert_x_sf_reshaped = expert_x_sf_viewed[:num_actual_tokens, :]
        
        # 打印比较结果
        diff = (expert_x_bf16_local.cpu() - expert_x_dequantized).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[DEBUG generate_payloads] Expert {local_expert_id}:")
        print(f"  Original shape: {expert_x_bf16_local.shape}, dtype: {expert_x_bf16_local.dtype}")
        print(f"  Quantized shape: {expert_x_q.shape}, dtype: {expert_x_q.dtype}")
        print(f"  Dequantized shape: {expert_x_dequantized.shape}, dtype: {expert_x_dequantized.dtype}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        print(f"  Original min: {expert_x_bf16_local.min().item():.6f}, max: {expert_x_bf16_local.max().item():.6f}")
        print(f"  Dequantized min: {expert_x_dequantized.min().item():.6f}, max: {expert_x_dequantized.max().item():.6f}")
        
        # 检查是否相同（考虑量化误差）
        if max_diff > 0.1:
            assert 0, f"  ⚠ WARNING: Large difference detected! Max diff: {max_diff:.6f}"
        else:
            print(f"  ✓ Quantization/dequantization check passed")
        
        expert_x[local_expert_id, :num_actual_tokens, :] = expert_x_q
        expert_x_scale[local_expert_id, :num_actual_tokens, :] = expert_x_sf
        expert_num_tokens[local_expert_id] = num_actual_tokens
    
    payload = ExpertForwardPayload(
        expert_x=expert_x,
        expert_x_origin_dtype=torch.bfloat16,
        expert_x_scale=expert_x_scale,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
    )
    
    # Generate weights in bfloat16, then quantize to NVFP4
    w1_bf16 = torch.randn(
        (num_local_experts, N, K), device="cuda", dtype=torch.bfloat16
    ) * 0.1
    w2_bf16 = torch.randn(
        (num_local_experts, K, N // 2), device="cuda", dtype=torch.bfloat16
    ) * 0.1
    
    # Quantize weights to NVFP4
    w1 = torch.zeros(
        (num_local_experts, N, K // 2), device="cuda", dtype=torch.uint8
    )
    w1_scale = torch.zeros(
        (num_local_experts, N, K // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    w2 = torch.zeros(
        (num_local_experts, K, (N // 2) // 2), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.zeros(
        (num_local_experts, K, (N // 2) // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    
    # Compute global scales for weights
    # NOTE: These are quantization global scales (used for quantization)
    # For dequantization, we need the inverse (1.0 / quantization_global_scale)
    w1_amax = torch.abs(w1_bf16).max().to(torch.float32)
    w2_amax = torch.abs(w2_bf16).max().to(torch.float32)
    w1_quantization_global_scale = input_global_scale.clone()
    w2_quantization_global_scale = input_global_scale.clone()
    # For dequantization and executor, we need the inverse
    w1_global_scale = 1.0 / w1_quantization_global_scale
    w2_global_scale = 1.0 / w2_quantization_global_scale
    
    for local_expert_id in range(num_local_experts):
        # Quantize w1
        w1_q, w1_sf = fp4_quantize(
            w1_bf16[local_expert_id],
            w1_quantization_global_scale,  # Use w1 quantization scale
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        w1_sf_viewed = w1_sf.view(torch.float8_e4m3fn)
        if w1_sf_viewed.ndim == 2:
            w1_sf_reshaped = w1_sf_viewed
        else:
            w1_sf_reshaped = w1_sf_viewed.reshape(N, K // NVFP4_BLOCK_SIZE)
        
        # Verify w1 quantization/dequantization
        w1_dequantized = e2m1_and_ufp8sf_scale_to_float(
            w1_q,
            w1_sf,
            w1_global_scale,
            sf_vec_size=16,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        )
        w1_dequantized_bf16 = w1_dequantized.to(device=w1_q.device).to(torch.bfloat16)
        w1_dequantized_bf16 = w1_dequantized_bf16.reshape(N, K)
        
        w1_diff = (w1_bf16[local_expert_id].cpu() - w1_dequantized_bf16.cpu()).abs()
        w1_max_diff = w1_diff.max().item()
        w1_mean_diff = w1_diff.mean().item()
        print(f"[DEBUG generate_weights] Expert {local_expert_id} w1:")
        print(f"  Original shape: {w1_bf16[local_expert_id].shape}, dtype: {w1_bf16[local_expert_id].dtype}")
        print(f"  Quantized shape: {w1_q.shape}, dtype: {w1_q.dtype}")
        print(f"  Dequantized shape: {w1_dequantized_bf16.shape}, dtype: {w1_dequantized_bf16.dtype}")
        print(f"  Max diff: {w1_max_diff:.6f}, Mean diff: {w1_mean_diff:.6f}")
        print(f"  Original min: {w1_bf16[local_expert_id].min().item():.6f}, max: {w1_bf16[local_expert_id].max().item():.6f}")
        print(f"  Dequantized min: {w1_dequantized_bf16.min().item():.6f}, max: {w1_dequantized_bf16.max().item():.6f}")
        if w1_max_diff > 0.1:
            print(f"w1_q: {w1_bf16[local_expert_id]}")
            print(f"w1_dequantized: {w1_dequantized_bf16}")
            assert 0, f"  ⚠ WARNING: Large w1 difference detected! Max diff: {w1_max_diff:.6f}"
        else:
            print(f"  ✓ w1 Quantization/dequantization check passed")
        
        assert 0
        w1[local_expert_id] = w1_q
        w1_scale[local_expert_id] = w1_sf
        
        # Quantize w2
        w2_q, w2_sf = fp4_quantize(
            w2_bf16[local_expert_id],
            w2_quantization_global_scale,  # Use w2 quantization scale
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        # fp4_quantize returns scale factors as [M, K // sf_vec_size] after reshape
        w2_sf_viewed = w2_sf.view(torch.float8_e4m3fn)
        if w2_sf_viewed.ndim == 2:
            w2_sf_reshaped = w2_sf_viewed
        else:
            w2_sf_reshaped = w2_sf_viewed.reshape(K, (N // 2) // NVFP4_BLOCK_SIZE)
        
        # Verify w2 quantization/dequantization
        w2_dequantized = e2m1_and_ufp8sf_scale_to_float(
            w2_q,
            w2_sf,
            w2_global_scale,
            sf_vec_size=16,
            ufp8_type=1,
            is_sf_swizzled_layout=True,
        )
        w2_dequantized_bf16 = w2_dequantized.to(device=w2_q.device).to(torch.bfloat16)
        w2_dequantized_bf16 = w2_dequantized_bf16.reshape(K, N // 2)
        
        w2_diff = (w2_bf16[local_expert_id].cpu() - w2_dequantized_bf16.cpu()).abs()
        w2_max_diff = w2_diff.max().item()
        w2_mean_diff = w2_diff.mean().item()
        print(f"[DEBUG generate_weights] Expert {local_expert_id} w2:")
        print(f"  Original shape: {w2_bf16[local_expert_id].shape}, dtype: {w2_bf16[local_expert_id].dtype}")
        print(f"  Quantized shape: {w2_q.shape}, dtype: {w2_q.dtype}")
        print(f"  Dequantized shape: {w2_dequantized_bf16.shape}, dtype: {w2_dequantized_bf16.dtype}")
        print(f"  Max diff: {w2_max_diff:.6f}, Mean diff: {w2_mean_diff:.6f}")
        print(f"  Original min: {w2_bf16[local_expert_id].min().item():.6f}, max: {w2_bf16[local_expert_id].max().item():.6f}")
        print(f"  Dequantized min: {w2_dequantized_bf16.min().item():.6f}, max: {w2_dequantized_bf16.max().item():.6f}")
        if w2_max_diff > 0.1:
            assert 0, f"  ⚠ WARNING: Large w2 difference detected! Max diff: {w2_max_diff:.6f}"
        else:
            print(f"  ✓ w2 Quantization/dequantization check passed")
        
        w2[local_expert_id] = w2_q
        w2_scale[local_expert_id] = w2_sf
    
    weights = {
        W.moe_w1: w1,
        W.moe_w2: w2,
        W.moe_s1: w1_scale,
        W.moe_s2: w2_scale,
        # Store global scales in weights dict for executor to use
        "moe_w1_global_scale": w1_global_scale,
        "moe_w2_global_scale": w2_global_scale,
        "moe_input_global_scale": input_global_scale,
    }
    
    # Store global scales and original tensors for reference output generation
    global_scales = {
        "input_global_scale": input_global_scale,
        "w1_global_scale": w1_global_scale,
        "w2_global_scale": w2_global_scale,
        "w1_bf16": w1_bf16,  # Store original w1 for verification
        "w2_bf16": w2_bf16,  # Store original w2 for verification
    }
    
    return payload, weights, global_scales


def _generate_ref_output(
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
    global_scales: Dict[str, torch.Tensor],
    use_nvfp4: bool,
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
    if not use_nvfp4:
        raise ValueError("This test is specifically for NVFP4 executor")
    
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
                print(f"  ⚠ WARNING: Large w1 dequantization difference! Max diff: {w1_max_diff:.6f}")
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
                print(f"  ⚠ WARNING: Large w2 dequantization difference! Max diff: {w2_max_diff:.6f}")
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


def test_nvfp4_masked_executor(use_nvfp4: bool = True):
    """
    Test NVFP4 masked executor for prefill scenario.
    
    Test scenario:
    - batch_size = 1 (single request)
    - seqlen = 128 (sequence length for prefill)
    - Single GPU: DP_SIZE=1, TP_SIZE=1, EP_SIZE=1
    - Prefill phase: all 128 tokens are processed in parallel
    - Total tokens = batch_size * seqlen = 1 * 128 = 128
    - Tokens are distributed across 128 experts based on routing (TOP_K=8)
    
    Note: The executor implementation is not yet available, so this test
    will fail until the executor is implemented.
    """
    if not FLASHINFER_AVAILABLE:
        import pytest
        pytest.skip("flashinfer is required for NVFP4 quantization")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    # Generate data
    config = _generate_config()
    payload, weights, global_scales = _generate_payload_and_weights(config, use_nvfp4)
    
    # Generate reference output
    ref_output = _generate_ref_output(payload, weights, global_scales, use_nvfp4)
    payload.expert_x_scale = payload.expert_x_scale.view(torch.float8_e4m3fn)
    weights[W.moe_s1] = weights[W.moe_s1].view(torch.float8_e4m3fn)
    weights[W.moe_s2] = weights[W.moe_s2].view(torch.float8_e4m3fn)
    
    # Create executor
    from rtp_llm.models_py.modules.moe.executors.trtllm_fp4_executor import (
        TrtllmFp4Executor,
    )
    
    # Generate dummy topk_ids and topk_weights for testing
    # In real usage, these would come from the router
    # For prefill scenario: total tokens = batch_size * seqlen = 1 * 128 = 128
    num_local_experts = NUM_EXPERTS // EP_SIZE
    total_tokens = BATCH_SIZE * SEQ_LEN  # Should be 128 for prefill
    top_k = TOP_K  # Use TOP_K constant matching bench_fp4_moe.py
    
    # Verify that total tokens match expected prefill scenario
    actual_total_tokens = payload.expert_tokens_meta.expert_num_tokens.sum().item()
    assert actual_total_tokens == total_tokens, (
        f"Total tokens mismatch: expected {total_tokens} (batch_size * seqlen), "
        f"got {actual_total_tokens} from expert_num_tokens"
    )
    
    # Create dummy topk_ids and topk_weights
    # This is a simplified version - in practice, these should come from routing
    # For prefill: each of the 128 tokens should be routed to TOP_K experts
    expert_topk_ids = torch.zeros(
        (total_tokens, top_k), device="cuda", dtype=torch.int32
    )
    expert_topk_weights = torch.ones(
        (total_tokens, top_k), device="cuda", dtype=torch.bfloat16
    ) / top_k
    
    # Assign expert IDs based on which expert has tokens
    # For simplicity, we assign tokens to experts in a round-robin fashion
    # In real scenario, this would come from router output
    token_idx = 0
    for expert_id in range(num_local_experts):
        num_tokens = payload.expert_tokens_meta.expert_num_tokens[expert_id].item()
        if num_tokens > 0:
            # Assign this expert as the primary expert for these tokens
            # Fill all top_k positions with the same expert for simplicity
            # In real scenario, each token would be routed to TOP_K different experts
            for k in range(top_k):
                expert_topk_ids[token_idx : token_idx + num_tokens, k] = expert_id
            token_idx += num_tokens
    
    # Verify all tokens are assigned
    assert token_idx == total_tokens, (
        f"Token assignment mismatch: assigned {token_idx} tokens, expected {total_tokens}"
    )
    
    # Update payload with topk information
    payload.expert_topk_ids = expert_topk_ids
    payload.expert_topk_weights = expert_topk_weights
    
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
    
    # Execute
    output = executor.execute(payload, "silu", None, None, False, None)
    
    # Debug: print output statistics
    print(f"[DEBUG Test] output shape: {output.shape}, dtype: {output.dtype}")
    print(f"[DEBUG Test] ref_output shape: {ref_output.shape}, dtype: {ref_output.dtype}")
    print(f"[DEBUG Test] output min: {output.min().item():.6f}, max: {output.max().item():.6f}, mean: {output.mean().item():.6f}")
    print(f"[DEBUG Test] ref_output min: {ref_output.min().item():.6f}, max: {ref_output.max().item():.6f}, mean: {ref_output.mean().item():.6f}")
    
    # Check for NaN or Inf
    output_nan = torch.isnan(output).sum().item()
    output_inf = torch.isinf(output).sum().item()
    ref_nan = torch.isnan(ref_output).sum().item()
    ref_inf = torch.isinf(ref_output).sum().item()
    print(f"[DEBUG Test] output NaN count: {output_nan}, Inf count: {output_inf}")
    print(f"[DEBUG Test] ref_output NaN count: {ref_nan}, Inf count: {ref_inf}")
    
    # Extract valid tokens from 3D tensors to 2D tensors for comparison
    # output and ref_output are 3D: (n_experts, max_num_token_per_expert, hidden_size)
    # We need to extract only valid tokens (excluding padding)
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    output_2d = _extract_valid_tokens_2d(output, expert_num_tokens)
    ref_output_2d = _extract_valid_tokens_2d(ref_output, expert_num_tokens)
    
    print(f"[DEBUG Test] output_2d shape: {output_2d.shape}, ref_output_2d shape: {ref_output_2d.shape}")
    assert output_2d.shape == ref_output_2d.shape, (
        f"Shape mismatch after extracting valid tokens: output_2d {output_2d.shape} vs ref_output_2d {ref_output_2d.shape}"
    )
    
    # Compute difference statistics on 2D tensors
    diff_2d = (output_2d - ref_output_2d).abs()
    print(f"[DEBUG Test] diff_2d min: {diff_2d.min().item():.6f}, max: {diff_2d.max().item():.6f}, mean: {diff_2d.mean().item():.6f}")
    print(f"[DEBUG Test] diff_2d > 1.0 count: {(diff_2d > 1.0).sum().item()}")
    
    print("output_2d: ", output_2d)
    print("ref_output_2d: ", ref_output_2d)
    # Compare using 2D tensors (only valid tokens)
    torch.testing.assert_close(output_2d, ref_output_2d, rtol=1e-1, atol=1e-1)
    
    # For now, just check that we can generate the data
    assert payload.expert_x.shape == (NUM_EXPERTS // EP_SIZE, M, K // 2)
    assert payload.expert_x_scale.shape == (
        NUM_EXPERTS // EP_SIZE,
        M,
        K // NVFP4_BLOCK_SIZE,
    )
    assert ref_output.shape == (NUM_EXPERTS // EP_SIZE, M, K)
    
    print("✓ NVFP4 test data generation successful")
    print("⚠ NVFP4 executor not yet implemented - test framework ready")


if __name__ == "__main__":
    test_nvfp4_masked_executor(use_nvfp4=True)

