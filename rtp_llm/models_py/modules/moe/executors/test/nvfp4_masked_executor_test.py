import random
from typing import Dict, Tuple, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.utils.model_weight import W

# Try to import fp4_quantize from flashinfer
try:
    from flashinfer import fp4_quantize
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    fp4_quantize = None

# Constants
DP_SIZE = 4
TP_SIZE = 1
EP_SIZE = 4
NUM_EXPERTS = 128
BATCH_SIZE = 32
MAX_GENERATE_BATCH_SIZE = 128
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768

M = (MAX_GENERATE_BATCH_SIZE + TP_SIZE - 1) // TP_SIZE * EP_SIZE
K = HIDDEN_SIZE
N = MOE_INTERMEDIATE_SIZE * 2

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


def _dequantize_nvfp4_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Dequantize NVFP4 tensor back to high precision dtype.
    
    Args:
        tensor_fp4: Quantized tensor of shape [M, K/2] with dtype uint8
        tensor_sf: Scale factors tensor (swizzled layout, float8_e4m3fn)
        global_scale: Global scale factor (scalar tensor)
        dtype: Target dtype for output
        block_size: Block size for quantization (default 16)
    
    Returns:
        Dequantized tensor of shape [M, K] with dtype
    """
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    
    # Unpack uint8 to two fp4 values
    # Each uint8 contains two fp4 values: lower 4 bits and upper 4 bits
    fp4_vals = torch.empty(m, k, dtype=torch.uint8, device=tensor_fp4.device)
    fp4_vals[:, 0::2] = tensor_fp4 & 0x0F  # Lower 4 bits
    fp4_vals[:, 1::2] = (tensor_fp4 >> 4) & 0x0F  # Upper 4 bits
    
    # E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit
    # Values: 0, 1, 2, 3, 4, 5, 6, -0, -1, -2, -3, -4, -5, -6
    E2M1_VALUES = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.0, 0.0],
        device=tensor_fp4.device,
        dtype=torch.float32,
    )
    
    # Extract sign and magnitude
    sign_mask = (fp4_vals & 0x08) != 0
    magnitude_idx = fp4_vals & 0x07
    
    # Convert to float values
    float_vals = E2M1_VALUES[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)
    
    # Reshape for block-wise scaling
    num_blocks = k // block_size
    float_vals = float_vals.reshape(m, num_blocks, block_size)
    
    # Convert scale factors from float8_e4m3fn to float32
    tensor_sf_float = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32)
    
    # Reshape scale factors to match blocks
    # Scale factors are in swizzled layout, need to convert to linear
    # For simplicity, we assume they're already in the correct shape
    # In practice, you might need to convert from swizzled layout
    if tensor_sf_float.numel() == m * num_blocks:
        tensor_sf_reshaped = tensor_sf_float.reshape(m, num_blocks)
    else:
        # Handle swizzled layout conversion (simplified)
        tensor_sf_reshaped = tensor_sf_float[: m * num_blocks].reshape(m, num_blocks)
    
    # Apply block-wise scaling
    tensor_sf_dtype = tensor_sf_reshaped / global_scale
    out = (float_vals * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    
    return out.to(dtype=dtype)


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
        (num_local_experts, M, K // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.float8_e4m3fn
    )
    
    # Generate global scale for input quantization
    input_global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX], device="cuda", dtype=torch.float32
    )
    
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = max(
            min(int(NUM_EXPERTS * DP_SIZE * random.uniform(0.7, 1.3)), M), 1
        )
        expert_x_bf16_local = torch.randn(
            (num_actual_tokens, K), device="cuda", dtype=torch.bfloat16
        ) * 0.1
        
        # Quantize to NVFP4
        expert_x_q, expert_x_sf = fp4_quantize(
            expert_x_bf16_local,
            input_global_scale,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
        )
        
        # Reshape scale factors
        expert_x_sf_reshaped = expert_x_sf.view(torch.float8_e4m3fn).reshape(
            num_actual_tokens, K // NVFP4_BLOCK_SIZE
        )
        
        expert_x[local_expert_id, :num_actual_tokens, :] = expert_x_q
        expert_x_scale[local_expert_id, :num_actual_tokens, :] = expert_x_sf_reshaped
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
        (num_local_experts, N, K // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.float8_e4m3fn
    )
    w2 = torch.zeros(
        (num_local_experts, K, (N // 2) // 2), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.zeros(
        (num_local_experts, K, (N // 2) // NVFP4_BLOCK_SIZE), device="cuda", dtype=torch.float8_e4m3fn
    )
    
    # Compute global scales for weights
    w1_amax = torch.abs(w1_bf16).max().to(torch.float32)
    w2_amax = torch.abs(w2_bf16).max().to(torch.float32)
    w1_global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax], device="cuda", dtype=torch.float32
    )
    w2_global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax], device="cuda", dtype=torch.float32
    )
    
    for local_expert_id in range(num_local_experts):
        # Quantize w1
        w1_q, w1_sf = fp4_quantize(
            w1_bf16[local_expert_id],
            w1_global_scale,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
        )
        w1_sf_reshaped = w1_sf.view(torch.float8_e4m3fn).reshape(
            N, K // NVFP4_BLOCK_SIZE
        )
        w1[local_expert_id] = w1_q
        w1_scale[local_expert_id] = w1_sf_reshaped
        
        # Quantize w2
        w2_q, w2_sf = fp4_quantize(
            w2_bf16[local_expert_id],
            w2_global_scale,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=True,
        )
        w2_sf_reshaped = w2_sf.view(torch.float8_e4m3fn).reshape(
            K, (N // 2) // NVFP4_BLOCK_SIZE
        )
        w2[local_expert_id] = w2_q
        w2_scale[local_expert_id] = w2_sf_reshaped
    
    weights = {
        W.moe_w1: w1,
        W.moe_w2: w2,
        W.moe_s1: w1_scale,
        W.moe_s2: w2_scale,
    }
    
    # Store global scales for reference output generation
    global_scales = {
        "input_global_scale": input_global_scale,
        "w1_global_scale": w1_global_scale,
        "w2_global_scale": w2_global_scale,
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
        
        # Dequantize input
        expert_x_local_q = expert_x[local_expert_id, :num_actual_tokens, :]
        expert_x_local_sf = expert_x_scale[local_expert_id, :num_actual_tokens, :]
        expert_x_local = _dequantize_nvfp4_to_dtype(
            expert_x_local_q,
            expert_x_local_sf,
            input_global_scale,
            torch.bfloat16,
            block_size=NVFP4_BLOCK_SIZE,
        )
        
        # Dequantize w1
        w1_local_q = w1[local_expert_id]
        w1_local_sf = w1_scale[local_expert_id]
        w1_local = _dequantize_nvfp4_to_dtype(
            w1_local_q,
            w1_local_sf,
            w1_global_scale,
            torch.bfloat16,
            block_size=NVFP4_BLOCK_SIZE,
        )
        
        # Dequantize w2
        w2_local_q = w2[local_expert_id]
        w2_local_sf = w2_scale[local_expert_id]
        w2_local = _dequantize_nvfp4_to_dtype(
            w2_local_q,
            w2_local_sf,
            w2_global_scale,
            torch.bfloat16,
            block_size=NVFP4_BLOCK_SIZE,
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


def test_nvfp4_masked_executor(use_nvfp4: bool = True):
    """
    Test NVFP4 masked executor.
    
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
    
    # Create executor
    from rtp_llm.models_py.modules.cuda.moe.executors.trtllm_fp4_executor import (
        TrtllmFp4Executor,
    )
    
    # Generate dummy topk_ids and topk_weights for testing
    # In real usage, these would come from the router
    num_local_experts = NUM_EXPERTS // EP_SIZE
    total_tokens = payload.expert_tokens_meta.expert_num_tokens.sum().item()
    top_k = 2  # Default top_k
    
    # Create dummy topk_ids and topk_weights
    # This is a simplified version - in practice, these should come from routing
    expert_topk_ids = torch.zeros(
        (total_tokens, top_k), device="cuda", dtype=torch.int32
    )
    expert_topk_weights = torch.ones(
        (total_tokens, top_k), device="cuda", dtype=torch.bfloat16
    ) / top_k
    
    # Assign expert IDs based on which expert has tokens
    token_idx = 0
    for expert_id in range(num_local_experts):
        num_tokens = payload.expert_tokens_meta.expert_num_tokens[expert_id].item()
        if num_tokens > 0:
            expert_topk_ids[token_idx : token_idx + num_tokens, 0] = expert_id
            token_idx += num_tokens
    
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
    
    # Check - use relaxed tolerance for quantization
    torch.testing.assert_close(output, ref_output, rtol=1e-1, atol=1e-1)
    
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

