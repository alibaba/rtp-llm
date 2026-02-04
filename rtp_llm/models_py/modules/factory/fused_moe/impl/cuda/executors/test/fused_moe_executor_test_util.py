import random
from typing import Dict, List, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoEQuantConfig,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


def calc_low_latency_max_token_per_rank(
    max_generate_batch_size: int,
    tp_size: int,
    quant_config: FusedMoEQuantConfig,
) -> int:
    ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size
    is_quantized = quant_config.is_quantized
    is_block_quantized = quant_config.is_block_quantized
    is_per_act_token = quant_config.is_per_act_token
    # Calculate max tokens per rank
    if not is_quantized or is_block_quantized:
        # deepgemm masked with max_m < 64 get incorrect result, related: https://github.com/deepseek-ai/DeepGEMM/issues/268
        matched_tokens = [64, 128]
    elif is_per_act_token:
        matched_tokens = [
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
        ]
    else:
        raise ValueError("Unsupported quantization config")
    if ll_num_max_token_per_rank > 128:
        ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
        return ll_num_max_token_per_rank
    for t in matched_tokens:
        if ll_num_max_token_per_rank <= t:
            ll_num_max_token_per_rank = t
            return ll_num_max_token_per_rank
    return 128


def generate_payload_and_weights(
    config: MoEConfigAdapter,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor]]:
    # Get necessary parameters
    top_k = config.moe_k
    K = config.hidden_size
    tp_size = config.tp_size
    ep_size = config.ep_size
    N = config.model_config.moe_inter_size * 2
    num_experts = config.expert_num
    num_local_experts = num_experts // ep_size
    max_generate_batch_size = config.max_generate_batch_size
    # Calculate M and expected_m
    if config.quant_config is None:
        config.quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
    quant_config = config.quant_config
    ll_num_max_token_per_rank = calc_low_latency_max_token_per_rank(
        max_generate_batch_size, tp_size, quant_config
    )
    expected_m = ll_num_max_token_per_rank * top_k * ep_size // num_experts
    M = ll_num_max_token_per_rank * ep_size
    # Generate payload
    expert_x = torch.zeros(
        (num_local_experts, M, K), device="cuda", dtype=torch.bfloat16
    )
    expert_topk_ids = (
        torch.ones((num_local_experts, M, top_k), device="cuda", dtype=torch.int) * -1
    )
    recv_topk_weights = torch.ones(
        (num_local_experts, M, top_k), device="cuda", dtype=torch.float32
    )
    expert_num_tokens = torch.empty(
        (num_local_experts,), device="cuda", dtype=torch.int32
    )
    # Build a deterministic token template: each consecutive 128 values are identical.
    # Example for K=6144 (K//128=48):
    #   [1]*128 + [2]*128 + ... + [48]*128
    assert K % 128 == 0, f"expected hidden_size K divisible by 128, got K={K}"
    token_template = (
        torch.randn((K // 128,), device="cuda", dtype=torch.float32)
        .repeat_interleave(128)
        .to(torch.bfloat16)
    ) * 0.01
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = max(
            min(int(expected_m * random.uniform(0.7, 1.3)), M),
            1,
        )
        expert_num_tokens[local_expert_id] = num_actual_tokens
        expert_topk_ids[local_expert_id, :num_actual_tokens, 0] = local_expert_id
        expert_x[local_expert_id, :num_actual_tokens, :].copy_(
            token_template.unsqueeze(0).expand(num_actual_tokens, -1)
        )
    payload = ExpertForwardPayload(
        expert_x=expert_x,
        expert_x_scale=None,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=expert_topk_ids,
        expert_topk_weights=recv_topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.tolist(),
        ),
    )
    # Generate weights
    assert N % 128 == 0 and K % 128 == 0 and (N // 2) % 128 == 0
    # Make weights block-constant within each 128x128 block to reduce quantization error.
    w1_block = torch.rand(
        (num_local_experts, N // 128, K // 128), device="cuda", dtype=torch.float32
    )
    w1 = (
        w1_block.repeat_interleave(128, dim=-2)
        .repeat_interleave(128, dim=-1)
        .to(torch.bfloat16)
        * 0.01
    )
    w2_block = torch.rand(
        (num_local_experts, K // 128, (N // 2) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    w2 = (
        w2_block.repeat_interleave(128, dim=-2)
        .repeat_interleave(128, dim=-1)
        .to(torch.bfloat16)
        * 0.01
    )
    weights = {
        W.moe_w1: w1,
        W.moe_w2: w2,
        W.moe_s1: None,
        W.moe_s2: None,
    }
    return payload, weights


def generate_ref_output(
    config: MoEConfigAdapter,
    payload: ExpertForwardPayload,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    # Get necessary parameters
    K = config.hidden_size
    tp_size = config.tp_size
    ep_size = config.ep_size
    N = config.model_config.moe_inter_size * 2
    num_experts = config.expert_num
    num_local_experts = num_experts // ep_size
    max_generate_batch_size = config.max_generate_batch_size
    quant_config = config.quant_config
    ll_num_max_token_per_rank = calc_low_latency_max_token_per_rank(
        max_generate_batch_size, tp_size, quant_config
    )
    M = ll_num_max_token_per_rank * ep_size
    # Get input data
    expert_x = payload.expert_x
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    w1 = weights[W.moe_w1]
    w2 = weights[W.moe_w2]
    # Initialize output
    ref_output = torch.zeros(
        (num_local_experts, M, K), device="cuda", dtype=torch.bfloat16
    )
    # Compute reference output
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = expert_num_tokens[local_expert_id].item()
        expert_x_local = expert_x[local_expert_id, :num_actual_tokens, :]
        w1_local = w1[local_expert_id]
        w2_local = w2[local_expert_id]
        workspace1 = expert_x_local @ w1_local.transpose(0, 1)
        gate = workspace1[..., N // 2 :].to(torch.float32)
        up = workspace1[..., : N // 2]
        gate = gate * (1.0 / (1.0 + torch.exp(-gate)))
        workspace2 = (gate * up).to(torch.bfloat16)
        ref_output[local_expert_id, :num_actual_tokens, :] = (
            workspace2 @ w2_local.transpose(0, 1)
        )
    return ref_output


def make_deepgemm_masked_test_config(
    use_fp8: bool,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    ep_size: int,
    max_generate_batch_size: int,
    enable_peo_level: int = 0,
    num_peo_rounds: int = 2,
    deep_ep_num_sm: int = 24,
) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.hidden_size = hidden_size
    model_config.moe_inter_size = moe_intermediate_size
    model_config.expert_num = num_experts
    model_config.moe_k = 8

    parallelism_config = ParallelismConfig()
    parallelism_config.world_size = ep_size
    parallelism_config.world_rank = 0
    parallelism_config.local_world_size = ep_size
    parallelism_config.local_rank = 0
    parallelism_config.dp_size = ep_size
    parallelism_config.dp_rank = 0
    parallelism_config.tp_size = 1
    parallelism_config.tp_rank = 0
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = 0

    moe_config = MoeConfig()
    moe_config.use_deepep_moe = False
    moe_config.use_deepep_internode = False
    moe_config.use_deepep_low_latency = True
    moe_config.enable_peo_level = enable_peo_level
    moe_config.num_peo_rounds = num_peo_rounds
    moe_config.deep_ep_num_sm = deep_ep_num_sm

    quant_config = (
        FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
        if use_fp8
        else FusedMoEQuantConfig(quant_dtype=None)
    )

    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        max_generate_batch_size=max_generate_batch_size,
        quant_config=quant_config,
        enable_comm_overlap=False,
    )


def make_dispatch_events(
    enable_peo_level: int, num_peo_rounds: int
) -> List[torch.cuda.Event]:
    # For overlap_4, executor ignores dispatch events; others need a list of events.
    if enable_peo_level == 4:
        return []
    events: List[torch.cuda.Event] = []
    for _ in range(num_peo_rounds):
        ev = torch.cuda.Event()
        torch.cuda.current_stream().record_event(ev)
        events.append(ev)
    return events


def quantize_weights_fp8_per_block(
    weights: Dict[str, torch.Tensor],
    use_ue8m0: bool,
) -> None:
    w1 = weights[W.moe_w1]
    w2 = weights[W.moe_w2]
    E, N, K = w1.shape
    assert w2.shape[0] == E and w2.shape[1] == K and w2.shape[2] == N // 2
    assert N % 128 == 0 and K % 128 == 0 and (N // 2) % 128 == 0

    w1_fp8 = torch.empty((E, N, K), device="cuda", dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty((E, K, N // 2), device="cuda", dtype=torch.float8_e4m3fn)
    s1 = torch.empty((E, N // 128, K // 128), device="cuda", dtype=torch.float32)
    s2 = torch.empty((E, K // 128, (N // 2) // 128), device="cuda", dtype=torch.float32)

    for i in range(E):
        w1_fp8[i], s1[i] = per_block_cast_to_fp8(w1[i], use_ue8m0=use_ue8m0)
        w2_fp8[i], s2[i] = per_block_cast_to_fp8(w2[i], use_ue8m0=use_ue8m0)

    weights[W.moe_w1] = w1_fp8
    weights[W.moe_w2] = w2_fp8
    weights[W.moe_s1] = s1
    weights[W.moe_s2] = s2


def quantize_payload_fp8_per_token_group(
    payload: ExpertForwardPayload,
    use_e8m0_scale: bool,
) -> None:
    # NOTE:
    # `sgl_per_token_group_quant_fp8` expects a 2D tensor; passing a 3D [E, M, K]
    # tensor will hit `create_per_token_group_quant_fp8_output_scale(...).permute(-1, -2)`
    # which only works for 2D tensors.
    x_bf16 = payload.expert_x.contiguous()
    assert (
        x_bf16.dim() == 3
    ), f"expected expert_x to be [E, M, K], got {tuple(x_bf16.shape)}"
    E, M, K = x_bf16.shape
    assert K % 128 == 0

    x_fp8 = torch.empty((E, M, K), device=x_bf16.device, dtype=torch.float8_e4m3fn)
    if use_e8m0_scale:
        assert (K // 128) % 4 == 0, f"UE8M0 requires K/128 divisible by 4, got K={K}"
        # DeepGEMM UE8M0 scale expects an MN-major (TMA-aligned) packed layout.
        # If we allocate a plain contiguous [E, M, K//128//4] tensor and copy into it,
        # we lose the required stride pattern and DeepGEMM will assert at runtime.
        k_packed = ((K // 128) + 3) // 4
        x_scale = torch.empty(
            (E, k_packed, M), device=x_bf16.device, dtype=torch.int32
        ).transpose(1, 2)
        # Match DeepGEMM layout invariant: stride(E) == stride(last) * size(last)
        assert x_scale.stride(-3) == x_scale.stride(-1) * x_scale.size(-1)
    else:
        x_scale = torch.empty(
            (E, K // 128, M), device=x_bf16.device, dtype=torch.float32
        ).transpose(1, 2)

    for i in range(E):
        q, s = sgl_per_token_group_quant_fp8(
            x_bf16[i],
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=use_e8m0_scale,
        )
        x_fp8[i] = q
        x_scale[i].copy_(s)

    payload.expert_x = x_fp8
    payload.expert_x_scale = x_scale


def compare_with_ref_per_expert(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    diff_th: float,
) -> None:
    for i, num_token in enumerate(expert_num_tokens.tolist()):
        num_token = int(num_token)
        if num_token <= 0:
            continue
        diff = calc_diff(output[i, :num_token], ref_output[i, :num_token])
        assert (
            diff < diff_th
        ), f"diff too large at expert {i}: {diff} >= {diff_th}, output={output[i, :num_token]}, ref_output={ref_output[i, :num_token]}"
