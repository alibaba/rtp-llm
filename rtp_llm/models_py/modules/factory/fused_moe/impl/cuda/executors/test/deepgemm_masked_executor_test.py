import random
from typing import Dict, Tuple

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
    DeepGemmMaskedExecutor,
)
from rtp_llm.utils.model_weight import W

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


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


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
    use_fp8: bool,
) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor]]:
    torch_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16
    # generate payload
    num_local_experts = config.expert_num // config.ep_size
    expert_x = torch.zeros((num_local_experts, M, K), device="cuda", dtype=torch_dtype)
    expert_num_tokens = torch.zeros(
        (num_local_experts,), device="cuda", dtype=torch.int32
    )
    expert_x_scale = None
    if use_fp8:
        expert_x_scale = torch.zeros(
            (num_local_experts, M, K // 128), device="cuda", dtype=torch.float32
        )
    for local_expert_id in range(num_local_experts):
        num_actual_tokens = max(
            min(int(NUM_EXPERTS * DP_SIZE * random.uniform(0.7, 1.3)), M), 1
        )
        expert_x[local_expert_id, :num_actual_tokens, :] = (
            torch.rand((num_actual_tokens, K), device="cuda", dtype=torch.float32).to(
                torch_dtype
            )
            * 2
            - 1
        )
        if use_fp8:
            expert_x_scale[local_expert_id, :num_actual_tokens, :] = torch.randn(
                (num_actual_tokens, K // 128), device="cuda", dtype=torch.float32
            )
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
    # generate weights
    w1_scale = None
    w2_scale = None
    if use_fp8:
        w1_scale = torch.randn(
            (num_local_experts, N // 128, K // 128), device="cuda", dtype=torch.float32
        )
        w2_scale = torch.randn(
            (num_local_experts, K // 128, N // 2 // 128),
            device="cuda",
            dtype=torch.float32,
        )
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


def _generate_ref_output(
    payload: ExpertForwardPayload, weights: Dict[str, torch.Tensor], use_fp8: bool
) -> torch.Tensor:
    num_local_experts = NUM_EXPERTS // EP_SIZE
    expert_x = payload.expert_x
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    w1 = weights[W.moe_w1]
    w2 = weights[W.moe_w2]
    if use_fp8:
        expert_x_scale = payload.expert_x_scale
        w1_scale = weights[W.moe_s1]
        w2_scale = weights[W.moe_s2]
        raise NotImplementedError("FP8 not implemented")
    else:
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


def test_deepgemm_masked_executor(use_fp8: bool):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    # generate data
    config = _generate_config()
    payload, weights = _generate_payload_and_weights(config, use_fp8=False)
    # generate ref output
    ref_output = _generate_ref_output(payload, weights, use_fp8=False)
    # create executor
    num_local_experts = config.expert_num // config.ep_size
    torch_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16

    if use_fp8:
        payload.expert_x_scale = torch.empty(
            (num_local_experts, M, K // 128), device="cuda", dtype=torch.float32
        )
        weights[W.moe_s1] = torch.empty(
            (num_local_experts, N // 128, K // 128), device="cuda", dtype=torch.float32
        )
        weights[W.moe_s2] = torch.empty(
            (num_local_experts, K // 128, N // 2 // 128),
            device="cuda",
            dtype=torch.float32,
        )
        new_expert_x = torch.zeros(
            (num_local_experts, M, K), device="cuda", dtype=torch_dtype
        )
        new_w1 = torch.zeros(
            (num_local_experts, N, K), device="cuda", dtype=torch_dtype
        )
        new_w2 = torch.zeros(
            (num_local_experts, K, N // 2), device="cuda", dtype=torch_dtype
        )

        for i in range(num_local_experts):
            new_expert_x[i], payload.expert_x_scale[i] = per_token_cast_to_fp8(
                payload.expert_x[i], use_ue8m0=is_deep_gemm_e8m0_used()
            )
            new_w1[i], weights[W.moe_s1][i] = per_block_cast_to_fp8(
                weights[W.moe_w1][i], use_ue8m0=is_deep_gemm_e8m0_used()
            )
            new_w2[i], weights[W.moe_s2][i] = per_block_cast_to_fp8(
                weights[W.moe_w2][i], use_ue8m0=is_deep_gemm_e8m0_used()
            )
        payload.expert_x = new_expert_x
        weights[W.moe_w1] = new_w1
        weights[W.moe_w2] = new_w2

    executor = DeepGemmMaskedExecutor(
        config,
        weights,
        (
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128],
            )
            if use_fp8
            else FusedMoEQuantConfig(quant_dtype=None)
        ),
    )
    # execute
    output = executor.execute(payload, "silu", None, None, False, None)
    expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
    for i, num_token in enumerate(expert_num_tokens):
        diff = calc_diff(output[i, :num_token], ref_output[i, :num_token])
        # print('diff:', diff)
        assert diff < 0.002


if __name__ == "__main__":
    test_deepgemm_masked_executor(use_fp8=True)
    if not is_deep_gemm_e8m0_used():
        test_deepgemm_masked_executor(use_fp8=False)
