import random
import unittest
from typing import Dict, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
    DeepGemmContinousExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.fused_moe_executor_test_util import (
    generate_payload_and_weights,
    generate_ref_output,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


class DeepGemmContinousExecutorTestBase:
    DP_SIZE = 4
    TP_SIZE = 1
    EP_SIZE = 4
    NUM_EXPERTS = 128
    MAX_GENERATE_BATCH_SIZE = 128
    HIDDEN_SIZE = 2048
    MOE_INTERMEDIATE_SIZE = 768

    M = (MAX_GENERATE_BATCH_SIZE + TP_SIZE - 1) // TP_SIZE * EP_SIZE
    K = HIDDEN_SIZE
    N = MOE_INTERMEDIATE_SIZE * 2

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)

    def _generate_config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 500000
        model_config.expert_num = self.NUM_EXPERTS
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.MOE_INTERMEDIATE_SIZE

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = self.DP_SIZE * self.EP_SIZE
        parallelism_config.dp_size = self.DP_SIZE
        parallelism_config.tp_size = self.TP_SIZE
        parallelism_config.ep_size = self.EP_SIZE
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()

        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=self.MAX_GENERATE_BATCH_SIZE,
        )

    def test_deepep_normal_executor(self, use_fp8: bool = True):
        # generate data
        config = self._generate_config()
        payload, weights = generate_payload_and_weights(config)
        # generate ref output
        ref_output = generate_ref_output(config, payload, weights)
        # create executor
        num_local_experts = config.expert_num // config.ep_size
        torch_dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16

        if use_fp8:
            if is_deep_gemm_e8m0_used():
                payload.expert_x_scale = torch.empty(
                    (num_local_experts, self.M, self.K // 128 // 4),
                    device="cuda",
                    dtype=torch.int,
                )
            else:
                payload.expert_x_scale = torch.empty(
                    (num_local_experts, self.M, self.K // 128),
                    device="cuda",
                    dtype=torch.float32,
                )
            weights[W.moe_s1] = torch.empty(
                (num_local_experts, self.N // 128, self.K // 128),
                device="cuda",
                dtype=torch.float32,
            )
            weights[W.moe_s2] = torch.empty(
                (num_local_experts, self.K // 128, self.N // 2 // 128),
                device="cuda",
                dtype=torch.float32,
            )
            new_expert_x = torch.zeros(
                (num_local_experts, self.M, self.K), device="cuda", dtype=torch_dtype
            )
            new_w1 = torch.zeros(
                (num_local_experts, self.N, self.K), device="cuda", dtype=torch_dtype
            )
            new_w2 = torch.zeros(
                (num_local_experts, self.K, self.N // 2),
                device="cuda",
                dtype=torch_dtype,
            )

            for i in range(num_local_experts):
                new_expert_x[i], payload.expert_x_scale[i] = (
                    sgl_per_token_group_quant_fp8(
                        payload.expert_x[i],
                        128,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        # scale_ue8m0=True,
                        scale_ue8m0=is_deep_gemm_e8m0_used(),
                    )
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

        executor = DeepGemmContinousExecutor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn if use_fp8 else None,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128] if use_fp8 else None,
            ),
            weights,
        )
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        payload.expert_x = torch.cat(
            [
                payload.expert_x[i, :num_token]
                for i, num_token in enumerate(expert_num_tokens)
            ],
            dim=0,
        )
        if use_fp8:
            payload.expert_x_scale = torch.cat(
                [
                    payload.expert_x_scale[i, :num_token]
                    for i, num_token in enumerate(expert_num_tokens)
                ],
                dim=0,
            )
        payload.expert_topk_ids = torch.cat(
            [
                payload.expert_topk_ids[i, :num_token]
                for i, num_token in enumerate(expert_num_tokens)
            ],
            dim=0,
        )
        payload.expert_topk_weights = torch.cat(
            [
                payload.expert_topk_weights[i, :num_token]
                for i, num_token in enumerate(expert_num_tokens)
            ],
            dim=0,
        )
        # execute
        combine_payload = executor.execute(payload, "silu", None, None, False, None)
        token_idx = 0
        for i, num_token in enumerate(expert_num_tokens):
            diff = calc_diff(
                combine_payload.fused_expert_output[token_idx : token_idx + num_token],
                ref_output[i, :num_token],
            )
            # print('diff:', diff, output[token_idx : token_idx + num_token], ref_output[i, :num_token])
            token_idx += num_token
            assert diff < 0.0022


class DeepGemmContinousExecutorTestBase(
    DeepGemmContinousExecutorTestBase, unittest.TestCase
):
    pass


if __name__ == "__main__":
    unittest.main()
