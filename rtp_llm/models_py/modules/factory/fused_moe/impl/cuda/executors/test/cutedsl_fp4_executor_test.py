# SPDX-License-Identifier: Apache-2.0

import random
import unittest
from typing import Dict, Tuple, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.ops import ParallelismConfig, MoeConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
    CutedslFp4Executor,
)
from rtp_llm.utils.model_weight import W

from flashinfer import fp4_quantize
from torch.nn import functional as F
from flashinfer import scaled_fp4_grouped_quantize


class CutedslFp4ExecutorTestBase:
    DP_SIZE = 1
    TP_SIZE = 1
    EP_SIZE = 1
    NUM_EXPERTS = 128
    BATCH_SIZE = 128
    MAX_GENERATE_BATCH_SIZE = 128
    HIDDEN_SIZE = 4096
    MOE_INTERMEDIATE_SIZE = 1536
    TOP_K = 8  # Number of experts per token

    M = (MAX_GENERATE_BATCH_SIZE + TP_SIZE - 1) // TP_SIZE * EP_SIZE
    K = HIDDEN_SIZE
    N = MOE_INTERMEDIATE_SIZE * 2

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = 448.0

    kE2M1ToFloat = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )

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
        model_config.moe_k = self.TOP_K
        
        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = self.DP_SIZE * self.EP_SIZE
        parallelism_config.dp_size = self.DP_SIZE
        parallelism_config.tp_size = self.TP_SIZE
        parallelism_config.ep_size = self.EP_SIZE
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_rank = 0
        parallelism_config.local_world_size = 1
        
        moe_config = MoeConfig()
        
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=self.MAX_GENERATE_BATCH_SIZE,
        )

    def _compute_routing(self, router_logits: torch.Tensor, top_k: int):
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.float()
        return routing_weights, selected_experts

    def _prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts: int,
        topk: int,
    ):
        routing_weights, topk_idx = self._compute_routing(router_logits, topk)

        masked_m = []
        for i in range(num_experts):
            mask = topk_idx.view(-1) == i
            masked_m.append(mask.sum())

        masked_m = torch.tensor(masked_m, dtype=torch.int32)
        hidden_states_3d = torch.empty(
            (num_experts, max(masked_m), hidden_states.shape[1]), dtype=hidden_states.dtype
        )
        for i in range(num_experts):
            hidden_states_3d[i, : masked_m[i], :] = hidden_states[topk_idx.view(-1) == i]

        return hidden_states_3d, masked_m, topk_idx, routing_weights

    def _convert_swizzled_to_linear(self, a_sf_swizzled: torch.Tensor, m, k, block_size):
        m_tiles = (m + 128 - 1) // 128
        f = block_size * 4
        k_tiles = (k + f - 1) // f
        tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
        tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
        out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
        return out[0:m, 0:k]

    def _dequantize_nvfp4_to_dtype(
        self, tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
    ):
        assert tensor_fp4.dtype == torch.uint8
        m, packed_k = tensor_fp4.shape
        k = packed_k * 2
        tensor_f32 = self._break_fp4_bytes(tensor_fp4, dtype)
        tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
        tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
        tensor_sf = self._convert_swizzled_to_linear(tensor_sf, m, k, block_size)
        tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
        out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
        return out.to(dtype=dtype)

    def _break_fp4_bytes(self, a, dtype):
        assert a.dtype == torch.uint8
        m, n = a.shape
        a_flat = a.flatten()
        high = (a_flat & 0xF0) >> 4
        low = a_flat & 0x0F
        combined = torch.stack((low, high), dim=1).flatten()
        signs = (combined & 0x08).to(torch.bool)
        abs_vals = (combined & 0x07).to(torch.long)
        kE2M1 = self.kE2M1ToFloat.to(device=a.device)
        values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
        return values.reshape(m, n * 2).to(dtype=dtype)

    def _torch_moe_nvfp4(self, a, w1, w2, topk, topk_weight, topk_ids):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                m = w1[i].shape[0]
                assert m % 2 == 0
                w3_expert, w1_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
                inter = F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())
                inter_gs = torch.tensor(1.0).cuda()
                inter_q, inter_blockscale = fp4_quantize(inter, inter_gs)
                inter = self._dequantize_nvfp4_to_dtype(
                    inter_q,
                    inter_blockscale,
                    inter_gs,
                    dtype=inter.dtype,
                    device=inter.device,
                    block_size=16,
                ).cuda()
                out[mask] = inter @ w2[i].transpose(0, 1)
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)

    def _generate_payload_and_weights(
        self,
        config: MoEConfigAdapter,
    ) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_local_experts = config.expert_num // config.ep_size
        num_tokens = self.BATCH_SIZE
        
        routing_logits = torch.rand(num_tokens, config.expert_num, device="cuda").to(torch.bfloat16)
        hidden_states = torch.randn(num_tokens, self.K, device="cuda").to(torch.bfloat16) * 0.1
        
        hidden_states_expanded = (
            hidden_states.view(num_tokens, -1, self.K)
            .repeat(1, self.TOP_K, 1)
            .reshape(-1, self.K)
        )
        
        hidden_states_3d, masked_m, topk_idx, routing_weights = self._prepare_inputs(
            hidden_states_expanded, routing_logits, config.expert_num, self.TOP_K
        )
        expert_x = hidden_states_3d[:num_local_experts].to("cuda")
        expert_num_tokens = masked_m[:num_local_experts].to("cuda").to(torch.int32)

        payload = ExpertForwardPayload(
            expert_x=expert_x.contiguous(),
            expert_x_origin_dtype=torch.bfloat16,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=None,
            ),
        )
        
        intermediate_size = self.MOE_INTERMEDIATE_SIZE
        w1_bf16 = (
            torch.randn(num_local_experts, 2 * intermediate_size, self.K, device="cuda")
            .to(torch.bfloat16)
            * 0.1
        )
        w2_bf16 = (
            torch.randn(num_local_experts, self.K, intermediate_size, device="cuda")
            .to(torch.bfloat16)
            * 0.1
        )
        
        w1_amax = w1_bf16.abs().amax(dim=(1, 2)).to(torch.float32).to(w1_bf16.device)
        w2_amax = w2_bf16.abs().amax(dim=(1, 2)).to(torch.float32).to(w2_bf16.device)
        
        input_global_scale = torch.ones(
            (num_local_experts,), dtype=torch.float32, device="cuda"
        )
        a2_global_scale = torch.ones(
            (num_local_experts,), dtype=torch.float32, device="cuda"
        )
        
        w1_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / w1_amax
        w2_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / w2_amax
        w1_fp4, w1_blockscale = scaled_fp4_grouped_quantize(
            w1_bf16,
            torch.ones(num_local_experts, dtype=torch.int32, device=w1_bf16.device)
            * 2
            * intermediate_size,
            w1_global_scale,
        )
        w2_fp4, w2_blockscale = scaled_fp4_grouped_quantize(
            w2_bf16,
            torch.ones(num_local_experts, dtype=torch.int32, device=w2_bf16.device)
            * self.K,
            w2_global_scale,
        )
        
        w1_quantized = w1_fp4.permute(2, 0, 1)
        w2_quantized = w2_fp4.permute(2, 0, 1)

        weights = {
            W.moe_w1: w1_quantized,
            W.moe_w2: w2_quantized,
            W.moe_s1: w1_blockscale,
            W.moe_s2: w2_blockscale,
            W.moe_w1_s2: 1 / w1_global_scale,
            W.moe_w2_s2: 1 / w2_global_scale,
            W.moe_w1_i_s: input_global_scale,
            W.moe_w2_i_s: a2_global_scale,
        }
        return payload, weights, w1_bf16, w2_bf16, topk_idx, routing_weights, hidden_states, w1_global_scale, w2_global_scale

    def _generate_ref_output(
        self,
        hidden_states: torch.Tensor,
        w1_bf16: torch.Tensor,
        w2_bf16: torch.Tensor,
        topk_idx: torch.Tensor,
        routing_weights: torch.Tensor,
        topk: int,
        input_global_scale: torch.Tensor,
        w1_global_scale: torch.Tensor,
        w2_global_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Dequantize inputs
        a_fp4, a_scale_interleaved = fp4_quantize(hidden_states, input_global_scale[0])
        a_in_dtype = self._dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            input_global_scale[0],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            block_size=16,
        )
        
        num_experts = w1_bf16.shape[0]
        w1_d = torch.empty(
            (num_experts, w1_bf16.shape[1], w1_bf16.shape[2]),
            device=w1_bf16.device,
            dtype=w1_bf16.dtype,
        )
        w2_d = torch.empty(
            (num_experts, w2_bf16.shape[1], w2_bf16.shape[2]),
            device=w2_bf16.device,
            dtype=w2_bf16.dtype,
        )
        
        for idx in range(num_experts):
            w1_fp4_sliced, w1_blockscale_sliced = fp4_quantize(
                w1_bf16[idx], w1_global_scale[idx]
            )
            w2_fp4_sliced, w2_blockscale_sliced = fp4_quantize(
                w2_bf16[idx], w2_global_scale[idx]
            )
            w1_d[idx] = self._dequantize_nvfp4_to_dtype(
                w1_fp4_sliced,
                w1_blockscale_sliced,
                w1_global_scale[idx],
                dtype=w1_bf16.dtype,
                device=w1_bf16.device,
                block_size=16,
            )
            w2_d[idx] = self._dequantize_nvfp4_to_dtype(
                w2_fp4_sliced,
                w2_blockscale_sliced,
                w2_global_scale[idx],
                dtype=w2_bf16.dtype,
                device=w2_bf16.device,
                block_size=16,
            )
        ref_output = self._torch_moe_nvfp4(
            a_in_dtype,
            w1_d,
            w2_d,
            topk,
            routing_weights.to(a_in_dtype.device),
            topk_idx.to(a_in_dtype.device),
        )
        
        return ref_output

    def _filter_valid_tokens(
        self, output: torch.Tensor, expert_num_tokens: torch.Tensor
    ) -> torch.Tensor:
        num_local_experts = output.shape[0]
        for expert_id in range(num_local_experts):
            num_valid_tokens = expert_num_tokens[expert_id].item()
            if num_valid_tokens < output.shape[1]:
                output[expert_id, num_valid_tokens:, :] = 0
        return output

    def test_cutedsl_fp4_executor(self):
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("Nvfp4 Requires compute capability of 10 or above.")
        
        config = self._generate_config()
        payload, weights, w1_bf16, w2_bf16, topk_idx, routing_weights, hidden_states, w1_global_scale, w2_global_scale = self._generate_payload_and_weights(config)
        
        executor = CutedslFp4Executor(
            config,
            weights,
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[16, 16],
            ),
        )
        output = executor.execute(payload, "silu", None, None, False, None)
        
        input_global_scale = weights[W.moe_w1_i_s]
        ref_output = self._generate_ref_output(
            hidden_states,
            w1_bf16,
            w2_bf16,
            topk_idx,
            routing_weights,
            self.TOP_K,
            input_global_scale,
            w1_global_scale,
            w2_global_scale,
        )
        
        output = self._filter_valid_tokens(output, payload.expert_tokens_meta.expert_num_tokens)
        
        num_local_experts = self.NUM_EXPERTS // self.EP_SIZE
        num_tokens = self.BATCH_SIZE
        output_aggregated = torch.zeros(
            num_tokens, self.K, device=output.device, dtype=output.dtype
        )
        expert_positions = torch.zeros(num_local_experts, dtype=torch.long, device="cuda")
        
        for batch_idx in range(num_tokens):
            for k_pos in range(self.TOP_K):
                expert_id = topk_idx[batch_idx, k_pos].item()
                if expert_id < num_local_experts:
                    weight = routing_weights[batch_idx, k_pos].item()
                    expert_pos = expert_positions[expert_id].item()
                    if expert_pos < payload.expert_tokens_meta.expert_num_tokens[expert_id]:
                        output_aggregated[batch_idx] += (
                            output[expert_id, expert_pos, :] * weight
                        )
                        expert_positions[expert_id] += 1
        
        print("output_aggregated shape: ", output_aggregated.shape)
        print("ref_output shape: ", ref_output.shape)
        print("output_aggregated: ", output_aggregated)
        print("ref_output: ", ref_output)
        print("diff: ", (output_aggregated - ref_output).abs().max())
        print("diff mean: ", (output_aggregated - ref_output).abs().mean())
        torch.testing.assert_close(output_aggregated, ref_output, atol=5e-2, rtol=5e-2)


class CutedslFp4ExecutorTest(CutedslFp4ExecutorTestBase, unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
