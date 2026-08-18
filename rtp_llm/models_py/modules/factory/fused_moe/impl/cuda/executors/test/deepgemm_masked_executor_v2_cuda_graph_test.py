import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_per_block import (
    CudaFp8PerBlockNoDPMaskedStrategy,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class DeepGemmMaskedExecutorV2CudaGraphTest(unittest.TestCase):
    NUM_EXPERTS = 4
    TOP_K = 2
    NUM_TOKENS = 64
    # ep_gather uses a fixed 512-element block and requires the hidden
    # dimension to be an exact multiple of it.  The target model uses 1024;
    # keep the unit-test fixture small while preserving that kernel contract.
    HIDDEN_SIZE = 512
    INTERMEDIATE_SIZE = 512

    def setUp(self) -> None:
        torch.manual_seed(20260820)
        torch.cuda.manual_seed(20260820)
        self.config = self._make_config()
        self.fused_moe = self._make_fused_moe()

    def _make_config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.quant_config = Fp8BlockWiseQuantConfig()
        model_config.data_type = "bf16"
        model_config.expert_num = self.NUM_EXPERTS
        model_config.moe_k = self.TOP_K
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.INTERMEDIATE_SIZE

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.moe_strategy = "fp8_per_block_no_dp_masked"
        moe_config.use_all_gather = True
        moe_config.use_deepep_moe = False
        moe_config.use_deepep_internode = False
        moe_config.use_deepep_low_latency = False

        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=True,
        )

    def _make_fused_moe(self) -> FusedMoe:
        gate_up_size = self.INTERMEDIATE_SIZE * 2
        gate_up = torch.randn(
            (self.NUM_EXPERTS, gate_up_size, self.HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        down = torch.randn(
            (self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )

        gate_up_fp8 = []
        gate_up_scales = []
        down_fp8 = []
        down_scales = []
        for expert_id in range(self.NUM_EXPERTS):
            quantized, scales = per_block_cast_to_fp8(
                gate_up[expert_id], use_ue8m0=False
            )
            gate_up_fp8.append(quantized)
            gate_up_scales.append(scales)
            quantized, scales = per_block_cast_to_fp8(down[expert_id], use_ue8m0=False)
            down_fp8.append(quantized)
            down_scales.append(scales)

        weights = {
            W.moe_w1: torch.stack(gate_up_fp8),
            W.moe_s1: torch.stack(gate_up_scales),
            W.moe_w2: torch.stack(down_fp8),
            W.moe_s2: torch.stack(down_scales),
        }
        if is_deep_gemm_e8m0_used():
            weights[W.moe_w1], weights[W.moe_s1] = requant_weight_ue8m0(
                weights[W.moe_w1], weights[W.moe_s1]
            )
            weights[W.moe_w2], weights[W.moe_s2] = requant_weight_ue8m0(
                weights[W.moe_w2], weights[W.moe_s2]
            )

        strategy = CudaFp8PerBlockNoDPMaskedStrategy()
        return FusedMoe(
            strategy.create_router(self.config),
            strategy.create_executor(self.config, weights),
            expert_num=self.NUM_EXPERTS,
        )

    def _make_inputs(self, offset: int) -> tuple[torch.Tensor, ...]:
        hidden_states = torch.randn(
            (self.NUM_TOKENS, self.HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
        token_ids = torch.arange(self.NUM_TOKENS, device="cuda")
        topk_ids = torch.stack(
            (
                (token_ids + offset) % self.NUM_EXPERTS,
                (token_ids + offset + 1) % self.NUM_EXPERTS,
            ),
            dim=1,
        ).to(torch.int64)
        topk_weights = torch.rand(
            (self.NUM_TOKENS, self.TOP_K), device="cuda", dtype=torch.float32
        )
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)
        return hidden_states, topk_weights, topk_ids

    def test_replay_with_updated_hidden_states_and_routing(self) -> None:
        static_hidden, static_weights, static_ids = self._make_inputs(offset=0)

        # Warm DeepGEMM/Triton compilation outside capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            self.fused_moe(static_hidden, static_weights, static_ids)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self.fused_moe(static_hidden, static_weights, static_ids)

        replay_hidden, replay_weights, replay_ids = self._make_inputs(offset=2)
        static_hidden.copy_(replay_hidden)
        static_weights.copy_(replay_weights)
        static_ids.copy_(replay_ids)
        graph.replay()
        captured = graph_output.clone()

        eager = self.fused_moe(static_hidden, static_weights, static_ids)
        torch.testing.assert_close(captured, eager, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
