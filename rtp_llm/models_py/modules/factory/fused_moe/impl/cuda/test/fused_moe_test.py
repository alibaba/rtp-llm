import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
    BatchedTritonExperts,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
    BatchedDataRouter,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig, RuntimeConfig


def torch_sparse_block_forward(
    hidden_states: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_ids: torch.Tensor,
):
    sequence_length = hidden_states.shape[0]
    num_experts = up_proj.shape[0]
    hidden_dim = down_proj.shape[1]
    inter_dim = down_proj.shape[2]

    final_hidden_states = torch.zeros(
        (sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = F.one_hot(expert_ids.long(), num_classes=num_experts).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        # in torch it is faster to index using lists than torch tensors
        top_x_list = top_x.tolist()
        idx_list = idx.tolist()

        routing_weight = (
            # routing_weights [num_tokens, top_k]
            routing_weights[top_x_list, idx_list, None]
        )

        current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

        up_proj_x, up_proj_g = torch.split(up_proj[expert_idx], inter_dim, dim=0)

        current_hidden_states = F.silu(F.linear(current_state, up_proj_g)) * F.linear(
            current_state, up_proj_x
        )
        current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
        current_hidden_states = current_hidden_states * routing_weight

        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(hidden_states.dtype)
        )

    final_hidden_states = final_hidden_states.reshape(sequence_length, hidden_dim)
    return final_hidden_states


class FusedMoeBatchedTest(TestCase):
    # Test parameters
    DTYPES = [torch.float16, torch.bfloat16]
    NUM_TOKENS = [64, 256, 1024]
    HIDDEN_SIZES = [256, 512, 1024]
    NUM_EXPERTS = [4, 8]
    TOP_K = [2, 4]
    INTER_SIZES = [512, 1024, 2048]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_moe_batched_test(
        self,
        num_tokens: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        inter_size: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        # Create model configuration objects
        model_config = ModelConfig()
        model_config.attn_config.head_num = 4
        model_config.attn_config.size_per_head = 64
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 32000
        model_config.hidden_size = hidden_size
        model_config.expert_num = num_experts
        model_config.moe_k = top_k
        model_config.inter_size = (
            inter_size  # Use inter_size instead of moe_inter_padding_size
        )
        model_config.has_moe_norm = True
        model_config.activation_type = "SiGLU"

        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 1
        parallelism_config.ep_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.tp_rank = 0
        parallelism_config.dp_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.world_size = 1
        parallelism_config.world_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        runtime_config = RuntimeConfig()
        runtime_config.max_generate_batch_size = num_tokens

        # Create router and experts
        router = BatchedDataRouter(
            max_num_tokens=num_tokens,
            num_local_experts=num_experts // parallelism_config.ep_size,
            ep_rank=parallelism_config.ep_rank,
            tp_size=parallelism_config.tp_size,
        )
        scaling_factor = 0.1
        # Create test weights
        w1 = (
            torch.randn(
                num_experts, inter_size * 2, hidden_size, dtype=dtype, device="cuda"
            )
            * scaling_factor
        )
        w2 = (
            torch.randn(
                num_experts, hidden_size, inter_size, dtype=dtype, device="cuda"
            )
            * scaling_factor
        )

        experts = BatchedTritonExperts(
            max_num_tokens=num_tokens, num_dispatchers=1, w1=w1, w2=w2
        )

        fused_moe = FusedMoe(router, experts, num_experts)

        hidden_states = (
            torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
            * scaling_factor
        )

        # Create routing weights and ids
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, dtype=torch.float32, device="cuda"), dim=1
        )
        gating_logits = torch.randn(num_tokens, num_experts, device="cuda")
        _, topk_ids = torch.topk(gating_logits, top_k, dim=1)

        # Run forward pass
        output = fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )

        # Compute reference output using torch_sparse_block_forward
        ref_output = torch_sparse_block_forward(
            hidden_states, w1, w2, topk_weights, topk_ids
        )

        # Verify output shape
        self.assertEqual(output.shape, (num_tokens, hidden_size))
        self.assertEqual(ref_output.shape, (num_tokens, hidden_size))

        # Compare outputs
        self.assertTrue(torch.allclose(output, ref_output, atol=1e-1, rtol=1e-1))

    def test_fused_moe_batched(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.NUM_EXPERTS,
            self.TOP_K,
            self.INTER_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                num_experts=params[2],
                top_k=params[3],
                inter_size=params[4],
                dtype=params[5],
            ):
                self._run_fused_moe_batched_test(*params)

    def test_fused_moe_batched_edge_cases(self):
        """Test FusedMoe with edge cases."""
        # Test with small tensors
        self._run_fused_moe_batched_test(1, 32, 2, 1, 64, torch.float16)

        # Test with large tensors
        self._run_fused_moe_batched_test(2048, 1024, 8, 4, 2048, torch.float16)

        # Test with different dtypes
        self._run_fused_moe_batched_test(128, 256, 4, 2, 512, torch.bfloat16)


if __name__ == "__main__":
    main()
