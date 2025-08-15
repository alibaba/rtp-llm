import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe import (
    BatchedDataRouter,
    FusedMoe,
    NaiveBatchedExperts,
)


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

        # Model configuration
        model_param = GptInitModelParameters(
            head_num=4,
            size_per_head=64,
            layer_num=2,
            max_seq_len=2048,
            vocab_size=32000,
            hidden_size=hidden_size,
        )
        model_param.expert_num = num_experts
        model_param.moe_k = top_k
        model_param.moe_inter_padding_size = inter_size
        model_param.has_moe_norm = True
        model_param.activation_type = "SiGLU"
        model_param.ep_size = 1
        model_param.ep_rank = 0

        # Create router and experts
        router = BatchedDataRouter(
            max_num_tokens=num_tokens,
            num_local_experts=num_experts,
            num_dispatchers=1,
            rank=0,
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

        experts = NaiveBatchedExperts(
            max_num_tokens=num_tokens, num_dispatchers=1, w1=w1, w2=w2
        )

        fused_moe = FusedMoe(router, experts)

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
            global_num_experts=num_experts,
        )

        # Verify output shape
        self.assertEqual(output.shape, (num_tokens, hidden_size))

        # Verify output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        print(
            f"FusedMoe with BatchedDataRouter and NaiveBatchedExperts test passed. "
            f"Output shape: {output.shape}, dtype: {output.dtype}"
        )

    def test_fused_moe_batched(self):
        """Test FusedMoe with BatchedDataRouter and NaiveBatchedExperts for various configurations."""
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
        self._run_fused_moe_batched_test(128, 512, 4, 2, 1024, torch.bfloat16)


if __name__ == "__main__":
    main()
