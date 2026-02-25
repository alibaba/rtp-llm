import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype
from torch.profiler import ProfilerActivity, profile

from rtp_llm.config.model_config import ModelConfig

from rtp_llm.ops.compute_ops import SelectTopkOp  # isort:skip


class SelectTopkOpTest(TestCase):
    # DTYPES = [torch.float32, torch.float16]
    # NUM_TOKENS = [7, 83, 4096, 5120]
    # NUM_EXPERT = [128]
    # TOP_K = [2, 5, 10, 32, 128]

    DTYPES = [torch.float32, torch.bfloat16, torch.float16]
    NUM_TOKENS = [10]
    NUM_EXPERT = [128]
    TOP_K = [16]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_select_topk_op_test(
        self, num_tokens: int, num_expert: int, top_k: int, dtype: _dtype
    ):
        torch.manual_seed(1)
        model_config = ModelConfig()
        model_config.attn_config.head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 1
        model_config.max_seq_len = 1
        model_config.vocab_size = 5120
        model_config.expert_num = num_expert
        model_config.moe_k = top_k
        model_config.has_moe_norm = True
        # Use default values for fake_balance_expert and dp_rank
        select_topk_op = SelectTopkOp(
            model_config, fake_balance_expert=False, dp_rank=0
        )

        router_logits = torch.randn(num_tokens, num_expert, dtype=dtype).to("cuda")
        router_logits_fp32 = router_logits.float()
        op_routing_weights = torch.zeros(
            (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
        )
        op_selected_experts = torch.zeros(
            (num_tokens, top_k), dtype=torch.int32, device=router_logits.device
        )

        for _ in range(5):
            select_topk_op.forward(
                router_logits_fp32, op_selected_experts, op_routing_weights
            )

            torch_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            torch_routing_weights, torch_selected_experts = torch.topk(
                torch_routing_weights, top_k, dim=-1
            )
            torch_selected_experts = torch_selected_experts.int()
            if top_k != 1:
                torch_routing_weights /= torch_routing_weights.sum(dim=-1, keepdim=True)
            print(f"op_routing_weights = {op_routing_weights}")
            print(f"torch_routing_weights = {torch_routing_weights}")
            print(f"op_selected_experts = {op_selected_experts}")
            print(f"torch_selected_experts = {torch_selected_experts}")
            # self.assertTrue(torch.allclose(op_routing_weights, torch_routing_weights, atol=1e-1, rtol=1e-1))
            # self.assertTrue(torch.allclose(op_selected_experts, torch_selected_experts, atol=1e-1, rtol=1e-1))

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(10):
                select_topk_op.forward(
                    router_logits_fp32, op_selected_experts, op_routing_weights
                )

                torch_routing_weights = F.softmax(
                    router_logits, dim=1, dtype=torch.float
                )
                torch_routing_weights, torch_selected_experts = torch.topk(
                    torch_routing_weights, top_k, dim=-1
                )
                torch_selected_experts = torch_selected_experts.int()
                if top_k != 1:
                    torch_routing_weights /= torch_routing_weights.sum(
                        dim=-1, keepdim=True
                    )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    def test_select_topk(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.NUM_EXPERT, self.TOP_K, self.DTYPES
        ):
            with self.subTest(
                num_tokens=params[0],
                num_expert=params[1],
                top_k=params[2],
                dtype=params[3],
            ):
                self._run_select_topk_op_test(*params)

    def _run_convert_logical_to_physical_experts_test(
        self,
        num_tokens: int,
        log_exp_num: int,
        phy_exp_num: int,
        ep_rank: int,
        expert_ids_dtype: torch.dtype,
    ):
        """Test convert_logical_to_physical_experts function"""
        torch.manual_seed(42)
        model_config = ModelConfig()
        model_config.attn_config.head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 1
        model_config.max_seq_len = 1
        model_config.vocab_size = 5120
        model_config.expert_num = log_exp_num
        model_config.moe_k = 2
        model_config.has_moe_norm = True

        select_topk_op = SelectTopkOp(
            model_config, fake_balance_expert=False, dp_rank=0
        )

        # Create test data
        # expert_ids: logical expert IDs [num_tokens]
        expert_ids = torch.randint(
            0, log_exp_num, (num_tokens,), dtype=expert_ids_dtype, device="cuda"
        )
        expert_ids_original = expert_ids.clone()

        # max_exp_num = phy_exp_num - log_exp_num + 1
        max_exp_num = phy_exp_num - log_exp_num + 1

        # logic_expert_cnt: [log_exp_num]
        # For simplicity, distribute physical experts evenly
        logic_expert_cnt = torch.ones((log_exp_num,), dtype=torch.int32, device="cuda")
        # Ensure at least one physical expert per logical expert
        # and total doesn't exceed phy_exp_num
        total_phy = log_exp_num
        remaining = phy_exp_num - total_phy
        if remaining > 0:
            # Distribute remaining physical experts
            for i in range(remaining):
                logic_expert_cnt[i % log_exp_num] += 1

        # log2phy: [log_exp_num * max_exp_num]
        # Initialize with -1 (invalid)
        log2phy = torch.full(
            (log_exp_num * max_exp_num,), -1, dtype=torch.int32, device="cuda"
        )

        # Build log2phy mapping: for each logical expert, assign physical experts
        phy_id_counter = 0
        for log_exp_id in range(log_exp_num):
            cnt = logic_expert_cnt[log_exp_id].item()
            for replica_idx in range(cnt):
                if phy_id_counter < phy_exp_num:
                    idx = log_exp_id * max_exp_num + replica_idx
                    log2phy[idx] = phy_id_counter
                    phy_id_counter += 1

        # Create expected result by manually computing the conversion
        # Formula: idx = log_exp_id * max_exp_num + (i + ep_rank) % cnt
        expected_expert_ids = expert_ids_original.clone()
        for i in range(num_tokens):
            log_exp_id = int(expert_ids_original[i].item())
            cnt = int(logic_expert_cnt[log_exp_id].item())
            idx = log_exp_id * max_exp_num + (i + ep_rank) % cnt
            phy_exp_id = int(log2phy[idx].item())
            expected_expert_ids[i] = phy_exp_id

        # Call the function (modifies expert_ids in-place)
        select_topk_op.convert_logical_to_physical_experts(
            expert_ids,
            log2phy,
            logic_expert_cnt,
            log_exp_num,
            phy_exp_num,
            ep_rank,
        )

        # Synchronize to ensure kernel execution completes
        torch.cuda.synchronize()

        # Verify results
        self.assertTrue(
            torch.equal(expert_ids, expected_expert_ids),
            f"Conversion failed. Expected: {expected_expert_ids}, Got: {expert_ids}",
        )

        # Verify that all converted IDs are valid (>= 0 and < phy_exp_num)
        self.assertTrue(
            torch.all((expert_ids >= 0) & (expert_ids < phy_exp_num)),
            f"Invalid physical expert IDs found: {expert_ids}",
        )

    def test_convert_logical_to_physical_experts(self):
        """Test convert_logical_to_physical_experts with various configurations"""
        test_configs = [
            # (num_tokens, log_exp_num, phy_exp_num, ep_rank, dtype)
            (10, 8, 16, 0, torch.int32),
            (10, 8, 16, 0, torch.int64),
            (100, 16, 32, 0, torch.int32),
            (100, 16, 32, 0, torch.int64),
            (50, 8, 12, 1, torch.int32),
            (50, 8, 12, 1, torch.int64),
            (1, 4, 8, 0, torch.int32),  # Single token
            (256, 64, 128, 0, torch.int32),  # Larger scale
        ]

        for params in test_configs:
            with self.subTest(
                num_tokens=params[0],
                log_exp_num=params[1],
                phy_exp_num=params[2],
                ep_rank=params[3],
                dtype=params[4],
            ):
                self._run_convert_logical_to_physical_experts_test(*params)

    def test_convert_logical_to_physical_experts_invalid_dtype(self):
        """Test that invalid dtype raises an error"""
        model_config = ModelConfig()
        model_config.attn_config.head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 1
        model_config.max_seq_len = 1
        model_config.vocab_size = 5120
        model_config.expert_num = 8
        model_config.moe_k = 2
        model_config.has_moe_norm = True

        select_topk_op = SelectTopkOp(
            model_config, fake_balance_expert=False, dp_rank=0
        )

        # Create tensors with invalid dtype (float32 instead of int32/int64)
        expert_ids = torch.randint(0, 8, (10,), dtype=torch.float32, device="cuda")
        log2phy = torch.zeros((8 * 9,), dtype=torch.int32, device="cuda")
        logic_expert_cnt = torch.ones((8,), dtype=torch.int32, device="cuda")

        with self.assertRaises(RuntimeError) as context:
            select_topk_op.convert_logical_to_physical_experts(
                expert_ids, log2phy, logic_expert_cnt, 8, 16, 0
            )

        self.assertIn("int32 or int64", str(context.exception))


if __name__ == "__main__":
    main()
