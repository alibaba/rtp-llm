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
        select_topk_op = SelectTopkOp(model_config, fake_balance_expert=False, dp_rank=0)

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


if __name__ == "__main__":
    main()
