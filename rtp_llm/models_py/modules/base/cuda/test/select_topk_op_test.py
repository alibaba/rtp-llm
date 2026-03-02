import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype
from torch.profiler import ProfilerActivity, profile

from rtp_llm.config.model_config import ModelConfig

from rtp_llm.ops.compute_ops import SelectTopkOp  # isort:skip


class SelectTopkOpTest(TestCase):
    DTYPES = [torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    NUM_EXPERT = [128]
    TOP_K = [8]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _assert_token_distribution_balanced(
        self, expert_token_counts: torch.Tensor, total_assignments: int
    ):
        # expert_token_counts: shape [num_expert], int64 on CPU or CUDA
        self.assertEqual(int(expert_token_counts.sum().item()), int(total_assignments))
        min_c = int(expert_token_counts.min().item())
        max_c = int(expert_token_counts.max().item())
        num_expert = int(expert_token_counts.numel())

        # Metric: standard deviation (population variance, unbiased=False)
        observed_std = float(
            expert_token_counts.to(torch.float32).std(unbiased=False).item()
        )

        # For integer assignments, the theoretically minimal std (most even distribution) is:
        # N=total_assignments, E=num_expert, r=N%E
        # Var_min = r*(E-r)/E^2, Std_min = sqrt(Var_min)
        r = int(total_assignments % num_expert)
        expected_min_std = ((r * (num_expert - r)) ** 0.5) / float(num_expert)

        # Allow tiny floating-point error
        eps = 1e-6
        self.assertLessEqual(
            observed_std,
            expected_min_std + eps,
            msg=f"token distribution is not even enough (std exceeds theoretical minimum): observed_std={observed_std}, "
            f"expected_min_std={expected_min_std}, r={r}, min={min_c}, max={max_c}, "
            f"total={total_assignments}, num_expert={num_expert}",
        )

        # Extra diagnostic: for the most even integer distribution, max-min must be <= 1
        self.assertLessEqual(
            max_c - min_c,
            1,
            msg=f"token distribution is imbalanced (max-min>1): min={min_c}, max={max_c}, total={total_assignments}, "
            f"num_expert={num_expert}",
        )

    def _bincount_compat(self, x: torch.Tensor, minlength: int) -> torch.Tensor:
        """
        In some PyTorch builds, `torch.bincount` might not support CUDA tensors.
        Try bincount on the original device first; on failure, fall back to CPU and move back.
        """
        x_i64 = x.to(torch.int64)
        try:
            return torch.bincount(x_i64, minlength=minlength)
        except Exception:
            return torch.bincount(x_i64.cpu(), minlength=minlength).to(x.device)

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
            model_config, fake_balance_expert=False, dp_rank=0, dp_size=1, ep_size=1
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
            # NOTE:
            # torch.topk guarantees sorted by value, but the order of indices for ties is not specified.
            # SelectTopkOp may return the same top-k set but with a different order when weights are equal.
            # Compare by (expert_id -> weight) mapping, rather than requiring identical ordering.
            op_sort_idx = torch.argsort(op_selected_experts, dim=-1)
            torch_sort_idx = torch.argsort(torch_selected_experts, dim=-1)
            op_selected_experts_sorted = torch.gather(
                op_selected_experts, 1, op_sort_idx
            )
            torch_selected_experts_sorted = torch.gather(
                torch_selected_experts, 1, torch_sort_idx
            )
            op_routing_weights_sorted = torch.gather(op_routing_weights, 1, op_sort_idx)
            torch_routing_weights_sorted = torch.gather(
                torch_routing_weights, 1, torch_sort_idx
            )
            self.assertTrue(
                torch.equal(op_selected_experts_sorted, torch_selected_experts_sorted)
            )
            self.assertTrue(
                torch.allclose(
                    op_routing_weights_sorted,
                    torch_routing_weights_sorted,
                    atol=1e-1,
                    rtol=1e-1,
                )
            )

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

    def test_select_topk_fake_balance_expert(self):
        torch.manual_seed(1)

        NUM_EXPERTS = [128]
        EP_SIZES = [1, 2, 4, 8, 16, 32, 64]
        TP_SIZES = [1, 2, 4, 8]
        BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        NUM_TOPK = [4, 8, 16]

        # Validate basic constraints first (so misconfigurations fail loudly).
        for num_expert, ep_size in itertools.product(NUM_EXPERTS, EP_SIZES):
            self.assertEqual(
                num_expert % ep_size,
                0,
                msg=f"num_expert={num_expert} must be divisible by ep_size={ep_size}",
            )

        # Cache ops by (num_expert, ep_size, tp_size, top_k) to avoid rebuilding them per batch_size.
        op_cache = {}

        for num_expert, ep_size, tp_size, top_k in itertools.product(
            NUM_EXPERTS, EP_SIZES, TP_SIZES, NUM_TOPK
        ):
            if ep_size % tp_size != 0:
                continue
            dp_size = ep_size // tp_size

            cache_key = (num_expert, ep_size, tp_size, top_k)
            select_topk_ops = op_cache.get(cache_key)
            if select_topk_ops is None:
                model_config = ModelConfig()
                model_config.attn_config.head_num = 1
                model_config.attn_config.size_per_head = 128
                model_config.num_layers = 1
                model_config.max_seq_len = 1
                model_config.vocab_size = 5120
                model_config.expert_num = num_expert
                model_config.moe_k = top_k
                model_config.has_moe_norm = True

                # Build ops for all dp_rank upfront to avoid recreating them inside batch loops
                select_topk_ops = [
                    SelectTopkOp(
                        model_config,
                        fake_balance_expert=True,
                        dp_rank=dp_rank,
                        dp_size=dp_size,
                        ep_size=ep_size,
                    )
                    for dp_rank in range(dp_size)
                ]
                op_cache[cache_key] = select_topk_ops

            for batch_size in BATCH_SIZES:
                with self.subTest(
                    num_expert=num_expert,
                    ep_size=ep_size,
                    tp_size=tp_size,
                    dp_size=dp_size,
                    batch_size=batch_size,
                    top_k=top_k,
                ):
                    # fake_balance_expert overwrites expert_ids / expert_scales;
                    # SelectTopkOp.forward still requires a router_logits input.
                    router_logits_fp32 = torch.zeros(
                        (batch_size, num_expert), dtype=torch.float32
                    )

                    global_counts = torch.zeros((num_expert,), dtype=torch.int64)
                    topk_idx = torch.empty((batch_size, top_k), dtype=torch.int32)
                    topk_w = torch.empty((batch_size, top_k), dtype=torch.float32)
                    for op in select_topk_ops:
                        op.forward(router_logits_fp32, topk_idx, topk_w)

                        # fake_balance_expert is expected to set weights to 1.0
                        self.assertTrue(bool(torch.all(topk_w == 1.0).item()))

                        # Aggregate token -> expert assignment counts across all dp_rank
                        global_counts += self._bincount_compat(
                            topk_idx.reshape(-1), minlength=num_expert
                        )

                    total_assignments = batch_size * top_k * dp_size
                    self._assert_token_distribution_balanced(
                        global_counts, total_assignments
                    )


if __name__ == "__main__":
    main()
