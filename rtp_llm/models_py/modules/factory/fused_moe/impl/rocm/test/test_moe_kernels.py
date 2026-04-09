"""Unit tests for aiter MoE kernels: moe_sorting_fwd, ck_moe_stage1, ck_moe_stage2."""

import itertools
from unittest import SkipTest, TestCase, main

import torch
from aiter.ops.shuffle import shuffle_weight
from torch import dtype as _dtype

from rtp_llm.ops import MoeConfig, ParallelismConfig


class MoEKernelsTest(TestCase):
    """Test aiter MoE kernels against torch reference."""

    DTYPES = [torch.bfloat16]
    TOKEN_NUM = [32, 64]
    HIDDEN_DIM = [256, 512]
    EXPERT_NUM = [8, 32]
    INTER_DIM = [512, 1024]
    TOP_K = [1, 4]
    BLOCK_SIZE_M = 32

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _torch_moe_ref(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        apply_router_weight: bool = True,
    ) -> torch.Tensor:
        """Reference MoE forward: w1(gate)*w1(up) -> silu -> w2 -> sum(topk)."""
        import torch.nn.functional as F

        M, D = hidden_states.shape
        top_k = topk_ids.shape[1]

        if M == 0:
            return torch.empty(
                (0, D), dtype=hidden_states.dtype, device=hidden_states.device
            )

        compute_dtype = hidden_states.dtype
        w1 = w1.to(compute_dtype)
        w2 = w2.to(compute_dtype)

        hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1)
        ffn_out = torch.zeros(
            (M, top_k, D), dtype=compute_dtype, device=hidden_states.device
        )

        for expert_id in range(w1.size(0)):
            mask = topk_ids == expert_id
            if not mask.any():
                continue
            tokens = hidden_expanded[mask]
            upgate = torch.matmul(tokens, w1[expert_id].t())
            gate, up = upgate.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            out = torch.matmul(activated, w2[expert_id].t())
            ffn_out[mask] = out

        if apply_router_weight:
            ffn_out = ffn_out * topk_weights.unsqueeze(-1)

        return ffn_out.sum(dim=1).to(hidden_states.dtype)

    def _run_kernel_test(
        self,
        dtype: _dtype,
        token_num: int,
        hidden_dim: int,
        expert_num: int,
        inter_dim: int,
        top_k: int,
    ):
        import aiter

        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
            BLOCK_SIZE_M,
        )

        torch.manual_seed(42)
        hidden_states = (
            torch.randn(token_num, hidden_dim, device="cuda").to(dtype) * 0.03
        )
        topk_ids = torch.topk(
            torch.rand(token_num, expert_num, device="cuda"), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(token_num, top_k, device="cuda"), dim=-1
        ).to(torch.float32)

        w1 = (
            torch.randn(expert_num, inter_dim, hidden_dim, device="cuda").to(dtype)
            * 0.02
        )
        w2 = (
            torch.randn(expert_num, hidden_dim, inter_dim // 2, device="cuda").to(dtype)
            * 0.02
        )

        w1_shuffled = shuffle_weight(w1, layout=(16, 16))
        w2_shuffled = shuffle_weight(w2, layout=(16, 16))

        # === Step 1: Test moe_sorting_fwd ===
        device = "cuda"
        M = token_num
        block_size = BLOCK_SIZE_M
        max_num_tokens_padded = topk_ids.numel() + expert_num * block_size - top_k
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=device
        )
        sorted_weights = torch.empty(
            (max_num_tokens_padded,), dtype=torch.float32, device=device
        )
        sorted_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
        moe_buf = torch.empty((M, hidden_dim), dtype=dtype, device=device)

        aiter.moe_sorting_fwd(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            moe_buf=moe_buf,
            num_experts=expert_num,
            unit_size=block_size,
            local_expert_mask=None,
        )

        # Verify sorting produces valid outputs
        self.assertEqual(sorted_ids.dtype, torch.int32)
        self.assertEqual(sorted_weights.dtype, torch.float32)
        self.assertEqual(sorted_expert_ids.dtype, torch.int32)
        self.assertGreater(num_valid_ids[0].item(), 0)
        self.assertEqual(moe_buf.shape, (M, hidden_dim))

        # === Step 2: Test ck_moe_stage1 + ck_moe_stage2 with kernel-side weighting ===
        a2 = torch.empty((M, top_k, inter_dim // 2), dtype=dtype, device=device)

        aiter.ck_moe_stage1(
            hidden_states=hidden_states,
            w1=w1_shuffled,
            w2=w2_shuffled,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=a2,
            topk=top_k,
            kernelName="",
            w1_scale=None,
            a1_scale=None,
            block_m=block_size,
            sorted_weights=sorted_weights,  # Enable kernel-side weighting
        )

        a2 = a2.view(-1, a2.shape[-1])

        aiter.ck_moe_stage2(
            inter_states=a2,
            w1=w1_shuffled,
            w2=w2_shuffled,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=moe_buf,
            topk=top_k,
            kernelName="",
            w2_scale=None,
            a2_scale=None,
            block_m=block_size,
            sorted_weights=sorted_weights,  # Enable kernel-side weighting
        )

        # === Step 3: Compare against torch reference ===
        ref_out = self._torch_moe_ref(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1=w1,
            w2=w2,
            apply_router_weight=True,
        )

        torch.testing.assert_close(moe_buf, ref_out, atol=1e-2, rtol=1e-2)

    def test_moe_kernels(self):
        """Test moe_sorting_fwd + ck_moe_stage1/2 against torch reference."""
        for params in itertools.product(
            self.DTYPES,
            self.TOKEN_NUM,
            self.HIDDEN_DIM,
            self.EXPERT_NUM,
            self.INTER_DIM,
            self.TOP_K,
        ):
            with self.subTest(
                dtype=params[0],
                token_num=params[1],
                hidden_dim=params[2],
                expert_num=params[3],
                inter_dim=params[4],
                top_k=params[5],
            ):
                self._run_kernel_test(*params)


if __name__ == "__main__":
    main()
