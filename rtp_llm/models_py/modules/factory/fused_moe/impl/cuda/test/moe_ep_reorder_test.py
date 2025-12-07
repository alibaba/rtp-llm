import itertools
import math
import random
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.util import (
    moe_permute,
    moe_unpermute,
)


class MoeReorderTest(TestCase):
    DTYPES = [torch.bfloat16, torch.float16]
    NUM_TOKENS = [1, 2, 8, 33, 2049, 5120]
    HIDDEN_DIMS = [2048, 7168]
    NUM_EXPERT = [16, 128]
    TOP_K = [2, 6, 8]
    EP_SIZE = [1, 4, 16]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def torch_fused_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ):
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weight = F.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(topk_weight, topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids

    def torch_moe_permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk: int,
        num_experts: int,
        num_local_experts: int,
        start_expert: int,
        expert_map: Optional[int] = None,
    ) -> List[torch.Tensor]:
        num_tokens, hidden_dim = hidden_states.shape[0], hidden_states.shape[1]
        if expert_map is not None:
            topk_ids = (expert_map[topk_ids] != -1) * (topk_ids - start_expert) + (
                expert_map[topk_ids] == -1
            ) * (topk_ids + num_experts)

        token_expert_indices = torch.arange(
            0, num_tokens * topk, dtype=torch.int32, device=hidden_states.device
        ).reshape((num_tokens, topk))

        sorted_topk_ids, sorted_indices = torch.sort(topk_ids.flatten(), stable=True)
        dst_row_id2src_row_id_map = token_expert_indices.flatten()[sorted_indices]

        expert_first_token_offset = torch.zeros(
            num_local_experts + 1, dtype=torch.int64, device="cuda"
        )
        idx = 0
        for i in range(0, num_local_experts):
            cnt = 0
            while idx < sorted_topk_ids.numel() and sorted_topk_ids[idx] == i:
                cnt += 1
                idx += 1
            expert_first_token_offset[i + 1] = expert_first_token_offset[i] + cnt

        _, src2dst_idx = torch.sort(dst_row_id2src_row_id_map)
        valid_row_idx = []

        permuted_hidden_states = hidden_states[dst_row_id2src_row_id_map // topk, ...]

        permuted_row_size = permuted_hidden_states.shape[0]

        m_indices = torch.empty(
            permuted_row_size, device="cuda", dtype=torch.int32
        ).fill_(0)

        for i in range(1, num_local_experts + 1):
            first_token_offset = expert_first_token_offset[i - 1]
            last_token_offset = expert_first_token_offset[i]
            m_indices[first_token_offset:last_token_offset] = i - 1
        src_row_id2dst_row_id_map = torch.arange(
            0, num_tokens * topk, device="cuda", dtype=torch.int32
        )[src2dst_idx].reshape((num_tokens, topk))
        valid_row_idx += [i for i in range(expert_first_token_offset[-1])]
        dst_row_id2src_row_id_map[expert_first_token_offset[-1] :] = num_tokens * topk
        return [
            permuted_hidden_states,
            expert_first_token_offset,
            src_row_id2dst_row_id_map,
            dst_row_id2src_row_id_map,
            m_indices,
            valid_row_idx,
        ]

    def torch_moe_unpermute(
        self,
        permuted_hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        src_row_id2dst_row_id_map: torch.Tensor,
        valid_row_idx: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        hidden_dim = permuted_hidden_states.shape[1]
        mask = torch.zeros(permuted_hidden_states.shape[0], dtype=bool, device="cuda")
        mask[valid_row_idx] = True
        permuted_hidden_states[~mask] = 0

        permuted_hidden_states = permuted_hidden_states[
            src_row_id2dst_row_id_map.flatten(), ...
        ]
        permuted_hidden_states = permuted_hidden_states.view(-1, topk, hidden_dim)
        output = (
            (permuted_hidden_states * topk_weights.unsqueeze(2))
            .sum(1)
            .to(permuted_hidden_states.dtype)
        )
        return output

    def _run_moe_ep_reorder_test(
        self,
        num_tokens: int,
        num_expert: int,
        hidden_dim: int,
        top_k: int,
        dtype: _dtype,
        ep_size: int,
    ):
        ep_rank = random.randint(0, ep_size - 1)

        print(
            f"testing >>>>> num_tokens:{num_tokens}, num_expert:{num_expert}, hidden_dim: {hidden_dim}, top_k:{top_k}, ep_size: {ep_size}, ep_rank: {ep_rank}>>>>>>"
        )
        # ep_rank = 0
        num_local_experts = num_expert
        expert_map = None

        if ep_size > 1:
            num_local_experts = num_expert // ep_size
            expert_map = torch.full((num_expert,), fill_value=-1, dtype=torch.int32)
            start_idx = ep_rank * num_local_experts
            end_idx = start_idx + num_local_experts
            expert_map[start_idx:end_idx] = torch.arange(
                0, num_local_experts, dtype=torch.int32, device="cuda"
            )

        hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype).to("cuda")
        router_logits = torch.randn(num_tokens, num_expert, dtype=dtype).to("cuda")
        topk_weights, topk_ids = self.torch_fused_topk(
            hidden_states, router_logits, top_k, False
        )
        # test pre reorder kernel
        (
            torch_permuted_hidden_states,
            torch_expert_first_token_offset,
            src_row_id2dst_row_id_map,
            _,
            _,
            valid_row_idx,
        ) = self.torch_moe_permute(
            hidden_states,
            topk_ids,
            top_k,
            num_expert,
            num_local_experts,
            start_expert=ep_rank * num_local_experts,
            expert_map=expert_map,
        )
        permuted_hidden_states, _, expert_first_token_offset, inv_permuted_idx = (
            moe_permute(
                hidden_states=hidden_states,
                a1q_scale=None,
                topk_ids=topk_ids,
                num_experts=num_expert,
                num_local_experts=num_local_experts,
                expert_map=expert_map,
                permuted_hidden_states=None,
            )
        )

        self.assertTrue(
            torch.equal(
                torch_expert_first_token_offset, expert_first_token_offset.flatten()
            )
        )
        # finalizeMoeRoutingKernel中inv_permuted_idx的shape为[topk, num_tokens], 所以在这里要做一次reshape
        self.assertTrue(
            torch.equal(src_row_id2dst_row_id_map.t().flatten(), inv_permuted_idx)
        )
        self.assertTrue(
            torch.equal(
                torch_permuted_hidden_states[valid_row_idx],
                permuted_hidden_states[valid_row_idx],
            )
        )

        # test post reorder kernel
        cuda_out = torch.empty_like(hidden_states)

        moe_unpermute(
            out=cuda_out,
            permuted_hidden_states=permuted_hidden_states,
            topk_weights=topk_weights,
            inv_permuted_idx=src_row_id2dst_row_id_map.t().flatten(),
            expert_first_token_offset=expert_first_token_offset,
        )

        torch_out = self.torch_moe_unpermute(
            permuted_hidden_states,
            topk_weights,
            src_row_id2dst_row_id_map,
            valid_row_idx,
            top_k,
        )

        self.assertTrue(torch.allclose(torch_out, cuda_out, atol=2e-2, rtol=0))

    def test_moe_ep_reorder(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.NUM_EXPERT,
            self.HIDDEN_DIMS,
            self.TOP_K,
            self.DTYPES,
            self.EP_SIZE,
        ):
            with self.subTest(
                num_tokens=params[0],
                num_expert=params[1],
                hidden_dim=params[2],
                top_k=params[3],
                dtype=params[4],
                ep_size=params[5],
            ):
                top_k, num_expert = params[3], params[1]
                if top_k > num_expert:
                    continue
                self._run_moe_ep_reorder_test(*params)


if __name__ == "__main__":
    main()
