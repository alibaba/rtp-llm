import itertools
import os
import sys
from typing import Callable, Optional
from unittest import SkipTest, TestCase, main

import torch

device = torch.device(f"cuda")
from rtp_llm.models_py.modules import GroupTopK


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    # TODO: NPU can't support directly evaluating a comparison for now
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
):
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        topk_weights, topk_ids = biased_grouped_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            correction_bias=correction_bias,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
        )
        return topk_weights, topk_ids
    else:
        raise ValueError(f"Unsupported grouped topk: {use_grouped_topk}")


class MLATest(TestCase):
    SEQ_LEN = [7]
    NUM_EXPERT = [256]
    NUM_EXPERT_GROUP = [8]
    TOPK_GROUP = [4]
    TOPK = [8]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)

    def _run_mla_test(
        self, seq_length, num_experts, num_expert_group, topk_group, topk
    ):
        dtype = torch.float32

        torch.manual_seed(seq_length)
        gating_output = torch.rand(
            (seq_length, num_experts), dtype=dtype, device="cuda"
        )
        bias = torch.rand(num_experts, dtype=dtype, device="cuda")
        scores = gating_output
        group_topk = GroupTopK()

        output = torch.empty(
            (seq_length, topk),
            dtype=torch.float32,
            device=scores.device,
        )
        indices = torch.empty(
            (seq_length, topk),
            dtype=torch.int64,
            device=scores.device,
        )

        group_topk(
            topk_weights=output,
            topk_ids=indices,
            scores=scores,
            correction_bias=bias,
            n_group=num_expert_group,
            topk_group=topk_group,
            topk=topk,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        indices = indices.to(torch.int32)

        ref_output, ref_indices = select_experts(
            hidden_states=scores,
            router_logits=scores,
            correction_bias=bias,
            routed_scaling_factor=2.5,
            use_grouped_topk=True,
            top_k=topk,
            renormalize=True,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=1,
        )
        torch.testing.assert_close(ref_output, output, atol=2e-2, rtol=0)
        torch.testing.assert_close(ref_indices, indices, atol=0, rtol=0)

    def test_mlp(self):
        for params in itertools.product(
            self.SEQ_LEN,
            self.NUM_EXPERT,
            self.NUM_EXPERT_GROUP,
            self.TOPK_GROUP,
            self.TOPK,
        ):
            with self.subTest(
                seq_length=params[0],
                num_experts=params[1],
                num_expert_group=params[2],
                topk_group=params[3],
                topk=params[4],
            ):
                self._run_mla_test(*params)

    def test_kimi_k3_fused_sigmoid_topk(self):
        torch.manual_seed(20260810)
        num_tokens = 17
        num_experts = 896
        topk = 16
        scale = 2.5
        logits = torch.randn(
            num_tokens,
            num_experts,
            dtype=torch.float32,
            device="cuda",
        )
        bias = 0.1 * torch.randn(
            num_experts,
            dtype=torch.float32,
            device="cuda",
        )
        scores = logits.sigmoid()
        expected_ids = (scores + bias.unsqueeze(0)).topk(
            topk, dim=-1, sorted=True
        ).indices
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = (
            expected_weights
            / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
            * scale
        )

        group_topk = GroupTopK()
        self.assertTrue(
            group_topk.fused_sigmoid_supported(
                logits,
                bias,
                n_group=1,
                topk_group=1,
                topk=topk,
            )
        )
        for index_dtype in (torch.int32, torch.int64):
            with self.subTest(index_dtype=index_dtype):
                actual_weights = torch.empty(
                    num_tokens,
                    topk,
                    dtype=torch.float32,
                    device="cuda",
                )
                actual_ids = torch.empty(
                    num_tokens,
                    topk,
                    dtype=index_dtype,
                    device="cuda",
                )
                group_topk.forward_fused_sigmoid(
                    actual_weights,
                    actual_ids,
                    logits,
                    bias,
                    topk,
                    True,
                    scale,
                )

                torch.testing.assert_close(
                    actual_ids.long(), expected_ids, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_weights,
                    expected_weights,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_kimi_k3_fused_sigmoid_topk_near_boundary(self):
        num_experts = 896
        topk = 16
        # Non-zero, all-distinct logits so sigmoid(logit) genuinely varies —
        # a broken tanh/sigmoid computation can no longer pass by luck. The
        # bias then compresses the top-16 boundary to ~1e-5 deterministically.
        logits = torch.linspace(
            -3.0, 3.0, num_experts, dtype=torch.float32, device="cuda"
        ).unsqueeze(0)
        scores = logits.sigmoid()
        # Rank experts purely by an index-monotone bias so the selected set is
        # deterministic, while keeping consecutive choice scores ~1e-5 apart.
        bias = (
            torch.arange(num_experts, dtype=torch.float32, device="cuda") * 1e-5
            - scores.squeeze(0)
        )
        choice_scores = scores + bias.unsqueeze(0)
        # Highest bias-index experts win; verify the boundary is genuinely tight.
        expected_ids = choice_scores.topk(topk, dim=-1, sorted=True).indices
        boundary = choice_scores.topk(topk + 1, dim=-1).values
        self.assertLess(
            float(boundary[0, topk - 1] - boundary[0, topk]), 5e-5
        )
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = expected_weights / (
            expected_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
        actual_weights = torch.empty_like(expected_weights)
        actual_ids = torch.empty_like(expected_ids)

        GroupTopK().forward_fused_sigmoid(
            actual_weights,
            actual_ids,
            logits,
            bias,
            topk,
            True,
            1.0,
        )

        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_weights,
            expected_weights,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    main()
