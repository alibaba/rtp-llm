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

    def _run_group_topk(
        self, gating_output, bias, num_expert_group, topk_group, topk, renormalize,
        routed_scaling_factor,
    ):
        group_topk = GroupTopK()
        output = torch.empty(
            (gating_output.shape[0], topk),
            dtype=torch.float32,
            device=gating_output.device,
        )
        indices = torch.empty(
            (gating_output.shape[0], topk),
            dtype=torch.int64,
            device=gating_output.device,
        )
        group_topk(
            topk_weights=output,
            topk_ids=indices,
            scores=gating_output,
            correction_bias=bias,
            n_group=num_expert_group,
            topk_group=topk_group,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
        return output, indices

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

        output, indices = self._run_group_topk(
            gating_output, bias, num_expert_group, topk_group, topk,
            renormalize=True, routed_scaling_factor=2.5,
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

    def _run_bf16_gate_tests(
        self, seq_length, num_experts, num_expert_group, topk_group, topk, seed,
        bias_dtype, logit_range=(0.0, 1.0),
    ):
        # Xing4 call form (see model_desc/generic_moe.py): raw bf16 router
        # logits plus the e_score_correction_bias are handed to the GroupTopK
        # wrapper; the fp32 cast + fp32 sigmoid + fp32 bias add are fused
        # inside the CUDA kernels now. Xing4/DeepSeek checkpoints store the
        # bias in fp32 while the gate emits bf16, so the fp32-bias variant is
        # the real serving configuration; the bf16-bias variant covers
        # checkpoints that keep the bias in the model dtype.
        torch.manual_seed(seed)
        gating_output = (
            torch.rand((seq_length, num_experts), device="cuda") * (logit_range[1] - logit_range[0]) + logit_range[0]
        ).to(torch.bfloat16)
        bias = (
            torch.rand(num_experts, device="cuda") * 0.2 - 0.1
        ).to(bias_dtype)

        # 1) Numerics gate (bitwise): with renormalize=False and
        #    routed_scaling_factor=1.0 the kernel writes the fp32 sigmoid
        #    values of the selected experts verbatim (topk_sum stays 1.0),
        #    so the output must be bitwise equal to the removed python path
        #    `scores.float().sigmoid()` gathered at the kernel's own indices.
        output, indices = self._run_group_topk(
            gating_output, bias, num_expert_group, topk_group, topk,
            renormalize=False, routed_scaling_factor=1.0,
        )
        sigmoid_ref = gating_output.float().sigmoid()
        self.assertTrue(
            torch.equal(output, sigmoid_ref.gather(1, indices)),
            "kernel fp32 sigmoid deviates from torch fp32 sigmoid on the "
            "widened bf16 logits",
        )
        # The routing must not have taken the non-finite fallback path
        # (that path writes a constant 1/topk).
        self.assertFalse(torch.equal(output, torch.full_like(output, 1.0 / topk)))

        # 2) Semantics gate: renormalize + routed_scaling_factor, compared
        #    against the pure-torch reference fed with the exact old-path
        #    fp32 tensors (gating_output.float(), bias.float()).
        output, indices = self._run_group_topk(
            gating_output, bias, num_expert_group, topk_group, topk,
            renormalize=True, routed_scaling_factor=2.5,
        )
        indices = indices.to(torch.int32)
        ref_output, ref_indices = select_experts(
            hidden_states=gating_output.float(),
            router_logits=gating_output.float(),
            correction_bias=bias.float(),
            routed_scaling_factor=2.5,
            use_grouped_topk=True,
            top_k=topk,
            renormalize=True,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=1,
        )
        torch.testing.assert_close(ref_output, output, atol=1e-5, rtol=0)
        torch.testing.assert_close(ref_indices, indices, atol=0, rtol=0)

    def test_bf16_sigmoid_gate_xing4(self):
        # Real Xing4 runtime form: bf16 router logits + fp32 correction bias
        # (observed live in the Xing4 prefill perf run).
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
                self._run_bf16_gate_tests(
                    *params, seed=100 + params[0], bias_dtype=torch.float32
                )

    def test_bf16_sigmoid_gate_xing4_shape(self):
        # Xing4.0-29B-A4B production shape: 64 routed experts, n_group=1,
        # top-4 — this exercises the >WARP_SIZE per-lane top-2 branch (the
        # group-select degenerates with a single group), unlike the 256/8
        # DeepSeek shape above which takes the group loop. Logits span
        # negative and large values (real router scores are signed).
        for seq_length, seed in ((7, 11), (129, 12)):
            with self.subTest(seq_length=seq_length):
                self._run_bf16_gate_tests(
                    seq_length, 64, 1, 1, 4, seed=seed, bias_dtype=torch.float32,
                    logit_range=(-8.0, 8.0),
                )

    def test_bf16_sigmoid_gate_same_dtype_bias(self):
        # Checkpoints that keep e_score_correction_bias in the model dtype.
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
                self._run_bf16_gate_tests(
                    *params, seed=300 + params[0], bias_dtype=torch.bfloat16
                )

    def test_fp16_sigmoid_gate(self):
        # fp16 logits widen to fp32 exactly like bf16; pin the half
        # instantiation with one full reference comparison.
        seq_length, num_experts = 7, 256
        num_expert_group, topk_group, topk = 8, 4, 8
        torch.manual_seed(200 + seq_length)
        gating_output = torch.rand(
            (seq_length, num_experts), dtype=torch.float16, device="cuda"
        )
        bias = torch.rand(num_experts, dtype=torch.float16, device="cuda")

        output, indices = self._run_group_topk(
            gating_output, bias, num_expert_group, topk_group, topk,
            renormalize=True, routed_scaling_factor=2.5,
        )
        indices = indices.to(torch.int32)
        ref_output, ref_indices = select_experts(
            hidden_states=gating_output.float(),
            router_logits=gating_output.float(),
            correction_bias=bias.float(),
            routed_scaling_factor=2.5,
            use_grouped_topk=True,
            top_k=topk,
            renormalize=True,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=1,
        )
        torch.testing.assert_close(ref_output, output, atol=1e-5, rtol=0)
        torch.testing.assert_close(ref_indices, indices, atol=0, rtol=0)

    def test_non_finite_scores_use_safe_fallback(self):
        num_experts = self.NUM_EXPERT[0]
        topk = self.TOPK[0]
        # Raw router logits (sigmoid is applied inside the op now): NaN logits
        # yield NaN biased scores in the kernel, so routing must fall back.
        scores = torch.full((1, num_experts), float("nan"), dtype=torch.float32)
        bias = torch.rand(num_experts, dtype=torch.float32)
        output = torch.empty((1, topk), dtype=torch.float32)
        indices = torch.empty((1, topk), dtype=torch.int64)

        GroupTopK().group_topk_op.forward(
            output,
            indices,
            scores,
            bias,
            self.NUM_EXPERT_GROUP[0],
            self.TOPK_GROUP[0],
            topk,
            True,
            2.5,
        )

        torch.testing.assert_close(indices, torch.arange(topk).unsqueeze(0))
        torch.testing.assert_close(output, torch.full_like(output, 1.0 / topk))

    def test_logits_bias_dtype_mismatch_raises(self):
        # Kernel dtype contract: for fp32 router logits only an fp32
        # correction bias is instantiated (bf16/fp16 logits accept fp32 or
        # same-dtype bias — the real Xing4 form is bf16 logits + fp32 bias).
        scores = torch.rand((1, 64), dtype=torch.float32)
        bias = torch.rand(64, dtype=torch.bfloat16)
        output = torch.empty((1, 8), dtype=torch.float32)
        indices = torch.empty((1, 8), dtype=torch.int64)
        with self.assertRaises(RuntimeError):
            GroupTopK().group_topk_op.forward(
                output,
                indices,
                scores,
                bias,
                self.NUM_EXPERT_GROUP[0],
                self.TOPK_GROUP[0],
                self.TOPK[0],
                True,
                2.5,
            )


if __name__ == "__main__":
    main()
