import itertools
import os
import sys
from unittest import SkipTest, TestCase, main

import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), "../../../"))
device = torch.device(f"cuda")
from rtp_llm.models_py.modules import GroupTopK
from rtp_llm.models_py.modules.ep.topk import select_experts


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

        output = torch.zeros(
            (seq_length, topk),
            dtype=torch.float32,
            device=scores.device,
        )
        indices = torch.zeros(
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
            apply_routed_scaling_factor_on_output=True,
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


if __name__ == "__main__":
    main()
