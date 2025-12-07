import itertools
from unittest import SkipTest, TestCase, main

import torch
from aiter.ops.shuffle import shuffle_weight
from torch import dtype as _dtype

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
    FusedMoeExecutor,
    torch_moe_ref,
)
from rtp_llm.utils.model_weight import W


class FusedMoeTest(TestCase):
    DTYPES = [torch.bfloat16]
    TOKEN_NUM = [32]
    HIDDEN_DIM = [512]
    EXPERT_NUM = [32]
    INTER_DIM = [1024]
    TOP_K = [4]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_moe_test(
        self,
        dtype: _dtype,
        token_num: int,
        hidden_dim: int,
        expert_num: int,
        inter_dim: int,
        top_k: int,
    ):

        torch.manual_seed(42)
        hidden_states = torch.randn(token_num, hidden_dim).to(dtype) * 0.03
        topk_ids = torch.topk(
            torch.rand(token_num, expert_num), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(torch.randn(token_num, top_k), dim=-1).to(
            torch.float32
        )

        w1 = torch.randn(expert_num, inter_dim, hidden_dim).to(dtype) * 0.02
        w2 = torch.randn(expert_num, hidden_dim, inter_dim // 2).to(dtype) * 0.02

        # print("hidden_states: ", hidden_states)
        # print("topk_ids: ", topk_ids)
        # print("topk_weights: ", topk_weights)
        # print("w1: ", w1)
        # print("w2: ", w2)

        payload = ExpertForwardPayload(
            expert_x=hidden_states,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        ref_out = torch_moe_ref(
            payload=payload,
            activation="silu",
            global_num_experts=expert_num,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
            w1=w1,
            w2=w2,
        )

        # Model configuration
        model_param = GptInitModelParameters(
            head_num=4,
            size_per_head=64,
            layer_num=2,
            max_seq_len=2048,
            vocab_size=32000,
        )
        model_param.ep_size = 1
        model_param.ep_rank = 0
        model_param.expert_num = expert_num
        model_param.moe_k = top_k
        model_param.moe_inter_padding_size = inter_dim
        model_param.activation_type = "silu"

        w1 = shuffle_weight(w1, layout=(16, 16))
        w2 = shuffle_weight(w2, layout=(16, 16))

        weights = {W.moe_w1: w1, W.moe_w2: w2}

        fused_moe_executors = FusedMoeExecutor(model_param, weights)

        exec_out = fused_moe_executors.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        # 数值比对
        torch.testing.assert_close(exec_out, ref_out, atol=1e-3, rtol=1e-3)

    def test_fused_moe(self):
        for params in itertools.product(
            self.DTYPES,
            self.TOKEN_NUM,
            self.HIDDEN_DIM,
            self.EXPERT_NUM,
            self.INTER_DIM,
            self.TOP_K,
        ):
            with self.subTest(
                dype=params[0],
                token_num=params[1],
                hidden_dim=params[2],
                expert_num=params[3],
                inter_dim=params[4],
                top_k=params[5],
            ):
                self._run_fused_moe_test(*params)


if __name__ == "__main__":
    main()
