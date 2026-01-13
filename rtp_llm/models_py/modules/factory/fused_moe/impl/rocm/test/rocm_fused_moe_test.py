import itertools
from unittest import SkipTest, TestCase, main

import torch
import pytest
aiter = pytest.importorskip("aiter")
from torch import dtype as _dtype

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
    FusedMoeExecutor,
    torch_moe_ref,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
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
        model_config = ModelConfig()
        model_config.attn_config.head_num = 4
        model_config.attn_config.size_per_head = 64
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 32000
        model_config.expert_num = expert_num
        model_config.moe_k = top_k
        model_config.inter_size = inter_dim
        model_config.activation_type = "silu"

        # Create ParallelismConfig
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 1
        parallelism_config.ep_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.tp_rank = 0
        parallelism_config.dp_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.world_size = 1
        parallelism_config.world_rank = 0
        parallelism_config.local_rank = 0
        parallelism_config.local_world_size = 1

        # Create MoEConfigAdapter
        moe_config = MoeConfig()
        config_adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=0,
        )

        w1 = aiter.ops.shuffle.shuffle_weight(w1, layout=(16, 16))
        w2 = aiter.ops.shuffle.shuffle_weight(w2, layout=(16, 16))

        weights = {W.moe_w1: w1, W.moe_w2: w2}

        fused_moe_executors = FusedMoeExecutor(
            config_adapter, FusedMoEQuantConfig(), weights
        )

        combine_payload = fused_moe_executors.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        # 数值比对
        torch.testing.assert_close(
            combine_payload.fused_expert_output, ref_out, atol=1e-3, rtol=1e-3
        )

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
