"""Tests for the MegaMoE passthrough router and strategy (CPU-only, no GPU required)."""

import os
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


def _make_config(
    expert_num=64,
    moe_k=8,
    ep_size=8,
    ep_rank=0,
    use_megamoe=True,
    use_mori_ep=False,
    use_deepep_moe=False,
    hidden_size=4096,
    moe_inter_size=2048,
) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = expert_num
    model_config.moe_k = moe_k
    model_config.hidden_size = hidden_size
    model_config.inter_size = moe_inter_size
    model_config.moe_inter_size = moe_inter_size

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = ep_rank
    parallelism_config.tp_size = ep_size
    parallelism_config.tp_rank = ep_rank
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = ep_size
    parallelism_config.world_rank = ep_rank
    parallelism_config.local_rank = ep_rank
    parallelism_config.local_world_size = ep_size

    moe_config = MoeConfig()
    moe_config.use_mori_ep = use_mori_ep
    moe_config.use_deepep_moe = use_deepep_moe
    moe_config.ll_num_max_token = 256

    # use_megamoe is controlled by the USE_MEGAMOE env var (the C++ MoeConfig
    # has no such field yet), so drive it through the environment.
    with patch.dict(os.environ, {"USE_MEGAMOE": "1" if use_megamoe else "0"}):
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )


class TestMegaMoePassthroughRouter(TestCase):
    """Tests for MegaMoePassthroughRouter.prepare and .finalize."""

    def _make_router(self, **kwargs):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.megamoe_router import (
            MegaMoePassthroughRouter,
        )

        config = _make_config(**kwargs)
        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return MegaMoePassthroughRouter(config, quant_config)

    def test_router_type(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.megamoe_router import (
            MegaMoePassthroughRouter,
        )

        self.assertEqual(MegaMoePassthroughRouter.router_type(), RouterType.MEGAMOE)

    def test_prepare_passthrough(self):
        router = self._make_router()
        num_tokens, hidden = 16, 4096
        topk = 8
        a1 = torch.randn(num_tokens, hidden)
        topk_weights = torch.rand(num_tokens, topk)
        topk_ids = torch.randint(0, 64, (num_tokens, topk))

        payload = router.prepare(a1, None, None, topk_weights, topk_ids)

        # Passthrough: expert_x IS a1, global routing passed through.
        self.assertTrue(torch.equal(payload.expert_x, a1))
        self.assertTrue(torch.equal(payload.expert_topk_ids, topk_ids))
        self.assertTrue(torch.equal(payload.expert_topk_weights, topk_weights))
        self.assertFalse(payload.expert_ids_are_local)

    def test_finalize_returns_output(self):
        router = self._make_router()
        num_tokens, hidden = 16, 4096
        output = torch.randn(num_tokens, hidden)
        combine_payload = CombineForwardPayload(fused_expert_output=output)
        topk_weights = torch.rand(num_tokens, 8)
        topk_ids = torch.randint(0, 64, (num_tokens, 8))

        result = router.finalize(
            combine_payload,
            topk_weights,
            topk_ids,
            False,
            {"original_num_tokens": num_tokens, "a1_shape": (num_tokens, hidden)},
        )
        self.assertTrue(torch.equal(result, output))

    def test_finalize_truncates_to_original_tokens(self):
        router = self._make_router()
        num_tokens, hidden = 16, 4096
        # output is padded (extra rows).
        output = torch.randn(num_tokens + 10, hidden)
        combine_payload = CombineForwardPayload(fused_expert_output=output)
        topk_weights = torch.rand(num_tokens, 8)
        topk_ids = torch.randint(0, 64, (num_tokens, 8))

        result = router.finalize(
            combine_payload,
            topk_weights,
            topk_ids,
            False,
            {"original_num_tokens": num_tokens, "a1_shape": (num_tokens, hidden)},
        )
        self.assertEqual(result.shape[0], num_tokens)


class TestMegaMoeStrategy(TestCase):
    """Tests for RocmMegaMoeStrategy condition checking."""

    def test_cannot_handle_when_disabled(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy.megamoe import (
            RocmMegaMoeStrategy,
        )

        strategy = RocmMegaMoeStrategy()
        config = _make_config(use_megamoe=False)
        self.assertFalse(strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.distributed.megamoe_wrapper.MegaMoeWrapper.supported",
        return_value=True,
    )
    def test_can_handle_when_enabled(self, mock_supported):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy.megamoe import (
            RocmMegaMoeStrategy,
        )

        strategy = RocmMegaMoeStrategy()
        config = _make_config(use_megamoe=True)
        self.assertTrue(strategy.can_handle(config))

    def test_priority_is_highest(self):
        """MegaMoE should have the highest priority among ROCm strategies."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy.megamoe import (
            RocmMegaMoeStrategy,
        )

        strategy = RocmMegaMoeStrategy()
        # priority = router_type.value * 10 + executor_type.value
        # MEGAMOE=8, MEGAMOE_FUSED=9 -> 8*10+9=89
        with patch(
            "rtp_llm.models_py.distributed.megamoe_wrapper.MegaMoeWrapper.supported",
            return_value=True,
        ):
            attrs = strategy.get_attributes()
            p = attrs.calculate_priority()
            self.assertEqual(p, 8 * 10 + 9)


class TestMegaMoeConfigAdapter(TestCase):
    """Tests for the use_megamoe field in MoEConfigAdapter."""

    def test_env_var_controls_flag(self):
        moe_config = MoeConfig()
        model_config = ModelConfig()
        model_config.attn_config.head_num = 4
        model_config.expert_num = 64
        model_config.moe_k = 8
        model_config.hidden_size = 4096

        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 8
        parallelism_config.world_size = 8
        parallelism_config.local_world_size = 8

        with patch.dict(os.environ, {"USE_MEGAMOE": "1"}):
            adapter = MoEConfigAdapter(
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
            )
            self.assertTrue(adapter.use_megamoe)

        with patch.dict(os.environ, {"USE_MEGAMOE": "0"}):
            adapter = MoEConfigAdapter(
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
            )
            self.assertFalse(adapter.use_megamoe)


class TestMegaMoeWrapperConfig(TestCase):
    """Tests for MegaMoeWrapperConfig."""

    def test_next_pow2(self):
        from rtp_llm.models_py.distributed.megamoe_wrapper import _next_pow2

        self.assertEqual(_next_pow2(1), 1)
        self.assertEqual(_next_pow2(2), 2)
        self.assertEqual(_next_pow2(3), 4)
        self.assertEqual(_next_pow2(255), 256)
        self.assertEqual(_next_pow2(256), 256)
        self.assertEqual(_next_pow2(257), 512)

    def test_from_config_adapter(self):
        from rtp_llm.models_py.distributed.megamoe_wrapper import MegaMoeWrapperConfig

        config = _make_config()
        wrapper_config = MegaMoeWrapperConfig.from_config_adapter(config)
        self.assertEqual(wrapper_config.rank, 0)
        self.assertEqual(wrapper_config.world_size, 8)
        self.assertEqual(wrapper_config.model_dim, 4096)
        self.assertEqual(wrapper_config.inter_dim, 2048)
        self.assertEqual(wrapper_config.experts, 64)
        self.assertEqual(wrapper_config.topk, 8)
        self.assertEqual(wrapper_config.max_tok_per_rank, 256)


if __name__ == "__main__":
    main()
