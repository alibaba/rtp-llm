"""Tests for Pure TP Router prepare/finalize flow (CPU-only, no ROCm required)."""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import CombineForwardPayload
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig


def _make_config(expert_num=8, moe_k=2, tp_size=1, tp_rank=0,
                 ep_size=1, ep_rank=0) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = expert_num
    model_config.moe_k = moe_k
    model_config.inter_size = 256

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = ep_rank
    parallelism_config.tp_size = tp_size
    parallelism_config.tp_rank = tp_rank
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = max(tp_size, ep_size)
    parallelism_config.world_rank = tp_rank
    parallelism_config.local_rank = tp_rank
    parallelism_config.local_world_size = max(tp_size, ep_size)

    moe_config = MoeConfig()
    moe_config.use_all_gather = True
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


def _make_router(expert_num=8, moe_k=2, tp_size=1, tp_rank=0,
                 ep_size=1, ep_rank=0):
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
        PureTpRouterNoQuant,
    )
    config = _make_config(expert_num=expert_num, moe_k=moe_k,
                          tp_size=tp_size, tp_rank=tp_rank,
                          ep_size=ep_size, ep_rank=ep_rank)
    quant_config = FusedMoEQuantConfig(quant_dtype=None)
    return PureTpRouterNoQuant(config, quant_config)


class PureTpRouterPrepareTest(TestCase):
    """Test PureTpRouterNoQuant.prepare flow."""

    def test_passthrough_and_dtype_preserved(self):
        router = _make_router()
        hidden_states = torch.randn(16, 64, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, 8, (16, 2), dtype=torch.int32)
        topk_weights = torch.softmax(torch.randn(16, 2), dim=-1).float()

        payload = router.prepare(
            a1=hidden_states, a1_scale=None, a2_scale=None,
            topk_weights=topk_weights, topk_ids=topk_ids,
        )

        torch.testing.assert_close(payload.expert_x, hidden_states)
        self.assertIsNone(payload.expert_x_scale)
        self.assertEqual(payload.expert_x.dtype, torch.bfloat16)
        self.assertEqual(payload.expert_x_origin_dtype, torch.bfloat16)
        torch.testing.assert_close(payload.expert_topk_ids, topk_ids)
        torch.testing.assert_close(payload.expert_topk_weights, topk_weights)

    def test_rejects_prescaled_input(self):
        router = _make_router()
        hidden_states = torch.randn(4, 32)
        topk_ids = torch.randint(0, 8, (4, 2), dtype=torch.int32)
        topk_weights = torch.ones(4, 2, dtype=torch.float32)

        with self.assertRaises(AssertionError):
            router.prepare(
                a1=hidden_states, a1_scale=torch.ones(4, 1),
                a2_scale=None, topk_weights=topk_weights, topk_ids=topk_ids,
            )


class PureTpRouterFinalizeTest(TestCase):
    """Test PureTpRouterNoQuant.finalize flow."""

    def test_single_gpu_passthrough(self):
        router = _make_router()
        expert_output = torch.randn(16, 64)
        payload = CombineForwardPayload(fused_expert_output=expert_output)

        result = router.finalize(
            payload=payload,
            topk_weights=torch.ones(16, 2, dtype=torch.float32),
            topk_ids=torch.randint(0, 8, (16, 2), dtype=torch.int32),
            apply_router_weight_on_input=False,
            extra_finalize_args=None,
        )
        torch.testing.assert_close(result, expert_output)

    @patch("rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router.all_reduce")
    def test_multi_gpu_calls_all_reduce(self, mock_all_reduce):
        router = _make_router(tp_size=2, tp_rank=0)
        expert_output = torch.randn(16, 64)
        reduced_output = torch.randn(16, 64)
        mock_all_reduce.return_value = reduced_output

        payload = CombineForwardPayload(fused_expert_output=expert_output)
        result = router.finalize(
            payload=payload,
            topk_weights=torch.ones(16, 2, dtype=torch.float32),
            topk_ids=torch.randint(0, 8, (16, 2), dtype=torch.int32),
            apply_router_weight_on_input=False,
            extra_finalize_args=None,
        )
        mock_all_reduce.assert_called_once()
        torch.testing.assert_close(result, reduced_output)


class PureTpRouterCheckConditionsTest(TestCase):
    """Regression tests for PureTpRouterBase.check_conditions strategy selection.

    Ensures pure-TP router correctly matches/rejects parallelism configurations,
    preventing the bug where ep_size == tp_size > 1 (EP scenario) was incorrectly
    matched by the pure-TP router.
    """

    def _check_passes(self, config: MoEConfigAdapter) -> bool:
        """Run PureTpRouterNoQuant.check_conditions and return whether all checks pass."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
            ConditionChecker,
        )
        checker = ConditionChecker("PureTpRouterNoQuant")
        PureTpRouterNoQuant.check_conditions(checker, config)
        return checker.all_passed()

    def test_pure_tp_matches_tp4_dp1_ep1(self):
        """tp=4, dp=1, ep=1 should match pure-TP (each rank holds all experts)."""
        config = _make_config(tp_size=4, ep_size=1)
        self.assertTrue(self._check_passes(config))

    def test_pure_tp_rejects_tp4_dp1_ep4(self):
        """tp=4, dp=1, ep=4 must NOT match pure-TP (weights split by EP)."""
        config = _make_config(tp_size=4, ep_size=4)
        self.assertFalse(self._check_passes(config))

    def test_pure_tp_matches_single_gpu(self):
        """tp=1, dp=1, ep=1 (single GPU) should match pure-TP."""
        config = _make_config(tp_size=1, ep_size=1)
        self.assertTrue(self._check_passes(config))

    def test_pure_tp_rejects_ep2_tp2(self):
        """tp=2, dp=1, ep=2 must NOT match pure-TP."""
        config = _make_config(tp_size=2, ep_size=2)
        self.assertFalse(self._check_passes(config))

if __name__ == "__main__":
    main()