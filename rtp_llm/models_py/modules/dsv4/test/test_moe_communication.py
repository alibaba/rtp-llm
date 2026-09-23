"""Unit tests for the DSV4 collective MoE communication layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe.sm120_fused_moe import build_sm120_fused_moe
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import (
    MoeCfg,
    RoutedExpertsStrategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_fused_moe import (
    _validate_world_collective_topology,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
    PureCpRouterNoQuant,
)


class _FakeGroupedFp4Strategy(RoutedExpertsStrategy):
    name = "fake_grouped_fp4"

    def setup_weights(self, layer_weights):
        return None

    @classmethod
    def can_handle(cls, cfg):
        return True

    def forward(self, x, weights, indices):
        self.last_weights = weights
        self.last_indices = indices
        return weights.sum(dim=-1, keepdim=True).expand_as(x).float()


class Sm120FusedMoeTest(unittest.TestCase):
    def _cfg(self) -> MoeCfg:
        return MoeCfg(
            layer_id=0,
            dim=2,
            moe_inter_dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            swiglu_limit=7.0,
            ep_size=2,
            ep_rank=1,
            n_local_experts=2,
            local_expert_start=2,
            local_expert_end=4,
            max_tokens_per_rank=16,
        )

    def test_fused_moe_masks_and_remaps_global_expert_ids(self):
        cfg = self._cfg()
        local = _FakeGroupedFp4Strategy(cfg)
        fused_moe = build_sm120_fused_moe(
            cfg,
            local,
            uses_grouped_fp4=True,
        )
        self.assertIsInstance(fused_moe, FusedMoe)

        x = torch.ones((2, 2), dtype=torch.bfloat16)
        weights = torch.tensor([[0.2, 0.7], [0.6, 0.4]])
        global_ids = torch.tensor([[0, 2], [3, 1]])
        output = fused_moe(x, weights, global_ids)

        self.assertTrue(torch.allclose(output, torch.tensor([[0.7, 0.7], [0.6, 0.6]])))
        self.assertEqual(local.last_indices.tolist(), [[0, 0], [1, 0]])
        self.assertTrue(
            torch.allclose(
                local.last_weights,
                torch.tensor([[0.0, 0.7], [0.6, 0.0]]),
            )
        )

    def test_world_collective_rejects_non_ep_topology(self):
        cfg = self._cfg()
        dist = SimpleNamespace(
            get_world_size=lambda _group: 4,
            get_rank=lambda _group: 1,
        )
        with self.assertRaisesRegex(RuntimeError, "does not match the expert"):
            _validate_world_collective_topology(cfg, dist, object())

    def test_cp_uses_common_factory_pure_cp_router(self):
        cfg = self._cfg()
        cfg = cfg.__class__(
            **{
                **cfg.__dict__,
                "ep_rank": 0,
                "local_expert_start": 0,
                "local_expert_end": 2,
                "moe_tp_size": 2,
                "cp_size": 2,
                "cp_enabled": True,
            }
        )
        local = _FakeGroupedFp4Strategy(cfg)

        gather_calls = []

        def fake_all_gather(tensor, group):
            gather_calls.append(tensor)
            # Activation all-gather must preserve rank order; metadata gathers
            # are duplicated because their values are rank-invariant in this UT.
            if len(gather_calls) == 1:
                return torch.cat((tensor, tensor + 1), dim=0)
            return torch.cat((tensor, tensor.clone()), dim=0)

        def fake_reduce_scatter(tensor, group):
            return tensor.view(2, -1, *tensor.shape[1:]).sum(dim=0)

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.all_gather",
            side_effect=fake_all_gather,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.reduce_scatter",
            side_effect=fake_reduce_scatter,
        ):
            fused_moe = build_sm120_fused_moe(
                cfg,
                local,
                uses_grouped_fp4=True,
                router_cls=PureCpRouterNoQuant,
            )
            output = fused_moe(
                torch.ones((2, 2), dtype=torch.bfloat16),
                torch.tensor([[0.2, 0.7], [0.6, 0.4]]),
                torch.tensor([[0, 2], [1, 3]]),
            )

        self.assertIsInstance(fused_moe.router, PureCpRouterNoQuant)
        self.assertTrue(torch.allclose(output, torch.tensor([[0.4, 0.4], [1.2, 1.2]])))


if __name__ == "__main__":
    unittest.main()
