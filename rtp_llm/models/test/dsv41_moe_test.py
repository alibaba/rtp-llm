"""Local route ownership, bounded execution and numerical strategy selection."""

import contextlib
import dataclasses
import sys
import types
import unittest
from unittest.mock import PropertyMock, patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_expert_test import expert
from rtp_llm.models.test.dsv41_mxfp8_test import linear
from rtp_llm.models.test.dsv41_routed_test import (
    reference_expert,
    routed,
    routed_weights,
)

MODULES = {
    "rtp_llm.models_py.modules.dsv4.v41_expert": expert,
    "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    "rtp_llm.models_py.modules.dsv4.v41_routed": routed,
}
with patch.dict(sys.modules, MODULES):
    moe = load_codec_source("_dsv41_moe_test", "v41_moe.py")
    base = load_codec_source("_dsv41_moe_base_test", "moe/strategies/base.py")
MODULES.update(
    {
        "rtp_llm.models_py.modules.dsv4.v41_moe": moe,
        "rtp_llm.models_py.modules.dsv4.moe.strategies.base": base,
    }
)
with patch.dict(sys.modules, MODULES):
    strategy = load_codec_source(
        "rtp_llm.models_py.modules.dsv4.moe.strategies.v41_grouped_fp4",
        "moe/strategies/v41_grouped_fp4.py",
    )


def cfg(recipe=None):
    with patch.dict(sys.modules, MODULES):
        return base.MoeCfg(
            layer_id=0,
            dim=128,
            moe_inter_dim=128,
            n_routed_experts=3,
            n_activated_experts=3,
            swiglu_limit=10.0,
            ep_size=1,
            ep_rank=0,
            n_local_experts=3,
            local_expert_start=10,
            local_expert_end=13,
            max_tokens_per_rank=16,
            expert_recipe=recipe,
        )


def local_reference(x, weights, ids, valid, tensors, start):
    result = torch.zeros_like(x, dtype=torch.float32)
    for e in range(tensors["w2_w"].shape[0]):
        for row in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                if (
                    valid[row]
                    and ids[row, slot] == start + e
                    and weights[row, slot] != 0
                ):
                    result[row] += reference_expert(
                        x[row : row + 1],
                        tensors,
                        e,
                        torch.ones(1, dtype=torch.bool),
                        weights[row, slot : slot + 1],
                    )[0].float()
    return result


class V41MoETest(unittest.TestCase):
    def fixtures(self, *, dim=64, inter=96):
        tensors = routed_weights(dim=dim, inter=inter)
        local = moe.V41LocalMoE(
            routed.V41RoutedExperts(tensors, backend="eager"),
            expert_start_id=10,
            max_tokens=16,
            masked_token_limit=8,
        )
        x = torch.randn(
            5, dim, generator=torch.Generator().manual_seed(4101)
        ).bfloat16()
        ids = torch.tensor(
            [[12, 10, 7], [11, 12, 10], [10, -1, 12], [11, 12, 10], [7, 14, -1]]
        )
        weights = torch.tensor(
            [
                [0.37, 0.25, 0.5],
                [0.5, 0.7, 0.37],
                [0, float("nan"), 0],
                [1.5, 0.3, 0.1],
                [1, 1, float("nan")],
            ]
        )
        valid = torch.tensor([True, True, True, False, True])
        x[2:] = float("nan")
        return local, tensors, x, weights, ids, valid

    def test_masked_and_contiguous_routes_match_global_id_reference(self):
        local, tensors, x, weights, ids, valid = self.fixtures()
        expected = local_reference(x, weights, ids, valid, tensors, 10)
        for layout in ("masked", "contiguous"):
            actual = local(x, weights, ids, valid_rows=valid, layout=layout)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(bool((actual[2:] == 0).all()))

    def test_route_slot_order_cp_permutation_and_chunks_preserve_results(self):
        local, _, x, weights, ids, valid = self.fixtures()
        expected = local(x, weights, ids, valid_rows=valid)
        order = torch.tensor([3, 0, 4, 1, 2])
        permuted = local(
            x[order],
            weights[order],
            ids[order],
            valid_rows=valid[order],
            layout="masked",
        )
        torch.testing.assert_close(permuted[order.argsort()], expected, rtol=0, atol=0)
        swapped = local(x, weights.flip(1), ids.flip(1), valid_rows=valid)
        torch.testing.assert_close(swapped, expected, rtol=0, atol=0)
        chunked = torch.cat(
            [
                local(x[:2], weights[:2], ids[:2], valid_rows=valid[:2]),
                local(x[2:], weights[2:], ids[2:], valid_rows=valid[2:]),
            ]
        )
        torch.testing.assert_close(chunked, expected, rtol=0, atol=0)

    def test_partitioned_shards_combine_without_second_weight_multiply(self):
        local, tensors, x, weights, ids, valid = self.fixtures()
        expected = local(x, weights, ids, valid_rows=valid)
        parts = []
        for begin, end in ((0, 1), (1, 3)):
            shard = {name: tensor[begin:end] for name, tensor in tensors.items()}
            op = moe.V41LocalMoE(
                routed.V41RoutedExperts(shard, backend="eager"),
                expert_start_id=10 + begin,
            )
            parts.append(op(x, weights, ids, valid_rows=valid))
        torch.testing.assert_close(parts[0] + parts[1], expected, rtol=0, atol=0)
        self.assertGreater(
            int((expected[0] != expected[0].bfloat16().float()).sum()), 0
        )

    def test_empty_and_unowned_rows_and_masked_planning_without_host_counts(self):
        local, _, x, weights, ids, valid = self.fixtures()
        for layout in ("masked", "contiguous"):
            self.assertEqual(
                local(x[:0], weights[:0], ids[:0], layout=layout).shape, (0, 64)
            )
            empty = local(
                x, weights, ids, valid_rows=torch.zeros_like(valid), layout=layout
            )
            self.assertTrue(bool((empty == 0).all()))
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("device counts read")
        ), patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("device counts read")
        ):
            result = local(x, weights, ids, valid_rows=valid, layout="masked")
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_invalid_routes_bounds_and_duplicate_experts_are_rejected(self):
        local, _, x, weights, ids, valid = self.fixtures()
        for bad_weights in (weights.bfloat16(), weights[:, :2], -weights):
            with self.assertRaises(ValueError):
                local(x, bad_weights, ids, valid_rows=valid)
        bad_ids = ids.clone()
        bad_ids[0, 1] = bad_ids[0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            local(x, weights, bad_ids, valid_rows=valid)
        safe_weights = weights.clone()
        safe_weights[0, 1] = 0
        self.assertTrue(
            bool(
                torch.isfinite(local(x, safe_weights, bad_ids, valid_rows=valid)).all()
            )
        )
        with self.assertRaisesRegex(ValueError, "bound"):
            local(x.repeat(4, 1), weights.repeat(4, 1), ids.repeat(4, 1))
        with self.assertRaisesRegex(ValueError, "bound"):
            local(
                x.repeat(2, 1), weights.repeat(2, 1), ids.repeat(2, 1), layout="masked"
            )
        with self.assertRaises(ValueError):
            local(x, weights, ids.float())
        with self.assertRaises(ValueError):
            local(x, weights, ids, valid_rows=valid.long())

    def test_contiguous_graph_capture_is_rejected_before_host_counts(self):
        local, _, x, weights, ids, valid = self.fixtures()
        with patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "capture requires masked"):
                local(x, weights, ids, valid_rows=valid)

    def test_recipe_is_part_of_configuration_identity_and_survives_partition_copy(self):
        legacy, current = cfg(), cfg(expert.V41_EXPERT_RECIPE)
        self.assertNotEqual(legacy, current)
        self.assertNotEqual(hash(legacy), hash(current))
        with patch.dict(sys.modules, MODULES):
            copied = dataclasses.replace(current, layer_id=14)
            with self.assertRaisesRegex(ValueError, "recipe"):
                dataclasses.replace(current, expert_recipe="group32")
        self.assertIs(copied.expert_recipe, expert.V41_EXPERT_RECIPE)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            current.expert_recipe = None

    def test_selection_and_direct_construction_reject_legacy_arithmetic(self):
        class Legacy(base.RoutedExpertsStrategy):
            name = "legacy"

            @classmethod
            def can_handle(cls, cfg):
                return True

        current, legacy = cfg(expert.V41_EXPERT_RECIPE), cfg()
        selected = strategy.V41GroupedFP4Strategy
        with patch.object(base, "_STRATEGY_PRIORITY", [Legacy, selected]), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            self.assertIs(base.select_strategy(legacy), Legacy)
            self.assertIs(base.select_strategy(current), selected)
            for forced in ("legacy", "sm120_decode", "mega"):
                with self.assertRaisesRegex(RuntimeError, "recipe"):
                    base.select_strategy(current, forced=forced, strict=False)
            with patch.dict(sys.modules, MODULES):
                distributed = dataclasses.replace(current, ep_size=8)
            with self.assertRaisesRegex(RuntimeError, "transport"):
                base.select_strategy(distributed)
        with self.assertRaisesRegex(ValueError, "recipe"):
            Legacy(current)
        with self.assertRaisesRegex(ValueError, "recipe"):
            selected(legacy)

    def test_strategy_weight_binding_is_gate_first_and_failure_preserves_loader_state(
        self,
    ):
        local, tensors, x, weights, ids, valid = self.fixtures(dim=128, inter=128)
        names = {
            "v4_routed_" + part + "_" + kind: part + "_" + kind
            for part in ("w1", "w2", "w3")
            for kind in ("w", "s")
        }
        fake_w = types.SimpleNamespace(W=types.SimpleNamespace(**names))
        layer = {"keep": object()}
        for suffix in ("w", "s"):
            layer["w1_" + suffix] = tensors["w13_" + suffix][:, :128].contiguous()
            layer["w3_" + suffix] = tensors["w13_" + suffix][:, 128:].contiguous()
            layer["w2_" + suffix] = tensors["w2_" + suffix]
        op = strategy.V41GroupedFP4Strategy(cfg(expert.V41_EXPERT_RECIPE))
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            op(x, weights, ids)
        with patch.dict(
            sys.modules, {"rtp_llm.utils.model_weight": fake_w}
        ), patch.object(
            strategy, "V41RoutedExperts", side_effect=RuntimeError("missing provider")
        ):
            with self.assertRaisesRegex(RuntimeError, "missing provider"):
                op.setup_weights(layer)
        self.assertEqual(len(layer), 7)
        real = routed.V41RoutedExperts
        with patch.dict(
            sys.modules, {"rtp_llm.utils.model_weight": fake_w}
        ), patch.object(
            strategy,
            "V41RoutedExperts",
            side_effect=lambda tensors, **kw: real(tensors, backend="eager", **kw),
        ):
            op.setup_weights(layer)
            with self.assertRaisesRegex(RuntimeError, "already initialized"):
                op.setup_weights(layer)
        self.assertEqual(list(layer), ["keep"])
        torch.testing.assert_close(
            op.local.experts.w13, tensors["w13_w"], rtol=0, atol=0
        )
        safe_weights = torch.where(valid[:, None], weights, 0)
        torch.testing.assert_close(
            op(x, safe_weights, ids),
            local(x, weights, ids, valid_rows=valid),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
