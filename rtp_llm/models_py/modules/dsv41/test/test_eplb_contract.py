"""CPU contracts for the private V4.1 MoE mapping and resident ownership."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.moe import moe_layer
from rtp_llm.models_py.modules.dsv41.moe.strategies.base import (
    MoeCfg,
    RoutedExpertsStrategy,
    select_strategy,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies.deepep import DeepEPStrategy
from rtp_llm.models_py.modules.dsv41.moe.strategies.grouped_fp4 import (
    GroupedFP4Strategy,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies.local_loop import LocalLoopStrategy
from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy
from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_se import MegaMoEStrategySE
from rtp_llm.utils.model_weight import W


class _Gate(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        experts,
        topk,
        score_func,
        route_scale,
        n_hash_layers,
        vocab_size,
        layer_weights,
    ):
        super().__init__()
        self.experts = experts
        self.route_scale = route_scale

    def forward(self, x, ids):
        indices = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)[: len(x)]
        return torch.ones_like(indices, dtype=torch.float32), indices


class _Strategy(RoutedExpertsStrategy):
    routed_includes_shared = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = []

    def can_use_gate_pack_static(self, gate):
        return True

    def setup_weights(self, weights):
        self.weight = torch.arange(self.cfg.n_local_experts).reshape(-1, 1)

    def get_eplb_weights(self):
        return (self.weight,)

    def setup_runtime(self):
        self.warmup_gate_pack = self._gate_pack_warmup_enabled

    def forward(self, x, weights, indices, expert_load=None, eplb_layer=None):
        self.finalize_routing(indices, weights, eplb_layer)
        self.calls.append((indices.clone(), expert_load))
        return x + 1

    def forward_with_gate_pack(
        self, x, gate, input_ids, expert_load=None, eplb_layer=None
    ):
        weights, indices = gate(x, input_ids)
        self.finalize_routing(indices, weights, eplb_layer)
        self.calls.append(("gate_pack", expert_load))
        return x + 1


class _Standalone(_Strategy):
    routed_includes_shared = False


class _Shared:
    def prepare(self, module):
        pass

    def start(self, module, x):
        self.result = torch.zeros_like(x)

    def finish(self):
        return self.result


def _cfg(physical=6):
    return MoeCfg(0, 8, 128, 4, 2, 10.0, 2, 1, 3, 3, 6, 16, n_physical_experts=physical)


class EplbPrivateMoeContractTest(unittest.TestCase):
    def build(
        self,
        *,
        mapped=True,
        standalone=False,
        rank=1,
        physical=6,
        missing_map=None,
        ep_size=2,
    ):
        weights = {
            W.v4_shared_w13_w: torch.zeros(4, 2),
            W.v4_shared_w13_s: torch.ones(1, 1),
            W.v4_shared_w2_w: torch.zeros(2, 2),
            W.v4_shared_w2_s: torch.ones(1, 1),
        }
        if mapped:
            mapping = torch.full((4, max(physical - 4 + 1, 1)), -1, dtype=torch.int32)
            counts = torch.zeros(4, dtype=torch.int32)
            for slot in range(physical):
                logical = slot % 4
                mapping[logical, counts[logical]] = slot
                counts[logical] += 1
            weights[W.log2phy] = mapping
            weights[W.logic_expert_cnt] = counts
        if missing_map is not None:
            del weights[missing_map]
        with mock.patch.dict(os.environ, {"MOEDBG": "0"}), mock.patch.object(
            moe_layer, "Gate", _Gate
        ), mock.patch.object(
            moe_layer,
            "select_strategy",
            return_value=_Standalone if standalone else _Strategy,
        ), mock.patch.object(
            moe_layer, "W13SharedExpert", return_value=nn.Identity()
        ), mock.patch.object(
            moe_layer, "get_shared_expert_executor", return_value=_Shared()
        ):
            return moe_layer.MoE(
                0,
                2,
                2,
                4,
                2,
                1,
                "sqrtsoftplus",
                1.5,
                10.0,
                0,
                8,
                layer_weights=weights,
                ep_size=ep_size,
                ep_rank=rank,
                n_physical_experts=physical if mapped else None,
            )

    def test_physical_capacity_and_none_path(self):
        mapped = self.build()
        self.assertEqual(mapped.gate.experts, 4)
        self.assertEqual(mapped._strategy.cfg.physical_expert_num, 6)
        self.assertEqual(
            (
                mapped.n_local_experts,
                mapped.local_expert_start,
                mapped.local_expert_end,
            ),
            (3, 3, 6),
        )
        self.assertIs(mapped.eplb_layer.weights[0], mapped._strategy.weight)
        self.assertTrue(mapped._gate_pack_static)
        self.assertTrue(mapped._strategy.warmup_gate_pack)
        unmapped = self.build(mapped=False)
        self.assertIsNone(unmapped.eplb_layer)
        self.assertEqual(unmapped._strategy.cfg.physical_expert_num, 4)
        self.assertTrue(unmapped._gate_pack_static)
        self.assertTrue(unmapped._strategy.warmup_gate_pack)
        with mock.patch.object(unmapped, "_should_chunk", return_value=False):
            unmapped(torch.zeros(2, 2), torch.arange(2), expert_load=None)
        self.assertEqual(unmapped._strategy.calls, [("gate_pack", None)])

    def test_invalid_capacity_and_partial_maps_are_rejected(self):
        for physical in (3, 5):
            with self.subTest(physical=physical), self.assertRaisesRegex(
                ValueError, "Physical experts must cover logical experts"
            ):
                self.build(physical=physical)
        for key in (W.log2phy, W.logic_expert_cnt):
            with self.subTest(missing=key), self.assertRaisesRegex(
                ValueError, "requires both log2phy and logic_expert_cnt"
            ):
                self.build(missing_map=key)

    def test_both_materialized_gate_paths_map_before_dispatch(self):
        context = object()
        x = torch.zeros(2, 2)
        ids = torch.arange(2)
        for standalone in (False, True):
            for chunk in (False, True):
                with self.subTest(standalone=standalone, chunk=chunk):
                    model = self.build(standalone=standalone)
                    model._gate_pack_static = False
                    with mock.patch.object(
                        model, "_should_chunk", return_value=False
                    ), mock.patch.object(
                        moe_layer,
                        "combine_routed_and_shared",
                        side_effect=lambda routed, shared, dtype, out: out.copy_(
                            routed + shared
                        ),
                    ):

                        def forward():
                            if chunk:
                                out = torch.empty_like(x)
                                model._run_chunk(x, ids, out, expert_load=context)
                                return out
                            return model(x, ids, expert_load=context)

                        torch.testing.assert_close(forward(), x + 1, rtol=0, atol=0)
                        # Rank 1 independent inputs: replica=(1+token*23333)%2.
                        # Both top-k columns of token 1 select replica zero.
                        self.assertEqual(
                            model._strategy.calls[-1][0].tolist(), [[4, -1], [1, 0]]
                        )
                        self.assertIs(model._strategy.calls[-1][1], context)
                        pointer = model.eplb_layer.logical_to_physical.data_ptr()
                        model.eplb_layer.logical_to_physical[[0, 1]] = (
                            model.eplb_layer.logical_to_physical[[1, 0]]
                        )
                        forward()
                        self.assertEqual(
                            model._strategy.calls[-1][0].tolist(), [[5, -1], [0, 1]]
                        )
                        self.assertEqual(
                            model.eplb_layer.logical_to_physical.data_ptr(), pointer
                        )

    def test_mega_input_streams_rotate_replicas_by_rank(self):
        # Even equal input rows are independent requests to Mega on each rank.
        ids = torch.tensor([[0, -1], [1, 0]])
        for rank, expected in (
            (0, [[0, -1], [5, 4]]),
            (1, [[4, -1], [1, 0]]),
        ):
            with self.subTest(rank=rank):
                layer = self.build(rank=rank).eplb_layer
                self.assertEqual(layer.replica_offset, rank)
                self.assertEqual(layer.map_ids(ids).tolist(), expected)

    def test_mega_entries_map_final_logical_ids_once_in_place(self):
        from rtp_llm.models_py.modules.dsv41.moe import mega_gate
        from rtp_llm.models_py.modules.dsv41.moe.strategies import mega, mega_se

        for module, cls in ((mega, MegaMoEStrategy), (mega_se, MegaMoEStrategySE)):
            for fake_balance in (False, True):
                with self.subTest(strategy=cls.__name__, fake=fake_balance):
                    strategy = cls.__new__(cls)
                    RoutedExpertsStrategy.__init__(strategy, _cfg())
                    x = torch.zeros(2, 8, dtype=torch.bfloat16)
                    buf = SimpleNamespace(
                        x=x,
                        x_sf=torch.zeros(2, 1),
                        shared_l1_acts_sf=torch.zeros(2, 1),
                        topk_idx=torch.full((2, 2), -1, dtype=torch.int64),
                        topk_weights=torch.zeros(2, 2),
                    )
                    strategy._mega_buf = buf
                    strategy._mega_y = torch.empty_like(x)
                    strategy._block_m = lambda _: 32
                    strategy._maybe_pre_kernel_barrier = lambda _: None
                    strategy._shared_recipe_kwargs = {}
                    for name in (
                        "_mega_l1_w",
                        "_mega_l1_sf",
                        "_mega_l2_w",
                        "_mega_l2_sf",
                        "_se_l1_w",
                        "_se_l1_sf",
                        "_se_l2_w",
                        "_se_l2_sf",
                    ):
                        setattr(strategy, name, None)
                    logical = torch.tensor([[0, 1], [1, 0]])
                    overridden = torch.tensor([[2, 3], [3, 2]])

                    def pack(*args, **kwargs):
                        buf.topk_idx.copy_(logical)
                        buf.topk_weights.fill_(0.25)

                    def override(indices, weights):
                        indices.copy_(overridden)
                        weights.fill_(1)

                    if fake_balance:
                        strategy.fake_balance_expert = override
                    final_logical = overridden if fake_balance else logical
                    layer = SimpleNamespace(
                        map_ids=mock.Mock(side_effect=lambda ids: ids + 10)
                    )
                    pointer = buf.topk_idx.data_ptr()
                    context = SimpleNamespace(
                        expert_token_counts=torch.zeros(3, dtype=torch.int32)
                    )

                    def consume(y, *args, **kwargs):
                        self.assertEqual(buf.topk_idx.data_ptr(), pointer)
                        torch.testing.assert_close(buf.topk_idx, final_logical + 10)
                        self.assertTrue(
                            torch.all(buf.topk_weights == (1 if fake_balance else 0.25))
                        )
                        self.assertIs(
                            kwargs["cumulative_local_expert_recv_stats"],
                            context.expert_token_counts,
                        )
                        y.fill_(2)

                    strategy._input_packer = SimpleNamespace(pack=pack)
                    gate = SimpleNamespace(
                        hash=False,
                        route_scale=1.5,
                        bias=torch.zeros(4),
                        tid2eid=torch.zeros(2, 2, dtype=torch.int64),
                        _weight_bf16=lambda: torch.zeros(4, 8, dtype=torch.bfloat16),
                        _project_scores=lambda x, w: x @ w.T,
                    )
                    getter = (
                        "_get_mega_se_gate_pack_kernels"
                        if module is mega_se
                        else "_get_gate_pack_kernels"
                    )
                    with mock.patch.object(
                        module, getter, return_value=(pack, pack, None)
                    ), mock.patch.object(
                        module, "sync_cuda_graph_warmup_ranks"
                    ), mock.patch.dict(
                        "sys.modules",
                        {"deep_gemm": SimpleNamespace(fp8_fp4_mega_moe=consume)},
                    ):
                        entries = [
                            lambda: strategy(
                                x,
                                buf.topk_weights,
                                buf.topk_idx,
                                expert_load=context,
                                eplb_layer=layer,
                            ),
                            lambda: strategy.forward_with_gate_pack(
                                x,
                                gate,
                                torch.arange(2),
                                expert_load=context,
                                eplb_layer=layer,
                            ),
                        ]
                        if module is mega_se:
                            inputs = SimpleNamespace(
                                owner=strategy,
                                x=x,
                                topk_idx=buf.topk_idx,
                                topk_weights=buf.topk_weights,
                            )
                            entries.append(
                                lambda: strategy.forward_from_mhc(
                                    x,
                                    gate,
                                    inputs,
                                    expert_load=context,
                                    eplb_layer=layer,
                                )
                            )
                        with mock.patch.object(
                            mega_gate, "mega_gate", side_effect=pack
                        ):
                            for entry in entries:
                                layer.map_ids.reset_mock()
                                torch.testing.assert_close(
                                    entry(), torch.full_like(x, 2)
                                )
                                layer.map_ids.assert_called_once()

    def test_four_mega_input_streams_use_physical_rank_phase(self):
        # Three replicas expose wrong phases on both ranks 2 and 3.
        ids = torch.tensor([[0, -1], [1, 0]])
        for rank, expected in (
            (0, [[0, -1], [9, 8]]),
            (1, [[4, -1], [1, 0]]),
            (2, [[8, -1], [5, 4]]),
            (3, [[0, -1], [9, 8]]),
        ):
            with self.subTest(rank=rank):
                model = self.build(rank=rank, ep_size=4, physical=12)
                self.assertEqual(model.n_local_experts, 3)
                self.assertEqual(model.local_expert_start, rank * 3)
                self.assertEqual(model.eplb_layer.replica_offset, rank)
                self.assertEqual(model.eplb_layer.map_ids(ids).tolist(), expected)

    def test_ep_rejects_strategies_without_global_dispatch(self):
        for strategy in ("local_loop", "deepep"):
            with self.subTest(strategy=strategy), mock.patch.dict(
                os.environ, {"DSV4_MOE_STRATEGY": strategy}
            ), self.assertRaisesRegex(RuntimeError, "would bypass Mega"):
                select_strategy(_cfg(), forced=strategy)

    def test_all_resident_weight_owners_are_exposed(self):
        cases = (
            (
                MegaMoEStrategy,
                ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"),
            ),
            (
                MegaMoEStrategySE,
                ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"),
            ),
            (
                GroupedFP4Strategy,
                ("_w13", "_s13", "_w2", "_s2", "_s13_dense_t", "_s2_dense_t"),
            ),
            (
                LocalLoopStrategy,
                (
                    "_W1_w",
                    "_W1_s",
                    "_W1_s_gemm",
                    "_W2_w",
                    "_W2_s",
                    "_W2_s_gemm",
                    "_W3_w",
                    "_W3_s",
                    "_W3_s_gemm",
                ),
            ),
        )
        for cls, names in cases:
            with self.subTest(strategy=cls.__name__):
                strategy = cls.__new__(cls)
                RoutedExpertsStrategy.__init__(strategy, _cfg())
                expected = tuple(
                    torch.arange(12).reshape(3, 2, 2).transpose(1, 2) + i
                    for i in range(len(names))
                )
                for name, tensor in zip(names, expected):
                    setattr(strategy, name, tensor)
                self.assertEqual(len(strategy.get_eplb_weights()), len(expected))
                for actual, tensor in zip(strategy.get_eplb_weights(), expected):
                    self.assertIs(actual, tensor)
                if cls is LocalLoopStrategy:
                    deepep = DeepEPStrategy(_cfg())
                    deepep._local = strategy
                    self.assertEqual(
                        tuple(map(id, deepep.get_eplb_weights())),
                        tuple(map(id, expected)),
                    )

    def test_mega_runtime_and_block_m_use_physical_capacity(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies import mega, mega_se

        for module, cls, allocator, output in (
            (
                mega,
                MegaMoEStrategy,
                "_get_or_create_mega_buf",
                "_get_or_create_mega_output",
            ),
            (
                mega_se,
                MegaMoEStrategySE,
                "_get_or_create_mega_se_buf",
                "_get_or_create_mega_se_output",
            ),
        ):
            with self.subTest(strategy=cls.__name__):
                strategy = cls.__new__(cls)
                RoutedExpertsStrategy.__init__(strategy, _cfg())
                strategy._mega_runtime_device = torch.device("cpu")
                buf = SimpleNamespace(num_max_tokens_per_rank=16)
                with mock.patch.object(
                    torch.distributed, "is_initialized", return_value=True
                ), mock.patch.object(
                    module, allocator, return_value=buf
                ) as allocate, mock.patch.object(
                    module, output, return_value=torch.zeros(16, 8)
                ), mock.patch.object(
                    strategy, "_maybe_warmup_jit_once"
                ), mock.patch.object(
                    mega, "_mega_intermediate_size", return_value=128
                ), mock.patch.object(
                    module,
                    (
                        "get_mega_moe_se_input_packer"
                        if cls is MegaMoEStrategySE
                        else "get_mega_moe_input_packer"
                    ),
                    return_value=object(),
                ):
                    strategy.setup_runtime()
                self.assertEqual(allocate.call_args.kwargs["num_experts"], 6)
                if cls is MegaMoEStrategySE:
                    get_block = mock.Mock(return_value=64)
                    with mock.patch.dict(
                        "sys.modules",
                        {
                            "deep_gemm": SimpleNamespace(
                                get_block_m_for_mega_moe=get_block
                            )
                        },
                    ):
                        self.assertEqual(strategy._block_m(2), 64)
                    self.assertEqual(get_block.call_args.args[:2], (2, 6))


if __name__ == "__main__":
    unittest.main()
