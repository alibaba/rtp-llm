import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.pp_layout import (
    even_split_counts,
    get_pp_partitioner,
    pp_layer_range_from_counts,
    register_pp_partitioner,
    resolve_pp_partition,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.ops import TaskType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def make_load_config(
    num_layers: int = 8,
    pp_size: int = 1,
    pp_rank: int = 0,
    pp_stage_layer_counts=None,
) -> LoadConfig:
    # Fixtures materialize the even split for pp>1 unless a case supplies its own.
    if pp_size > 1 and pp_stage_layer_counts is None:
        pp_stage_layer_counts = even_split_counts(num_layers, pp_size)
    database = MagicMock(spec=CkptDatabase)
    return LoadConfig(
        database=database,
        num_layers=num_layers,
        hidden_size=16,
        head_num=4,
        head_num_kv=4,
        size_per_head=4,
        moe_pure_tp_mode=False,
        align_size=1,
        moe_align_size=1,
        moe_layer_index=[],
        moe_n_group=0,
        expert_num=0,
        enable_eplb=False,
        phy_exp_num=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=1,
        lm_head_tp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        num_nodes=1,
        compute_dtype=torch.float16,
        pp_size=pp_size,
        pp_rank=pp_rank,
        pp_stage_layer_counts=pp_stage_layer_counts,
    )


def make_loader(
    load_config: LoadConfig, task_type=TaskType.LANGUAGE_MODEL,
    apply_pp_partition: bool = True,
) -> ModelLoader:
    weights_info = MagicMock(
        is_attn_model=False,
        pp_size=load_config.pp_size,
        pp_rank=load_config.pp_rank,
        pp_stage_layer_counts=load_config.pp_stage_layer_counts,
    )
    weights_info.create_load_config.return_value = load_config
    # Exercise scope selection in the real constructor without device/EPLB setup.
    with patch.object(ModelLoader, "create_eplb", return_value=(None, None)), patch(
        "rtp_llm.device.get_current_device"
    ):
        return ModelLoader(
            SimpleNamespace(
                task_type=task_type,
                compute_dtype=load_config.compute_dtype,
                num_layers=load_config.num_layers,
            ),
            weights_info, [], load_config.database,
            apply_pp_partition=apply_pp_partition,
        )


class FakeWeight:
    def __init__(self, name: str):
        self.name = name


def make_eplb_loader(cfg, moe_layers, apply_pp_partition=True, enabled=True):
    """Exercise the real loader and balancer with small metadata and no device I/O."""
    cfg.moe_layer_index = moe_layers
    cfg.expert_num = 2
    cfg.phy_exp_num = 4
    cfg.enable_eplb = enabled
    model_config = SimpleNamespace(
        task_type=TaskType.LANGUAGE_MODEL,
        compute_dtype=cfg.compute_dtype,
        num_layers=cfg.num_layers,
        expert_num=cfg.expert_num,
        phy2log_path=None,
        ckpt_path="unused-checkpoint",
        eplb_config=SimpleNamespace(
            balance_method="mix", eplb_stats_window_size=2, eplb_force_repack=0
        ),
    )
    weights_info = MagicMock(
        is_attn_model=False,
        pp_size=cfg.pp_size,
        pp_rank=cfg.pp_rank,
        pp_stage_layer_counts=cfg.pp_stage_layer_counts,
        expert_num_=cfg.expert_num,
        phy_exp_num_=cfg.phy_exp_num,
        ep_size=cfg.ep_size,
        num_nodes=cfg.num_nodes,
        moe_layer_index_=moe_layers,
        moe_style_=1 if moe_layers else 0,
        enable_eplb_=enabled,
    )

    def load_config(**kwargs):
        cfg.phy2log = kwargs["phy2log"]
        cfg.exported_device = kwargs["exported_device"]
        return cfg

    weights_info.create_load_config.side_effect = load_config
    device = SimpleNamespace(support_dio_load=False)
    with patch("rtp_llm.model_loader.loader.CkptDatabase", return_value=cfg.database), patch(
        "rtp_llm.device.get_current_device", return_value=device
    ), patch("rtp_llm.eplb.ep_balancer.get_current_device", return_value=device):
        return ModelLoader(
            model_config, weights_info, None, cfg.database,
            apply_pp_partition=apply_pp_partition,
        )


class PPEplbLoadingTest(unittest.TestCase):
    def test_balancer_scope_and_mapping_initialization(self):
        cases = [
            # Global MoE layers are [1, 3, 5, 7]; stage 0 has none here.
            (8, 3, 0, [1, 3, 4], True, [1, 3, 5, 7], []),
            (8, 3, 1, [1, 3, 4], True, [1, 3, 5, 7], [1, 3]),
            (8, 3, 2, [1, 3, 4], True, [1, 3, 5, 7], [5, 7]),
            (8, 1, 0, None, True, [1, 3, 5, 7], [1, 3, 5, 7]),
            (8, 1, 0, None, True, [], []),
            # A draft owns both of its layers despite the target's PP layout.
            (2, 3, 2, [1, 3, 4], False, [0, 1], [0, 1]),
        ]
        for layers, pp, rank, counts, partition, moe_layers, expected in cases:
            with self.subTest(layers=layers, pp=pp, rank=rank, partition=partition):
                cfg = make_load_config(layers, pp, rank, counts)
                loader = make_eplb_loader(cfg, moe_layers, partition)
                if expected:
                    self.assertEqual(loader._py_eplb.moe_layer_index, expected)
                else:
                    self.assertIsNone(loader._py_eplb)
                weights = ModelWeights(layers, "cpu", torch.float32)
                loader._init_eplb_weight(weights, "cpu")
                self.assertEqual(
                    [i for i, row in enumerate(weights.weights) if W.log2phy in row],
                    expected,
                )
                for layer_id in expected:
                    # EP=1 initial layout is [0, 1, 0, 1].
                    torch.testing.assert_close(
                        weights.weights[layer_id][W.logic_expert_cnt],
                        torch.tensor([2, 2], dtype=torch.int32),
                    )
                    torch.testing.assert_close(
                        weights.weights[layer_id][W.log2phy],
                        torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32),
                    )

    def test_redundant_mapping_without_dynamic_balancer(self):
        cfg = make_load_config(8, 2, 1, [4, 4])
        loader = make_eplb_loader(cfg, [1, 3, 5, 7], enabled=False)
        self.assertIsNone(loader._py_eplb)
        weights = ModelWeights(8, "cpu", torch.float32)
        loader._init_eplb_weight(weights, "cpu")
        self.assertEqual(
            [i for i, row in enumerate(weights.weights) if W.log2phy in row], [5, 7]
        )

    def test_balancer_never_selects_another_stage(self):
        from rtp_llm.eplb.ep_balancer import SelectLayerMethod

        loader = make_eplb_loader(make_load_config(8, 2, 1), [1, 3, 5, 7])
        balancer = loader._py_eplb
        gpu_loads = torch.tensor([[0], [1000], [0], [900], [0], [10], [0], [20]])
        balancer.select_layer_method = SelectLayerMethod.ROUND
        self.assertEqual([balancer.get_balanced_layer(gpu_loads) for _ in range(4)], [5, 7, 5, 7])
        balancer.select_layer_method = SelectLayerMethod.MOST_UNBALANCED_LAYER
        self.assertEqual(balancer.get_balanced_layer(gpu_loads), 7)
        for method in (SelectLayerMethod.RANDOM, SelectLayerMethod.MIX):
            balancer.select_layer_method = method
            for step in range(8):
                balancer.update_cnt = step
                self.assertIn(balancer.get_balanced_layer(gpu_loads), [5, 7])

    def test_static_and_dynamic_use_the_same_ep_node_count(self):
        for ep_size, num_nodes in ((1, 1), (2, 1), (2, 2)):
            with self.subTest(ep_size=ep_size, num_nodes=num_nodes):
                cfg = make_load_config(8, 2, 1)
                cfg.ep_size = ep_size
                cfg.num_nodes = num_nodes
                with patch.object(
                    LoadConfig, "create_redundant_expert",
                    wraps=LoadConfig.create_redundant_expert,
                ) as static_placement:
                    loader = make_eplb_loader(cfg, [1, 3, 5, 7])
                self.assertEqual(static_placement.call_args.kwargs["num_nodes"], num_nodes)
                self.assertEqual(loader._py_eplb.num_nodes, num_nodes)
                self.assertEqual(loader._py_eplb.num_gpu, ep_size)

    def test_ft_loading_reads_selected_layers_and_globals(self):
        for pp, rank, partition, expected in (
            (2, 0, True, [0, 1]),
            (2, 1, True, list(range(2, 12))),
            (2, 1, False, list(range(12))),
            (1, 0, True, list(range(12))),
        ):
            with self.subTest(pp=pp, rank=rank, partition=partition):
                cfg = make_load_config(12, pp, rank, [2, 10] if pp > 1 else None)
                cfg.exported_device = SimpleNamespace(support_dio_load=False)
                loader = make_loader(cfg, apply_pp_partition=partition)
                layer_prefix = ModelWeights.layer_weight_prefix(0, 0, 0)
                global_prefix = ModelWeights.global_weight_prefix(0, 0, 0)
                checkpoint = {
                    f"{layer_prefix}{i}.weight": [torch.tensor([i])] for i in range(12)
                }
                checkpoint[f"{global_prefix}{W.embedding}"] = [torch.tensor([99])]
                checkpoint["rank_01_00_00.layers.0.weight"] = [torch.tensor([-1])]
                cfg.database.load_tensors_by_prefix.side_effect = (
                    lambda prefixes, device, direct_io: {
                        key: value for key, value in checkpoint.items() if key.startswith(prefixes)
                    }
                )
                result = loader._load_from_ft_style("cpu")
                self.assertEqual([i for i, row in enumerate(result.weights) if row], expected)
                for i in expected:
                    self.assertEqual(result.weights[i]["weight"].item(), i)
                self.assertEqual(result.global_weights[W.embedding].item(), 99)


class PPLayerRangeTest(unittest.TestCase):
    def test_pp1_degenerate_full_range(self):
        cfg = make_load_config(num_layers=8, pp_size=1, pp_rank=0)
        self.assertEqual(list(cfg.pp_layer_range()), list(range(8)))

    def test_ranges_tile_completely(self):
        for num_layers, pp_size in [(1, 1), (7, 3), (64, 4), (61, 5)]:
            covered = []
            for pp_rank in range(pp_size):
                cfg = make_load_config(
                    num_layers=num_layers, pp_size=pp_size, pp_rank=pp_rank
                )
                covered.extend(cfg.pp_layer_range())
            self.assertEqual(
                covered, list(range(num_layers)), f"layers={num_layers} pp={pp_size}"
            )

    def test_capability_flags(self):
        cfg = make_load_config(pp_size=1, pp_rank=0)
        self.assertTrue(cfg.has_pp_embedding)
        self.assertTrue(cfg.has_pp_lm_head)
        flags = []
        for pp_rank in range(3):
            cfg = make_load_config(pp_size=3, pp_rank=pp_rank)
            flags.append((cfg.has_pp_embedding, cfg.has_pp_lm_head))
        self.assertEqual(flags, [(True, False), (False, False), (False, True)])


class DraftPPLoadingTest(unittest.TestCase):
    def test_draft_keeps_all_layers_and_globals_with_target_partition(self):
        for pp_size, pp_rank, counts in [(1, 0, None), (3, 2, [2, 3, 2])]:
            with self.subTest(pp_size=pp_size):
                cfg = make_load_config(
                    num_layers=2,
                    pp_size=pp_size,
                    pp_rank=pp_rank,
                    pp_stage_layer_counts=counts,
                )
                cfg.tp_size, cfg.tp_rank = 2, 1
                cfg.dp_size, cfg.dp_rank = 2, 1
                loader = make_loader(cfg, apply_pp_partition=False)
                for name in PPSkipWeightTest.GLOBAL_NAMES:
                    self.assertFalse(loader._maybe_skip_weight(FakeWeight(name)))
                self.assertEqual((cfg.tp_size, cfg.tp_rank, cfg.dp_size, cfg.dp_rank), (2, 1, 2, 1))
                self.assertEqual((cfg.pp_size, cfg.pp_rank, cfg.pp_stage_layer_counts), (pp_size, pp_rank, counts))

    def test_both_loader_paths_follow_selected_range(self):
        for num_layers, pp_size, pp_rank, apply_pp, expected in (
            (7, 3, 0, True, [0, 1]),
            (7, 3, 1, True, [2, 3, 4]),
            (7, 3, 2, True, [5, 6]),
            (2, 3, 2, False, [0, 1]),
            (2, 1, 0, False, [0, 1]),
            (2, 1, 0, True, [0, 1]),
        ):
            with self.subTest(pp_size=pp_size, pp_rank=pp_rank, apply_pp=apply_pp):
                cfg = make_load_config(
                    num_layers=num_layers, pp_size=pp_size, pp_rank=pp_rank,
                    pp_stage_layer_counts=[2, 3, 2] if pp_size > 1 else None,
                )
                loader = make_loader(cfg, apply_pp_partition=apply_pp)
                weights = [MagicMock() for _ in range(num_layers)]
                for idx, weight in enumerate(weights):
                    weight.get_components.return_value = [weight]
                    weight.get_tensor_names.return_value = [f"layer.{idx}"]
                loader._model_weights_info = SimpleNamespace(
                    layer_weights=[[weight] for weight in weights], weights=[]
                )
                loader._load_layer_weights = MagicMock(return_value={"weight": torch.ones(1)})
                self.assertEqual([row[0] for row in loader.prepare_weights("cpu")], expected)
                with patch("rtp_llm.model_loader.loader.TensorCollector"):
                    _, weight_infos = loader._generate_weight_info()
                self.assertEqual([info.layer_id for info in weight_infos], expected)

    def test_full_loading_preserves_alias_and_task_filters(self):
        cfg = make_load_config(num_layers=2, pp_size=3, pp_rank=2, pp_stage_layer_counts=[2, 3, 2])
        loader = make_loader(cfg, apply_pp_partition=False)
        loader._global_weight_aliases = {W.embedding: torch.ones(1)}
        self.assertTrue(loader._maybe_skip_weight(FakeWeight(W.embedding)))
        self.assertFalse(loader._maybe_skip_weight(FakeWeight(W.lm_head)))
        loader = make_loader(cfg, TaskType.DENSE_EMBEDDING, apply_pp_partition=False)
        self.assertTrue(loader._maybe_skip_weight(FakeWeight(W.lm_head)))

    def test_lm_head_fallback_uses_selected_ownership(self):
        for apply_pp in (True, False):
            with self.subTest(apply_pp=apply_pp):
                cfg = make_load_config(num_layers=2, pp_size=3, pp_rank=0, pp_stage_layer_counts=[2, 3, 2])
                loader = make_loader(cfg, apply_pp_partition=apply_pp)
                loader._weights_info.model_config = SimpleNamespace(
                    normalize_lm_head_weight=False, logit_scale=1.0, vocab_size=4,
                )
                embedding = torch.ones(4, 4)
                weights = MagicMock(global_weights={W.embedding: embedding})
                weights.steal_global_weight.return_value = None
                loader._load_dynamic_weights(weights, "cpu")
                if apply_pp:
                    weights.steal_global_weight.assert_not_called()
                    weights.set_global_weight.assert_not_called()
                else:
                    weights.steal_global_weight.assert_called_once_with(W.lm_head)
                    weights.set_global_weight.assert_called_once_with(W.lm_head, embedding)


class PPMaterializedPartitionTest(unittest.TestCase):
    """The materialized partition is consumed by prefix-sum lookup."""

    def test_lookup_follows_materialized_data(self):
        # counts disagree with the even split on purpose: lookup must follow the data.
        stage0 = make_load_config(
            num_layers=8, pp_size=2, pp_rank=0, pp_stage_layer_counts=[3, 5]
        )
        stage1 = make_load_config(
            num_layers=8, pp_size=2, pp_rank=1, pp_stage_layer_counts=[3, 5]
        )
        self.assertEqual(list(stage0.pp_layer_range()), [0, 1, 2])
        self.assertEqual(list(stage1.pp_layer_range()), [3, 4, 5, 6, 7])

    def test_counts_materialized_even_split_matches_lookup(self):
        expected = {
            (65, 4): [(0, 17), (17, 33), (33, 49), (49, 65)],
            (9, 2): [(0, 5), (5, 9)],
            (64, 4): [(0, 16), (16, 32), (32, 48), (48, 64)],
        }
        for (num_layers, pp_size), ranges in expected.items():
            counts = resolve_pp_partition(num_layers, pp_size)
            for pp_rank in range(pp_size):
                cfg = make_load_config(
                    num_layers=num_layers,
                    pp_size=pp_size,
                    pp_rank=pp_rank,
                    pp_stage_layer_counts=counts,
                )
                begin, end = ranges[pp_rank]
                self.assertEqual(
                    list(cfg.pp_layer_range()),
                    list(range(begin, end)),
                    f"layers={num_layers} pp={pp_size} rank={pp_rank}",
                )

    def test_pp_gt_1_without_counts_is_an_error(self):
        # pp>1 must carry materialized data; no silent algorithm fallback.
        cfg = make_load_config(
            num_layers=8, pp_size=2, pp_rank=0, pp_stage_layer_counts=None
        )
        # The fixture auto-materializes by default; force the empty state.
        cfg.pp_stage_layer_counts = None
        with self.assertRaises(ValueError):
            cfg.pp_layer_range()

    def test_range_from_counts_rejects_bad_rank(self):
        with self.assertRaises(ValueError):
            pp_layer_range_from_counts([4, 4], 2)
        with self.assertRaises(ValueError):
            pp_layer_range_from_counts([4, 4], -1)


class PPResolvePartitionTest(unittest.TestCase):
    """resolve_pp_partition: even split, validation, partitioner registry."""

    def test_default_even_split_golden_values(self):
        self.assertEqual(resolve_pp_partition(65, 4), [17, 16, 16, 16])
        self.assertEqual(resolve_pp_partition(64, 4), [16, 16, 16, 16])
        self.assertEqual(even_split_counts(9, 2), [5, 4])
        self.assertEqual(resolve_pp_partition(64, 1), [64])

    def test_rejects_bad_partitioner_output(self):
        register_pp_partitioner("__bad_len__", lambda n, p, mc: [n])
        try:
            with self.assertRaises(ValueError):
                resolve_pp_partition(8, 2, MagicMock(model_type="__bad_len__"))
        finally:
            self._unregister("__bad_len__")

        register_pp_partitioner("__bad_sum__", lambda n, p, mc: [4, 5])
        try:
            with self.assertRaises(ValueError):
                resolve_pp_partition(8, 2, MagicMock(model_type="__bad_sum__"))
        finally:
            self._unregister("__bad_sum__")

        register_pp_partitioner("__bad_zero__", lambda n, p, mc: [8, 0])
        try:
            with self.assertRaises(ValueError):
                resolve_pp_partition(8, 2, MagicMock(model_type="__bad_zero__"))
        finally:
            self._unregister("__bad_zero__")

    def test_registered_partitioner_wins(self):
        register_pp_partitioner("__shaped__", lambda n, p, mc: [2, 6])
        try:
            counts = resolve_pp_partition(8, 2, MagicMock(model_type="__shaped__"))
            self.assertEqual(counts, [2, 6])
            self.assertIsNotNone(get_pp_partitioner("__shaped__"))
        finally:
            self._unregister("__shaped__")
        # Unregistered model type falls back to the even split.
        self.assertIsNone(get_pp_partitioner("__shaped__"))
        self.assertEqual(
            resolve_pp_partition(8, 2, MagicMock(model_type="__other__")), [4, 4]
        )

    @staticmethod
    def _unregister(model_type: str):
        from rtp_llm.config import pp_layout

        pp_layout._PP_PARTITIONERS.pop(model_type, None)


class PPSkipWeightTest(unittest.TestCase):
    GLOBAL_NAMES = [
        W.embedding,
        W.positional_embedding,
        W.lm_head,
        W.final_ln_gamma,
        W.final_ln_beta,
        "some_other_global",
    ]

    def skipped_names(
        self, pp_size: int, pp_rank: int, task_type=TaskType.LANGUAGE_MODEL
    ):
        loader = make_loader(
            make_load_config(pp_size=pp_size, pp_rank=pp_rank), task_type
        )
        return {
            name
            for name in self.GLOBAL_NAMES
            if loader._maybe_skip_weight(FakeWeight(name))
        }

    def test_pp1_skips_nothing(self):
        self.assertEqual(self.skipped_names(pp_size=1, pp_rank=0), set())

    def test_first_stage_skips_tail_parts(self):
        skipped = self.skipped_names(pp_size=2, pp_rank=0)
        self.assertEqual(skipped, {W.lm_head, W.final_ln_gamma, W.final_ln_beta})
        self.assertNotIn(W.embedding, skipped)

    def test_last_stage_skips_head_parts(self):
        skipped = self.skipped_names(pp_size=2, pp_rank=1)
        self.assertEqual(skipped, {W.embedding, W.positional_embedding})
        self.assertNotIn(W.lm_head, skipped)
        self.assertNotIn(W.final_ln_gamma, skipped)

    def test_middle_stage_skips_both_ends(self):
        skipped = self.skipped_names(pp_size=3, pp_rank=1)
        self.assertEqual(
            skipped,
            {
                W.embedding,
                W.positional_embedding,
                W.lm_head,
                W.final_ln_gamma,
                W.final_ln_beta,
            },
        )

    def test_non_lm_task_still_skips_lm_head(self):
        skipped = self.skipped_names(
            pp_size=1, pp_rank=0, task_type=TaskType.DENSE_EMBEDDING
        )
        self.assertIn(W.lm_head, skipped)


if __name__ == "__main__":
    unittest.main()
