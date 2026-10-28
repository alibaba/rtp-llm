import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.ops import TaskType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def make_load_config(
    num_layers: int = 8,
    layer_range: range | None = None,
) -> LoadConfig:
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
        layer_range=range(num_layers) if layer_range is None else layer_range,
    )


def make_loader(
    load_config: LoadConfig,
    task_type=TaskType.LANGUAGE_MODEL,
    misc_weights_info=None,
) -> ModelLoader:
    weights_info = MagicMock(is_attn_model=False, layer_range=load_config.layer_range)
    weights_info.create_load_config.return_value = load_config
    with patch.object(ModelLoader, "create_eplb", return_value=(None, None)), patch(
        "rtp_llm.device.get_current_device"
    ):
        return ModelLoader(
            SimpleNamespace(
                task_type=task_type,
                compute_dtype=load_config.compute_dtype,
                num_layers=load_config.num_layers,
            ),
            weights_info,
            misc_weights_info,
            load_config.database,
        )


class FakeWeight:
    def __init__(self, name: str):
        self.name = name


def make_eplb_loader(cfg, moe_layers, *, enabled=True):
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
        layer_range=cfg.layer_range,
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
    with patch(
        "rtp_llm.model_loader.loader.CkptDatabase", return_value=cfg.database
    ), patch("rtp_llm.device.get_current_device", return_value=device), patch(
        "rtp_llm.eplb.ep_balancer.get_current_device", return_value=device
    ):
        return ModelLoader(
            model_config,
            weights_info,
            None,
            cfg.database,
        )


class PPEplbLoadingTest(unittest.TestCase):
    def test_balancer_scope_and_mapping_initialization(self):
        cases = [
            (8, range(1), [1, 3, 5, 7], []),
            (8, range(1, 4), [1, 3, 5, 7], [1, 3]),
            (8, range(4, 8), [1, 3, 5, 7], [5, 7]),
            (8, range(8), [1, 3, 5, 7], [1, 3, 5, 7]),
            (8, range(8), [], []),
            (2, range(2), [0, 1], [0, 1]),
        ]
        for layers, layer_range, moe_layers, expected in cases:
            with self.subTest(layers=layers, layer_range=layer_range):
                cfg = make_load_config(layers, layer_range)
                loader = make_eplb_loader(cfg, moe_layers)
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
        cfg = make_load_config(8, range(4, 8))
        loader = make_eplb_loader(cfg, [1, 3, 5, 7], enabled=False)
        self.assertIsNone(loader._py_eplb)
        weights = ModelWeights(8, "cpu", torch.float32)
        loader._init_eplb_weight(weights, "cpu")
        self.assertEqual(
            [i for i, row in enumerate(weights.weights) if W.log2phy in row], [5, 7]
        )

    def test_balancer_never_selects_another_stage(self):
        from rtp_llm.eplb.ep_balancer import SelectLayerMethod

        loader = make_eplb_loader(make_load_config(8, range(4, 8)), [1, 3, 5, 7])
        balancer = loader._py_eplb
        gpu_loads = torch.tensor([[0], [1000], [0], [900], [0], [10], [0], [20]])
        balancer.select_layer_method = SelectLayerMethod.ROUND
        self.assertEqual(
            [balancer.get_balanced_layer(gpu_loads) for _ in range(4)], [5, 7, 5, 7]
        )
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
                cfg = make_load_config(8, range(4, 8))
                cfg.ep_size = ep_size
                cfg.num_nodes = num_nodes
                with patch.object(
                    LoadConfig,
                    "create_redundant_expert",
                    wraps=LoadConfig.create_redundant_expert,
                ) as static_placement:
                    loader = make_eplb_loader(cfg, [1, 3, 5, 7])
                self.assertEqual(
                    static_placement.call_args.kwargs["num_nodes"], num_nodes
                )
                self.assertEqual(loader._py_eplb.num_nodes, num_nodes)
                self.assertEqual(loader._py_eplb.num_gpu, ep_size)


EMBEDDING_NAMES = {W.embedding, W.positional_embedding}
CUSTOM_WEIGHT_NAME = CustomAtomicWeight.prefix + "classifier.weight"
OUTPUT_NAMES = {
    W.lm_head,
    W.lm_head_b,
    W.final_ln_gamma,
    W.final_ln_beta,
}
OTHER_GLOBAL_NAMES = {CUSTOM_WEIGHT_NAME, "some_other_global"}
GLOBAL_NAMES = EMBEDDING_NAMES | OUTPUT_NAMES | OTHER_GLOBAL_NAMES


class LayerScopedLoadingTest(unittest.TestCase):
    def test_raw_and_fast_loading_follow_resolved_range(self):
        for layers, scope, expected in (
            (7, range(2), [0, 1]),
            (7, range(2, 5), [2, 3, 4]),
            (7, range(5, 7), [5, 6]),
            (2, range(2), [0, 1]),
        ):
            with self.subTest(layers=layers, scope=scope):
                cfg = make_load_config(layers, scope)
                loader = make_loader(cfg)
                weights = [MagicMock() for _ in range(layers)]
                for idx, weight in enumerate(weights):
                    weight.get_components.return_value = [weight]
                    weight.get_tensor_names.return_value = [f"layer.{idx}"]
                loader._model_weights_info = SimpleNamespace(
                    layer_weights=[[weight] for weight in weights], weights=[]
                )
                loader._load_layer_weights = MagicMock(
                    return_value={"weight": torch.ones(1)}
                )
                self.assertEqual(
                    [row[0] for row in loader.prepare_weights("cpu")], expected
                )
                with patch("rtp_llm.model_loader.loader.TensorCollector"):
                    _, weight_infos = loader._generate_weight_info()
                self.assertEqual([info.layer_id for info in weight_infos], expected)

    def test_ft_loading_reads_selected_layers_and_globals(self):
        for scope, expected, globals_expected in (
            (range(2), [0, 1], EMBEDDING_NAMES | OTHER_GLOBAL_NAMES),
            (range(2, 12), list(range(2, 12)), OUTPUT_NAMES | OTHER_GLOBAL_NAMES),
            (range(2, 6), [2, 3, 4, 5], OTHER_GLOBAL_NAMES),
            (range(12), list(range(12)), GLOBAL_NAMES),
        ):
            with self.subTest(scope=scope):
                cfg = make_load_config(12, scope)
                cfg.exported_device = SimpleNamespace(support_dio_load=False)
                loader = make_loader(cfg)
                layer_prefix = ModelWeights.layer_weight_prefix(0, 0, 0)
                global_prefix = ModelWeights.global_weight_prefix(0, 0, 0)
                checkpoint = {
                    f"{layer_prefix}{i}.weight": [torch.tensor([i])] for i in range(12)
                }
                for name in GLOBAL_NAMES:
                    checkpoint[f"{global_prefix}{name}"] = [torch.tensor([99])]
                checkpoint["rank_01_00_00.layers.0.weight"] = [torch.tensor([-1])]
                cfg.database.get_pretrain_tensor_names.return_value = list(checkpoint)
                cfg.database.load_tensors_by_prefix.side_effect = (
                    lambda prefixes, device, direct_io: {
                        key: value
                        for key, value in checkpoint.items()
                        if key.startswith(prefixes)
                    }
                )
                result = loader._load_from_ft_style("cpu")
                self.assertEqual(
                    [i for i, row in enumerate(result.weights) if row], expected
                )
                for i in expected:
                    self.assertEqual(result.weights[i]["weight"].item(), i)
                self.assertEqual(set(result.global_weights), globals_expected)
                for name in globals_expected:
                    self.assertEqual(result.global_weights[name].item(), 99)
                requested = cfg.database.load_tensors_by_prefix.call_args.args[0]
                for name in GLOBAL_NAMES - globals_expected:
                    self.assertFalse(f"{global_prefix}{name}".startswith(requested))

    def test_custom_weights_load_in_every_scope_in_both_loading_paths(self):
        for scope in (range(2), range(2, 5), range(5, 8), range(8)):
            with self.subTest(scope=scope):
                custom_weight = MagicMock()
                custom_weight.name = CUSTOM_WEIGHT_NAME
                custom_weight.load.return_value = {CUSTOM_WEIGHT_NAME: torch.ones(1)}
                custom_weight.get_tensor_names.return_value = ["classifier.weight"]
                loader = make_loader(
                    make_load_config(8, scope), misc_weights_info=[custom_weight]
                )
                loader._is_attn_model = True
                loader._model_weights_info = SimpleNamespace(
                    layer_weights=[], weights=[]
                )
                self.assertEqual(len(list(loader.prepare_weights("cpu"))), 1)
                self.assertEqual(custom_weight.load.call_count, 1)
                with patch("rtp_llm.model_loader.loader.TensorCollector"):
                    _, weight_infos = loader._generate_weight_info()
                self.assertEqual(len(weight_infos), 1)
                self.assertEqual(custom_weight.get_tensor_names.call_count, 1)

    def test_lora_loading_uses_local_global_layer_ids(self):
        cfg = make_load_config(7, range(2, 5))
        cfg.database.get_lora_config.return_value = SimpleNamespace(
            lora_alpha=2, rank=1
        )
        loader = make_loader(cfg)
        loader._load_layer_lora_weights = MagicMock(
            side_effect=lambda adapter, layer, device: {
                "attn.lora_A": torch.tensor([[float(layer)]]),
                "attn.lora_B": torch.ones(1, 1),
            }
        )
        result = loader.load_lora_weights("adapter", "unused-path")
        self.assertEqual(
            [call.args[1] for call in loader._load_layer_lora_weights.call_args_list],
            [2, 3, 4],
        )
        self.assertEqual(
            [idx for idx, row in enumerate(result.lora_a_weights) if row], [2, 3, 4]
        )
        for layer in (2, 3, 4):
            self.assertEqual(result.lora_a_weights[layer]["attn"].item(), layer)
            self.assertEqual(result.lora_b_weights[layer]["attn"].item(), 2)

    def test_lm_head_postprocessing_follows_output_layer(self):
        for layers, scope, owns_embedding, owns_lm_head in (
            (7, range(2), True, False),
            (7, range(2, 5), False, False),
            (7, range(5, 7), False, True),
            (2, range(2), True, True),
        ):
            with self.subTest(layers=layers, scope=scope):
                loader = make_loader(make_load_config(layers, scope))
                loader._weights_info.model_config = SimpleNamespace(
                    normalize_lm_head_weight=False,
                    logit_scale=1.0,
                    vocab_size=4,
                )
                loader._weights_info.create_dynamic_weights.return_value = []
                embedding = torch.ones(4, 4)
                lm_head = torch.full((4, 4), 2.0)
                globals_ = {W.embedding: embedding} if owns_embedding else {}
                if owns_lm_head and not owns_embedding:
                    globals_[W.lm_head] = lm_head
                weights = MagicMock(global_weights=globals_)
                weights.steal_global_weight.return_value = globals_.get(W.lm_head)
                loader._load_dynamic_weights(weights, "cpu")
                if not owns_lm_head:
                    weights.steal_global_weight.assert_not_called()
                    weights.set_global_weight.assert_not_called()
                else:
                    weights.steal_global_weight.assert_called_once_with(W.lm_head)
                    weights.set_global_weight.assert_called_once_with(
                        W.lm_head, embedding if owns_embedding else lm_head
                    )


class GlobalWeightScopeTest(unittest.TestCase):
    def test_global_scope_follows_boundary_layers(self):
        for scope, expected in (
            (range(2), EMBEDDING_NAMES | OTHER_GLOBAL_NAMES),
            (range(2, 5), OTHER_GLOBAL_NAMES),
            (range(5, 7), OUTPUT_NAMES | OTHER_GLOBAL_NAMES),
            (range(7), GLOBAL_NAMES),
        ):
            with self.subTest(scope=scope):
                loader = make_loader(make_load_config(7, scope))
                for name in GLOBAL_NAMES:
                    self.assertEqual(
                        loader.is_global_weight_local(name), name in expected
                    )
                    self.assertEqual(
                        loader._maybe_skip_weight(FakeWeight(name)),
                        name not in expected,
                    )

    def test_aliases_and_non_language_model_heads_are_skipped(self):
        loader = make_loader(make_load_config(2))
        loader._global_weight_aliases = {W.embedding: torch.ones(1)}
        self.assertTrue(loader._maybe_skip_weight(FakeWeight(W.embedding)))
        self.assertFalse(loader._maybe_skip_weight(FakeWeight(W.lm_head)))
        loader = make_loader(make_load_config(2), TaskType.DENSE_EMBEDDING)
        self.assertTrue(loader._maybe_skip_weight(FakeWeight(W.lm_head)))


if __name__ == "__main__":
    unittest.main()
