"""Exercise the real target forward loop with tied toy weights and separate KV."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc import generic_moe as gm
from rtp_llm.utils.glm5_layer_repeat import (
    build_execution_layers,
    checkpoint_layer_config,
    configure_glm5_layer_repeat,
    expand_repeated_layer_weights,
)


class ToyAttention(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layer_idx = i
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.indexer = nn.Linear(1, 1, bias=False)
        self.indexer.layer_idx = i

    def forward(self, hidden, cache):
        if cache.layer_id != self.layer_idx:
            raise AssertionError("attention/cache layer ID mismatch")
        hidden = hidden + self.weight
        cache.value = hidden.clone()
        return hidden


class ToyLayer(gm.GenericMoeDecoderLayer):
    def __init__(self, i):
        nn.Module.__init__(self)
        self.layer_idx = i
        self.self_attn = ToyAttention(i)
        self.mlp = nn.Linear(1, 1, bias=False)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.cmp = None

    def forward(self, hidden, residual, fmha_impl, kv_cache, **kwargs):
        return gm.DecodeLayerOutput(self.self_attn(hidden, kv_cache), residual)


class IdentityNorm(nn.Module):
    def forward(self, hidden, residual):
        return hidden, residual


def runtime_config():
    c = ModelConfig()
    c.model_type = "glm_5"
    c.num_layers = 78
    c.moe_layer_index = list(range(3, 78))
    c.attn_config.use_mla = True
    c.attn_config.is_sparse = True
    c.index_topk_freq = 4
    c.index_skip_topk_offset = 3
    with patch.dict(os.environ, {"GLM5_REPEAT_MOE_LAYERS": "middle27"}):
        configure_glm5_layer_repeat(c)
    return c


class Glm5LayerRepeatModelTest(unittest.TestCase):
    def test_cpp_config_and_checkpoint_view(self):
        c = runtime_config()
        self.assertEqual(c.num_layers, 105)
        source = checkpoint_layer_config(c)
        self.assertEqual(source.num_layers, 78)
        source.embedding_size = 2048
        self.assertEqual(c.embedding_size, 2048)
        self.assertEqual(len(source.indexer_types), 78)

    def test_real_checkpoint_weight_info_keeps_78_layers(self):
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig

        c = runtime_config()
        parallel = ParallelismConfig()
        parallel.local_world_size = 1
        parallel.world_size = 1
        info = DeepSeekV2Weight(c, parallel, HWKernelConfig(), KVCacheConfig())
        self.assertEqual(info._num_layers, 78)
        self.assertEqual(info.moe_layer_index_, list(range(3, 78)))
        self.assertEqual(c.num_layers, 105)

    def test_loader_expands_only_after_loading_source_weights(self):
        from unittest.mock import Mock

        from rtp_llm.model_loader.loader import ModelLoader

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = runtime_config()
        source = [dict(weight=torch.ones(1)) for _ in range(78)]
        weights = SimpleNamespace(weights=source)
        loader._load_config = SimpleNamespace(is_ft_style_weight=True)
        loader._load_from_ft_style = Mock(return_value=weights)
        loader._load_dynamic_weights = Mock()
        loader._init_eplb_weight = Mock()
        result = loader.load_weights("cpu")
        self.assertEqual(len(result.weights), 105)
        for logical_id, source_id in enumerate(
            loader.model_config._glm5_layer_source_ids
        ):
            self.assertIs(result.weights[logical_id], source[source_id])
        loader._load_from_ft_style.assert_called_once_with("cpu")

    def test_real_model_constructor_builds_each_source_only_once(self):
        c = runtime_config()
        weights = SimpleNamespace(
            weights=[{"source": i} for i in range(78)],
            global_weights={},
            get_global_weight=lambda name: torch.ones(1),
        )
        expand_repeated_layer_weights(c, weights)
        constructed = []

        def base_init(model, *args, **kwargs):
            nn.Module.__init__(model)
            model.layer_num = c.num_layers

        def create(config, parallel, layer_weights, globals_, idx, *args, **kwargs):
            constructed.append((idx, layer_weights["source"]))
            return ToyLayer(idx)

        with patch.object(gm.GptModelBase, "__init__", base_init), patch.object(
            gm, "GenericMoeDecoderLayer", side_effect=create
        ), patch.object(gm, "Embedding", return_value=nn.Identity()), patch.object(
            gm, "RMSResNorm", return_value=IdentityNorm()
        ):
            model = gm.GenericMoeModel(c, None, weights, None, 1)
        self.assertEqual(len(model.layers), 105)
        self.assertEqual([source for _, source in constructed], list(range(78)))
        self.assertEqual([layer.layer_idx for layer in model.layers], list(range(105)))
        self.assertIs(model.layers[27].mlp, model.layers[28].mlp)
        self.assertIsNot(model.layers[27].self_attn, model.layers[28].self_attn)

    def test_non_ft_loader_expands_after_source_postprocessing(self):
        from rtp_llm.model_loader.loader import ModelLoader

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = runtime_config()
        weights = SimpleNamespace(weights=[{} for _ in range(78)])
        loader._load_config = SimpleNamespace(is_ft_style_weight=False)
        loader._load_weight = Mock(return_value=weights)
        loader.force_clean_cuda_memory = Mock()
        seen = []
        loader._load_dynamic_weights = Mock(
            side_effect=lambda w, d: seen.append(len(w.weights))
        )
        loader._init_eplb_weight = Mock(
            side_effect=lambda w, d: seen.append(len(w.weights))
        )
        result = loader.load_weights("cpu")
        self.assertEqual(seen, [78, 78])
        self.assertEqual(len(result.weights), 105)
        loader.force_clean_cuda_memory.assert_called_once_with()

    def test_static_expert_layout_uses_checkpoint_layers(self):
        from rtp_llm.model_loader.loader import ModelLoader

        c = runtime_config()
        c.expert_num = 4
        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = c
        loader._weights_info = SimpleNamespace(
            model_config=checkpoint_layer_config(c),
            expert_num_=4,
            phy_exp_num_=4,
            ep_size=1,
            num_nodes=1,
            enable_eplb_=False,
        )
        placement = [
            [(expert + layer) % 4 for expert in range(4)] for layer in range(78)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(placement, f)
            f.flush()
            c.phy2log_path = f.name
            balancer, actual = loader.create_eplb()
        self.assertIsNone(balancer)
        self.assertEqual(actual, placement)
        self.assertEqual(c.num_layers, 105)

    def test_mtp_keeps_checkpoint_offset_with_repeat_enabled(self):
        from rtp_llm.models.deepseek_v2 import DeepSeekV2, Glm5Mtp

        c = ModelConfig()
        c.num_layers = 78
        with patch.object(DeepSeekV2, "_create_config", return_value=c), patch(
            "rtp_llm.models.deepseek_v2.os.path.exists", return_value=False
        ), patch.dict(os.environ, {"GLM5_REPEAT_MOE_LAYERS": "middle27"}):
            draft = Glm5Mtp._create_config("unused")
            configure_glm5_layer_repeat(draft)
        self.assertEqual(draft.num_layers, 1)
        self.assertEqual(draft.mtp_layer_offset, 78)
        self.assertIs(checkpoint_layer_config(draft), draft)
        self.assertFalse(hasattr(draft, "_glm5_layer_source_ids"))

    def test_clone_without_indexer_and_cuda_graph_preserves_logical_id(self):
        layer = ToyLayer(27)
        layer.self_attn.indexer = None
        layer._drain_pinned_mla_before_moe = True
        layer._fuse_input_norm_quant = False
        layer._fuse_input_scale_ue8m0 = False
        layer._fuse_post_norm_quant = False
        layer._fuse_post_norm_quant_moe = False
        repeated = layer.clone_for_layer_repeat(28)
        graph = repeated.clone_for_cuda_graph()
        self.assertIsNone(repeated.self_attn.indexer)
        self.assertEqual(layer.self_attn.layer_idx, 27)
        self.assertEqual(graph.self_attn.layer_idx, 28)
        self.assertIs(graph.mlp, layer.mlp)
        self.assertIs(graph.self_attn.weight, layer.self_attn.weight)
        self.assertTrue(graph._drain_pinned_mla_before_moe)

    def test_cmp_clone_has_separate_coordinator_and_logical_attention(self):
        layer = ToyLayer(27)
        layer.cmp = SimpleNamespace(
            config=runtime_config(), parallelism_config=object()
        )
        with patch.object(gm, "Glm5Cmp") as constructor:
            repeated = layer.clone_for_layer_repeat(28)
        self.assertIs(repeated.cmp, constructor.return_value)
        self.assertIsNot(repeated.cmp, layer.cmp)
        self.assertEqual(constructor.call_args.kwargs["layer_idx"], 28)
        self.assertIs(constructor.call_args.kwargs["self_attn"], repeated.self_attn)
        self.assertIs(constructor.call_args.kwargs["mlp"], layer.mlp)

    def test_forward_105_passes_with_tied_weights_and_independent_kv(self):
        c = runtime_config()
        layers = build_execution_layers(
            c, ToyLayer, lambda layer, i: layer.clone_for_layer_repeat(i)
        )
        self.assertEqual(len(layers), 105)
        self.assertEqual(
            len({layer.self_attn.weight.data_ptr() for layer in layers}), 78
        )
        first, repeated = layers[27:29]
        self.assertIsNot(first.self_attn, repeated.self_attn)
        self.assertEqual(
            (first.self_attn.layer_idx, repeated.self_attn.layer_idx), (27, 28)
        )
        self.assertEqual(
            (first.self_attn.indexer.layer_idx, repeated.self_attn.indexer.layer_idx),
            (27, 28),
        )
        self.assertIs(first.self_attn.weight, repeated.self_attn.weight)
        self.assertIs(first.self_attn.indexer.weight, repeated.self_attn.indexer.weight)
        self.assertIs(first.mlp, repeated.mlp)
        caches = [SimpleNamespace(layer_id=i, value=None) for i in range(105)]
        model = gm.GenericMoeModel.__new__(gm.GenericMoeModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList(layers)
        model.layer_num = c.num_layers
        model.embed_tokens = nn.Identity()
        model.norm = IdentityNorm()
        model.pinned_mla_groups = {}
        model.layered_prefill_cache = None
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda i: caches[i])
        inputs = SimpleNamespace(
            input_ids=torch.zeros(2, 1),
            attention_inputs=SimpleNamespace(
                kv_cache_kernel_block_id_device_by_group=None
            ),
        )
        fmha = SimpleNamespace(fmha_params=None)
        with patch.object(
            gm, "should_enable_glm5_cmp", return_value=False
        ), patch.object(gm, "select_block_map_for_layer"), patch.object(
            gm, "PyModelOutputs", side_effect=lambda h, p: h
        ):
            # Run both the ordinary path and the upstream layered-prefill path.
            output = model.forward(inputs, fmha)
            layered = Mock(active=True)
            model.layered_prefill_cache = layered
            output = model.forward(inputs, fmha)
            layered.begin_forward.assert_called_once_with(fmha)
            self.assertEqual(
                [call.args[0] for call in layered.before_layer.call_args_list],
                list(range(105)),
            )
            layered.end_forward.assert_called_once_with(failed=False)
            layered.reset_mock()
            with patch.object(
                layers[28], "forward", side_effect=RuntimeError("layer failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "layer failed"):
                    model.forward(inputs, fmha)
            layered.end_forward.assert_called_once_with(failed=True)
            # Restore full cache contents for the checks below.
            model.forward(inputs, fmha)
        torch.testing.assert_close(output, torch.full((2, 1), 105.0))
        for i, cache in enumerate(caches):
            torch.testing.assert_close(cache.value, torch.full((2, 1), float(i + 1)))
        self.assertEqual(len({cache.value.data_ptr() for cache in caches}), 105)


if __name__ == "__main__":
    unittest.main()
