"""Exercise the real target forward loop with tied toy weights and separate KV."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc import generic_moe as gm
from rtp_llm.utils.glm5_layer_repeat import (
    build_execution_layers,
    checkpoint_layer_config,
    configure_glm5_layer_repeat,
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
            output = model.forward(inputs, fmha)
        torch.testing.assert_close(output, torch.full((2, 1), 105.0))
        for i, cache in enumerate(caches):
            torch.testing.assert_close(cache.value, torch.full((2, 1), float(i + 1)))
        self.assertEqual(len({cache.value.data_ptr() for cache in caches}), 105)


if __name__ == "__main__":
    unittest.main()
