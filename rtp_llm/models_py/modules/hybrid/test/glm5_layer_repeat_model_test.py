"""Exercise the real target forward loop with tied toy weights and separate KV."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.models_py.model_desc import glm5 as gm
from rtp_llm.ops import ParallelismConfig, ProfilingDebugLoggingConfig, TaskType
from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer
from rtp_llm.utils.glm5_layer_repeat import (
    build_execution_layers,
    checkpoint_layer_config,
    configure_glm5_layer_repeat,
    expand_repeated_layer_weights,
)


class ToyAttention(gm.MlaAttention):
    def __init__(self, i, has_indexer=True, *, source_id=None):
        nn.Module.__init__(self)
        self.layer_idx = i
        self.has_indexer = has_indexer
        source_id = i if source_id is None else source_id
        self.weight = nn.Parameter(
            torch.tensor([1 + source_id / 100]), requires_grad=False
        )
        self.scale = 0.95 if source_id % 2 else 1.0
        self.indexer = nn.Linear(1, 1, bias=False) if has_indexer else None
        if self.indexer is not None:
            self.indexer.layer_idx = i

    def forward(self, hidden, cache):
        if cache.layer_id != self.layer_idx:
            raise AssertionError("attention/cache layer ID mismatch")
        hidden = hidden * self.scale + self.weight
        cache.value = hidden.clone()
        return hidden


class ToyLayer(gm.Glm5MoeDecoderLayer):
    def __init__(self, i, has_indexer=True, *, source_id=None):
        nn.Module.__init__(self)
        self.layer_idx = i
        self.self_attn = ToyAttention(i, has_indexer, source_id=source_id)
        self.mlp = nn.Linear(1, 1, bias=False)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.cmp = None

    def forward(self, hidden, residual, fmha_impl, kv_cache, **kwargs):
        if self.self_attn.has_indexer:
            if kv_cache["indexer_kv"].layer_id != self.layer_idx:
                raise AssertionError("indexer/cache layer ID mismatch")
            topk = self.layer_idx
        else:
            if "indexer_kv" in kv_cache or kwargs["prev_topk_indices"] is None:
                raise AssertionError("shared indexer must receive previous top-k")
            topk = kwargs["prev_topk_indices"]
        kv_cache["default"].topk = topk
        return gm.DecodeLayerOutput(
            self.self_attn(hidden, kv_cache["default"]), residual, topk
        )


class IdentityNorm(nn.Module):
    def forward(self, hidden, residual):
        return hidden, residual


def runtime_config(*, expand=True):
    c = ModelConfig()
    c.model_type = "glm_5"
    c.num_layers = 78
    c.moe_layer_index = list(range(3, 78))
    c.attn_config.use_mla = True
    c.attn_config.is_sparse = True
    c.index_topk_freq = 4
    c.index_skip_topk_offset = 3
    c._glm5_loop_layer_range = (26, 54)
    if expand:
        configure_glm5_layer_repeat(c)
    return c


class Glm5LayerRepeatModelTest(unittest.TestCase):
    def test_weight_and_parameter_estimates_keep_physical_layers(self):
        config = runtime_config(expand=False)
        config.task_type = TaskType.LANGUAGE_MODEL
        config.max_seq_len = 256
        config.hidden_size = 64
        config.vocab_size = 128
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = 16
        config.moe_style = 1
        config.expert_num = 4
        config.moe_inter_size = 16
        config.inter_size = 32
        config.expert_weight_dtype = "fp4"
        weight_bytes = config.eval_model_weight_size()
        per_rank_bytes = config.eval_model_weight_size_per_rank(1, 4)
        parameters = config.model_param_count()
        kv_bytes = config._eval_kv_cache_mem_size()
        self.assertGreater(weight_bytes, 0)
        self.assertGreater(kv_bytes, 0)

        configure_glm5_layer_repeat(config)

        self.assertEqual(config.eval_model_weight_size(), weight_bytes)
        self.assertEqual(config.eval_model_weight_size_per_rank(1, 4), per_rank_bytes)
        self.assertEqual(config.model_param_count(), parameters)
        self.assertEqual(config._eval_kv_cache_mem_size() * 78, kv_bytes * 106)

    def test_capture_override_accepts_logical_layer_ids(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            args = ModelArgs()
            args.model_type = "glm_5"
            args.ckpt_path = args.tokenizer_path = ckpt_path
            kv_config = KVCacheConfig()
            debug_config = ProfilingDebugLoggingConfig()
            config = runtime_config(expand=False)
            args.json_model_override_args = json.dumps(
                {"hidden_state_capture_layer_ids": [105, 54, 26]}
            )

            build_model_config(config, args, kv_config, debug_config)

            self.assertEqual(config.num_layers, 106)
            self.assertEqual(config.hidden_state_capture_layer_ids, [105, 54, 26])
            args.json_model_override_args = json.dumps(
                {"hidden_state_capture_layer_ids": [106]}
            )
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                build_model_config(
                    runtime_config(expand=False), args, kv_config, debug_config
                )

    def test_cpp_config_and_checkpoint_view(self):
        c = runtime_config()
        self.assertEqual(c.num_layers, 106)
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
        self.assertEqual(c.num_layers, 106)

    def test_loader_expands_only_after_loading_source_weights(self):
        from rtp_llm.model_loader.loader import ModelLoader

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = runtime_config()
        source = [dict(weight=torch.ones(1)) for _ in range(78)]
        weights = SimpleNamespace(weights=source)
        loader._load_config = SimpleNamespace(is_ft_style_weight=True)
        loader._model_weights_info = SimpleNamespace(weights=[])
        loader._load_from_ft_style = Mock(return_value=weights)
        loader._load_dynamic_weights = Mock()
        loader._init_eplb_weight = Mock()
        result = loader.load_weights("cpu")
        self.assertEqual(len(result.weights), 106)
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

        def base_init(model, config, parallelism, model_weights, **kwargs):
            nn.Module.__init__(model)
            model.config = config
            model.parallelism_config = parallelism
            model.weight = model_weights
            model.py_hw_kernel_config = kwargs.get("py_hw_kernel_config")
            model.layer_num = config.num_layers

        def create(config, parallel, layer_weights, globals_, idx, *args, **kwargs):
            constructed.append((idx, layer_weights["source"]))
            return ToyLayer(idx)

        with patch.object(gm.GptModelBase, "__init__", base_init), patch.object(
            gm, "Glm5MoeDecoderLayer", side_effect=create
        ), patch.object(gm, "Embedding", return_value=nn.Identity()), patch.object(
            gm, "RMSResNorm", return_value=IdentityNorm()
        ):
            model = gm.Glm5MoeModel(c, None, weights, None, 1)
        self.assertEqual(len(model.layers), 106)
        self.assertEqual([source for _, source in constructed], list(range(78)))
        self.assertEqual([layer.layer_idx for layer in model.layers], list(range(106)))
        self.assertIs(model.layers[26].mlp, model.layers[54].mlp)
        self.assertIsNot(model.layers[26].self_attn, model.layers[54].self_attn)

    def test_non_ft_loader_expands_after_source_postprocessing(self):
        from rtp_llm.model_loader.loader import ModelLoader

        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = runtime_config()
        weights = SimpleNamespace(weights=[{} for _ in range(78)])
        loader._load_config = SimpleNamespace(is_ft_style_weight=False)
        loader._model_weights_info = SimpleNamespace(weights=[])
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
        self.assertEqual(len(result.weights), 106)
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
            moe_layer_index_=list(range(3, 78)),
            layer_range=range(78),
        )
        placement = [
            [(expert + layer) % 4 for expert in range(4)] for layer in range(78)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f, patch(
            "rtp_llm.model_loader.loader.CkptDatabase"
        ):
            json.dump(placement, f)
            f.flush()
            c.phy2log_path = f.name
            balancer, actual = loader.create_eplb()
        self.assertIsNone(balancer)
        self.assertEqual(actual, placement)
        self.assertEqual(c.num_layers, 106)

    def test_mtp_keeps_checkpoint_offset_with_repeat_enabled(self):
        from rtp_llm.models.deepseek_v2 import DeepSeekV2
        from rtp_llm.models.glm5 import Glm5Mtp

        c = ModelConfig()
        c.num_layers = 78
        with patch.object(DeepSeekV2, "_create_config", return_value=c), patch(
            "rtp_llm.models.glm5.os.path.exists", return_value=False
        ):
            draft = Glm5Mtp._create_config("unused")
            draft._glm5_loop_layer_range = (26, 54)
            configure_glm5_layer_repeat(draft)
        self.assertEqual(draft.num_layers, 1)
        self.assertEqual(draft.mtp_layer_offset, 78)
        self.assertIs(checkpoint_layer_config(draft), draft)
        self.assertFalse(hasattr(draft, "_glm5_layer_source_ids"))

    def test_clone_without_indexer_and_cuda_graph_preserves_logical_id(self):
        layer = ToyLayer(27)
        layer.self_attn.indexer = None
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

    def test_cmp_shares_packed_weights_across_passes_and_graph_warmup(self):
        layer = ToyLayer(26)
        attention = layer.self_attn
        projection = SimpleNamespace(weight=torch.ones(1), weight_scales=torch.ones(1))
        attention.fused_qkv_a_proj = attention.q_b_proj = attention.o_proj = projection
        attention.indexer.wk = attention.indexer.wq_b = projection
        attention.indexer.weights_proj = nn.Linear(2, 2, bias=False)
        weight = torch.ones((6144, 256), dtype=torch.bfloat16).T
        weight[0, 0] = 5
        self.assertFalse(weight.is_contiguous())
        layer.mlp.gate = SimpleNamespace(weight=weight)
        layer.cmp = gm.Glm5Cmp(
            layer_idx=layer.layer_idx,
            config=runtime_config(),
            parallelism_config=ParallelismConfig(),
            self_attn=attention,
            input_layernorm=layer.input_layernorm,
            mlp=layer.mlp,
            post_attention_layernorm=layer.post_attention_layernorm,
        )
        repeated = layer.clone_for_layer_repeat(54)
        graph = repeated.cmp.clone_for_cuda_graph(mlp=repeated.mlp)
        self.assertIsNot(repeated.cmp, layer.cmp)
        self.assertEqual(repeated.cmp.layer_idx, 54)
        self.assertIs(repeated.cmp.self_attn, repeated.self_attn)
        self.assertIs(repeated.cmp.mlp, layer.mlp)
        layer.cmp._events = object()
        self.assertIsNone(repeated.cmp._events)
        self.assertIsNone(graph._events)

        ops = Mock()
        ops.pack_head_gate_weight.side_effect = torch.clone
        with patch(
            "rtp_llm.models_py.modules.hybrid.glm5_cmp._is_capturing",
            return_value=False,
        ) as is_capturing:
            # A graph can be warmed before the eager model has run any decode.
            graph._initialize_for_cmp(ops)
            graph_head_gate = graph._get_packed_head_gate_weight(ops)
            layer.cmp._initialize_for_cmp(ops)
            repeated.cmp._initialize_for_cmp(ops)
            self.assertIs(layer.cmp._get_packed_head_gate_weight(ops), graph_head_gate)
            self.assertIs(repeated.cmp._get_packed_head_gate_weight(ops), graph_head_gate)
            is_capturing.assert_called_once()
        ops.pack_head_gate_weight.assert_called_once()
        self.assertIs(layer.cmp._router_weight, graph._router_weight)
        self.assertIs(repeated.cmp._router_weight, graph._router_weight)
        self.assertTrue(graph._router_weight.is_contiguous())
        torch.testing.assert_close(graph._router_weight, weight)

    def test_forward_block_order_with_tied_weights_and_independent_kv(self):
        c = runtime_config()
        layers = build_execution_layers(
            c,
            lambda i: ToyLayer(
                i, dsa_layer_has_indexer(c, i), source_id=c._glm5_layer_source_ids[i]
            ),
            lambda layer, i: layer.clone_for_layer_repeat(i),
        )
        self.assertEqual(len(layers), 106)
        self.assertEqual(
            len({layer.self_attn.weight.data_ptr() for layer in layers}), 78
        )
        first, repeated = layers[26], layers[54]
        self.assertIsNot(first.self_attn, repeated.self_attn)
        self.assertEqual(
            (first.self_attn.layer_idx, repeated.self_attn.layer_idx), (26, 54)
        )
        self.assertEqual(
            (first.self_attn.indexer.layer_idx, repeated.self_attn.indexer.layer_idx),
            (26, 54),
        )
        self.assertIs(first.self_attn.weight, repeated.self_attn.weight)
        self.assertIs(first.self_attn.indexer.weight, repeated.self_attn.indexer.weight)
        self.assertIs(first.mlp, repeated.mlp)
        caches = [
            SimpleNamespace(tag="default", layer_id=i, value=None) for i in range(106)
        ]
        indexer_caches = [
            SimpleNamespace(tag="indexer_kv", layer_id=i) for i in range(106)
        ]
        model = gm.Glm5MoeModel.__new__(gm.Glm5MoeModel)
        nn.Module.__init__(model)
        model.config = c
        model.layers = nn.ModuleList(layers)
        model.layer_num = c.num_layers
        model.embed_tokens = nn.Identity()
        model.norm = IdentityNorm()
        model._init_capture_context(
            model._capture_canonical_layer, model._capture_canonical_final
        )

        def layer_caches(i):
            if dsa_layer_has_indexer(c, i):
                return [caches[i], indexer_caches[i]]
            return [caches[i]]

        model.kv_cache = SimpleNamespace(get_layer_cache_groups=layer_caches)
        inputs = SimpleNamespace(
            input_ids=torch.zeros(2, 1),
            capture_hidden_states=False,
        )
        fmha = {"default": object(), "indexer_kv": object()}
        with patch.object(gm, "should_enable_glm5_cmp", return_value=False):
            output = model.forward(inputs, fmha).hidden_states
        expected = torch.zeros_like(output)
        expected_topk_layer = None
        original_config = checkpoint_layer_config(c)
        order = (*range(54), *range(26, 54), *range(54, 78))
        for cache, source in zip(caches, order):
            expected = expected * (0.95 if source % 2 else 1.0) + (1 + source / 100)
            torch.testing.assert_close(cache.value, expected)
            if dsa_layer_has_indexer(original_config, source):
                expected_topk_layer = cache.layer_id
            self.assertEqual(cache.topk, expected_topk_layer)
        torch.testing.assert_close(output, expected)
        self.assertEqual(len({cache.value.data_ptr() for cache in caches}), 106)

    def test_forward_injects_vision_features_only_at_selected_rows(self):
        config = runtime_config()
        config.mm_model_config.is_multimodal = True
        embedding_table = torch.arange(18, dtype=torch.float32).reshape(6, 3) + 1
        weights = SimpleNamespace(get_global_weight=lambda _: embedding_table)
        parallelism = ParallelismConfig()
        parallelism.world_size = parallelism.local_world_size = 1
        parallelism.tp_size = 1

        def lookup(output, ids, table, positions, token_types, text_mask):
            output.copy_(table[ids])

        def pass_through(hidden, *args, **kwargs):
            return gm.DecodeLayerOutput(hidden, torch.zeros_like(hidden), None)

        ids = torch.tensor([3, 1, 0, 0, 2, 0])
        text_only = embedding_table[ids]
        first_image = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        second_image = torch.tensor([[-7.0, -8.0, -9.0]])
        mixed = text_only.clone()
        mixed[2:4] = first_image
        mixed[5:6] = second_image
        inputs = SimpleNamespace(
            input_ids=ids,
            capture_hidden_states=False,
            combo_position_ids=torch.arange(len(ids)),
            embedding_inputs=SimpleNamespace(
                combo_tokens_type_ids=None,
                text_tokens_mask=None,
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[first_image, second_image],
                mm_features_locs=torch.tensor([2, 5]),
            ),
        )
        fmha = {"default": object(), "indexer_kv": object()}
        with (
            patch.object(gm.Glm5MoeModel, "_initialize_decoder_layers"),
            patch.object(
                gm.Glm5MoeModel, "_forward_decoder_layers", side_effect=pass_through
            ),
            patch.object(gm, "RMSResNorm", return_value=IdentityNorm()),
            patch(
                "rtp_llm.models_py.modules.base.common.embedding.rtp_llm_ops.embedding",
                side_effect=lookup,
            ),
        ):
            model = gm.Glm5MoeModel(config, parallelism, weights, None, 1)
            torch.testing.assert_close(model.forward(inputs, fmha).hidden_states, mixed)
            inputs.multimodal_inputs.multimodal_features = []
            torch.testing.assert_close(
                model.forward(inputs, fmha).hidden_states, text_only
            )
            inputs.multimodal_inputs = None
            torch.testing.assert_close(
                model.forward(inputs, fmha).hidden_states, text_only
            )


if __name__ == "__main__":
    unittest.main()
