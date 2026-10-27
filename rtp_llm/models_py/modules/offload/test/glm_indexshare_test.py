import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.utils.dsa_indexing import build_dsa_indexer_groups, dsa_layer_has_indexer
from rtp_llm.utils.model_weight import W


def config():
    value = ModelConfig()
    value.num_layers = 78
    value.attn_config.use_mla = True
    value.attn_config.is_sparse = True
    value.attn_config.indexer_head_dim = 128
    value.attn_config.indexer_topk = 2048
    value.attn_config.tokens_per_block = 128
    value.index_topk_freq = 4
    value.index_skip_topk_offset = 3
    return value


class GlmIndexShareTest(unittest.TestCase):
    def test_optional_drain_runs_before_moe(self):
        for enabled, final_layer in ((True, False), (False, False), (True, True)):
            events = []
            layer = GenericMoeDecoderLayer.__new__(GenericMoeDecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.layer_idx = 3
            layer._uses_sparse_mla = True
            layer._drain_offload_before_moe = enabled
            layer.input_layernorm = lambda hidden, residual: (hidden, residual)
            layer.self_attn = lambda **kwargs: (
                events.append("attention") or "hidden",
                "topk",
            )
            layer.post_attention_layernorm = lambda hidden, residual: (
                events.append("norm") or hidden,
                residual,
            )
            layer.mlp = lambda hidden: events.append("moe") or hidden
            plan = SimpleNamespace(
                drain=final_layer,
                runtime=SimpleNamespace(drain=lambda: events.append("drain")),
            )
            layer.self_attn.offload_plan = plan
            output = layer(
                "input",
                "residual",
                None,
            )
            self.assertEqual(output.topk_indices, "topk")
            self.assertEqual(
                events,
                [
                    "attention",
                    *(["drain"] if enabled and not final_layer else []),
                    "norm",
                    "moe",
                ],
            )

    def test_reference_glm53_groups(self):
        value = config()
        groups = build_dsa_indexer_groups(value, 78)
        self.assertEqual(list(groups), [0, 1, 2, *range(6, 78, 4)])
        self.assertEqual(groups[2], (2, 3, 4, 5))
        self.assertEqual(groups[74], (74, 75, 76, 77))
        self.assertEqual(sum(map(len, groups.values())), 78)

    def test_explicit_pattern_precedence_and_invalid_first_layer(self):
        value = config()
        value.indexer_types = ["full", "shared", "compute", "skip"]
        self.assertEqual(build_dsa_indexer_groups(value, 4), {0: (0, 1), 2: (2, 3)})
        value.indexer_types = None
        value.index_topk_pattern = "FSFS"
        self.assertEqual(build_dsa_indexer_groups(value, 4), {0: (0, 1), 2: (2, 3)})
        value.index_topk_pattern = "SFFF"
        with self.assertRaisesRegex(ValueError, "layer 0"):
            build_dsa_indexer_groups(value, 4)
        value.index_topk_pattern = "XFFF"
        with self.assertRaisesRegex(ValueError, "Unsupported DSA"):
            build_dsa_indexer_groups(value, 4)

    def test_old_models_and_mtp_keep_per_layer_indexer(self):
        value = config()
        value.index_topk_freq = 1
        self.assertEqual(
            build_dsa_indexer_groups(value, 3), {0: (0,), 1: (1,), 2: (2,)}
        )
        value.is_mtp = True
        value.indexer_types = ["shared"]
        self.assertTrue(dsa_layer_has_indexer(value, 0))
        value.attn_config.is_sparse = False
        self.assertFalse(dsa_layer_has_indexer(value, 0))

    def test_source_only_cache_descriptors_and_offload(self):
        value = config()
        DeepSeekV2._post_build_model_config(value)
        sources = [
            layer
            for layer, descs in enumerate(value.kv_cache_spec_descs)
            if any(desc.tag == "indexer_kv" for desc in descs)
        ]
        self.assertEqual(sources, [0, 1, 2, *range(6, 78, 4)])
        model = SimpleNamespace(
            model_config=value,
            max_generate_batch_size=2,
        )
        DeepSeekV2._configure_sparse_kv_offload(model)
        runtime_groups = {
            layer: source
            for source, layers in build_dsa_indexer_groups(value, 78).items()
            for layer in layers
        }
        for layer, descs in enumerate(value.kv_cache_spec_descs):
            for desc in descs:
                if desc.tag == "default":
                    self.assertTrue(desc.offload.enabled)
                    self.assertEqual(desc.offload.max_selected_rows, 4096)
                    self.assertEqual(desc.offload_runtime_group, runtime_groups[layer])
                else:
                    self.assertIsNone(desc.offload)
                    self.assertIsNone(desc.offload_runtime_group)

    def test_shared_layers_do_not_request_missing_indexer_weights(self):
        weight = DeepSeekV2Weight.__new__(DeepSeekV2Weight)
        weight.model_config = config()
        weight._head_num = 128
        weight.nope_head_dim = weight.v_head_dim = 128
        weight.rope_head_dim = 64
        weight.kv_lora_rank = 512
        weight.q_use_lora = True
        weight._get_hf_ffn_layer_weight_info = lambda _: []
        indexer_names = {
            W.mla_indexer_qb_w,
            W.mla_indexer_k_w,
            W.mla_indexer_k_norm_w,
            W.mla_indexer_k_norm_b,
            W.mla_indexer_weights_proj_w,
        }
        for layer in (0, 2, 3, 5, 6, 77):
            names = {item.name for item in weight._get_hf_layer_weight_info(layer)}
            self.assertEqual(
                names & indexer_names, indexer_names if layer in (0, 2, 6) else set()
            )


if __name__ == "__main__":
    unittest.main()
