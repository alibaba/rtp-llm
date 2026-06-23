from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v4 import _refresh_dsv4_kv_cache_spec_descs
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase
from rtp_llm.ops import CacheType, DataType, HybridAttentionType, KvCacheDataType


def _by_tag(layer_descs):
    result = {}
    for descs in layer_descs.values():
        for desc in descs:
            result.setdefault(desc.tag, desc)
    return result


class KVCacheSpecsTest(TestCase):
    def _basic_model_config(self) -> ModelConfig:
        config = ModelConfig()
        config.num_layers = 3
        config.attn_config.kv_head_num = 2
        config.attn_config.size_per_head = 16
        config.attn_config.tokens_per_block = 8
        config.attn_config.use_mla = False
        return config

    def test_base_model_builds_default_mha_desc_per_layer(self):
        config = self._basic_model_config()

        BaseModel._post_build_model_config(config)

        self.assertEqual(sorted(config.kv_cache_spec_descs.keys()), [0, 1, 2])
        for layer_id in range(3):
            self.assertEqual(len(config.kv_cache_spec_descs[layer_id]), 1)
            desc = config.kv_cache_spec_descs[layer_id][0]
            self.assertEqual(desc.tag, "default")
            self.assertEqual(desc.cache_type, CacheType.MHA)
            self.assertEqual(desc.seq_size_per_block, 8)
            self.assertEqual(desc.size_per_head, 16)

    def test_qwen3_next_builds_one_desc_per_layer(self):
        config = self._basic_model_config()
        config.num_layers = 4
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        config.linear_attention_config.linear_key_head_dim = 32
        config.linear_attention_config.linear_value_head_dim = 32
        config.linear_attention_config.linear_conv_kernel_dim = 4

        Qwen3NextBase._post_build_model_config(config)

        self.assertEqual(sorted(config.kv_cache_spec_descs.keys()), [0, 1, 2, 3])
        self.assertEqual(
            [config.kv_cache_spec_descs[i][0].tag for i in range(4)],
            ["linear", "full", "linear", "full"],
        )
        full_desc = config.kv_cache_spec_descs[1][0]
        linear_desc = config.kv_cache_spec_descs[0][0]
        self.assertEqual(full_desc.cache_type, CacheType.MHA)
        self.assertEqual(full_desc.seq_size_per_block, 8)
        self.assertEqual(full_desc.size_per_head, 16)
        self.assertEqual(linear_desc.cache_type, CacheType.LINEAR)
        self.assertEqual(linear_desc.head_k_dim, 32)
        self.assertEqual(linear_desc.head_v_dim, 32)
        self.assertEqual(linear_desc.conv_kernel_dim, 4)

    def test_deepseek_v4_builds_layer_wise_desc_declarations(self):
        config = self._basic_model_config()
        config.num_layers = 5
        config.attn_config.size_per_head = 512
        config.attn_config.indexer_head_dim = 128
        config.attn_config.layer_compress_ratios = [4, 128, 4, 128, 0]

        _refresh_dsv4_kv_cache_spec_descs(config)

        self.assertEqual(sorted(config.kv_cache_spec_descs.keys()), [0, 1, 2, 3, 4])
        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[0]],
            ["csa_kv", "indexer_kv", "indexer_state", "csa_state", "swa_kv"],
        )
        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[1]],
            ["hca_kv", "hca_state", "swa_kv"],
        )
        self.assertEqual([desc.tag for desc in config.kv_cache_spec_descs[4]], ["swa_kv"])

        descs = _by_tag(config.kv_cache_spec_descs)
        self.assertEqual(descs["csa_kv"].cache_type, CacheType.COMPRESSED_KV)
        self.assertEqual(descs["indexer_state"].cache_type, CacheType.FIXED_STATE)
        self.assertTrue(
            all(
                descs[tag].dtype == DataType.TYPE_UINT8
                for tag in ["csa_kv", "hca_kv", "indexer_kv", "swa_kv"]
            )
        )
        self.assertTrue(
            all(
                descs[tag].dtype == DataType.TYPE_FP32
                for tag in ["indexer_state", "csa_state", "hca_state"]
            )
        )
        self.assertEqual(descs["csa_kv"].entry_elems, 1024)
        self.assertEqual(descs["indexer_kv"].entry_elems, 256)
        self.assertEqual(descs["indexer_state"].entry_elems, 512)
        self.assertEqual(descs["csa_state"].entry_elems, 2048)
        self.assertEqual(descs["hca_state"].entry_elems, 1024)
        self.assertEqual(descs["swa_kv"].entry_elems, 1024)

    def test_deepseek_v4_refresh_updates_fp8_entry_sizes(self):
        config = self._basic_model_config()
        config.num_layers = 2
        config.attn_config.size_per_head = 512
        config.attn_config.indexer_head_dim = 128
        config.attn_config.layer_compress_ratios = [4, 128]
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8

        _refresh_dsv4_kv_cache_spec_descs(config)

        desc_by_tag = _by_tag(config.kv_cache_spec_descs)
        self.assertEqual(desc_by_tag["csa_kv"].entry_elems, 584)
        self.assertEqual(desc_by_tag["hca_kv"].entry_elems, 584)
        self.assertEqual(desc_by_tag["swa_kv"].entry_elems, 584)
        self.assertEqual(desc_by_tag["indexer_kv"].entry_elems, 132)
        self.assertEqual(desc_by_tag["indexer_state"].entry_elems, 512)
        self.assertEqual(desc_by_tag["csa_state"].entry_elems, 2048)
        self.assertEqual(desc_by_tag["hca_state"].entry_elems, 1024)
        self.assertEqual(desc_by_tag["csa_kv"].cache_type, CacheType.COMPRESSED_KV)
        self.assertEqual(desc_by_tag["swa_kv"].cache_type, CacheType.FIXED_STATE)
        self.assertTrue(desc_by_tag["swa_kv"].is_state_cache)

    def test_deepseek_v4_refresh_is_desc_order_stable(self):
        config = self._basic_model_config()
        config.num_layers = 2
        config.attn_config.size_per_head = 512
        config.attn_config.indexer_head_dim = 128
        config.attn_config.layer_compress_ratios = [4, 128]
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8

        _refresh_dsv4_kv_cache_spec_descs(config)

        desc_by_tag = _by_tag(config.kv_cache_spec_descs)
        self.assertEqual(desc_by_tag["csa_kv"].entry_elems, 584)
        self.assertEqual(desc_by_tag["hca_kv"].entry_elems, 584)
        self.assertEqual(desc_by_tag["swa_kv"].entry_elems, 584)
        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "csa_kv")
        self.assertEqual(config.kv_cache_spec_descs[1][0].tag, "hca_kv")


if __name__ == "__main__":
    main()
