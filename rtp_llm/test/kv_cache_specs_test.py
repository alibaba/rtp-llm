from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import (
    DataType,
    KVCacheSpecType,
    LinearKVCacheSpec,
    MHAKVCacheSpec,
    MLAKVCacheSpec,
)


class KVCacheSpecsTest(TestCase):
    def test_model_config_accepts_layer_wise_mha_specs(self):
        config = ModelConfig()
        config.num_layers = 3

        spec = MHAKVCacheSpec()
        spec.tag = "default"
        spec.type = KVCacheSpecType.MultiHeadAttention
        spec.dtype = DataType.TYPE_FP16
        spec.local_head_num_kv = 2
        spec.seq_size_per_block = 8
        spec.size_per_head = 16

        config.kv_cache_specs = {layer_id: [spec] for layer_id in range(config.num_layers)}

        self.assertEqual(sorted(config.kv_cache_specs.keys()), [0, 1, 2])
        for layer_id in range(config.num_layers):
            layer_spec = config.kv_cache_specs[layer_id][0]
            self.assertIsInstance(layer_spec, MHAKVCacheSpec)
            self.assertEqual(layer_spec.tag, "default")
            self.assertEqual(layer_spec.type, KVCacheSpecType.MultiHeadAttention)
            self.assertEqual(layer_spec.dtype, DataType.TYPE_FP16)
            self.assertEqual(layer_spec.seq_size_per_block, 8)
            self.assertEqual(layer_spec.size_per_head, 16)

    def test_model_config_accepts_mixed_full_and_linear_specs(self):
        config = ModelConfig()
        config.num_layers = 4

        full_spec = MHAKVCacheSpec()
        full_spec.tag = "full"
        full_spec.type = KVCacheSpecType.MultiHeadAttention
        full_spec.seq_size_per_block = 8
        full_spec.size_per_head = 16

        linear_spec = LinearKVCacheSpec()
        linear_spec.tag = "linear"
        linear_spec.type = KVCacheSpecType.LinearAttention
        linear_spec.seq_size_per_block = 8
        linear_spec.head_k_dim = 32
        linear_spec.head_v_dim = 32
        linear_spec.conv_kernel_dim = 4

        config.kv_cache_specs = {
            0: [linear_spec],
            1: [full_spec],
            2: [linear_spec],
            3: [full_spec],
        }

        self.assertEqual([config.kv_cache_specs[i][0].tag for i in range(4)], ["linear", "full", "linear", "full"])
        self.assertIsInstance(config.kv_cache_specs[0][0], LinearKVCacheSpec)
        self.assertEqual(config.kv_cache_specs[0][0].type, KVCacheSpecType.LinearAttention)
        self.assertEqual(config.kv_cache_specs[0][0].head_k_dim, 32)
        self.assertEqual(config.kv_cache_specs[1][0].type, KVCacheSpecType.MultiHeadAttention)

    def test_model_config_accepts_mla_specs(self):
        config = ModelConfig()
        config.num_layers = 2

        spec = MLAKVCacheSpec()
        spec.tag = "default"
        spec.type = KVCacheSpecType.MultiHeadLatentAttention
        spec.seq_size_per_block = 16
        spec.kv_lora_rank = 512
        spec.rope_head_dim = 64

        config.kv_cache_specs = {0: [spec], 1: [spec]}

        self.assertEqual(config.kv_cache_specs[0][0].type, KVCacheSpecType.MultiHeadLatentAttention)
        self.assertEqual(config.kv_cache_specs[0][0].kv_lora_rank, 512)
        self.assertEqual(config.kv_cache_specs[1][0].rope_head_dim, 64)


if __name__ == "__main__":
    main()
