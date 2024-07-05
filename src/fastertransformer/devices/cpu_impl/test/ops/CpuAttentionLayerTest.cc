#include "src/fastertransformer/devices/base_tests/AttentionLayerTest.hpp"
#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

class CpuAttentionLayerTestFP16: public AttentionLayerTest<float> {};

TEST_F(CpuAttentionLayerTestFP16, testSimpleContextAttention) {
    AttentionConfigs attention_conf;
    attention_conf.head_num = 16;
    attention_conf.kv_head_num = 16;
    attention_conf.size_per_head = 32; 
    attention_conf.tokens_per_block = 4;

    attention_conf.mask_type = AttentionMaskType::causalMask;
    
    attention_conf.rope_config.embedding_style = RopeType::Base;
    attention_conf.rope_config.embedding_base = 10000;
    attention_conf.rope_config.dynamic_embedding_max_pos = 4096;
    attention_conf.rope_config.embedding_dim = attention_conf.size_per_head;

    const size_t layer_num = 1;
    const size_t block_num = 1024;
    CacheConfig cache_conf(
        layer_num, block_num, attention_conf.kv_head_num, attention_conf.size_per_head,
        attention_conf.tokens_per_block, getTensorType<TestType>());
    
    testAttentionLayer(cache_conf, attention_conf, {5}, {});
}