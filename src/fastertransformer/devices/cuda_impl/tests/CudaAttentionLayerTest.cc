#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/AttentionLayerTest.hpp"

class AttentionLayerTestFp16 : public AttentionLayerTest<half> {};

TEST_F(AttentionLayerTestFp16, testSimpleContextAttention) {
    AttentionConfigs attention_conf;
    attention_conf.head_num = 4;
    attention_conf.kv_head_num = 4;
    attention_conf.size_per_head = 8;
    attention_conf.tokens_per_block = 4;

    attention_conf.rope_config.style = RopeStyle::Base;
    attention_conf.rope_config.dim = attention_conf.size_per_head;
    attention_conf.rope_config.base = 1000000;

    const size_t layer_num = 2;
    const size_t block_num = 1024;
    CacheConfig cache_conf(
        layer_num, block_num, attention_conf.kv_head_num, attention_conf.size_per_head,
        attention_conf.tokens_per_block, getTensorType<TestType>());
    testAttentionLayer(cache_conf, attention_conf, {5}, {});
}

TEST_F(AttentionLayerTestFp16, testSimpleContextAttention2) {
    AttentionConfigs attention_conf;
    attention_conf.head_num = 16;
    attention_conf.kv_head_num = 16;
    attention_conf.size_per_head = 64;
    attention_conf.tokens_per_block = 4;
    attention_conf.mask_type = AttentionMaskType::causalMask;
    attention_conf.rope_config.style = RopeStyle::Base;
    attention_conf.rope_config.dim = attention_conf.size_per_head;
    attention_conf.rope_config.base = 1000000;

    const size_t layer_num = 2;
    const size_t block_num = 1024;
    CacheConfig cache_conf(
        layer_num, block_num, attention_conf.kv_head_num, attention_conf.size_per_head,
        attention_conf.tokens_per_block, getTensorType<TestType>());
    testAttentionLayer(cache_conf, attention_conf, {3}, {});
}
