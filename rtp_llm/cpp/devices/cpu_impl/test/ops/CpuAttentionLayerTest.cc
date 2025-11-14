#include "rtp_llm/cpp/devices/base_tests/AttentionLayerTest.hpp"
#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"

using namespace rtp_llm;

class CpuAttentionLayerTestFP16: public AttentionLayerTest<float> {};

TEST_F(CpuAttentionLayerTestFP16, testSimpleContextAttention) {
    AttentionConfigs attention_conf;
    attention_conf.head_num         = 16;
    attention_conf.kv_head_num      = 16;
    attention_conf.size_per_head    = 32;
    attention_conf.tokens_per_block = 4;

    attention_conf.is_causal = true;

    attention_conf.rope_config.style   = RopeStyle::Base;
    attention_conf.rope_config.base    = 10000;
    attention_conf.rope_config.max_pos = 4096;
    attention_conf.rope_config.dim     = attention_conf.size_per_head;

    const size_t layer_num = 1;
    const size_t block_num = 1024;
    CacheConfig  cache_conf(KVCacheParam{layer_num,
                                        block_num,
                                        (uint)attention_conf.kv_head_num,
                                        (uint)attention_conf.size_per_head,
                                        (uint)attention_conf.tokens_per_block,
                                        getTensorType<TestType>()});

    testAttentionLayer(cache_conf, attention_conf, {5}, {});
}
