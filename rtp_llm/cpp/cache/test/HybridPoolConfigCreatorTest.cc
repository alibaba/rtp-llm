#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

ModelConfig makeHybridAttentionModelConfig() {
    ModelConfig model_config;
    model_config.num_layers                                                = 4;
    model_config.hidden_size                                               = 128;
    model_config.attn_config.head_num                                      = 4;
    model_config.attn_config.kv_head_num                                   = 2;
    model_config.attn_config.size_per_head                                 = 16;
    model_config.attn_config.tokens_per_block                              = 8;
    model_config.hybrid_attention_config.enable_hybrid_attention           = true;
    model_config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    model_config.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    model_config.linear_attention_config.linear_conv_kernel_dim = 4;
    model_config.linear_attention_config.linear_key_head_dim    = 16;
    model_config.linear_attention_config.linear_value_head_dim  = 16;
    model_config.linear_attention_config.linear_num_key_heads   = 2;
    model_config.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(model_config);
    return model_config;
}

TEST(HybridPoolConfigCreatorTest, MhaPhysicalStrideIsNotRepeatedPerKernelBlock) {
    auto model_config = makeHybridAttentionModelConfig();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 2;

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 8;
    kv_cache_config.kernel_seq_size_per_block = 2;
    auto config = HybridPoolConfigCreator::createConfig(model_config, parallelism_config, kv_cache_config, false, 0);

    const auto full_gid = config.groupIdForTag("full");
    ASSERT_EQ(config.specForGroup(full_gid)->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(config.kernelBlocksPerKvBlockForGroup(full_gid), 4u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(full_gid), config.specForGroup(full_gid)->block_size_bytes());
}

}  // namespace
}  // namespace rtp_llm::test
