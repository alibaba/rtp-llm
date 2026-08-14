#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {
namespace {

ModelConfig makeGlm54HybridModelConfig() {
    ModelConfig mc;
    mc.num_layers                             = 6;
    mc.hidden_size                            = 1024;
    mc.attn_config.use_mla                    = true;
    mc.attn_config.is_sparse                  = true;
    mc.attn_config.head_num                   = 8;
    mc.attn_config.kv_head_num                = 1;
    mc.attn_config.size_per_head              = 128;
    mc.attn_config.kv_lora_rank               = 512;
    mc.attn_config.rope_head_dim              = 64;
    mc.attn_config.tokens_per_block           = 128;
    mc.attn_config.kernel_tokens_per_block    = 128;
    mc.attn_config.kv_cache_dtype             = KvCacheDataType::FP8;
    mc.attn_config.indexer_head_dim            = 128;
    mc.attn_config.indexer_head_num            = 64;
    mc.attn_config.indexer_topk                = 512;
    mc.attn_config.indexer_compress_ratio      = 4;
    mc.attn_config.indexer_compressor_overlap = 1;
    mc.attn_config.sparse_attention_topk       = 2048;

    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    mc.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::NONE,
        HybridAttentionType::LINEAR,
        HybridAttentionType::NONE,
        HybridAttentionType::NONE,
    };
    mc.linear_attention_config.linear_conv_kernel_dim = 4;
    mc.linear_attention_config.linear_key_head_dim    = 64;
    mc.linear_attention_config.linear_value_head_dim  = 64;
    mc.linear_attention_config.linear_num_key_heads   = 4;
    mc.linear_attention_config.linear_num_value_heads = 4;
    return mc;
}

KVCacheConfig makeGlm54KvCacheConfig() {
    KVCacheConfig config;
    config.seq_size_per_block        = 128;
    config.kernel_seq_size_per_block = 128;
    config.dsv4_fixed_pool_blocks    = 32;
    return config;
}

}  // namespace

TEST(GLM54CacheConfigTest, HybridMlaKdaAddsOnlyIndexerSidePools) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeGlm54HybridModelConfig(), pc, makeGlm54KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 4u);
    ASSERT_EQ(config.group_region_names.size(), 4u);
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[1], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[2], KVCacheRegionName::INDEXER_KV);
    EXPECT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    EXPECT_NE(dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[0].get()), nullptr);
    EXPECT_NE(dynamic_cast<LinearKVCacheSpec*>(config.cache_specs[1].get()), nullptr);

    auto* indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->layer_num, 4u);
    EXPECT_EQ(indexer_kv->entry_elems, 132u);
    EXPECT_EQ(indexer_kv->entries_per_block, 32u);
    EXPECT_EQ(indexer_kv->block_size_bytes(), 32u * 132u);

    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->layer_num, 4u);
    EXPECT_EQ(indexer_state->state_dim, 512u);
    EXPECT_EQ(indexer_state->entries_per_block, 8u);
    EXPECT_EQ(indexer_state->block_size_bytes(), 8u * 512u * sizeof(float));

    EXPECT_TRUE(config.use_typed_cache_regions);
    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_TRUE(config.use_opaque_kv_cache_store);
    EXPECT_EQ(config.seq_size_per_block, 128u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
}

TEST(GLM54CacheConfigTest, KdaLayersDoNotOwnIndexerRegions) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeGlm54HybridModelConfig(), pc, makeGlm54KvCacheConfig(), false, 0);

    const auto idx_kv_region = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
    const auto idx_state_region = static_cast<size_t>(KVCacheRegionName::INDEXER_STATE);
    ASSERT_EQ(config.layer_region_to_group_id.size(), 6u);
    for (int layer : {0, 3}) {
        EXPECT_EQ(config.layer_region_to_group_id[static_cast<size_t>(layer)][idx_kv_region], -1);
        EXPECT_EQ(config.layer_region_to_group_id[static_cast<size_t>(layer)][idx_state_region], -1);
    }
    for (int layer : {1, 2, 4, 5}) {
        EXPECT_GE(config.layer_region_to_group_id[static_cast<size_t>(layer)][idx_kv_region], 0);
        EXPECT_GE(config.layer_region_to_group_id[static_cast<size_t>(layer)][idx_state_region], 0);
    }
}

TEST(GLM54CacheConfigTest, ExplicitIndexerLayersSupportFutureIndexShareSchedule) {
    ParallelismConfig pc;
    auto mc                          = makeGlm54HybridModelConfig();
    mc.attn_config.indexer_layer_ids = {2, 5};
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeGlm54KvCacheConfig(), false, 0);

    ASSERT_EQ(config.global_layer_ids.size(), 4u);
    EXPECT_EQ(config.global_layer_ids[2], std::vector<int>({2, 5}));
    EXPECT_EQ(config.global_layer_ids[3], std::vector<int>({2, 5}));
}

TEST(GLM54CacheConfigTest, StateRingIncludesSpeculativeDecodeSlack) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeGlm54HybridModelConfig(), pc, makeGlm54KvCacheConfig(), false, 3);
    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->entries_per_block, 12u);
}

TEST(GLM54CacheConfigTest, PureMlaModelAlsoGetsCompressedIndexerPools) {
    ParallelismConfig pc;
    auto mc = makeGlm54HybridModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention = false;
    mc.hybrid_attention_config.hybrid_attention_types.clear();
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeGlm54KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 3u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 6u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 6u);
    EXPECT_EQ(config.global_layer_ids[2].size(), 6u);
}

}  // namespace test
}  // namespace rtp_llm
