#include <gtest/gtest.h>

// GLM-5.3 compressed-indexer cache geometry tests.

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

ModelConfig makeGlm53HybridModelConfig() {
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
    mc.attn_config.indexer_head_dim           = 128;
    mc.attn_config.indexer_head_num           = 64;
    mc.attn_config.indexer_topk               = 512;
    mc.attn_config.indexer_compress_ratio     = 4;
    mc.attn_config.indexer_compressor_overlap = 0;
    mc.attn_config.sparse_attention_topk      = 2051;

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
    mc.linear_attention_config.ssm_state_dtype        = DataType::TYPE_FP32;
    return mc;
}

KVCacheConfig makeGlm53KvCacheConfig() {
    KVCacheConfig config;
    config.seq_size_per_block        = 128;
    config.kernel_seq_size_per_block = 128;
    config.dsv4_fixed_pool_blocks    = 32;
    return config;
}

}  // namespace

TEST(GLM53CacheConfigTest, HybridMlaKdaAddsOnlyIndexerSidePools) {
    ParallelismConfig pc;
    auto              config =
        HybridPoolConfigCreator::createConfig(makeGlm53HybridModelConfig(), pc, makeGlm53KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 4u);
    ASSERT_EQ(config.group_region_names.size(), 4u);
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[1], KVCacheRegionName::DEFAULT);
    EXPECT_EQ(config.group_region_names[2], KVCacheRegionName::INDEXER_KV);
    EXPECT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    EXPECT_NE(dynamic_cast<MLAKVCacheSpec*>(config.cache_specs[0].get()), nullptr);
    auto* linear_cache = dynamic_cast<LinearKVCacheSpec*>(config.cache_specs[1].get());
    ASSERT_NE(linear_cache, nullptr);
    EXPECT_EQ(linear_cache->ssm_state_dtype, DataType::TYPE_FP32);

    auto* indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->layer_num, 4u);
    EXPECT_EQ(indexer_kv->entry_elems, 132u);
    EXPECT_EQ(indexer_kv->entries_per_block, 32u);
    EXPECT_EQ(indexer_kv->block_size_bytes(), 32u * 132u);

    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->layer_num, 4u);
    EXPECT_EQ(indexer_state->state_dim, 256u);
    EXPECT_EQ(indexer_state->entries_per_block, 4u);
    EXPECT_EQ(indexer_state->block_size_bytes(), 4u * 256u * sizeof(float));

    EXPECT_TRUE(config.use_typed_cache_regions);
    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_TRUE(config.use_opaque_kv_cache_store);
    EXPECT_EQ(config.seq_size_per_block, 128u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
}

TEST(GLM53CacheConfigTest, IndexerPoolStaysFp8WhenMainMlaCacheIsBf16) {
    ParallelismConfig pc;
    auto              mc          = makeGlm53HybridModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::BASE;
    auto config                   = HybridPoolConfigCreator::createConfig(mc, pc, makeGlm53KvCacheConfig(), false, 0);

    auto* indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->entry_elems, DSV4_FP8_INDEXER_ENTRY_BYTES);
}

TEST(GLM53CacheConfigTest, KdaLayersDoNotOwnIndexerRegions) {
    ParallelismConfig pc;
    auto              config =
        HybridPoolConfigCreator::createConfig(makeGlm53HybridModelConfig(), pc, makeGlm53KvCacheConfig(), false, 0);

    const auto idx_kv_region    = static_cast<size_t>(KVCacheRegionName::INDEXER_KV);
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

TEST(GLM53CacheConfigTest, ExplicitIndexerLayersSupportMtpSharingSchedule) {
    ParallelismConfig pc;
    auto              mc             = makeGlm53HybridModelConfig();
    mc.attn_config.indexer_layer_ids = {2, 5};
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeGlm53KvCacheConfig(), false, 0);

    ASSERT_EQ(config.global_layer_ids.size(), 4u);
    EXPECT_EQ(config.global_layer_ids[2], std::vector<int>({2, 5}));
    EXPECT_EQ(config.global_layer_ids[3], std::vector<int>({2, 5}));
}

TEST(GLM53CacheConfigTest, StateRingIncludesSpeculativeDecodeSlack) {
    ParallelismConfig pc;
    auto              config =
        HybridPoolConfigCreator::createConfig(makeGlm53HybridModelConfig(), pc, makeGlm53KvCacheConfig(), false, 3);
    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->entries_per_block, 8u);
}

TEST(GLM53CacheConfigTest, PureMlaModelAlsoGetsCompressedIndexerPools) {
    ParallelismConfig pc;
    auto              mc                               = makeGlm53HybridModelConfig();
    mc.hybrid_attention_config.enable_hybrid_attention = false;
    mc.hybrid_attention_config.hybrid_attention_types.clear();
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeGlm53KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 3u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 6u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 6u);
    EXPECT_EQ(config.global_layer_ids[2].size(), 6u);
}

TEST(GLM53CacheConfigTest, SplitPhysicalBlockScalesOnlyFullAttentionPools) {
    ParallelismConfig pc;
    auto              kv_config         = makeGlm53KvCacheConfig();
    kv_config.seq_size_per_block        = 256;
    kv_config.kernel_seq_size_per_block = 128;
    auto config = HybridPoolConfigCreator::createConfig(makeGlm53HybridModelConfig(), pc, kv_config, false, 0);

    ASSERT_EQ(config.cache_specs.size(), 4u);
    EXPECT_EQ(config.seq_size_per_block, 256u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    auto* indexer_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    ASSERT_NE(indexer_kv, nullptr);
    EXPECT_EQ(indexer_kv->entries_per_block, 32u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], 2u * 32u * DSV4_FP8_INDEXER_ENTRY_BYTES);

    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(config.group_kv_block_stride_bytes[3], indexer_state->block_size_bytes());
}

TEST(GLM53CacheConfigTest, RejectsInvalidCompressedIndexerGeometry) {
    ParallelismConfig pc;
    auto              kv_config = makeGlm53KvCacheConfig();

    auto bad_ratio                               = makeGlm53HybridModelConfig();
    bad_ratio.attn_config.indexer_compress_ratio = 2;
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(bad_ratio, pc, kv_config, false, 0), std::exception);

    auto bad_topk                              = makeGlm53HybridModelConfig();
    bad_topk.attn_config.sparse_attention_topk = 2048;
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(bad_topk, pc, kv_config, false, 0), std::exception);

    kv_config.seq_size_per_block        = 64;
    kv_config.kernel_seq_size_per_block = 64;
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(makeGlm53HybridModelConfig(), pc, kv_config, false, 0),
                 std::exception);
}

TEST(GLM53CacheConfigTest, RejectsInvalidExplicitIndexerLayerOwnership) {
    ParallelismConfig pc;
    auto              kv_config = makeGlm53KvCacheConfig();

    auto duplicate                          = makeGlm53HybridModelConfig();
    duplicate.attn_config.indexer_layer_ids = {1, 1};
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(duplicate, pc, kv_config, false, 0), std::exception);

    auto linear_owner                          = makeGlm53HybridModelConfig();
    linear_owner.attn_config.indexer_layer_ids = {0};
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(linear_owner, pc, kv_config, false, 0), std::exception);

    auto out_of_range                          = makeGlm53HybridModelConfig();
    out_of_range.attn_config.indexer_layer_ids = {6};
    EXPECT_THROW(HybridPoolConfigCreator::createConfig(out_of_range, pc, kv_config, false, 0), std::exception);
}

}  // namespace test
}  // namespace rtp_llm
