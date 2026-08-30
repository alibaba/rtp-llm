#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

CacheConfig makeKimiHybridConfig(bool legacy_independent_flag = false) {
    ModelConfig config;
    config.num_layers                                                = 4;
    config.attn_config.head_num                                      = 4;
    config.attn_config.kv_head_num                                   = 2;
    config.attn_config.size_per_head                                 = 32;
    config.attn_config.tokens_per_block                              = 8;
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = legacy_independent_flag;
    config.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    config.linear_attention_config.linear_conv_kernel_dim = 4;
    config.linear_attention_config.linear_key_head_dim    = 16;
    config.linear_attention_config.linear_value_head_dim  = 16;
    config.linear_attention_config.linear_num_key_heads   = 2;
    config.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(config);

    ParallelismConfig parallelism;
    return CacheConfigCreator::createBasicConfig(config, parallelism, KVCacheConfig{}, /*gen_num_per_cycle=*/0);
}

CacheConfig makeDeepSeekV4HybridPoolConfig() {
    ModelConfig config;
    config.num_layers                                                = 2;
    config.attn_config.head_num                                      = 128;
    config.attn_config.kv_head_num                                   = 1;
    config.attn_config.size_per_head                                 = 512;
    config.attn_config.indexer_head_dim                              = 128;
    config.attn_config.tokens_per_block                              = 128;
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(config, {128, 4});

    ParallelismConfig parallelism;
    return CacheConfigCreator::createBasicConfig(config, parallelism, KVCacheConfig{}, /*gen_num_per_cycle=*/0);
}

TEST(CacheSemanticSnapshotTest, SingleMhaMatchesGolden) {
    const auto config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                 /*block_num=*/7,
                                                 /*tokens_per_block=*/8,
                                                 DataType::TYPE_FP16,
                                                 /*local_head_num_kv=*/2,
                                                 /*size_per_head=*/4);

    const CacheSemanticSnapshot expected = {{"default",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0, 1},
                                             7,
                                             8,
                                             8,
                                             512,
                                             256,
                                             0}};

    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheSemanticSnapshotTest, KimiHybridMatchesGolden) {
    const CacheSemanticSnapshot expected = {{"full",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {1, 3},
                                             0,
                                             8,
                                             8,
                                             4096,
                                             2048,
                                             0},
                                            {"linear",
                                             KVCacheSpecType::LinearAttention,
                                             CacheGroupType::LINEAR,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             1,
                                             true,
                                             CpBlockMappingMode::NONE,
                                             CpBlockSliceMode::NONE,
                                             {0, 2},
                                             0,
                                             8,
                                             8,
                                             3200,
                                             1600,
                                             0}};

    const auto legacy_flag_off = makeKimiHybridConfig(false);
    const auto legacy_flag_on  = makeKimiHybridConfig(true);
    EXPECT_EQ(snapshotCacheConfig(legacy_flag_off), expected);
    EXPECT_EQ(snapshotCacheConfig(legacy_flag_on), expected);
}

TEST(CacheSemanticSnapshotTest, DeepSeekV4HybridPoolMatchesGolden) {
    const CacheGroupSemanticSnapshot full_csa      = {"csa_kv",
                                                      KVCacheSpecType::OpaqueKV,
                                                      CacheGroupType::FULL,
                                                      true,
                                                      CacheEvictPolicy::CHAIN,
                                                      true,
                                                      0,
                                                      0,
                                                      true,
                                                      CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                      CpBlockSliceMode::NONE,
                                                      {1},
                                                      0,
                                                      128,
                                                      128,
                                                      32768,
                                                      32768,
                                                      0};
    const CacheGroupSemanticSnapshot state_csa     = {"csa_state",
                                                      KVCacheSpecType::OpaqueState,
                                                      CacheGroupType::SWA,
                                                      true,
                                                      CacheEvictPolicy::INDEPENDENT,
                                                      true,
                                                      0,
                                                      2,
                                                      true,
                                                      CpBlockMappingMode::COMPACT_LAST_RANK,
                                                      CpBlockSliceMode::PAYLOAD_BYTES,
                                                      {1},
                                                      0,
                                                      128,
                                                      128,
                                                      65536,
                                                      65536,
                                                      0};
    const CacheGroupSemanticSnapshot full_hca      = {"hca_kv",
                                                      KVCacheSpecType::OpaqueKV,
                                                      CacheGroupType::FULL,
                                                      true,
                                                      CacheEvictPolicy::CHAIN,
                                                      true,
                                                      0,
                                                      0,
                                                      true,
                                                      CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                      CpBlockSliceMode::NONE,
                                                      {0},
                                                      0,
                                                      128,
                                                      128,
                                                      1024,
                                                      1024,
                                                      0};
    const CacheGroupSemanticSnapshot state_hca     = {"hca_state",
                                                      KVCacheSpecType::OpaqueState,
                                                      CacheGroupType::SWA,
                                                      false,
                                                      CacheEvictPolicy::INDEPENDENT,
                                                      true,
                                                      256,
                                                      1,
                                                      false,
                                                      CpBlockMappingMode::COMPACT_LAST_RANK,
                                                      CpBlockSliceMode::PAYLOAD_BYTES,
                                                      {0},
                                                      0,
                                                      128,
                                                      128,
                                                      524288,
                                                      524288,
                                                      0};
    const CacheGroupSemanticSnapshot full_indexer  = {"indexer_kv",
                                                      KVCacheSpecType::OpaqueKV,
                                                      CacheGroupType::FULL,
                                                      true,
                                                      CacheEvictPolicy::CHAIN,
                                                      true,
                                                      0,
                                                      0,
                                                      true,
                                                      CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                      CpBlockSliceMode::NONE,
                                                      {1},
                                                      0,
                                                      128,
                                                      128,
                                                      8192,
                                                      8192,
                                                      0};
    const CacheGroupSemanticSnapshot state_indexer = {"indexer_state",
                                                      KVCacheSpecType::OpaqueState,
                                                      CacheGroupType::SWA,
                                                      true,
                                                      CacheEvictPolicy::INDEPENDENT,
                                                      true,
                                                      0,
                                                      2,
                                                      true,
                                                      CpBlockMappingMode::COMPACT_LAST_RANK,
                                                      CpBlockSliceMode::PAYLOAD_BYTES,
                                                      {1},
                                                      0,
                                                      128,
                                                      128,
                                                      16384,
                                                      16384,
                                                      0};
    const CacheGroupSemanticSnapshot state_swa     = {"swa_kv",
                                                      KVCacheSpecType::OpaqueState,
                                                      CacheGroupType::SWA,
                                                      true,
                                                      CacheEvictPolicy::INDEPENDENT,
                                                      true,
                                                      0,
                                                      2,
                                                      true,
                                                      CpBlockMappingMode::COMPACT_LAST_RANK,
                                                      CpBlockSliceMode::EQUAL_BYTES,
                                                      {0, 1},
                                                      0,
                                                      128,
                                                      128,
                                                      262144,
                                                      131072,
                                                      0};
    const CacheSemanticSnapshot      expected      = {
        full_csa, state_csa, full_hca, state_hca, full_indexer, state_indexer, state_swa};

    EXPECT_EQ(snapshotCacheConfig(makeDeepSeekV4HybridPoolConfig()), expected);
}

}  // namespace
}  // namespace rtp_llm::test
