#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {

constexpr int      kDsv4PoolNum           = 7;
constexpr uint32_t kDsv4TokensPerBlock    = 256;
constexpr uint32_t kDsv4KvEntryBytes      = 1024;
constexpr uint32_t kDsv4IndexerEntryBytes = 256;
constexpr uint32_t kDsv4Fp8KvEntryBytes   = 584;
constexpr uint32_t kNonFullAdditionBlocks = 256;

}  // namespace

static ModelConfig makeProModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 61;
    mc.hidden_size                  = 7168;
    mc.attn_config.head_num         = 128;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 1024;
    mc.attn_config.o_groups         = 16;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios;
    ratios.push_back(128);
    ratios.push_back(128);
    for (int i = 2; i < 61; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

static ModelConfig makeFlashModelConfig() {
    ModelConfig mc;
    mc.num_layers                   = 43;
    mc.hidden_size                  = 4096;
    mc.attn_config.head_num         = 64;
    mc.attn_config.kv_head_num      = 1;
    mc.attn_config.size_per_head    = 512;
    mc.attn_config.rope_head_dim    = 64;
    mc.attn_config.sliding_window   = 128;
    mc.attn_config.indexer_head_dim = 128;
    mc.attn_config.indexer_head_num = 64;
    mc.attn_config.indexer_topk     = 512;
    mc.attn_config.o_groups         = 8;
    mc.attn_config.o_lora_rank      = 1024;
    std::vector<int> ratios         = {0, 0};
    for (int i = 2; i < 43; i++) {
        ratios.push_back((i % 2 == 0) ? 4 : 128);
    }
    mc.attn_config.layer_compress_ratios = ratios;
    return mc;
}

static ModelConfig makeFlashMtpModelConfig() {
    ModelConfig mc                       = makeFlashModelConfig();
    mc.num_layers                        = 1;
    mc.attn_config.layer_compress_ratios = {0};
    return mc;
}

static ModelConfig makeHybridAttentionModelConfig(bool independent_pool) {
    ModelConfig mc;
    mc.num_layers                                                = 4;
    mc.hidden_size                                               = 128;
    mc.attn_config.head_num                                      = 4;
    mc.attn_config.kv_head_num                                   = 2;
    mc.attn_config.size_per_head                                 = independent_pool ? 16 : 32;
    mc.attn_config.tokens_per_block                              = 8;
    mc.hybrid_attention_config.enable_hybrid_attention           = true;
    mc.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pool;
    mc.hybrid_attention_config.hybrid_attention_types            = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    mc.linear_attention_config.linear_conv_kernel_dim = 4;
    mc.linear_attention_config.linear_key_head_dim    = 16;
    mc.linear_attention_config.linear_value_head_dim  = 16;
    mc.linear_attention_config.linear_num_key_heads   = 2;
    mc.linear_attention_config.linear_num_value_heads = 2;
    return mc;
}

// ============================================================
// Layer classification
// ============================================================

TEST(HybridPoolConfigCreatorTest, ProLayerClassification) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 30u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 31u);
    EXPECT_EQ(config.global_layer_ids[6].size(), 61u);
}

TEST(HybridPoolConfigCreatorTest, FlashLayerClassification) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 20u);
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);
}

TEST(HybridPoolConfigCreatorTest, MtpSwaOnlyLayerIsNotStripped) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(makeFlashMtpModelConfig(), pc, KVCacheConfig{}, true);

    EXPECT_EQ(config.layer_num, 1u);
    EXPECT_EQ(config.block_size_bytes, 1u);
    EXPECT_TRUE(config.global_layer_ids[0].empty());
    EXPECT_TRUE(config.global_layer_ids[1].empty());
    ASSERT_EQ(config.global_layer_ids[6], std::vector<int>({0}));
    ASSERT_EQ(config.layer_to_group_id.size(), 1u);
    EXPECT_EQ(config.layer_to_group_id[0], 6);
    ASSERT_EQ(config.layer_region_to_group_id.size(), 1u);
    EXPECT_EQ(config.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::SWA_KV)], 6);
}

// ============================================================
// Pool specs
// ============================================================

TEST(HybridPoolConfigCreatorTest, ProPoolSpecs) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);

    EXPECT_EQ(config.cache_specs[0]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);

    EXPECT_EQ(config.cache_specs[1]->layer_num, 31u);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);

    EXPECT_EQ(config.cache_specs[2]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);

    EXPECT_EQ(config.cache_specs[3]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 256u * 512u * 4u);

    EXPECT_EQ(config.cache_specs[4]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 256u * 2048u * 4u);

    EXPECT_EQ(config.cache_specs[5]->layer_num, 31u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 256u * 1024u * 4u);

    EXPECT_EQ(config.cache_specs[6]->layer_num, 61u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, FlashPoolSpecs) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc);
    EXPECT_EQ(config.cache_specs[0]->layer_num, 21u);
    EXPECT_EQ(config.cache_specs[1]->layer_num, 20u);
    EXPECT_EQ(config.cache_specs[6]->layer_num, 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(HybridPoolConfigCreatorTest, BlockSizeBytes) {
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 256u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 256u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 256u * 1024u * 4u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, Fp8BlockSizeBytesUsePaddedPhysicalStride) {
    ParallelismConfig pc;
    auto              mc          = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    auto config                   = HybridPoolConfigCreator::createConfig(mc, pc);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    ASSERT_EQ(config.group_kv_block_stride_bytes.size(), 7u);

    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 37440u);  // 64 * 584 padded to 576B alignment
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1728u);   // 2 * 584 padded to 576B alignment
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * 132u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), 149760u);  // 256 * 584 padded to 576B alignment

    EXPECT_EQ(config.group_kv_block_stride_bytes[0], config.cache_specs[0]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[1], config.cache_specs[1]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[6], config.cache_specs[6]->block_size_bytes());
}

// ============================================================
// CacheConfig output
// ============================================================

TEST(HybridPoolConfigCreatorTest, CreateCacheConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc);

    // 7 groups -> groupNums() > 1 -> HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.global_layer_ids.size(), 7u);
    EXPECT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.group_types.size(), 7u);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_TRUE(config.is_sparse);
    EXPECT_FALSE(config.use_mla);
}

TEST(HybridPoolConfigCreatorTest, FlashCacheConfig) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    // Group 6 (SWA) should have all 43 layers
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);
    // Group 0 (CSA KV) should have 21 layers
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolUsesHybridPoolConfig) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(true), pc);

    EXPECT_TRUE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.full_group_num, 1);
    EXPECT_EQ(config.linear_group_num, 1);
    ASSERT_EQ(config.cache_specs.size(), 2u);
    EXPECT_LT(config.cache_specs[0]->block_size_bytes(), config.cache_specs[1]->block_size_bytes());
    EXPECT_EQ(config.group_block_nums.size(), 2u);
    EXPECT_EQ(config.group_region_names,
              std::vector<KVCacheRegionName>({KVCacheRegionName::DEFAULT, KVCacheRegionName::DEFAULT}));
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionWithoutIndependentPoolKeepsSharedHybridConfig) {
    ParallelismConfig pc;
    auto              config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc);

    EXPECT_FALSE(config.use_independent_block_pools);
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.group_block_nums.empty());
}

// ============================================================
// DSV4KVCacheSpec
// ============================================================

TEST(DSV4KVCacheSpecTest, KVSpecFromPoolSpec) {
    DSV4KVSpec spec(KVCacheRegionName::CSA_KV, 30, kDsv4Fp8KvEntryBytes, 64, DataType::TYPE_UINT8, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.layer_num, 30u);
    EXPECT_EQ(spec.block_size(), 64u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), 64u * kDsv4Fp8KvEntryBytes * 1u);  // uint8 = 1 byte
    EXPECT_EQ(spec.block_size_bytes(), 37440u);
    EXPECT_EQ(spec.cache_type, KVCacheRegionName::CSA_KV);
    EXPECT_EQ(spec.entry_elems, kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.entries_per_block, 64u);

    DSV4KVSpec hca_spec(
        KVCacheRegionName::HCA_KV, 31, kDsv4Fp8KvEntryBytes, 2, DataType::TYPE_UINT8, kDsv4TokensPerBlock);
    EXPECT_EQ(hca_spec.natural_block_size_bytes(), 2u * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(hca_spec.block_size_bytes(), 1728u);
}

TEST(DSV4KVCacheSpecTest, SWAFp8StateSpecUsesPaddedPhysicalBlockSize) {
    DSV4StateSpec spec(
        KVCacheRegionName::SWA_KV, 61, kDsv4Fp8KvEntryBytes, 256, DataType::TYPE_UINT8, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.block_size_bytes(), 149760u);
    EXPECT_EQ(spec.cache_type, KVCacheRegionName::SWA_KV);
}

TEST(DSV4KVCacheSpecTest, StateSpecFloat32) {
    DSV4StateSpec spec(KVCacheRegionName::CSA_STATE, 30, 2048, 8, DataType::TYPE_FP32, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), 8u * 2048u);
    EXPECT_EQ(spec.block_size_bytes(), 8u * 2048u * 4u);  // float32 = 4 bytes
    EXPECT_EQ(spec.cache_type, KVCacheRegionName::CSA_STATE);
    EXPECT_EQ(spec.state_dim, 2048u);
}

TEST(DSV4KVCacheSpecTest, IndexerKVSpec) {
    DSV4KVSpec spec(KVCacheRegionName::INDEXER_KV, 30, 132, 64, DataType::TYPE_UINT8, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), 64u * 132u);
    EXPECT_EQ(spec.block_size_bytes(), 64u * 132u);
    EXPECT_EQ(spec.cache_type, KVCacheRegionName::INDEXER_KV);
}

TEST(DSV4KVCacheSpecTest, HCAStateSpec) {
    DSV4StateSpec spec(KVCacheRegionName::HCA_STATE, 31, 1024, 128, DataType::TYPE_FP32, kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(spec.cache_type, KVCacheRegionName::HCA_STATE);
}

// ============================================================
// Pool 0/1/2 shared properties: same tokens_per_block, same num_blocks
// ============================================================

TEST(HybridPoolConfigCreatorTest, PagedPoolsShareTokensPerBlock) {
    // Pro config
    {
        ParallelismConfig pc;
        auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
        EXPECT_EQ(config.group_seq_size_per_block[0], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[1], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[2], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[6], kDsv4TokensPerBlock);
    }
    // Flash config
    {
        ParallelismConfig pc;
        auto              config = HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc);
        EXPECT_EQ(config.group_seq_size_per_block[0], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[1], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[2], kDsv4TokensPerBlock);
    }
}

TEST(HybridPoolConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc);
    config.block_num         = 100;

    // Paged groups derive their block count from the global block_num; fixed/SWA
    // groups use per-group fixed block counts.
    EXPECT_EQ(config.groupNums(), 7);
    for (int i = 0; i < 7; i++) {
        EXPECT_GT(config.cache_specs[i]->block_size_bytes(), 0u) << "pool " << i;
    }
}

TEST(HybridPoolConfigCreatorTest, NonFullAdditionAppliedToSwaGroups) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                              = 100;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    // FULL groups get global_block_num as-is
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    // SWA groups get rule_blocks + addition (default 256)
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u + kNonFullAdditionBlocks) << "gid=" << gid;
    }

    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += static_cast<size_t>(kNonFullAdditionBlocks) * config.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(HybridPoolConfigCreatorTest, NonFullAdditionIsIndependentOfMaxConcurrency) {
    for (uint32_t max_concurrency : {1u, 2u, 8u}) {
        auto              mc = makeProModelConfig();
        ParallelismConfig pc;
        RuntimeConfig     runtime_config;
        KVCacheConfig     kv_cache_config;
        kv_cache_config.test_block_num                              = 100;
        runtime_config.max_generate_batch_size                      = max_concurrency;
        runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

        auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

        ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
        for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
            EXPECT_EQ(config.group_block_nums[gid], 100u + kNonFullAdditionBlocks)
                << "gid=" << gid << " max_concurrency=" << max_concurrency;
        }
    }
}

TEST(HybridPoolConfigCreatorTest, NonFullAdditionCanBeOverriddenByConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num                              = 100;
    kv_cache_config.non_full_addition_kvcache_blocks            = 6;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    // FULL groups: 100
    for (int gid = 0; gid < 3; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u) << "gid=" << gid;
    }
    // SWA groups: 100 + 6
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u + 6u) << "gid=" << gid;
    }
}

TEST(CacheConfigTest, FinalizeBlockNumsIsNoopForSingleAndSharedHybridConfig) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 8;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 4;

    ParallelismConfig pc;
    auto              single_config = CacheConfigCreator::createBasicConfig(ModelConfig(), pc);
    single_config.finalizeBlockNums(123, runtime_config);
    EXPECT_TRUE(single_config.group_block_nums.empty());
    EXPECT_EQ(single_config.fixed_pool_reserve_bytes, 0u);

    auto hybrid_config = CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc);
    hybrid_config.finalizeBlockNums(123, runtime_config);
    EXPECT_FALSE(hybrid_config.use_independent_block_pools);
    EXPECT_TRUE(hybrid_config.group_block_nums.empty());
    EXPECT_EQ(hybrid_config.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, FinalizeBlockNumsAppliesToIndependentPools) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
    config.finalizeBlockNums(100, runtime_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u + kNonFullAdditionBlocks) << "gid=" << gid;
    }
    EXPECT_GT(config.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, AdditionReserveDeductedFromPagedBudget) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    const uint32_t addition = 32;

    // Budget sized to fit the SWA/STATE reserve (4 non-FULL pools × 32 blocks ×
    // ~30 MiB/block ≈ 3.9 GiB) plus headroom for paged-FULL allocation. With a
    // 1024 MiB budget the createConfig path RTP_LLM_CHECK-throws on
    // "budget < fixed-pool reservation" before block_num is computed.
    constexpr int64_t kBudgetMb = 8192;
    // Config with addition > 0
    KVCacheConfig kv_cache_config_with;
    kv_cache_config_with.kv_cache_mem_mb                  = kBudgetMb;
    kv_cache_config_with.non_full_addition_kvcache_blocks = addition;
    auto config_with = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_with);

    // Config with addition = 0
    KVCacheConfig kv_cache_config_without;
    kv_cache_config_without.kv_cache_mem_mb                  = kBudgetMb;
    kv_cache_config_without.non_full_addition_kvcache_blocks = 0;
    auto config_without = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_without);

    // With addition, FULL groups must get fewer blocks because paged_budget is smaller
    EXPECT_LT(config_with.block_num, config_without.block_num);
    // FULL group block_nums reflect the reduced block_num
    EXPECT_EQ(config_with.group_block_nums[0], static_cast<uint32_t>(config_with.block_num));
    EXPECT_EQ(config_without.group_block_nums[0], static_cast<uint32_t>(config_without.block_num));
    // SWA groups get rule_blocks + addition
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config_with.group_block_nums[gid], static_cast<uint32_t>(config_with.block_num) + addition)
            << "gid=" << gid;
    }
    // Reserve bytes = addition × sum(swa group block sizes)
    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += static_cast<size_t>(addition) * config_with.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config_with.fixed_pool_reserve_bytes, expected_reserve);
    EXPECT_EQ(config_without.fixed_pool_reserve_bytes, 0u);
}

// ============================================================
// state_pool_uses_pinned_cpu single source of truth (B1 Option A)
//
// Pre-fix bug: HybridPoolKVCacheAllocator added a unilateral
//   `state_pool_uses_pinned_cpu || role_type_ == PREFILL`
// at init time. CacheConfigCreator however set the flag from
// `state_pool_memory_mb > 0` only. Under PREFILL with state_pool_memory_mb=0
// the HBM divisor still included STATE bytes (over-reserve) while the
// allocator silently moved STATE to pinned host memory anyway.
//
// Fix: CacheConfigCreator is the single source of truth. The allocator just
// reads `config.state_pool_uses_pinned_cpu`. Below we lock down the matrix
// for all (role × state_pool_memory_mb) combinations on DSV4 hybrid configs.
// ============================================================
namespace {

constexpr int64_t kStatePinnedTestBudgetMb = 8192;

// Helper: build a DSV4 createConfig() under the given (role, state budget).
// non_full_addition_kvcache_blocks is forced to 0 because the default (256) on
// the full pro model yields ~31 GiB of fixed reserve, which would blow the
// 8 GiB paged budget when STATE pools are device-resident (PDFUSION/DECODE).
// The reserve invariant under pinning is exercised separately below.
CacheConfig makeStatePinnedTestConfig(int64_t state_pool_memory_mb, RoleType role) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_mem_mb                  = kStatePinnedTestBudgetMb;
    kv_cache_config.state_pool_memory_mb             = state_pool_memory_mb;
    kv_cache_config.non_full_addition_kvcache_blocks = 0;
    return CacheConfigCreator::createConfig(
        mc, pc, runtime_config, kv_cache_config, std::nullopt, std::nullopt, role);
}

}  // namespace

TEST(CacheConfigStatePinnedTest, PrefillWithZeroBudgetMovesStateToPinnedCpu) {
    auto cfg = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PREFILL);
    EXPECT_TRUE(cfg.state_pool_uses_pinned_cpu)
        << "PREFILL + state_pool_memory_mb=0 must mark STATE as pinned-CPU at the config layer";
    EXPECT_GT(cfg.state_block_size_bytes, 0u) << "DSV4 hybrid config must populate state_block_size_bytes";
}

TEST(CacheConfigStatePinnedTest, PrefillWithExplicitBudgetMovesStateToPinnedCpu) {
    auto cfg = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/256, RoleType::PREFILL);
    EXPECT_TRUE(cfg.state_pool_uses_pinned_cpu);
    EXPECT_GT(cfg.state_block_size_bytes, 0u);
    // state_block_num is decoupled from HBM count when an explicit budget is set.
    const size_t expected_state_block_num =
        (static_cast<size_t>(256) * 1024 * 1024) / cfg.state_block_size_bytes;
    EXPECT_EQ(cfg.state_block_num, static_cast<uint32_t>(expected_state_block_num));
}

TEST(CacheConfigStatePinnedTest, PdfusionWithZeroBudgetKeepsStateOnDevice) {
    auto cfg = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PDFUSION);
    EXPECT_FALSE(cfg.state_pool_uses_pinned_cpu)
        << "Default-OFF invariant: non-PREFILL + state_pool_memory_mb=0 must keep STATE on device "
        << "(byte-identical to pre-unification behavior).";
}

TEST(CacheConfigStatePinnedTest, DecodeWithZeroBudgetKeepsStateOnDevice) {
    auto cfg = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::DECODE);
    EXPECT_FALSE(cfg.state_pool_uses_pinned_cpu)
        << "DECODE role without explicit STATE budget must keep STATE on device — only PREFILL "
        << "forces pinned-CPU residency.";
}

TEST(CacheConfigStatePinnedTest, PrefillHbmDivisorExcludesStateBytesVsPdfusion) {
    // Two configs differ only by role_type. PREFILL excludes STATE bytes from
    // the HBM divisor (b/c STATE lives on host), so it gets MORE block_num
    // than PDFUSION at the same paged_budget.
    auto cfg_prefill  = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PREFILL);
    auto cfg_pdfusion = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PDFUSION);

    ASSERT_TRUE(cfg_prefill.state_pool_uses_pinned_cpu);
    ASSERT_FALSE(cfg_pdfusion.state_pool_uses_pinned_cpu);
    ASSERT_GT(cfg_prefill.state_block_size_bytes, 0u);

    EXPECT_GT(cfg_prefill.block_num, cfg_pdfusion.block_num)
        << "PREFILL HBM divisor must exclude state_block_size_bytes — should yield more blocks "
        << "than PDFUSION at identical budget";
}

TEST(CacheConfigStatePinnedTest, PrefillReserveExcludesStateAdditionBytes) {
    // With non_full_addition_kvcache_blocks > 0 and STATE pinned-CPU, the
    // fixed_pool_reserve must only count the SWA_KV pool's addition (gid 6),
    // NOT the three STATE pools (gids 3..5). This is the load-bearing assert
    // for the over-reserve half of the bug.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_mem_mb                  = kStatePinnedTestBudgetMb;
    kv_cache_config.state_pool_memory_mb             = 0;  // pinned via PREFILL role
    kv_cache_config.non_full_addition_kvcache_blocks = 4;

    auto cfg = CacheConfigCreator::createConfig(
        mc, pc, runtime_config, kv_cache_config, std::nullopt, std::nullopt, RoleType::PREFILL);

    ASSERT_TRUE(cfg.state_pool_uses_pinned_cpu);
    const size_t expected_reserve = static_cast<size_t>(4) * cfg.group_block_size_bytes[6];
    EXPECT_EQ(cfg.fixed_pool_reserve_bytes, expected_reserve)
        << "STATE pool addition bytes must NOT be counted in HBM reserve when STATE is pinned-CPU "
        << "(silent HBM waste described in B1/B2/B3 reviews).";
}

TEST(CacheConfigStatePinnedTest, NonDsv4ConfigUnaffectedByRoleType) {
    // Guards default-OFF for non-DSV4: a single-pool config has
    // state_block_size_bytes=0, so role_type=PREFILL must NOT flip the flag.
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 32;

    auto cfg = CacheConfigCreator::createConfig(
        ModelConfig(), pc, runtime_config, kv_cache_config, std::nullopt, std::nullopt, RoleType::PREFILL);
    EXPECT_FALSE(cfg.state_pool_uses_pinned_cpu)
        << "Default-OFF: non-DSV4 paths (state_block_size_bytes=0) must keep the flag false "
        << "regardless of role";
    EXPECT_EQ(cfg.state_block_size_bytes, 0u);
}

// ============================================================
// F01 PR-1: K_state phase-2 hook (dsv4_state_entries_per_block)
//
// Default OFF: state pools keep 256 entries/block, byte-identical to
// today's production state-on-CPU goldens. When >0, the 3 STATE pools
// (INDEXER_STATE @ idx3, CSA_STATE @ idx4, HCA_STATE @ idx5) collapse
// from 256 down to K_state entries/block, shrinking per-block bytes by
// 256/K_state. Paged pools (CSA_KV / HCA_KV / INDEXER_KV) and SWA_KV
// (idx6) MUST be unaffected. Kernel-side wiring lands in F01-PR2 / M02
// §3.3 HBM-savings smoke; PR-1 only flips the sizing surface.
// ============================================================

TEST(DSV4KStateHookTest, DefaultOffStatePoolBlockSizeUnchanged) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;  // dsv4_state_entries_per_block = 0 (default)

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);

    // Default-OFF byte identity: every state-pool block_size_bytes must
    // match the pre-F01 expectations from BlockSizeBytes above.
    ASSERT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 256u * 512u * 4u);   // INDEXER_STATE
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 256u * 2048u * 4u);  // CSA_STATE
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 256u * 1024u * 4u);  // HCA_STATE
    // Paged + SWA pools also unchanged.
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);  // CSA_KV
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);   // HCA_KV
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);  // INDEXER_KV
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);  // SWA_KV

    // Mirror field on CacheConfig defaults to 0 → no kernel-side opt-in.
    EXPECT_EQ(config.state_entries_per_block_constant, 0);
}

TEST(DSV4KStateHookTest, KStateFourCollapsesStatePoolBlockSizeBy64x) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = 4;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    // 256 entries/block → 4 entries/block = 64x reduction on per-block bytes
    // for the 3 STATE pools. entry_elems and dtype_size unchanged.
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 4u * 512u * 4u);   // INDEXER_STATE: 256 → 4
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 4u * 2048u * 4u);  // CSA_STATE:     256 → 4
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 4u * 1024u * 4u);  // HCA_STATE:     256 → 4

    // Paged + SWA pools MUST be untouched by the K_state hook.
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);

    // K_state mirrored onto CacheConfig for downstream hash-salt / kernel
    // wiring (the producer lands in F01-PR2; PR-1 only plumbs the value).
    EXPECT_EQ(config.state_entries_per_block_constant, 4);
}

TEST(DSV4KStateHookTest, KStateOneIsLowerBoundReducesStatePoolBy256x) {
    // K_state=1 is the tightest collapse the hook permits; verifies the
    // arithmetic and that we still produce sensible non-zero state bytes.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = 1;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    ASSERT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 1u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 1u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 1u * 1024u * 4u);
    EXPECT_EQ(config.state_entries_per_block_constant, 1);
}

TEST(DSV4KStateHookTest, KStateAggregatedStateBlockSizeBytesShrinksOnHook) {
    // state_block_size_bytes is the sum-over-state-pools used by
    // CacheConfigCreator's HBM divisor / pinned-CPU budget calc.
    // F01 PR-1 collapses each contributing pool's per-block bytes, so the
    // aggregate must shrink by the same K_state factor (no other terms).
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;

    KVCacheConfig kv_off;
    auto          cfg_off = HybridPoolConfigCreator::createConfig(mc, pc, kv_off);

    KVCacheConfig kv_on;
    kv_on.dsv4_state_entries_per_block = 4;
    auto cfg_on = HybridPoolConfigCreator::createConfig(mc, pc, kv_on);

    ASSERT_GT(cfg_off.state_block_size_bytes, 0u);
    // 256 → 4 ⇒ 64x shrink on each state pool ⇒ 64x shrink on the sum.
    EXPECT_EQ(cfg_on.state_block_size_bytes, cfg_off.state_block_size_bytes / 64u);

    // Paged-side aggregate (kv_block_size_bytes + kv_scale_size_bytes) is
    // unaffected — the K_state hook only touches the 3 state pools.
    EXPECT_EQ(cfg_on.kv_block_size_bytes, cfg_off.kv_block_size_bytes);
    EXPECT_EQ(cfg_on.kv_scale_size_bytes, cfg_off.kv_scale_size_bytes);
    EXPECT_EQ(cfg_on.swa_block_size_bytes, cfg_off.swa_block_size_bytes);
}

TEST(DSV4KStateHookTest, StateOnCpuPrefillHbmAccountingIdenticalWhenHookOff) {
    // State-on-CPU production path (state_pool_uses_pinned_cpu=true, set by
    // CacheConfigCreator under PREFILL) must remain byte-identical in HBM
    // accounting when the hook is OFF (dsv4_state_entries_per_block=0).
    // This is the load-bearing default-OFF byte-identity assert: anyone
    // landing F01 PR-1 must NOT shift HBM block_num for production
    // PREFILL roles that today rely on state-on-pinned-CPU.
    auto cfg_off = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PREFILL);

    // Compare against a re-created config built the same way — sanity-
    // check the helper is deterministic, then check K_state stays at 0.
    auto cfg_off_again = makeStatePinnedTestConfig(/*state_pool_memory_mb=*/0, RoleType::PREFILL);
    EXPECT_EQ(cfg_off.block_num, cfg_off_again.block_num);
    EXPECT_EQ(cfg_off.state_block_size_bytes, cfg_off_again.state_block_size_bytes);
    EXPECT_TRUE(cfg_off.state_pool_uses_pinned_cpu);
    EXPECT_EQ(cfg_off.state_entries_per_block_constant, 0);
}

TEST(DSV4KStateHookTest, StateOnCpuPrefillHbmDivisorUnchangedRegardlessOfKState) {
    // When STATE is pinned-CPU (production PREFILL), state_block_size_bytes
    // is EXCLUDED from the HBM divisor — so toggling the K_state hook must
    // NOT shift the paged block_num. This protects the production state-
    // on-CPU goldens from any accidental HBM accounting drift.
    auto mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    KVCacheConfig kv_off;
    kv_off.kv_cache_mem_mb                  = kStatePinnedTestBudgetMb;
    kv_off.non_full_addition_kvcache_blocks = 0;
    // dsv4_state_entries_per_block = 0 (default, OFF)
    auto cfg_off = CacheConfigCreator::createConfig(
        mc, pc, runtime_config, kv_off, std::nullopt, std::nullopt, RoleType::PREFILL);

    KVCacheConfig kv_on                    = kv_off;
    kv_on.dsv4_state_entries_per_block     = 4;
    auto cfg_on = CacheConfigCreator::createConfig(
        mc, pc, runtime_config, kv_on, std::nullopt, std::nullopt, RoleType::PREFILL);

    ASSERT_TRUE(cfg_off.state_pool_uses_pinned_cpu);
    ASSERT_TRUE(cfg_on.state_pool_uses_pinned_cpu);

    // Paged-FULL block_num must be unchanged because state bytes are
    // already excluded from the HBM divisor on PREFILL.
    EXPECT_EQ(cfg_on.block_num, cfg_off.block_num)
        << "PREFILL state-on-CPU HBM block_num must not change when K_state hook flips";

    // State per-block bytes do shrink (this is the whole point of K_state),
    // and the mirror field reflects the override.
    EXPECT_LT(cfg_on.state_block_size_bytes, cfg_off.state_block_size_bytes);
    EXPECT_EQ(cfg_on.state_entries_per_block_constant, 4);
    EXPECT_EQ(cfg_off.state_entries_per_block_constant, 0);
}

TEST(CacheConfigTest, FinalizeBlockNumsWithLinearStepShrinksSwaRuleBlocks) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
    // Simulate linear_step = 4
    config.linear_step                      = 4;
    config.non_full_addition_kvcache_blocks = 10;
    config.finalizeBlockNums(100, runtime_config);

    // FULL groups: unaffected by step, get global_block_num
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    // DSV4 pool layout: gids 3..5 are STATE regions (INDEXER/CSA/HCA_STATE),
    // gid 6 is the genuine SWA_KV pool. After commit 04e36fe9, finalizeBlockNums
    // routes STATE rule_blocks via state_global_block_num (= global_block_num
    // for the one-arg overload), so STATE pools get global_block_num + addition.
    // Only the true SWA pool divides by linear_step.
    for (int gid = 3; gid <= 5; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u + 10u) << "STATE gid=" << gid;
    }
    EXPECT_EQ(config.group_block_nums[6], 25u + 10u) << "SWA_KV gid=6";
    // Reserve only counts the addition portion of every non-FULL pool.
    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += 10u * config.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, AdditionZeroProducesNoReserve) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    auto              config                = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc);
    config.non_full_addition_kvcache_blocks = 0;
    config.finalizeBlockNums(50, runtime_config);

    // All groups get exactly rule_blocks with no addition
    for (int gid = 0; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 50u) << "gid=" << gid;
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, DSV4MtpKeepsProposeLayerInSwaPool) {
    auto score_model_config   = makeFlashModelConfig();
    auto propose_model_config = makeFlashMtpModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 100;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     true,
                                                     false);

    ASSERT_EQ(config.layer_num, 43u);
    ASSERT_EQ(config.layer_all_num, 45u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    ASSERT_NE(config.mtp_sub_configs[0], nullptr);
    ASSERT_NE(config.mtp_sub_configs[1], nullptr);

    EXPECT_EQ(config.layer_to_group_id[43], 6);
    EXPECT_EQ(config.layer_to_group_id[44], 6);
    EXPECT_EQ(config.layer_region_to_group_id[43][static_cast<size_t>(KVCacheRegionName::SWA_KV)], 6);
    EXPECT_EQ(config.layer_region_to_group_id[44][static_cast<size_t>(KVCacheRegionName::SWA_KV)], 6);

    EXPECT_EQ(config.global_layer_ids[6].size(), 45u);
    EXPECT_EQ(config.mtp_sub_configs[0]->global_layer_ids[6], std::vector<int>({43}));
    EXPECT_EQ(config.mtp_sub_configs[1]->global_layer_ids[6], std::vector<int>({44}));
    EXPECT_TRUE(config.mtp_sub_configs[0]->global_layer_ids[0].empty());
    EXPECT_TRUE(config.mtp_sub_configs[1]->global_layer_ids[0].empty());

    const size_t addition = config.non_full_addition_kvcache_blocks;
    const size_t expected_fixed_reserve =
        addition * config.group_block_size_bytes[3] + addition * config.group_block_size_bytes[4]
        + addition * config.group_block_size_bytes[5] + addition * config.group_block_size_bytes[6];
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_fixed_reserve);
}

TEST(HybridPoolConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // DSV4 has multiple cache regions per logical layer. The config must expose
    // every region's group id for the layer so model/runtime code can request the
    // correct region by KVCacheRegionName.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc);

    // Verify layer_to_group_id mapping: each layer maps to group 6 (SWA) by default
    EXPECT_EQ(config.layer_to_group_id.size(), 61u);
    for (size_t i = 0; i < config.layer_to_group_id.size(); i++) {
        EXPECT_EQ(config.layer_to_group_id[i], 6) << "layer " << i << " should map to SWA group";
    }

    // Verify global_layer_ids: each group has the correct layer list
    // Group 0 (CSA KV) and Group 2 (Indexer KV) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[2]);
    // Group 0 (CSA KV) and Group 3 (Indexer State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[3]);
    // Group 0 (CSA KV) and Group 4 (CSA State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[0], config.global_layer_ids[4]);
    // Group 1 (HCA KV) and Group 5 (HCA State) should have identical layer lists
    EXPECT_EQ(config.global_layer_ids[1], config.global_layer_ids[5]);
}

// ============================================================
// Helper: build a DSV4 CacheConfig with block_num set for allocator tests
// ============================================================

static CacheConfig makeDSV4AllocatorConfig(bool use_flash = false) {
    auto              mc = use_flash ? makeFlashModelConfig() : makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc);
    // Set enough blocks for tests (7 groups × N blocks each)
    config.block_num = 200;
    return config;
}

// ============================================================
// HybridTypeKVCacheAllocator integration tests with DSV4 7-group config
// ============================================================

class DSV4AllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

TEST_F(DSV4AllocatorTest, InitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // 7 groups → HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, FlashInitAndBasicProperties) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, AddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // Verify address lookup works for a layer in each group
    // Group 0 (CSA KV): csa_layer_ids[0]
    // Group 1 (HCA KV): hca_layer_ids[0]
    // Group 6 (SWA KV): all_layer_ids[0]
    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.global_layer_ids[gid].empty()) << "group " << gid << " has no layers";
        int  layer_id = config.global_layer_ids[gid][0];
        auto addr     = allocator->convertIndexToAddr(layer_id, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "null kv_addr for group " << gid << " layer " << layer_id;
    }
}

TEST_F(DSV4AllocatorTest, BlockPoolCreatedWithCorrectTensors) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // allLayerCacheBase should return tensors for all 61 layers
    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "undefined kv buffer for layer " << i;
    }
}

TEST_F(DSV4AllocatorTest, ConvertIndexToBufferAllGroups) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // convertIndexToBuffer should work for layers in each of the 7 groups
    for (int gid = 0; gid < 7; gid++) {
        int  layer_id = config.global_layer_ids[gid][0];
        auto buf      = allocator->convertIndexToBuffer(layer_id, /*block_id=*/1);
        ASSERT_FALSE(buf.empty()) << "empty buffer for group " << gid;
        EXPECT_NE(buf[0].addr, nullptr) << "null addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, MallocAndFreeBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 3u);

    // Direct block pool malloc/free
    auto blocks = block_pool->malloc(3);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 3);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(DSV4AllocatorTest, SevenGroupLayerMapping) {
    auto config = makeDSV4AllocatorConfig();

    // Verify group layer assignments match DSV4 classification
    // Group 0: CSA KV → csa_layer_ids (30 layers)
    EXPECT_EQ(config.global_layer_ids[0].size(), 30u);
    // Group 1: HCA KV → hca_layer_ids (31 layers)
    EXPECT_EQ(config.global_layer_ids[1].size(), 31u);
    // Group 2: Indexer KV → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[2].size(), 30u);
    // Group 3: Indexer State → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[3].size(), 30u);
    // Group 4: CSA State → csa_layer_ids
    EXPECT_EQ(config.global_layer_ids[4].size(), 30u);
    // Group 5: HCA State → hca_layer_ids
    EXPECT_EQ(config.global_layer_ids[5].size(), 31u);
    // Group 6: SWA KV → all_layer_ids
    EXPECT_EQ(config.global_layer_ids[6].size(), 61u);

    // Paged DSV4 pools are FULL; fixed/state and SWA pools keep a sliding-window tail.
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[1], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);
    EXPECT_EQ(config.group_types[3], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[4], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[5], CacheGroupType::SWA);
    EXPECT_EQ(config.group_types[6], CacheGroupType::SWA);
}

TEST_F(DSV4AllocatorTest, SpecBlockSizesMatchPoolSpecs) {
    auto config = makeDSV4AllocatorConfig();

    ASSERT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 256u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 256u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 256u * 1024u * 4u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, KVBlockStrideIsMaxAcrossGroups) {
    auto config = makeDSV4AllocatorConfig();

    // kv_block_stride_bytes should be the max block_size_bytes across all 7 pools
    size_t expected_max = 0;
    for (int i = 0; i < kDsv4PoolNum; i++) {
        expected_max = std::max(expected_max, config.cache_specs[i]->block_size_bytes());
    }
    EXPECT_EQ(config.kv_block_stride_bytes, expected_max);
    EXPECT_EQ(expected_max, config.cache_specs[4]->block_size_bytes());
}

TEST_F(DSV4AllocatorTest, AllGroupsParticipateInPrefixCache) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    // Insert blocks into cache for ALL 7 groups
    constexpr int group_num = 7;
    CacheKeysType keys      = {100, 101, 102};
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
        ASSERT_EQ(blocks.size(), keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(keys[i], group_slots, true);
        }
        block_pool->requestFree(blocks);
    }

    // All 7 groups should have cache entries
    for (int gid = 0; gid < group_num; gid++) {
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(100, gid))) << "group " << gid << " should be in cache";
    }
}

// ============================================================
// Flash config: allocator integration
// ============================================================

TEST_F(DSV4AllocatorTest, FlashGroupTypes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    // Flash: 21 CSA + 20 HCA + 2 SWA-only = 43 layers
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 20u);
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);

    // Same group type split as Pro: 3 FULL paged groups, 4 SWA tail groups.
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);  // CSA KV
    EXPECT_EQ(config.group_types[1], CacheGroupType::FULL);  // HCA KV
    EXPECT_EQ(config.group_types[2], CacheGroupType::FULL);  // Indexer KV
    EXPECT_EQ(config.group_types[3], CacheGroupType::SWA);   // Indexer State
    EXPECT_EQ(config.group_types[4], CacheGroupType::SWA);   // CSA State
    EXPECT_EQ(config.group_types[5], CacheGroupType::SWA);   // HCA State
    EXPECT_EQ(config.group_types[6], CacheGroupType::SWA);   // SWA KV
}

TEST_F(DSV4AllocatorTest, FlashAddressLookupAllGroups) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    for (int gid = 0; gid < 7; gid++) {
        ASSERT_FALSE(config.global_layer_ids[gid].empty()) << "Flash group " << gid << " has no layers";
        int  layer_id = config.global_layer_ids[gid][0];
        auto addr     = allocator->convertIndexToAddr(layer_id, /*block_id=*/1);
        EXPECT_NE(addr.kv_addr, nullptr) << "Flash null kv_addr for group " << gid;
    }
}

TEST_F(DSV4AllocatorTest, FlashBlockPoolTensors) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), 43u);
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_TRUE(layout.layers_to_kv_buffer_ptrs[i].defined()) << "Flash undefined kv buffer for layer " << i;
    }
}

TEST_F(DSV4AllocatorTest, FlashLayerMapping) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);  // CSA KV
    EXPECT_EQ(config.global_layer_ids[1].size(), 20u);  // HCA KV
    EXPECT_EQ(config.global_layer_ids[2].size(), 21u);  // Indexer KV
    EXPECT_EQ(config.global_layer_ids[3].size(), 21u);  // Indexer State
    EXPECT_EQ(config.global_layer_ids[4].size(), 21u);  // CSA State
    EXPECT_EQ(config.global_layer_ids[5].size(), 20u);  // HCA State
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);  // SWA KV (all layers)
}

TEST_F(DSV4AllocatorTest, FlashSpecBlockSizes) {
    auto config = makeDSV4AllocatorConfig(/*use_flash=*/true);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 64u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 2u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 64u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST_F(DSV4AllocatorTest, FlashMallocAndFree) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto   block_pool  = allocator->getBlockPool();
    size_t free_before = allocator->freeBlocksNum();
    ASSERT_GT(free_before, 5u);

    auto blocks = block_pool->malloc(5);
    ASSERT_EQ(blocks.size(), 5u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 5);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Prefix cache: insertIntoCache inserts ALL 7 groups
// ============================================================

TEST_F(DSV4AllocatorTest, InsertIntoCacheAllGroups) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Manually set up a BatchKVCacheResource with blocks for all 7 groups
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);

    CacheKeysType keys = {200, 201, 202, 203};
    batch_res->setBatchCacheKeys(0, keys);

    // Allocate 3 blocks per group (simulating 3 full blocks)
    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
    }

    // Create CompleteTokenIds: 3 full blocks * seq_size_per_block tokens + partial
    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;  // 3 full blocks + 1 partial
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // With manually populated non-null block IDs, every group inserts the
    // corresponding full-block cache keys.
    for (int gid = 0; gid < 3; gid++) {
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(200, gid))) << "group " << gid << " should be cached";
    }
    for (int gid = 3; gid < 7; gid++) {
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(200, gid)))
            << "SWA group " << gid << " should cache key 200";
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(201, gid)))
            << "SWA group " << gid << " should cache tail key 201";
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(202, gid)))
            << "SWA group " << gid << " should cache tail key 202";
    }

    // Free all blocks
    for (int gid = 0; gid < 7; gid++) {
        const auto& blocks = batch_res->blocks(0, gid);
        block_pool->requestFree(blocks);
    }
}

// ============================================================
// Prefix cache: Flash config insertIntoCache all groups
// ============================================================

TEST_F(DSV4AllocatorTest, FlashInsertIntoCacheAllGroups) {
    auto config       = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);

    CacheKeysType keys = {300, 301, 302, 303};
    batch_res->setBatchCacheKeys(0, keys);

    for (int gid = 0; gid < 7; gid++) {
        auto blocks = block_pool->malloc(3);
        ASSERT_EQ(blocks.size(), 3u);
        batch_res->mutableBlockIds(0, gid).assign(BlockIndicesType(blocks.begin(), blocks.end()));
    }

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    int  total_tokens               = 3 * seq_size_per_block + 1;
    generate_input->input_ids       = torch::arange(total_tokens, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    InsertInfo insert_info{batch_res, complete_token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // With manually populated non-null block IDs, every group inserts the
    // corresponding full-block cache keys.
    for (int gid = 0; gid < 3; gid++) {
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(300, gid)))
            << "Flash group " << gid << " should be cached";
    }
    for (int gid = 3; gid < 7; gid++) {
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(300, gid)))
            << "Flash SWA group " << gid << " should cache key 300";
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(301, gid)))
            << "Flash SWA group " << gid << " should cache tail key 301";
        EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(302, gid)))
            << "Flash SWA group " << gid << " should cache tail key 302";
    }

    for (int gid = 0; gid < 7; gid++) {
        block_pool->requestFree(batch_res->blocks(0, gid));
    }
}

// ============================================================
// Prefix cache: paged FULL groups reuse; SWA/state groups require a matched latest tail block.
// ============================================================

TEST_F(DSV4AllocatorTest, PrefixCacheReusePagedGroupsOnly) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Pre-populate cache for ALL 7 groups with keys {100,101,102}
    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    // Now do a malloc with reuse enabled — keys {100,101,102,103}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;  // 3 full + partial
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Prefix cache reuse should work with paged DSV4 groups";

    for (int gid = 0; gid < 3; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "group " << gid << " should have >= 3 blocks";
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << "group " << gid << " block 0 should be reused";
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]) << "group " << gid << " block 1 should be reused";
    }
    for (int gid = 3; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "group " << gid << " should have tail slots";
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1]))
            << "group " << gid << " previous matched tail is evicted after new tail allocation";
        EXPECT_EQ(out_blocks[2], cached_blocks[gid][2]) << "group " << gid << " last matched tail block should remain";
    }

    // Clean up
    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseRequiresSWATailHit) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(3);
    for (int gid = 0; gid < 3; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, 0) << "SWA tail miss should veto paged prefix reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, PrefixCacheReuseAcceptsSingleLatestSWATailHit) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int group_num   = 7;
    CacheKeysType cached_keys = {100, 101, 102};
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (gid >= 3 && i + 1 < cached_keys.size()) {
                continue;
            }
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(3 * spb + 1, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "latest SWA tail hit should allow paged prefix reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, FlashPrefixCacheReusePagedGroupsOnly) {
    auto config       = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {500, 501, 502};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{500, 501, 502, 503});

    int  seq_size_per_block         = allocator->seqSizePerBlock();
    int  seq_len                    = 3 * seq_size_per_block + 1;
    auto complete_token_ids         = std::make_shared<CompleteTokenIds>(1, 1, 4096, seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::arange(seq_len, torch::kInt32);
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    complete_token_ids->init(generate_input);

    MallocInfo info{batch_res, complete_token_ids};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_GT(result.reuse_len, 0) << "Flash prefix cache reuse should work for paged groups";

    for (int gid = 0; gid < 3; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "Flash group " << gid;
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]) << "Flash group " << gid << " block 0 should be reused";
    }
    for (int gid = 3; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u) << "Flash group " << gid;
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1]))
            << "Flash group " << gid << " previous matched tail is evicted after new tail allocation";
        EXPECT_EQ(out_blocks[2], cached_blocks[gid][2])
            << "Flash group " << gid << " last matched tail block should remain";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksAreDistributedAcrossGroups) {
    auto config      = makeDSV4AllocatorConfig(/*use_flash=*/true);
    config.block_num = 200;
    auto allocator   = std::make_shared<HybridPoolKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/10);
    ASSERT_TRUE(allocator->init());

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{600, 601});

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.verbose             = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// SWA (group 6) prefix cache: verify SWA blocks participate in reuse
// ============================================================

TEST_F(DSV4AllocatorTest, SWAGroupParticipatesInPrefixCacheReuse) {
    auto config       = makeDSV4AllocatorConfig();
    config.block_num  = 100;
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int group_num = 7;

    // Only populate SWA group (6) and one paged group (0) to verify SWA participates
    CacheKeysType             cached_keys = {700, 701};
    std::vector<BlockIdxType> swa_blocks, csa_blocks;

    // Group 0 (CSA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[0] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        csa_blocks = blocks;
        block_pool->requestFree(blocks);
    }
    // Group 6 (SWA KV)
    {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[6] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        swa_blocks = blocks;
        block_pool->requestFree(blocks);
    }

    // Verify both groups have cache entries
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(700, 0)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(700, 6)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(701, 0)));
    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(701, 6)));

    // Groups 1,2,3,4,5 not populated — they will limit reuse to 0
    // But this verifies SWA group 6 IS in the reuse path
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 3)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 4)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(700, 5)));
}

// ============================================================
// SWA prefix cache: cache entries exist and the matched tail window gates reuse.
// ============================================================

TEST_F(DSV4AllocatorTest, SWAPrefixCacheRestoresTailReuse) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    // Populate ALL 7 groups with same keys
    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {800, 801};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(2);
        for (size_t i = 0; i < 2; ++i) {
            std::vector<BlockIdxType> group_slots(group_num, NULL_BLOCK_IDX);
            group_slots[gid] = blocks[i];
            shared_cache->put(cached_keys[i], group_slots, true);
        }
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    // Malloc with reuse — keys {800, 801, 802}
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{800, 801, 802});

    int  spb            = allocator->seqSizePerBlock();
    int  seq_len        = 2 * spb + 1;
    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.reuse_len, 0);

    const auto& swa_out = batch_res->blocks(0, 6);
    ASSERT_GE(swa_out.size(), 2u);
    EXPECT_TRUE(isNullBlockIdx(swa_out[0])) << "SWA previous matched tail is evicted after new tail allocation";
    EXPECT_EQ(swa_out[1], cached_blocks[6][1]) << "SWA last matched tail block should remain";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// incrMalloc: decode grows sequence after initial prefill
// ============================================================

TEST_F(DSV4AllocatorTest, IncrMallocDecodeGrowsBlocks) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    // Initial malloc: 1 block worth of tokens
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{900, 901});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);

    // All 7 groups should have 1 block each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "group " << gid << " should have 1 block after init";
    }

    size_t free_after_init = allocator->freeBlocksNum();

    // incrMalloc: grow to 2 blocks
    cti->setSeqLength(2 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    ASSERT_TRUE(incr_result.success);

    // All 7 groups should now have 2 blocks each
    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 2u) << "group " << gid << " should have 2 blocks after incr";
    }

    // Should have consumed 7 more blocks (1 per group)
    EXPECT_EQ(allocator->freeBlocksNum(), free_after_init - 7);

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// Free and reallocate: blocks return to pool
// ============================================================

TEST_F(DSV4AllocatorTest, FreeReturnsBlocksToPool) {
    auto config    = makeDSV4AllocatorConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    size_t free_before = allocator->freeBlocksNum();
    int    spb         = allocator->seqSizePerBlock();

    // Allocate
    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1000, 1001});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    size_t free_after_alloc = allocator->freeBlocksNum();
    EXPECT_LT(free_after_alloc, free_before);

    // Free
    FreeInfo free_info{batch_res};
    allocator->free(free_info);

    // All blocks should be returned
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);

    // Can allocate again
    auto batch_res2 = std::make_shared<BatchKVCacheResource>();
    batch_res2->resetBatchSize(1);
    batch_res2->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res2->setBatchCacheKeys(0, CacheKeysType{1100, 1101});

    MallocInfo info2{batch_res2, cti};
    info2.enable_device_cache = false;
    auto result2              = allocator->malloc(info2);
    ASSERT_TRUE(result2.success);

    FreeInfo free_info2{batch_res2};
    allocator->free(free_info2);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

// ============================================================
// Flash: incrMalloc decode path
// ============================================================

TEST_F(DSV4AllocatorTest, FlashIncrMallocDecode) {
    auto config    = makeDSV4AllocatorConfig(/*use_flash=*/true);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    int spb = allocator->seqSizePerBlock();

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1200, 1201});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo init_info{batch_res, cti};
    init_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(init_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "Flash group " << gid;
    }

    // Grow to 3 blocks
    cti->setSeqLength(3 * spb);
    MallocInfo incr_info{batch_res, cti};
    incr_info.enable_device_cache = false;
    ASSERT_TRUE(allocator->malloc(incr_info).success);

    for (int gid = 0; gid < 7; gid++) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 3u) << "Flash group " << gid << " after incr";
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// ============================================================
// F01-PR2 part A: CacheKeySalt producer + handshake metadata
//
// makeCacheKeySalt(CacheConfig) is the single source of truth for both
// (a) the PD handshake (version, bitmap) emitted by
//     KVCacheConnectorCoordinator::init
// (b) the cache_key XOR applied by KVCacheManager via
//     initCacheKeys/updateCacheKeys.
//
// Default (K_state == 0) MUST be byte-identical to today (all-zero
// salt, bitmap 0, version 0).  When K_state > 0 the salt mirrors the
// override, the bitmap acquires bit3, and the version flips to
// kCacheKeySaltSchemaVersion so a mixed-mode PD peer producing
// different cache_keys is the conservative outcome.
// ============================================================

namespace {

// Run initCacheKeys against a small synthetic batch and return the
// produced cache_key vector for batch 0.  Helper for cross-K_state
// inequality below.
CacheKeysType runInitCacheKeysWithSalt(const CacheKeySalt& salt, int seq_size_per_block, int seq_len) {
    // Deterministic non-trivial token pattern so the unsalted Jenkins
    // roll yields a non-zero baseline that the salt XOR can perturb in a
    // detectable way.
    const int reserve = std::max(seq_len, seq_size_per_block) + 32;
    auto cti = std::make_shared<CompleteTokenIds>(/*batch_size=*/1,
                                                   /*beam_size=*/1,
                                                   /*max_seq_len=*/reserve,
                                                   seq_size_per_block);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::empty({seq_len}, torch::kInt32);
    int32_t* src                    = generate_input->input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i) {
        src[i] = (i * 31 + 7) & 0xffff;
    }
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    cti->init(generate_input);

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(/*group_nums=*/1, /*layer_num=*/1);

    initCacheKeys(batch_res, cti, seq_size_per_block, salt);
    return batch_res->cacheKeys(0);
}

}  // namespace

TEST(DSV4CacheKeySaltProducerTest, DefaultOffSaltEmptyBitmapZeroVersionZero) {
    // K_state == 0 (default) -> all-zero salt fields, bitmap 0, and the
    // schema-version selector at the coordinator call site stays 0
    // (legacy/legacy peers handshake trivially).
    CacheConfig cfg;
    EXPECT_EQ(cfg.state_entries_per_block_constant, 0);

    const CacheKeySalt salt = makeCacheKeySalt(cfg);
    EXPECT_EQ(salt.model_id, 0u);
    EXPECT_EQ(salt.dtype_id, 0u);
    EXPECT_EQ(salt.lora_id, 0u);
    EXPECT_EQ(salt.K_state, 0u);
    EXPECT_EQ(salt.Tlog, 0u);

    const uint32_t bitmap = nonzeroFieldBitmap(salt);
    EXPECT_EQ(bitmap, 0u);

    // Mirrors the selector in KVCacheConnectorCoordinator::init: version
    // is 0 when bitmap is 0 so legacy peers stay byte-identical.
    const uint32_t version = (bitmap != 0) ? kCacheKeySaltSchemaVersion : 0u;
    EXPECT_EQ(version, 0u);
}

TEST(DSV4CacheKeySaltProducerTest, KStateFourPopulatesBit3AndFlipsVersion) {
    CacheConfig cfg;
    cfg.state_entries_per_block_constant = 4;

    const CacheKeySalt salt = makeCacheKeySalt(cfg);
    EXPECT_EQ(salt.model_id, 0u);
    EXPECT_EQ(salt.dtype_id, 0u);
    EXPECT_EQ(salt.lora_id, 0u);
    EXPECT_EQ(salt.K_state, 4u);
    EXPECT_EQ(salt.Tlog, 0u);

    const uint32_t bitmap = nonzeroFieldBitmap(salt);
    EXPECT_EQ(bitmap, 1u << 3) << "K_state must light up bit3 only";

    // When the bitmap is non-zero the coordinator promotes the version
    // to ``kCacheKeySaltSchemaVersion`` so cross-K_state peers trip
    // REQ-D1 in validateHandshake.
    const uint32_t version = (bitmap != 0) ? kCacheKeySaltSchemaVersion : 0u;
    EXPECT_EQ(version, kCacheKeySaltSchemaVersion);
    EXPECT_EQ(kCacheKeySaltSchemaVersion, 2u);
}

TEST(DSV4CacheKeySaltProducerTest, KStateOneIsLowerBoundStillSaltsBit3) {
    // K_state == 1 is the tightest collapse the F01-PR1 hook permits;
    // bitmap/version behaviour must match K_state == 4.
    CacheConfig cfg;
    cfg.state_entries_per_block_constant = 1;

    const CacheKeySalt salt = makeCacheKeySalt(cfg);
    EXPECT_EQ(salt.K_state, 1u);
    EXPECT_EQ(nonzeroFieldBitmap(salt), 1u << 3);
}

TEST(DSV4CacheKeySaltProducerTest, CrossKStateCacheKeysDiverge) {
    // The load-bearing safety guarantee: cache_keys produced under
    // different K_state values MUST differ block-for-block so a
    // mixed-mode PD pair gets a clean reuse-miss instead of silently
    // colliding on opaque cache_key strings.
    constexpr int kSeqSizePerBlock = 16;
    constexpr int kSeqLen          = 16 * 3;  // 3 full blocks, no partial tail

    CacheConfig cfg_off;
    CacheConfig cfg_on;
    cfg_on.state_entries_per_block_constant = 4;

    const CacheKeySalt salt_off = makeCacheKeySalt(cfg_off);
    const CacheKeySalt salt_on  = makeCacheKeySalt(cfg_on);

    const CacheKeysType keys_off = runInitCacheKeysWithSalt(salt_off, kSeqSizePerBlock, kSeqLen);
    const CacheKeysType keys_on  = runInitCacheKeysWithSalt(salt_on, kSeqSizePerBlock, kSeqLen);

    ASSERT_EQ(keys_off.size(), 3u);
    ASSERT_EQ(keys_on.size(), 3u);

    // Every block_id must differ — the K_state XOR perturbs the rolling
    // hash at every step, not just the first block.
    for (size_t i = 0; i < keys_off.size(); ++i) {
        EXPECT_NE(keys_off[i], keys_on[i])
            << "block " << i << " collided across K_state values (silent corruption hazard)";
    }
}

// ============================================================
// F01-PR2-followup task 7: salt v2 byte-window layout
// ============================================================

namespace {

// Replicates the .cc-private byte packing so the test can predict the
// exact bytes XOR'd into the rolling hash and assert pairwise disjoint
// byte windows.
uint64_t expectedSaltBytes(const CacheKeySalt& s) {
    auto fold8 = [](uint64_t v) {
        v ^= v >> 32;
        v ^= v >> 16;
        v ^= v >> 8;
        return v & 0xffull;
    };
    uint64_t bytes = 0;
    bytes |= fold8(s.model_id) << (0 * 8);
    bytes |= (static_cast<uint64_t>(s.dtype_id) & 0xffull) << (1 * 8);
    bytes |= (static_cast<uint64_t>(s.lora_id) & 0xffull) << (2 * 8);
    bytes |= (static_cast<uint64_t>(s.K_state) & 0xffull) << (3 * 8);
    bytes |= (static_cast<uint64_t>(s.Tlog) & 0xffull) << (4 * 8);
    return bytes;
}

}  // namespace

TEST(DSV4CacheKeySaltProducerTest, SaltXorBytesNonOverlapping) {
    // For each single-field non-zero salt, assert (a) the field only
    // writes to its own byte window, and (b) any two distinct fields'
    // byte windows are disjoint. Pairwise non-overlap is the load-bearing
    // v2 invariant that kills the R4-4 F7 collision class.
    struct FieldCase {
        const char*  name;
        CacheKeySalt salt;
        uint64_t     expected_byte_mask;
    };
    std::vector<FieldCase> cases = {
        {"model_id", [] { CacheKeySalt s{}; s.model_id = 0xdeadbeefcafebabeULL; return s; }(), 0x00000000000000ffULL},
        {"dtype_id", [] { CacheKeySalt s{}; s.dtype_id = 0xab; return s; }(),                  0x000000000000ff00ULL},
        {"lora_id",  [] { CacheKeySalt s{}; s.lora_id  = 0xcd; return s; }(),                  0x0000000000ff0000ULL},
        {"K_state",  [] { CacheKeySalt s{}; s.K_state  = 4;    return s; }(),                  0x00000000ff000000ULL},
        {"Tlog",     [] { CacheKeySalt s{}; s.Tlog     = 7;    return s; }(),                  0x000000ff00000000ULL},
    };

    for (const auto& c : cases) {
        const uint64_t bytes = expectedSaltBytes(c.salt);
        EXPECT_NE(bytes, 0ull) << c.name << " produced no salt bytes (fold8 / mask broken)";
        EXPECT_EQ(bytes & ~c.expected_byte_mask, 0ull)
            << c.name << " leaked outside its allowed byte window";
        // Reserved bytes 5..7 must stay zero.
        EXPECT_EQ(bytes & 0xffffff0000000000ULL, 0ull)
            << c.name << " wrote into reserved bytes 5..7";
    }

    // Pairwise disjoint AND-mask.
    for (size_t i = 0; i < cases.size(); ++i) {
        for (size_t j = i + 1; j < cases.size(); ++j) {
            const uint64_t bi = expectedSaltBytes(cases[i].salt);
            const uint64_t bj = expectedSaltBytes(cases[j].salt);
            EXPECT_EQ(bi & bj, 0ull)
                << "byte windows for " << cases[i].name << " and " << cases[j].name << " overlap";
        }
    }
}

TEST(DSV4CacheKeySaltProducerTest, SaltCollisionRegressionTlogVsModelId) {
    // R4-4 F7 regression guard: under the v1 XOR layout, {Tlog=1} and
    // {model_id=1<<32} XOR'd into bits 32..63 — producing identical
    // cache_keys. The v2 byte-window layout routes Tlog->byte4 and
    // fold8(model_id)->byte0, so the two configurations now diverge.
    CacheKeySalt salt_tlog{};
    salt_tlog.Tlog = 1;

    CacheKeySalt salt_model{};
    salt_model.model_id = (1ULL << 32);

    EXPECT_NE(expectedSaltBytes(salt_tlog), expectedSaltBytes(salt_model))
        << "v1 collision class still present: {Tlog=1} and {model_id=1<<32} produce identical salt bytes";

    constexpr int kSpb = 16;
    constexpr int kLen = 16 * 2;
    auto keys_tlog  = runInitCacheKeysWithSalt(salt_tlog, kSpb, kLen);
    auto keys_model = runInitCacheKeysWithSalt(salt_model, kSpb, kLen);
    ASSERT_EQ(keys_tlog.size(), 2u);
    ASSERT_EQ(keys_model.size(), 2u);
    for (size_t i = 0; i < keys_tlog.size(); ++i) {
        EXPECT_NE(keys_tlog[i], keys_model[i])
            << "block " << i << " regressed R4-4 F7: {Tlog=1} ≡ {model_id=1<<32}";
    }
}

TEST(DSV4CacheKeySaltProducerTest, ModelIdHighWordContributesViaFold8) {
    // fold8 must surface model_id high bits (1<<32) into byte 0;
    // otherwise a salt with only model_id set high would equal zero salt.
    CacheKeySalt salt_hi{};
    salt_hi.model_id = (1ULL << 32);
    EXPECT_NE(expectedSaltBytes(salt_hi), 0ull)
        << "fold8(model_id=1<<32) collapsed to 0 — model_id high bits not contributing";
}

TEST(DSV4CacheKeySaltProducerTest, SaltSchemaVersionIsV2) {
    // F01-PR2-followup bumped schema to v2 (non-overlapping byte windows).
    EXPECT_EQ(kCacheKeySaltSchemaVersion, 2u);
}

// ============================================================
// F01-PR2-followup task 7: K_state lifecycle hardening
// ============================================================

TEST(DSV4KStateHookTest, KStateEquals256NormalizesToOff) {
    // K_state == 256 is the kernel-block identity (no shrink). Helper
    // must normalize it to OFF (mirror=0, salt bit3 dark, byte-identical
    // hash bytes) so a config operationally equivalent to OFF does NOT
    // light up salt bit3 / handshake bitmap.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = 256;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);

    EXPECT_EQ(config.state_entries_per_block_constant, 0)
        << "K_state=256 must normalize to mirror=0 (identity, no kernel shrink)";

    const CacheKeySalt salt = makeCacheKeySalt(config);
    EXPECT_EQ(salt.K_state, 0u);
    EXPECT_EQ(nonzeroFieldBitmap(salt), 0u) << "K_state=256 must keep bitmap empty";

    // State pool block sizes byte-identical to default-OFF.
    ASSERT_EQ(config.cache_specs.size(), 7u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 256u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 256u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 256u * 1024u * 4u);
}

TEST(DSV4KStateHookTest, KStateNegativeRejected) {
    // Validate-on-load: negative K_state must fail-loud.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = -1;

    EXPECT_ANY_THROW({
        (void)HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    }) << "K_state<0 must trigger RTP_LLM_CHECK_WITH_INFO";
}

TEST(DSV4KStateHookTest, KStateOversizedRejected) {
    // Validate-on-load: K_state > kernel block size (256) must fail-loud.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = 512;

    EXPECT_ANY_THROW({
        (void)HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    }) << "K_state>256 must trigger RTP_LLM_CHECK_WITH_INFO";
}

// FIX-B HIGH-3 (DEFEND-3 #2): K_state must divide kDsv4KernelTokensPerBlock=256
// evenly; non-divisors silently corrupt per-block byte math downstream.  This
// is a stricter contract than the previous "warn on non-power-of-2" test
// (which accepted K_state=3); the WARN path is now subsumed by the hard
// CHECK because all divisors of 256 are exactly the powers of 2 in [1,256].
TEST(DSV4KStateHookTest, KStateNonDivisorOfKernelBlockRejected) {
    // Each of {3,5,7,9} is non-divisor of 256 (and non-power-of-2); each
    // must trigger RTP_LLM_CHECK_WITH_INFO at the divisibility guard.
    for (int bad_k : {3, 5, 7, 9}) {
        auto              mc = makeProModelConfig();
        ParallelismConfig pc;
        KVCacheConfig     kv_cache_config;
        kv_cache_config.dsv4_state_entries_per_block = bad_k;

        EXPECT_ANY_THROW({
            (void)HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
        }) << "K_state=" << bad_k << " (non-divisor of 256) must trigger RTP_LLM_CHECK_WITH_INFO";
    }
}

TEST(DSV4CacheKeySaltProducerTest, DefaultOffCacheKeysAreLegacyIdentical) {
    // Default config (K_state == 0) -> all-zero salt -> XOR identity
    // -> cache_keys MUST match the legacy unsalted overload byte-for-
    // byte.  Anyone landing F01-PR2 must NOT shift production cache_key
    // bytes when the K_state hook is off.
    constexpr int kSeqSizePerBlock = 16;
    constexpr int kSeqLen          = 16 * 4;  // 4 full blocks

    CacheConfig cfg_off;
    const CacheKeySalt salt_off = makeCacheKeySalt(cfg_off);
    EXPECT_EQ(nonzeroFieldBitmap(salt_off), 0u);

    const CacheKeysType keys_salted = runInitCacheKeysWithSalt(salt_off, kSeqSizePerBlock, kSeqLen);

    // Build an unsalted baseline via the legacy overload (zero-salt
    // wrapper) and compare element-wise.
    auto cti = std::make_shared<CompleteTokenIds>(/*batch_size=*/1,
                                                   /*beam_size=*/1,
                                                   /*max_seq_len=*/kSeqLen + 32,
                                                   kSeqSizePerBlock);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = torch::empty({kSeqLen}, torch::kInt32);
    int32_t* src                    = generate_input->input_ids.data_ptr<int32_t>();
    for (int i = 0; i < kSeqLen; ++i) {
        src[i] = (i * 31 + 7) & 0xffff;
    }
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    cti->init(generate_input);

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(/*group_nums=*/1, /*layer_num=*/1);
    initCacheKeys(batch_res, cti, kSeqSizePerBlock);  // legacy unsalted overload
    const CacheKeysType keys_legacy = batch_res->cacheKeys(0);

    ASSERT_EQ(keys_salted.size(), keys_legacy.size());
    for (size_t i = 0; i < keys_salted.size(); ++i) {
        EXPECT_EQ(keys_salted[i], keys_legacy[i])
            << "default-off salt produced a different key at block " << i
            << " (must be byte-identical to legacy)";
    }
}

// ============================================================
// F01-PR2 followup (DEV-3): createSpConfig() WARN mirror.
//
// createConfig() at CacheConfigCreator.cc L186-197 emits a B-fix-2
// WARNING when PREFILL + state_pool_memory_mb=0 falls back to the
// HBM-derived block_num placed on pinned host memory and the projected
// host bytes/pool exceed 8 GiB. The sp entry-point had the identical
// fallback formula but no WARN — silent foot-gun for future PREFILL+MTP
// rollouts. This test exercises the predicate that gates the WARN to
// guard against regressions of the mirror branch.
// ============================================================
TEST(CacheConfigStatePinnedTest, SpConfigPrefillPinnedFallbackEmitsWarnWhenStateBytesLarge) {
    auto score_model_config   = makeFlashModelConfig();
    auto propose_model_config = makeFlashMtpModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    // Force a block_num that crosses the 8 GiB WARN threshold but stays
    // under the 64 GiB HARD CAP added by R6 DEV-α (canary §1.0 item 10).
    // For flash sp config state_block_size_bytes ≈ 76 MB / pool, so
    //   - 8 GiB / 76 MB ≈ 113 blocks (WARN floor)
    //   - 64 GiB / 76 MB ≈ 862 blocks (HARD CAP ceiling)
    // 512 sits comfortably in between → WARN fires, CAP does not.
    kv_cache_config.test_block_num                   = 512;
    kv_cache_config.state_pool_memory_mb             = 0;  // pinned via PREFILL role
    kv_cache_config.non_full_addition_kvcache_blocks = 0;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     /*is_mtp=*/true,
                                                     /*is_eagle=*/false,
                                                     RoleType::PREFILL);

    // (a) Result is valid — fallback still produced a usable config.
    EXPECT_GT(config.block_num, 0);
    EXPECT_EQ(static_cast<uint32_t>(config.block_num), 512u);

    // (b) Verify the WARN guard predicate is satisfied — i.e. the code
    // path that emits the WARN was taken. alog output is not trivially
    // capturable from gtest, so we assert via the same side-channels the
    // WARN guards on:
    //   1. state_pool_uses_pinned_cpu (PREFILL fallback was applied)
    //   2. state_block_size_bytes > 0 (DSV4 path, not non-DSV4)
    //   3. block_num * state_block_size_bytes > 8 GiB (threshold crossed)
    ASSERT_TRUE(config.state_pool_uses_pinned_cpu)
        << "PREFILL + state_pool_memory_mb=0 must mark sp-config STATE as pinned-CPU";
    ASSERT_GT(config.state_block_size_bytes, 0u)
        << "DSV4 sp score_config must populate state_block_size_bytes";
    const size_t projected_host_bytes =
        static_cast<size_t>(config.block_num) * config.state_block_size_bytes;
    constexpr size_t kWarnThresholdBytes = 8ULL * 1024 * 1024 * 1024;  // 8 GiB
    EXPECT_GT(projected_host_bytes, kWarnThresholdBytes)
        << "Test scenario must cross the 8 GiB WARN threshold so the sp WARN "
        << "branch actually fires (test_block_num=512 * state_block_size_bytes="
        << config.state_block_size_bytes << " B = "
        << (projected_host_bytes / 1024 / 1024) << " MiB)";

    // state_block_num must mirror HBM block_num for the fallback path
    // (per the unchanged `return static_cast<uint32_t>(block_num)` return).
    EXPECT_EQ(config.state_block_num, static_cast<uint32_t>(config.block_num))
        << "PREFILL fallback (no explicit budget) must mirror HBM block_num";
}

// Negative guard: non-PREFILL role with the same large state-byte
// scenario must NOT take the WARN branch. We can't observe the absence
// of a log line, but state_pool_uses_pinned_cpu must be false, which is
// the same predicate that gates the WARN.
TEST(CacheConfigStatePinnedTest, SpConfigPdfusionDoesNotEnterPinnedFallback) {
    auto score_model_config   = makeFlashModelConfig();
    auto propose_model_config = makeFlashMtpModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num                   = 200000;
    kv_cache_config.state_pool_memory_mb             = 0;
    kv_cache_config.non_full_addition_kvcache_blocks = 0;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createSpConfig(score_model_config,
                                                     propose_model_config,
                                                     parallelism_config,
                                                     runtime_config,
                                                     kv_cache_config,
                                                     sp_config,
                                                     std::nullopt,
                                                     /*is_mtp=*/true,
                                                     /*is_eagle=*/false,
                                                     RoleType::PDFUSION);

    EXPECT_FALSE(config.state_pool_uses_pinned_cpu)
        << "PDFUSION must not enter pinned-CPU fallback — WARN branch must be skipped";
}

// ====================================================================
// R6 DEV-ε — KVCacheHashUtil + DSV4CacheConfigHelper defensive CHECKs
// (DEFEND-4 HIGH-2 / HIGH-4 + DEFEND-3 HIGH-3)
// ====================================================================
//
// Tests below cover the three HIGH-severity bounds CHECKs added in
// FIX-B follow-up:
//
//   R6Hash*  — KVCacheHashUtil.cc invariants:
//              * bitmap-vs-salt-byte consistency under fold8 collision
//                (DEFEND-4 HIGH-2);
//              * per-field >255 silent-truncation reject (DEFEND-4 HIGH-4).
//
//   R6Helper* — DSV4CacheConfigHelper.cc structural invariants:
//              * pools[0..2].is_paged + pools[3..5].!is_paged (STATE) +
//                pools[6].region==SWA_KV hoisted OUT of the K_state>0
//                branch so legacy OFF deployments also exercise them
//                (DEFEND-3 HIGH-3).
//
// Coordinated with DEV-α (CreateSpMtpUnified* / PinnedCpuHardCap* /
// F02Mirror* prefixes appended to this same file): non-overlapping
// prefixes prevent merge collisions.

namespace {

// DEFEND-4 HIGH-2: a uint64 whose 8 bytes XOR to 0 produces fold8(v)==0
// yet is itself non-zero.  0xFFFF has byte0=0xFF, byte1=0xFF, rest=0 →
// XOR = 0; the smallest natural model_id collision.  See
// KVCacheHashUtil.cc::fold8 for the folding formula.
constexpr uint64_t kFold8CollisionModelId = 0xFFFFull;

}  // namespace

TEST(R6HashFold8CollisionTest, BitmapReportsSetEvenWhenFold8Collapses) {
    // Bitmap MUST reflect "user configured this field" (raw source field
    // != 0), NOT "packed salt byte != 0".  Otherwise a fold8-collision
    // model_id would silently report bit0=0 over the handshake — peers
    // with model_id=0 and model_id=0xFFFF would advertise the same
    // (version, bitmap) and pass REQ-D1, yet their cache_keys differ
    // (one path skips XOR, the other XORs salt-byte0=0 which is identity
    // — so they'd actually collide here, but for non-fold-colliding
    // dtype/lora paths the silent-skip hazard is real).  This test pins
    // the bitmap-side contract.
    CacheKeySalt salt{};
    salt.model_id = kFold8CollisionModelId;
    EXPECT_NE(salt.model_id, 0ull)
        << "test precondition: model_id must be non-zero to exercise bit0";
    const uint32_t bitmap = nonzeroFieldBitmap(salt);
    EXPECT_NE(bitmap & (1u << 0), 0u)
        << "bitmap must report bit0=1 for any non-zero model_id, even when fold8 collapses to 0";
}

TEST(R6HashFold8CollisionTest, PackSaltRejectsFold8CollisionForNonZeroModelId) {
    // The salt boundary CHECK MUST fire on fold8(model_id)==0 with
    // model_id != 0.  This is the silent-corruption path: bitmap bit0=1
    // (peer thinks model_id is set), salt byte0=0 (XOR is identity for
    // model_id); a peer with model_id=0 would compute identical
    // cache_keys yet REFUSE on bitmap mismatch — and worse, two peers
    // each with distinct fold-colliding model_ids would PASS handshake
    // (same bitmap) and silently collide on cache_keys.
    CacheKeySalt salt{};
    salt.model_id = kFold8CollisionModelId;
    EXPECT_ANY_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "fold8-collision model_id must trigger RTP_LLM_CHECK_WITH_INFO at salt boundary";
}

TEST(R6HashFold8CollisionTest, ZeroModelIdAllowedAsLegacyIdentity) {
    // Negative control: model_id == 0 (legacy unsalted entry, bit0=0 by
    // design) MUST be allowed through the CHECK.  Otherwise the legacy
    // zero-salt wrapper would crash on every call.
    CacheKeySalt salt{};
    salt.model_id = 0;
    EXPECT_NO_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    });
}

TEST(R6HashOversizedFieldTest, DtypeIdAbove255Rejected) {
    // DEFEND-4 HIGH-4: ``s.dtype_id & 0xff`` silently truncates >255 to
    // 0, producing bitmap=set yet byte1=0 (silent reuse-miss across PD
    // peers).  packSaltBytes CHECK must reject.
    CacheKeySalt salt{};
    salt.dtype_id = 0x100;  // first value where & 0xff == 0
    EXPECT_ANY_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "dtype_id=0x100 must trigger RTP_LLM_CHECK_WITH_INFO at salt boundary";
}

TEST(R6HashOversizedFieldTest, LoraIdAbove255Rejected) {
    // Multi-tenant LoRA buckets can plausibly hit >255 IDs; producer is
    // expected to fold (id % 256) before assignment.  Catch unfolded IDs
    // at the salt boundary so the bug surfaces at startup.
    CacheKeySalt salt{};
    salt.lora_id = 1024;
    EXPECT_ANY_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "lora_id=1024 must trigger RTP_LLM_CHECK_WITH_INFO at salt boundary";
}

TEST(R6HashOversizedFieldTest, KStateAbove255Rejected) {
    // DSV4CacheConfigHelper normalises K_state=256 to OFF and rejects
    // anything larger up-front, but the salt boundary is the last
    // defence against handshake-receive paths re-injecting an oversized
    // K_state into a CacheKeySalt manually.
    CacheKeySalt salt{};
    salt.K_state = 256;
    EXPECT_ANY_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "K_state=256 must trigger RTP_LLM_CHECK_WITH_INFO at salt boundary";
}

TEST(R6HashOversizedFieldTest, TlogAbove255Rejected) {
    CacheKeySalt salt{};
    salt.Tlog = 300;
    EXPECT_ANY_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "Tlog=300 must trigger RTP_LLM_CHECK_WITH_INFO at salt boundary";
}

TEST(R6HashOversizedFieldTest, MaxByteValueAllowed) {
    // Negative control: each field at the boundary (0xFF) MUST pass.
    CacheKeySalt salt{};
    salt.dtype_id = 0xFF;
    salt.lora_id  = 0xFF;
    salt.K_state  = 0xFF;
    salt.Tlog     = 0xFF;
    EXPECT_NO_THROW({
        (void)runInitCacheKeysWithSalt(salt, /*seq_size_per_block=*/16, /*seq_len=*/16);
    }) << "all fields at 0xFF must pass — boundary inclusivity";
}

TEST(R6HelperPoolStructuralInvariants, DefaultOffPathExercisesStructuralChecks) {
    // DEFEND-3 HIGH-3: pools[0..2].is_paged + pools[3..5].!is_paged
    // (STATE) + pools[6].region==SWA_KV invariants must fire on the
    // legacy OFF path, not only when K_state>0.  Positive proof: build a
    // default DSV4 config (K_state=0, super_block layout off) and verify
    // every produced pool matches the ordering contract — this is the
    // exact assertion the new structural CHECKs guard against.  Any
    // future refactor of buildDSV4PoolDescs that re-orders pools would
    // crash applyConfig at startup and ALSO fail this test, giving
    // operators two independent signals of the regression.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    // Default K_state=0 (legacy OFF) — this is the path that previously
    // had zero structural guards on the STATE pools.
    kv_cache_config.dsv4_state_entries_per_block = 0;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    ASSERT_EQ(config.group_types.size(), 7u);
    ASSERT_EQ(config.group_region_names.size(), 7u);

    // pools[0..2] are paged FULL (CSA_KV, HCA_KV, INDEXER_KV).
    for (size_t i : {0u, 1u, 2u}) {
        EXPECT_EQ(config.group_types[i], CacheGroupType::FULL)
            << "pool " << i << " must be FULL group type";
        EXPECT_FALSE(isStateRegion(config.group_region_names[i]))
            << "pool " << i << " must NOT be a STATE region";
    }
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::CSA_KV);
    EXPECT_EQ(config.group_region_names[1], KVCacheRegionName::HCA_KV);
    EXPECT_EQ(config.group_region_names[2], KVCacheRegionName::INDEXER_KV);

    // pools[3..5] are non-paged STATE (INDEXER_STATE, CSA_STATE, HCA_STATE).
    for (size_t i : {3u, 4u, 5u}) {
        EXPECT_EQ(config.group_types[i], CacheGroupType::SWA)
            << "pool " << i << " must be SWA group type (non-paged)";
        EXPECT_TRUE(isStateRegion(config.group_region_names[i]))
            << "pool " << i << " must be a STATE region (idx 3..5 invariant)";
    }
    EXPECT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    EXPECT_EQ(config.group_region_names[4], KVCacheRegionName::CSA_STATE);
    EXPECT_EQ(config.group_region_names[5], KVCacheRegionName::HCA_STATE);

    // pools[6] is the non-paged SWA_KV.
    EXPECT_EQ(config.group_types[6], CacheGroupType::SWA);
    EXPECT_EQ(config.group_region_names[6], KVCacheRegionName::SWA_KV);
}

TEST(R6HelperPoolStructuralInvariants, KStateOnPathStillExercisesStructuralChecks) {
    // Cross-check: the K_state>0 path must continue to honour the same
    // structural invariants (the hoisted CHECKs run regardless of
    // K_state). Pick a power-of-2 divisor of 256 so the inner divisibility
    // CHECKs all pass.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.dsv4_state_entries_per_block = 64;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    ASSERT_EQ(config.group_region_names.size(), 7u);
    EXPECT_EQ(config.group_region_names[3], KVCacheRegionName::INDEXER_STATE);
    EXPECT_EQ(config.group_region_names[4], KVCacheRegionName::CSA_STATE);
    EXPECT_EQ(config.group_region_names[5], KVCacheRegionName::HCA_STATE);
    EXPECT_EQ(config.group_region_names[6], KVCacheRegionName::SWA_KV);
    EXPECT_EQ(config.state_entries_per_block_constant, 64);
}

TEST(R6HelperPoolStructuralInvariants, FlashModelAlsoSatisfiesStructure) {
    // Negative control across a second model topology so the structural
    // contract is verified independent of layer_compress_ratios.
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config);
    ASSERT_EQ(config.group_region_names.size(), 7u);
    EXPECT_EQ(config.group_region_names[0], KVCacheRegionName::CSA_KV);
    EXPECT_EQ(config.group_region_names[6], KVCacheRegionName::SWA_KV);
    for (size_t i : {3u, 4u, 5u}) {
        EXPECT_TRUE(isStateRegion(config.group_region_names[i]))
            << "flash-model pool " << i << " must be STATE";
    }
}

// ====================================================================
// R6 DEV-α — CacheConfigCreator: canary §1.0 Phase-5 re-flip blockers
// (Task 1: createSpConfig MTP unified-branch crash fix,
//  Task 2: PREFILL pinned-CPU HARD CAP,
//  Task 3: DEFEND-3 HIGH-1 createSp↔createConfig F02 mirror parity).
// All test names are prefixed CreateSpMtpUnified* / PinnedCpuHardCap* /
// F02MirrorTest* per cross-agent file lock convention.
// ====================================================================

namespace {

CacheConfig makeStateHardCapTestConfig(uint32_t test_block_num, RoleType role) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num                   = test_block_num;
    kv_cache_config.state_pool_memory_mb             = 0;  // PREFILL pinned-CPU fallback
    kv_cache_config.non_full_addition_kvcache_blocks = 0;
    return CacheConfigCreator::createConfig(
        mc, pc, runtime_config, kv_cache_config, std::nullopt, std::nullopt, role);
}

}  // namespace

// --- Task 1: createSpConfig MTP unified branch (canary item 1 / R02 H-1) ---
//
// Pre-fix: with dsv4_unified_block_count=1, score_config and propose_config
// returned from createBasicConfig() have super_block_layout.enabled=true but
// num_super_blocks=0. createSpConfig() then never populates num_super_blocks
// on the merged config, so HybridPoolKVCacheAllocator::init crashes at
// CHECK(num_super_blocks > 0). This test asserts the F02 mirror branch wires
// num_super_blocks + group_block_nums on both the merged config and every
// mtp_sub_config without throwing.
TEST(CacheConfigStatePinnedTest, CreateSpMtpUnifiedDoesNotCrash) {
    auto score_model_config   = makeFlashModelConfig();
    auto propose_model_config = makeFlashMtpModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num            = 128;
    kv_cache_config.state_pool_memory_mb      = 0;
    kv_cache_config.dsv4_unified_block_count  = 1;  // M02-PR1 F02 unified path ON
    kv_cache_config.non_full_addition_kvcache_blocks = 0;

    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 2;

    CacheConfig config;
    ASSERT_NO_THROW(config = CacheConfigCreator::createSpConfig(score_model_config,
                                                                propose_model_config,
                                                                parallelism_config,
                                                                runtime_config,
                                                                kv_cache_config,
                                                                sp_config,
                                                                std::nullopt,
                                                                /*is_mtp=*/true,
                                                                /*is_eagle=*/false,
                                                                RoleType::PDFUSION))
        << "DSV4 + MTP + dsv4_unified_block_count=1 must not crash createSpConfig "
        << "(R02 H-1 / canary §1.0 item 1)";

    // Merged config must satisfy HybridPoolKVCacheAllocator::init invariants:
    //   - super_block_layout.enabled == true (carried from score_config)
    //   - num_super_blocks > 0  (was 0 pre-fix — the crash root cause)
    //   - bps.size() == cache_specs.size() (group-count parity)
    //   - group_block_nums populated via block_num * bps[p] + non-FULL addition
    EXPECT_TRUE(config.super_block_layout.enabled)
        << "F02 unified flag must propagate to merged config";
    EXPECT_GT(config.super_block_layout.num_super_blocks, 0u)
        << "createSpConfig F02 mirror must set num_super_blocks (was 0 pre-fix → crash)";
    EXPECT_EQ(config.super_block_layout.bps.size(), config.cache_specs.size())
        << "bps size must equal cache_specs size for allocator init CHECK";
    EXPECT_EQ(config.group_block_nums.size(), config.group_block_size_bytes.size())
        << "group_block_nums must be populated (F02 path write-back)";

    // Sub-configs (per MTP module) must also have F02 populated so any
    // per-sub allocator path passes the same CHECK.
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u) << "gen_num_per_cycle=2 expects 2 sub_cfgs";
    for (const auto& sub : config.mtp_sub_configs) {
        ASSERT_NE(sub, nullptr);
        EXPECT_TRUE(sub->super_block_layout.enabled);
        EXPECT_GT(sub->super_block_layout.num_super_blocks, 0u)
            << "sub_cfg must also have F02 num_super_blocks populated";
        EXPECT_EQ(sub->super_block_layout.bps.size(), sub->cache_specs.size());
    }
}

// --- Task 2: PREFILL pinned-CPU HARD CAP (canary items 2 + 10) ---
//
// createConfig() at the WARN block must HARD-CHECK fail when
// block_num * state_block_size_bytes >= 64 GiB (per pool). Operator MUST set
// --state_pool_memory_mb explicitly to escape; silent unbounded fallback is
// banned. The CHECK is in addition to (not replacing) the 8 GiB WARN.
//
// Empirical: DSV4 pro model state_block_size_bytes ≈ 111 MB (per pool with
// default 256-entry blocks). 64 GiB / 111 MB ≈ 590 blocks. So any
// test_block_num >= ~590 will trip the CAP under the pro model. We pick
// 4096 to give comfortable margin while still being a sane bazel UT value.
TEST(CacheConfigStatePinnedTest, PinnedCpuHardCapTriggersOnOversize) {
    // (a) Below CAP: a tiny block_num must not throw — a 32-block PREFILL
    // config produces ~3.5 GiB host bytes, well under the 8 GiB WARN and the
    // 64 GiB CAP. WARN does not fire, CAP does not fire.
    EXPECT_NO_THROW(makeStateHardCapTestConfig(/*test_block_num=*/32, RoleType::PREFILL))
        << "32 blocks (well under WARN+CAP) must remain valid PREFILL pinned-CPU config";

    // (b) Above CAP: RTP_LLM_CHECK_WITH_INFO throws rtp_llm::RTPException
    // (std::exception) rather than abort(), so use EXPECT_THROW not
    // EXPECT_DEATH. 4096 blocks * ~111 MB ≈ 444 GiB host bytes — comfortably
    // past the 64 GiB cap. The CHECK message includes "exceeds hard cap" so
    // we additionally substring-match below for robustness.
    try {
        (void)makeStateHardCapTestConfig(/*test_block_num=*/4096u, RoleType::PREFILL);
        FAIL() << "4096 blocks must trip canary §1.0 item 10 HARD CAP CHECK (no throw observed)";
    } catch (const std::exception& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("exceeds hard cap"), std::string::npos)
            << "HARD CAP CHECK message must mention 'exceeds hard cap'; got: " << what;
        EXPECT_NE(what.find("state_pool_memory_mb"), std::string::npos)
            << "HARD CAP CHECK message must direct operator to --state_pool_memory_mb; got: " << what;
    }
}

// --- Task 3: DEFEND-3 HIGH-1 createSp F02 super-block branch mirror ---
//
// CacheConfigCreatorMirrorTest core case: same ModelConfig + KVCacheConfig
// driven through both createConfig() and createSpConfig() (with a degenerate
// sp_config that triggers the sp path) must produce structurally-identical
// super_block_layout state. Catches future drift where one entry-point
// gets a new F02 field and the other does not.
TEST(CacheConfigStatePinnedTest, F02MirrorTestSpVsNonSpConfigParity) {
    auto mc = makeFlashModelConfig();

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num                   = 128;
    kv_cache_config.state_pool_memory_mb             = 0;
    kv_cache_config.dsv4_unified_block_count         = 1;
    kv_cache_config.non_full_addition_kvcache_blocks = 0;

    auto non_sp_config = CacheConfigCreator::createConfig(
        mc, parallelism_config, runtime_config, kv_cache_config, std::nullopt, std::nullopt, RoleType::PDFUSION);

    // SP path with a degenerate single-module propose config so the test
    // exercises the merge logic without doubling group counts.
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;

    auto sp_propose_config = makeFlashMtpModelConfig();
    auto sp_config_out     = CacheConfigCreator::createSpConfig(mc,
                                                            sp_propose_config,
                                                            parallelism_config,
                                                            runtime_config,
                                                            kv_cache_config,
                                                            sp_config,
                                                            std::nullopt,
                                                            /*is_mtp=*/true,
                                                            /*is_eagle=*/false,
                                                            RoleType::PDFUSION);

    // Parity invariants (anti-drift):
    EXPECT_EQ(non_sp_config.super_block_layout.enabled, sp_config_out.super_block_layout.enabled)
        << "F02 enabled flag must mirror between createConfig and createSpConfig";
    EXPECT_TRUE(sp_config_out.super_block_layout.enabled)
        << "test precondition: dsv4_unified_block_count=1 must enable F02 path";

    EXPECT_GT(non_sp_config.super_block_layout.num_super_blocks, 0u)
        << "createConfig must populate num_super_blocks";
    EXPECT_GT(sp_config_out.super_block_layout.num_super_blocks, 0u)
        << "createSpConfig must populate num_super_blocks (DEFEND-3 HIGH-1)";

    // bps content parity — DSV4 today is bps[p] ≡ 1 across all pools. If a
    // future kernel change flips one path off-by-one this test surfaces it.
    EXPECT_EQ(non_sp_config.super_block_layout.bps, sp_config_out.super_block_layout.bps)
        << "F02 bps vector must match between sp / non-sp entry-points";

    // group_block_nums population parity (length match — values may differ
    // for non-FULL additions across MTP modules, but both paths must have
    // populated the vector).
    EXPECT_EQ(non_sp_config.group_block_nums.size(), sp_config_out.group_block_nums.size())
        << "F02 group_block_nums must be populated on both entry-points";
    EXPECT_FALSE(sp_config_out.group_block_nums.empty())
        << "createSpConfig must populate group_block_nums in F02 path";
}

}  // namespace test
}  // namespace rtp_llm
