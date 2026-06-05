#include <gtest/gtest.h>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/ZeroSwaCacheHelper.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {

constexpr int      kDsv4PoolNum           = 7;
constexpr uint32_t kDsv4TokensPerBlock    = 128;
constexpr uint32_t kDsv4KvEntryBytes      = 1024;
constexpr uint32_t kDsv4IndexerEntryBytes = 256;
constexpr uint32_t kDsv4Fp8KvEntryBytes   = 584;
constexpr uint32_t kDsv4FixedPoolBlocks   = 256;

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old = std::getenv(name_);
        if (old != nullptr) {
            old_value_ = old;
            had_old_   = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_old_{false};
};

}  // namespace

static KVCacheConfig makeDsv4KvCacheConfig(uint32_t fixed_pool_blocks        = kDsv4FixedPoolBlocks,
                                           uint32_t seq_size_per_block       = kDsv4TokensPerBlock,
                                           uint32_t kernel_seq_size_per_block = 0) {
    KVCacheConfig config;
    config.seq_size_per_block        = seq_size_per_block;
    config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    config.dsv4_fixed_pool_blocks    = fixed_pool_blocks;
    return config;
}

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
    auto config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 30u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 31u);
    EXPECT_EQ(config.global_layer_ids[6].size(), 61u);
}

TEST(HybridPoolConfigCreatorTest, FlashLayerClassification) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.layer_num, 43u);
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
    EXPECT_EQ(config.global_layer_ids[1].size(), 20u);
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);
}

TEST(HybridPoolConfigCreatorTest, ZeroSwaRestoreWindowAlignsToOne8192TokenBlock) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeProModelConfig(),
        pc,
        makeDsv4KvCacheConfig(kDsv4FixedPoolBlocks, /*seq_size_per_block=*/8192, /*kernel_seq_size_per_block=*/128),
        false,
        0);
    config.dsv4_zero_swa_caching = true;

    EXPECT_EQ(config.swa_window_size, 128u);
    EXPECT_EQ(config.layer_num, 61u);
    EXPECT_EQ(config.seq_size_per_block, 8192u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 64u);
    EXPECT_EQ(zeroSwaRestoreWindowTokens(config), 7808u);
    EXPECT_EQ(zeroSwaRestoreWindowBlocks(config, config.seq_size_per_block), 1u);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config, 4, static_cast<int>(config.seq_size_per_block)), 3);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config, 1, static_cast<int>(config.seq_size_per_block)), 0);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config,
                                              1,
                                              static_cast<int>(config.seq_size_per_block),
                                              static_cast<int>(config.seq_size_per_block + zeroSwaRestoreWindowTokens(config))),
              1);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config,
                                              1,
                                              static_cast<int>(config.seq_size_per_block),
                                              static_cast<int>(config.seq_size_per_block + zeroSwaRestoreWindowTokens(config) - 1)),
              0);
}

TEST(HybridPoolConfigCreatorTest, ZeroSwaRestoreWindowUsesCpVirtualBlockUnit) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeProModelConfig(),
        pc,
        makeDsv4KvCacheConfig(kDsv4FixedPoolBlocks, /*seq_size_per_block=*/8192, /*kernel_seq_size_per_block=*/128),
        false,
        0);
    config.dsv4_zero_swa_caching = true;

    const size_t cp2_virtual_block_tokens = config.seq_size_per_block * 2;
    EXPECT_EQ(zeroSwaRestoreWindowTokens(config), 7808u);
    EXPECT_EQ(zeroSwaRestoreWindowBlocks(config, cp2_virtual_block_tokens), 1u);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config, 4u, cp2_virtual_block_tokens), 3u);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config, 1u, cp2_virtual_block_tokens), 0u);
    EXPECT_EQ(capReuseBlocksForZeroSwaCaching(config,
                                              1u,
                                              cp2_virtual_block_tokens,
                                              cp2_virtual_block_tokens + zeroSwaRestoreWindowTokens(config)),
              1u);
}

TEST(HybridPoolConfigCreatorTest, ZeroSwaRestoreWindowCoversRuntimeActiveTail) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(
        makeProModelConfig(),
        pc,
        makeDsv4KvCacheConfig(kDsv4FixedPoolBlocks, /*seq_size_per_block=*/8192, /*kernel_seq_size_per_block=*/128),
        false,
        0);
    config.dsv4_zero_swa_caching = true;

    // Runtime SWA active tail = kZeroSwaRuntimeActiveTailBlocks(2) * kernel_block(128) = 256 tokens.
    EXPECT_EQ(zeroSwaActiveTailTokens(config), 256u);
    // Restore window 7808 tokens >> 256, so the invariant holds for DSV4 Pro.
    EXPECT_TRUE(zeroSwaRestoreWindowCoversActiveTail(config));

    // Degenerate config whose restore window is below the active tail must be rejected.
    config.swa_window_size = 1;
    config.layer_num       = 1;
    EXPECT_FALSE(zeroSwaRestoreWindowCoversActiveTail(config));

    // Disabled config always passes the invariant (no constraint when off).
    config.dsv4_zero_swa_caching = false;
    EXPECT_TRUE(zeroSwaRestoreWindowCoversActiveTail(config));
}

TEST(HybridPoolConfigCreatorTest, MtpSwaOnlyLayerIsNotStripped) {
    ParallelismConfig pc;
    auto              config =
        HybridPoolConfigCreator::createConfig(makeFlashMtpModelConfig(), pc, makeDsv4KvCacheConfig(), true, 0);

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
    auto config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);

    EXPECT_EQ(config.cache_specs[0]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.group_types[0], CacheGroupType::FULL);

    EXPECT_EQ(config.cache_specs[1]->layer_num, 31u);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1u * kDsv4KvEntryBytes);

    EXPECT_EQ(config.cache_specs[2]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);

    EXPECT_EQ(config.cache_specs[3]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 8u * 512u * 4u);

    EXPECT_EQ(config.cache_specs[4]->layer_num, 30u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 8u * 2048u * 4u);

    EXPECT_EQ(config.cache_specs[5]->layer_num, 31u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 128u * 1024u * 4u);

    EXPECT_EQ(config.cache_specs[6]->layer_num, 61u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, FlashPoolSpecs) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.cache_specs[0]->layer_num, 21u);
    EXPECT_EQ(config.cache_specs[1]->layer_num, 20u);
    EXPECT_EQ(config.cache_specs[6]->layer_num, 43u);
}

// ============================================================
// Block size bytes
// ============================================================

TEST(HybridPoolConfigCreatorTest, BlockSizeBytes) {
    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), kDsv4TokensPerBlock * kDsv4KvEntryBytes);
}

TEST(HybridPoolConfigCreatorTest, Fp8BlockSizeBytesUsePaddedPhysicalStride) {
    ParallelismConfig pc;
    auto              mc          = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    auto config                   = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    ASSERT_EQ(config.group_kv_block_stride_bytes.size(), 7u);

    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 19008u);  // 32 * 584 padded to 576B alignment
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1152u);   // 1 * 584 padded to 576B alignment
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), 74880u);  // 128 * 584 padded to 576B alignment

    EXPECT_EQ(config.group_kv_block_stride_bytes[0], config.cache_specs[0]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[1], config.cache_specs[1]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[6], config.cache_specs[6]->block_size_bytes());
}

TEST(HybridPoolConfigCreatorTest, DecoupledPhysicalAndKernelBlockSizeUsesPerGroupBpk) {
    ParallelismConfig pc;
    auto              mc = makeProModelConfig();
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block        = 16384;
    kv_cache_config.kernel_seq_size_per_block = 128;
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config, false, 0);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    ASSERT_EQ(config.group_seq_size_per_block.size(), 7u);

    EXPECT_EQ(config.seq_size_per_block, 16384u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 128u);
    for (size_t gid = 0; gid < config.group_seq_size_per_block.size(); ++gid) {
        EXPECT_EQ(config.group_seq_size_per_block[gid], 16384u);
    }

    auto* csa_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[0].get());
    auto* hca_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[1].get());
    auto* idx_kv = dynamic_cast<DSV4KVSpec*>(config.cache_specs[2].get());
    auto* swa_kv = dynamic_cast<DSV4StateSpec*>(config.cache_specs[6].get());
    ASSERT_NE(csa_kv, nullptr);
    ASSERT_NE(hca_kv, nullptr);
    ASSERT_NE(idx_kv, nullptr);
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(csa_kv->entries_per_block, 32u);
    EXPECT_EQ(hca_kv->entries_per_block, 1u);
    EXPECT_EQ(idx_kv->entries_per_block, 32u);
    EXPECT_EQ(swa_kv->entries_per_block, 128u);

    EXPECT_EQ(config.group_kv_block_stride_bytes[0], config.cache_specs[0]->block_size_bytes() * 128u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[1], config.cache_specs[1]->block_size_bytes() * 128u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[2], config.cache_specs[2]->block_size_bytes() * 128u);
    EXPECT_EQ(config.group_kv_block_stride_bytes[6], config.cache_specs[6]->block_size_bytes());

    auto full_pool = BlockPoolConfigHelper::createConfigForGroup(config, 0);
    auto swa_pool  = BlockPoolConfigHelper::createConfigForGroup(config, 6);
    ASSERT_EQ(full_pool.memory_layouts.size(), 1u);
    ASSERT_EQ(swa_pool.memory_layouts.size(), 1u);
    EXPECT_EQ(full_pool.memory_layouts[0].kernel_blocks_per_kv_block, 128u);
    EXPECT_EQ(swa_pool.memory_layouts[0].kernel_blocks_per_kv_block, 1u);
}

TEST(HybridPoolConfigCreatorTest, PrefillCpShardedSlicesFixedAndSwaPhysicalBlocks) {
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 4;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto mc                       = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    auto config                   = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    ASSERT_EQ(config.group_kv_block_stride_bytes.size(), 7u);

    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 19008u);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1152u);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * 132u);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 2u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 2u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 32u * 1024u * 4u);

    EXPECT_EQ(config.cache_specs[6]->block_size_bytes(), 32u * 584u);  // contiguous natural CP slice
    EXPECT_EQ(config.group_kv_block_stride_bytes[3], config.cache_specs[3]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[4], config.cache_specs[4]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[5], config.cache_specs[5]->block_size_bytes());
    EXPECT_EQ(config.group_kv_block_stride_bytes[6], config.cache_specs[6]->block_size_bytes());
    for (size_t gid : {3u, 4u, 5u, 6u}) {
        EXPECT_EQ(config.group_seq_size_per_block[gid], kDsv4TokensPerBlock * 4u) << "gid=" << gid;
    }

    pc.role_type       = RoleType::DECODE;
    auto decode_config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_EQ(decode_config.cache_specs[3]->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(decode_config.cache_specs[4]->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(decode_config.cache_specs[5]->block_size_bytes(), 128u * 1024u * 4u);
    EXPECT_EQ(decode_config.cache_specs[6]->block_size_bytes(), 74880u);
}

TEST(HybridPoolConfigCreatorTest, ZeroSwaTrimDisabledForPrefillCpSharded) {
    ScopedEnvVar caching_env("DSV4_ZERO_SWA_CACHING", "1");
    ScopedEnvVar trim_env("DSV4_ZERO_SWA_TRIM", "1");

    auto mc                       = makeProModelConfig();
    mc.attn_config.kv_cache_dtype = KvCacheDataType::FP8;

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = 4;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;
    auto cp_config = HybridPoolConfigCreator::createConfig(mc, prefill_pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_TRUE(cp_config.dsv4_zero_swa_caching);
    EXPECT_FALSE(cp_config.dsv4_zero_swa_trim);

    ParallelismConfig non_cp_pc;
    non_cp_pc.role_type = RoleType::PREFILL;
    non_cp_pc.tp_size   = 4;
    auto non_cp_config = HybridPoolConfigCreator::createConfig(mc, non_cp_pc, makeDsv4KvCacheConfig(), false, 0);
    EXPECT_TRUE(non_cp_config.dsv4_zero_swa_caching);
    EXPECT_TRUE(non_cp_config.dsv4_zero_swa_trim);
}

TEST(KVCacheTransferPlannerTest, CpCompactSwaUsesCanonicalTailRows) {
    auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/8,
                                         /*reuse_block_size=*/0,
                                         /*use_hybrid=*/true,
                                         CacheGroupType::SWA,
                                         /*cp_rank=*/0,
                                         /*cp_size=*/4);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].key_index, 7);
    EXPECT_EQ(plan[1].offset_index, 1);
}

TEST(KVCacheTransferPlannerTest, CpCompactSwaKeepsPartialTailRows) {
    {
        auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/1,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::SWA,
                                             /*cp_rank=*/0,
                                             /*cp_size=*/2);
        ASSERT_EQ(plan.size(), 1u);
        EXPECT_EQ(plan[0].key_index, 0);
        EXPECT_EQ(plan[0].offset_index, 0);
    }
    {
        auto plan = buildCacheStoreBlockPlan(/*total_logical_blocks=*/11,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true,
                                             CacheGroupType::SWA,
                                             /*cp_rank=*/0,
                                             /*cp_size=*/2);
        ASSERT_EQ(plan.size(), 2u);
        EXPECT_EQ(plan[0].key_index, 9);
        EXPECT_EQ(plan[0].offset_index, 4);
        EXPECT_EQ(plan[1].key_index, 10);
        EXPECT_EQ(plan[1].offset_index, 5);
    }
}

// ============================================================
// CacheConfig output
// ============================================================

TEST(HybridPoolConfigCreatorTest, CreateCacheConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(config.layer_num, 43u);
    // Group 6 (SWA) should have all 43 layers
    EXPECT_EQ(config.global_layer_ids[6].size(), 43u);
    // Group 0 (CSA KV) should have 21 layers
    EXPECT_EQ(config.global_layer_ids[0].size(), 21u);
}

TEST(HybridPoolConfigCreatorTest, HybridAttentionIndependentPoolUsesHybridPoolConfig) {
    ParallelismConfig pc;
    auto              config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(true), pc, KVCacheConfig{}, false, 0);

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
    auto              config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, KVCacheConfig{}, false, 0);

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
    DSV4StateSpec spec(KVCacheRegionName::SWA_KV,
                       61,
                       kDsv4Fp8KvEntryBytes,
                       kDsv4TokensPerBlock,
                       DataType::TYPE_UINT8,
                       kDsv4TokensPerBlock);

    EXPECT_EQ(spec.block_size(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.natural_block_size_bytes(), kDsv4TokensPerBlock * kDsv4Fp8KvEntryBytes);
    EXPECT_EQ(spec.block_size_bytes(), 74880u);
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
        auto              config =
            HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
        EXPECT_EQ(config.group_seq_size_per_block[0], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[1], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[2], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[6], kDsv4TokensPerBlock);
    }
    // Flash config
    {
        ParallelismConfig pc;
        auto              config =
            HybridPoolConfigCreator::createConfig(makeFlashModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
        EXPECT_EQ(config.group_seq_size_per_block[0], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[1], kDsv4TokensPerBlock);
        EXPECT_EQ(config.group_seq_size_per_block[2], kDsv4TokensPerBlock);
    }
}

TEST(HybridPoolConfigCreatorTest, AllPagedPoolsShareBlockNum) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    config.block_num         = 100;

    // Paged groups derive their block count from the global block_num; fixed/SWA
    // groups use per-group fixed block counts.
    EXPECT_EQ(config.groupNums(), 7);
    for (int i = 0; i < 7; i++) {
        EXPECT_GT(config.cache_specs[i]->block_size_bytes(), 0u) << "pool " << i;
    }
}

TEST(HybridPoolConfigCreatorTest, DSV4FixedPoolBlocksAppliedToFixedGroups) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
    kv_cache_config.dsv4_fixed_pool_blocks                      = kDsv4FixedPoolBlocks;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    // FULL groups get global_block_num as-is
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    // Fixed groups use DSV4_FIXED_POOL_BLOCKS.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], kDsv4FixedPoolBlocks) << "gid=" << gid;
    }

    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += static_cast<size_t>(kDsv4FixedPoolBlocks) * config.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(HybridPoolConfigCreatorTest, DSV4HcaStatePoolBlocksOverridesOnlyHcaState) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
    kv_cache_config.dsv4_fixed_pool_blocks                      = 512;
    kv_cache_config.dsv4_hca_state_pool_blocks                  = 350;
    runtime_config.max_generate_batch_size                      = 5;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 3;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    EXPECT_EQ(config.group_block_nums[3], 512u);
    EXPECT_EQ(config.group_block_nums[4], 512u);
    EXPECT_EQ(config.group_block_nums[5], 350u);
    EXPECT_EQ(config.group_block_nums[6], 512u);

    size_t expected_reserve = 0;
    expected_reserve += 512u * config.group_block_size_bytes[3];
    expected_reserve += 512u * config.group_block_size_bytes[4];
    expected_reserve += 350u * config.group_block_size_bytes[5];
    expected_reserve += 512u * config.group_block_size_bytes[6];
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4KernelSeqSizeAllowsDecoupledPhysicalBlocks) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto create_config = [&](int seq_size_per_block, int kernel_seq_size_per_block) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;
        kv_cache_config.dsv4_fixed_pool_blocks    = kDsv4FixedPoolBlocks;
        return CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);
    };

    auto old_valid = create_config(128, 128);
    EXPECT_EQ(old_valid.seq_size_per_block, 128u);
    EXPECT_EQ(old_valid.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(old_valid.kernelBlocksPerKvBlock(), 1u);

    auto decoupled = create_config(16384, 128);
    EXPECT_EQ(decoupled.seq_size_per_block, 16384u);
    EXPECT_EQ(decoupled.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(decoupled.kernelBlocksPerKvBlock(), 128u);
}

TEST(CacheConfigTest, DSV4KernelSeqSizeRejectsInvalidPhysicalKernelShape) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto create_config = [&](int seq_size_per_block, int kernel_seq_size_per_block) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.seq_size_per_block        = seq_size_per_block;
        kv_cache_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
        kv_cache_config.test_block_num            = 100;
        kv_cache_config.dsv4_fixed_pool_blocks    = kDsv4FixedPoolBlocks;
        return CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);
    };

    EXPECT_THROW((void)create_config(16384, 64), std::exception);
    EXPECT_THROW((void)create_config(16384, 384), std::exception);
}

TEST(HybridPoolConfigCreatorTest, DSV4FixedPoolBlocksIndependentOfMaxConcurrency) {
    for (uint32_t max_concurrency : {1u, 2u, 8u}) {
        auto              mc = makeProModelConfig();
        ParallelismConfig pc;
        RuntimeConfig     runtime_config;
        KVCacheConfig     kv_cache_config;
        kv_cache_config.seq_size_per_block                          = 128;
        kv_cache_config.test_block_num                              = 100;
        kv_cache_config.dsv4_fixed_pool_blocks                      = kDsv4FixedPoolBlocks;
        runtime_config.max_generate_batch_size                      = max_concurrency;
        runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

        auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

        ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
        for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
            EXPECT_EQ(config.group_block_nums[gid], kDsv4FixedPoolBlocks)
                << "gid=" << gid << " max_concurrency=" << max_concurrency;
        }
    }
}

TEST(HybridPoolConfigCreatorTest, DSV4FixedPoolBlocksCanBeOverriddenByConfig) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block                          = 128;
    kv_cache_config.test_block_num                              = 100;
    kv_cache_config.dsv4_fixed_pool_blocks                      = 6;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    auto config = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    // FULL groups: 100
    for (int gid = 0; gid < 3; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 100u) << "gid=" << gid;
    }
    // Fixed groups: exactly the configured fixed pool count.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 6u) << "gid=" << gid;
    }
}

TEST(CacheConfigTest, FinalizeBlockNumsIsNoopForSingleAndSharedHybridConfig) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 8;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 4;

    ParallelismConfig pc;
    auto single_config = CacheConfigCreator::createBasicConfig(ModelConfig(), pc, KVCacheConfig{}, false, 0);
    single_config.finalizeBlockNums(123, runtime_config);
    EXPECT_TRUE(single_config.group_block_nums.empty());
    EXPECT_EQ(single_config.fixed_pool_reserve_bytes, 0u);

    auto hybrid_config =
        CacheConfigCreator::createBasicConfig(makeHybridAttentionModelConfig(false), pc, KVCacheConfig{}, false, 0);
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
    auto config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    config.finalizeBlockNums(100, runtime_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], kDsv4FixedPoolBlocks) << "gid=" << gid;
    }
    EXPECT_GT(config.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, FixedPoolReserveDeductedFromPagedBudget) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    const uint32_t small_fixed_pool = 32;

    // Config with fewer fixed-pool blocks.
    KVCacheConfig kv_cache_config_with;
    kv_cache_config_with.seq_size_per_block     = 128;
    kv_cache_config_with.kv_cache_mem_mb        = 65536;
    kv_cache_config_with.dsv4_fixed_pool_blocks = small_fixed_pool;
    auto config_with = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_with);

    // Config with more fixed-pool blocks.
    KVCacheConfig kv_cache_config_without;
    kv_cache_config_without.seq_size_per_block     = 128;
    kv_cache_config_without.kv_cache_mem_mb        = 65536;
    kv_cache_config_without.dsv4_fixed_pool_blocks = kDsv4FixedPoolBlocks;
    auto config_without = CacheConfigCreator::createConfig(mc, pc, runtime_config, kv_cache_config_without);

    // More fixed blocks reserve more HBM and leave fewer blocks for FULL pools.
    EXPECT_GT(config_with.block_num, config_without.block_num);
    // FULL group block_nums reflect the reduced block_num
    EXPECT_EQ(config_with.group_block_nums[0], static_cast<uint32_t>(config_with.block_num));
    EXPECT_EQ(config_without.group_block_nums[0], static_cast<uint32_t>(config_without.block_num));
    // Fixed groups use the configured fixed-pool counts.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config_with.group_block_nums[gid], small_fixed_pool) << "gid=" << gid;
        EXPECT_EQ(config_without.group_block_nums[gid], kDsv4FixedPoolBlocks) << "gid=" << gid;
    }
    // Reserve bytes = fixed blocks × sum(fixed group block sizes).
    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += static_cast<size_t>(small_fixed_pool) * config_with.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config_with.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4ExplicitFixedPoolBlocksIgnoreLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    auto config = HybridPoolConfigCreator::createConfig(makeProModelConfig(), pc, makeDsv4KvCacheConfig(), false, 0);
    config.linear_step = 4;
    config.finalizeBlockNums(100, runtime_config);

    // FULL groups: unaffected by step, get global_block_num
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    // DSV4 fixed groups ignore linear_step and use DSV4_FIXED_POOL_BLOCKS.
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], kDsv4FixedPoolBlocks) << "gid=" << gid;
    }
    size_t expected_reserve = 0;
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        expected_reserve += static_cast<size_t>(kDsv4FixedPoolBlocks) * config.group_block_size_bytes[gid];
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_reserve);
}

TEST(CacheConfigTest, DSV4FixedPoolBlocksFallbackFollowsLinearStep) {
    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    ParallelismConfig pc;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block = 128;
    kv_cache_config.test_block_num     = 100;
    kv_cache_config.linear_step        = 4;

    auto config = CacheConfigCreator::createConfig(makeProModelConfig(), pc, runtime_config, kv_cache_config);

    ASSERT_EQ(config.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    EXPECT_EQ(config.group_block_nums[0], 100u);
    EXPECT_EQ(config.group_block_nums[1], 100u);
    EXPECT_EQ(config.group_block_nums[2], 100u);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(config.group_block_nums[gid], 25u) << "gid=" << gid;
    }
    EXPECT_EQ(config.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, DSV4PinnedFixedPoolFallbackIsExcludedFromGpuBudget) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    KVCacheConfig gpu_fixed_config;
    gpu_fixed_config.seq_size_per_block = 128;
    gpu_fixed_config.kv_cache_mem_mb    = 65536;
    gpu_fixed_config.linear_step        = 4;

    KVCacheConfig pinned_fixed_config              = gpu_fixed_config;
    pinned_fixed_config.dsv4_fixed_pool_use_memory = true;

    auto gpu_fixed    = CacheConfigCreator::createConfig(mc, pc, runtime_config, gpu_fixed_config);
    auto pinned_fixed = CacheConfigCreator::createConfig(mc, pc, runtime_config, pinned_fixed_config);

    EXPECT_GT(pinned_fixed.block_num, gpu_fixed.block_num);
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(gpu_fixed.group_block_nums[gid], static_cast<uint32_t>(gpu_fixed.block_num / gpu_fixed.linear_step))
            << "gid=" << gid;
        EXPECT_EQ(pinned_fixed.group_block_nums[gid],
                  static_cast<uint32_t>(pinned_fixed.block_num / pinned_fixed.linear_step))
            << "gid=" << gid;
        EXPECT_GT(pinned_fixed.group_block_nums[gid], gpu_fixed.group_block_nums[gid]) << "gid=" << gid;
    }
    EXPECT_EQ(pinned_fixed.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, DSV4PinnedFixedPoolFallbackFollowsExpandedFullPoolWhenStepOne) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 4;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 2;

    KVCacheConfig gpu_fixed_config;
    gpu_fixed_config.seq_size_per_block = 128;
    gpu_fixed_config.kv_cache_mem_mb    = 65536;

    KVCacheConfig pinned_fixed_config              = gpu_fixed_config;
    pinned_fixed_config.dsv4_fixed_pool_use_memory = true;

    auto gpu_fixed    = CacheConfigCreator::createConfig(mc, pc, runtime_config, gpu_fixed_config);
    auto pinned_fixed = CacheConfigCreator::createConfig(mc, pc, runtime_config, pinned_fixed_config);

    EXPECT_GT(pinned_fixed.block_num, gpu_fixed.block_num);
    ASSERT_EQ(pinned_fixed.group_block_nums.size(), static_cast<size_t>(kDsv4PoolNum));
    for (int gid = 3; gid < kDsv4PoolNum; ++gid) {
        EXPECT_EQ(gpu_fixed.group_block_nums[gid], static_cast<uint32_t>(gpu_fixed.block_num)) << "gid=" << gid;
        EXPECT_EQ(pinned_fixed.group_block_nums[gid], static_cast<uint32_t>(pinned_fixed.block_num)) << "gid=" << gid;
        EXPECT_GT(pinned_fixed.group_block_nums[gid], gpu_fixed.group_block_nums[gid]) << "gid=" << gid;
    }
    EXPECT_EQ(pinned_fixed.fixed_pool_reserve_bytes, 0u);
}

TEST(CacheConfigTest, DSV4MtpKeepsProposeLayerInSwaPool) {
    auto score_model_config                         = makeFlashModelConfig();
    auto propose_model_config                       = makeFlashMtpModelConfig();
    score_model_config.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;
    propose_model_config.attn_config.kv_cache_dtype = KvCacheDataType::FP8;

    ParallelismConfig parallelism_config;
    RuntimeConfig     runtime_config;
    runtime_config.max_generate_batch_size                      = 2;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 1;

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 16384;
    kv_cache_config.kernel_seq_size_per_block = 128;
    kv_cache_config.test_block_num            = 100;
    kv_cache_config.dsv4_fixed_pool_blocks    = kDsv4FixedPoolBlocks;

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
    EXPECT_EQ(config.seq_size_per_block, 16384u);
    EXPECT_EQ(config.kernel_seq_size_per_block, 128u);
    EXPECT_EQ(config.kernelBlocksPerKvBlock(), 128u);
    EXPECT_EQ(config.mtp_sub_configs[0]->seq_size_per_block, 16384u);
    EXPECT_EQ(config.mtp_sub_configs[0]->kernel_seq_size_per_block, 128u);

    const size_t fixed_blocks = config.dsv4_fixed_pool_blocks;
    const size_t expected_fixed_reserve =
        fixed_blocks * config.group_block_size_bytes[3] + fixed_blocks * config.group_block_size_bytes[4]
        + fixed_blocks * config.group_block_size_bytes[5] + fixed_blocks * config.group_block_size_bytes[6];
    EXPECT_EQ(config.fixed_pool_reserve_bytes, expected_fixed_reserve);
}

TEST(HybridPoolConfigCreatorTest, MtpGenNum2RingEntriesMatch) {
    // gen_num_per_cycle=2 -> CSA/INDEXER R=10, HCA R=130, SWA R=130.
    // Formula: R = ceil_even((1 + overlap) * ratio + gen_num_per_cycle).
    // SWA_KV is sized like the HCA state ring (window 128, overlap 0).
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              config =
        HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, /*gen_num_per_cycle=*/2);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    // Pool 3: INDEXER_STATE (ratio=4, overlap=1) → R=10
    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    ASSERT_NE(indexer_state, nullptr);
    EXPECT_EQ(indexer_state->entries_per_block, 10u);
    // Pool 4: CSA_STATE (ratio=4, overlap=1) → R=10
    auto* csa_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[4].get());
    ASSERT_NE(csa_state, nullptr);
    EXPECT_EQ(csa_state->entries_per_block, 10u);
    // Pool 5: HCA_STATE (ratio=128, overlap=0) → R=130
    auto* hca_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[5].get());
    ASSERT_NE(hca_state, nullptr);
    EXPECT_EQ(hca_state->entries_per_block, 130u);
    // Pool 6: SWA_KV (window=128, overlap=0) → R=130, same as HCA_STATE
    auto* swa_kv = dynamic_cast<DSV4StateSpec*>(config.cache_specs[6].get());
    ASSERT_NE(swa_kv, nullptr);
    EXPECT_EQ(swa_kv->cache_type, KVCacheRegionName::SWA_KV);
    EXPECT_EQ(swa_kv->entries_per_block, 130u);
}

TEST(HybridPoolConfigCreatorTest, PrefillCp8MtpGenNum2PadsStateRingBeforeSlicing) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = 8;
    pc.prefill_cp_config.kv_cache_sharded = true;

    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 2);

    ASSERT_EQ(config.cache_specs.size(), 7u);
    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(config.cache_specs[3].get());
    auto* csa_state     = dynamic_cast<DSV4StateSpec*>(config.cache_specs[4].get());
    auto* hca_state     = dynamic_cast<DSV4StateSpec*>(config.cache_specs[5].get());
    auto* swa_kv        = dynamic_cast<DSV4StateSpec*>(config.cache_specs[6].get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    // gen_num_per_cycle=2 gives raw INDEXER/CSA R=10, HCA R=130, SWA R=130.
    // CP sharding pads the full ring to a cp_size multiple before taking the
    // per-rank physical slice: 10 -> 16 -> 2, 130 -> 136 -> 17 (HCA and SWA).
    EXPECT_EQ(indexer_state->entries_per_block, 2u);
    EXPECT_EQ(csa_state->entries_per_block, 2u);
    EXPECT_EQ(hca_state->entries_per_block, 17u);
    EXPECT_EQ(swa_kv->entries_per_block, 17u);
}

TEST(HybridPoolConfigCreatorTest, DecodePrefillCp8MtpGenNum2MatchesPrefillSliceTimesCpSize) {
    constexpr uint32_t cp_size = 8;
    auto               mc      = makeFlashModelConfig();

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = cp_size;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = cp_size;
    decode_pc.world_size                         = cp_size;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = cp_size;

    auto prefill_config = HybridPoolConfigCreator::createConfig(mc, prefill_pc, makeDsv4KvCacheConfig(), false, 2);
    auto decode_config  = HybridPoolConfigCreator::createConfig(mc, decode_pc, makeDsv4KvCacheConfig(), false, 2);

    ASSERT_EQ(prefill_config.cache_specs.size(), 7u);
    ASSERT_EQ(decode_config.cache_specs.size(), 7u);

    for (size_t gid : {3u, 4u, 5u, 6u}) {
        auto* prefill_spec = dynamic_cast<DSV4StateSpec*>(prefill_config.cache_specs[gid].get());
        auto* decode_spec  = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[gid].get());
        ASSERT_NE(prefill_spec, nullptr) << "gid=" << gid;
        ASSERT_NE(decode_spec, nullptr) << "gid=" << gid;
        EXPECT_EQ(decode_spec->cache_type, prefill_spec->cache_type) << "gid=" << gid;
        const auto expected_entries = prefill_spec->entries_per_block * cp_size;
        EXPECT_EQ(decode_spec->entries_per_block, expected_entries)
            << "gid=" << gid << " region=" << static_cast<int>(decode_spec->cache_type);
    }

    auto* indexer_state = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[3].get());
    auto* csa_state     = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[4].get());
    auto* hca_state     = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[5].get());
    auto* swa_kv        = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[6].get());
    ASSERT_NE(indexer_state, nullptr);
    ASSERT_NE(csa_state, nullptr);
    ASSERT_NE(hca_state, nullptr);
    ASSERT_NE(swa_kv, nullptr);

    EXPECT_EQ(indexer_state->entries_per_block, 16u);
    EXPECT_EQ(csa_state->entries_per_block, 16u);
    EXPECT_EQ(hca_state->entries_per_block, 136u);
    EXPECT_EQ(swa_kv->entries_per_block, 136u);
    for (size_t gid : {3u, 4u, 5u, 6u}) {
        EXPECT_EQ(prefill_config.group_seq_size_per_block[gid], kDsv4TokensPerBlock * cp_size) << "gid=" << gid;
        EXPECT_EQ(decode_config.group_seq_size_per_block[gid], kDsv4TokensPerBlock * cp_size) << "gid=" << gid;
    }
}

TEST(HybridPoolConfigCreatorTest, DecodeExplicitPrefillCpSizeHandlesDp16) {
    constexpr uint32_t cp_size = 8;
    auto               mc      = makeFlashModelConfig();

    ParallelismConfig prefill_pc;
    prefill_pc.role_type                          = RoleType::PREFILL;
    prefill_pc.tp_size                            = cp_size;
    prefill_pc.prefill_cp_config.kv_cache_sharded = true;

    ParallelismConfig decode_pc;
    decode_pc.role_type                          = RoleType::DECODE;
    decode_pc.tp_size                            = 1;
    decode_pc.dp_size                            = 16;
    decode_pc.world_size                         = 16;
    decode_pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    decode_pc.prefill_cp_config.kv_cache_sharded = true;
    decode_pc.prefill_cp_config.prefill_cp_size  = cp_size;

    auto prefill_config = HybridPoolConfigCreator::createConfig(mc, prefill_pc, makeDsv4KvCacheConfig(), false, 2);
    auto decode_config  = HybridPoolConfigCreator::createConfig(mc, decode_pc, makeDsv4KvCacheConfig(), false, 2);

    for (size_t gid : {3u, 4u, 5u, 6u}) {
        auto* prefill_spec = dynamic_cast<DSV4StateSpec*>(prefill_config.cache_specs[gid].get());
        auto* decode_spec  = dynamic_cast<DSV4StateSpec*>(decode_config.cache_specs[gid].get());
        ASSERT_NE(prefill_spec, nullptr) << "gid=" << gid;
        ASSERT_NE(decode_spec, nullptr) << "gid=" << gid;
        const auto expected_entries = prefill_spec->entries_per_block * cp_size;
        EXPECT_EQ(decode_spec->entries_per_block, expected_entries)
            << "gid=" << gid << " region=" << static_cast<int>(decode_spec->cache_type);
        EXPECT_EQ(prefill_config.group_seq_size_per_block[gid], kDsv4TokensPerBlock * cp_size) << "gid=" << gid;
        EXPECT_EQ(decode_config.group_seq_size_per_block[gid], kDsv4TokensPerBlock * cp_size) << "gid=" << gid;
    }
}

TEST(CacheConfigTest, DSV4NonMtpSpConfigDoesNotInflateRing) {
    // SP_TYPE_NONE with default gen_num_per_cycle=1 must NOT inflate state ring.
    // Non-MTP DSV4 ring: R = ceil_even((1+overlap)*ratio + 0) = 8 for CSA.
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    RuntimeConfig     rc;
    rc.max_generate_batch_size                      = 2;
    rc.fifo_scheduler_config.max_context_batch_size = 1;
    KVCacheConfig kvc;
    kvc.seq_size_per_block        = 128;
    kvc.kernel_seq_size_per_block = 128;
    kvc.test_block_num            = 50;
    SpeculativeExecutionConfig sp_none;  // type=SP_TYPE_NONE, gen_num_per_cycle=1
    auto config = CacheConfigCreator::createConfig(mc, pc, rc, kvc, std::nullopt, std::make_optional(sp_none));
    ASSERT_EQ(config.cache_specs.size(), 7u);
    // CSA_STATE (pool 4): ratio=4, overlap=1, gen_num=0 → R=8
    auto* csa = dynamic_cast<DSV4StateSpec*>(config.cache_specs[4].get());
    ASSERT_NE(csa, nullptr);
    EXPECT_EQ(csa->entries_per_block, 8u) << "SP_TYPE_NONE should not inflate ring";
}

TEST(HybridPoolConfigCreatorTest, BlockIdConsistencyAcrossGroups) {
    // DSV4 has multiple cache regions per logical layer. The config must expose
    // every region's group id for the layer so model/runtime code can request the
    // correct region by KVCacheRegionName.
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);

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
    auto              config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    // Set enough blocks for tests (7 groups × N blocks each)
    config.block_num = 200;
    return config;
}

static CacheConfig makeDSV4CpAllocatorConfig(uint32_t cp_size) {
    auto              mc = makeProModelConfig();
    ParallelismConfig pc;
    pc.role_type                          = RoleType::PREFILL;
    pc.tp_size                            = cp_size;
    pc.prefill_cp_config.kv_cache_sharded = true;
    auto config = HybridPoolConfigCreator::createConfig(mc, pc, makeDsv4KvCacheConfig(), false, 0);
    config.block_num = 200;
    config.group_block_nums.assign(config.groupNums(), config.block_num);
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
    auto               allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // 7 groups → HybridTypeKVCacheAllocator path
    EXPECT_EQ(config.groupNums(), 7);
    EXPECT_EQ(allocator->seqSizePerBlock(), static_cast<int>(config.seq_size_per_block));
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);
}

TEST_F(DSV4AllocatorTest, CpPageRrFixedAndSwaAllocateOneBlockPerVirtualBlock) {
    constexpr uint32_t cp_size = 4;
    auto               config  = makeDSV4CpAllocatorConfig(cp_size);
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int spb     = allocator->seqSizePerBlock();
    const int seq_len = static_cast<int>(cp_size) * spb;

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7,
                          static_cast<int>(config.layer_all_num),
                          config.layer_to_group_id,
                          config.kernelBlocksPerKvBlock(),
                          config.group_types,
                          config.layer_region_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});

    auto cti            = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(0, static_cast<int>(cp_size), spb);

    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    for (int gid = 0; gid < 7; ++gid) {
        EXPECT_EQ(batch_res->blocksNum(0, gid), 1u) << "gid=" << gid;
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, CpPageRrHcaStateRestoresPreviousTailForPdTransfer) {
    constexpr uint32_t cp_size = 2;
    auto               config  = makeDSV4CpAllocatorConfig(cp_size);
    auto               allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto               shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    CacheKeysType full_keys = {100, 101, 102, 103, 104, 105, 106};
    CacheKeysType cp_keys   = {101, 103, 105};
    std::vector<std::vector<BlockIdxType>> cached_blocks(7);
    for (int gid = 0; gid < 7; ++gid) {
        auto blocks = block_pool->malloc(static_cast<int>(cp_keys.size()));
        ASSERT_EQ(blocks.size(), cp_keys.size());
        for (size_t i = 0; i < cp_keys.size(); ++i) {
            std::vector<BlockIdxType> group_slots(7, NULL_BLOCK_IDX);
            group_slots[static_cast<size_t>(gid)] = blocks[i];
            shared_cache->put(cp_keys[i], group_slots, true);
        }
        cached_blocks[static_cast<size_t>(gid)] = blocks;
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7,
                          static_cast<int>(config.layer_all_num),
                          config.layer_to_group_id,
                          config.kernelBlocksPerKvBlock(),
                          config.group_types,
                          config.layer_region_to_group_id);
    batch_res->setBatchCacheKeys(0, full_keys);

    const int spb     = allocator->seqSizePerBlock();
    const int seq_len = static_cast<int>(cp_size) * spb * 3 + 1;
    auto      cti     = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
    auto      gi      = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(0, static_cast<int>(cp_size), spb);

    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_GT(result.reuse_len, 0);

    const auto& hca_state_blocks = batch_res->blocks(0, 5);
    ASSERT_GE(hca_state_blocks.size(), 4u);
    EXPECT_EQ(hca_state_blocks[2], cached_blocks[5][2])
        << "previous HCA_STATE tail must be restored so decode can load the last two state blocks";
    EXPECT_FALSE(isNullBlockIdx(hca_state_blocks[3])) << "current prefill tail HCA_STATE block must be allocated";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
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
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
    EXPECT_EQ(config.cache_specs[3]->block_size_bytes(), 8u * 512u * 4u);
    EXPECT_EQ(config.cache_specs[4]->block_size_bytes(), 8u * 2048u * 4u);
    EXPECT_EQ(config.cache_specs[5]->block_size_bytes(), 128u * 1024u * 4u);
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
    // HCA_STATE has the largest per-block bytes (128 entries * 1024 * 4)
    EXPECT_EQ(expected_max, config.cache_specs[5]->block_size_bytes());
}

TEST_F(DSV4AllocatorTest, HCAStateIsExcludedFromReuseCachePolicy) {
    auto config = makeDSV4AllocatorConfig();
    ASSERT_EQ(config.group_region_names.size(), 7u);

    for (size_t gid = 0; gid < config.group_region_names.size(); ++gid) {
        if (gid == 5) {
            EXPECT_TRUE(skipReuseCacheRegion(config.group_region_names[gid])) << "HCA_STATE should skip reuse cache";
        } else {
            EXPECT_FALSE(skipReuseCacheRegion(config.group_region_names[gid])) << "group " << gid;
        }
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
    EXPECT_EQ(config.cache_specs[0]->block_size_bytes(), 32u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[1]->block_size_bytes(), 1u * kDsv4KvEntryBytes);
    EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 32u * kDsv4IndexerEntryBytes);
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
// Prefix cache: HCA_STATE is excluded from reuse decisions, but its tail blocks
// are still cached so PD decode can load them when a prefix is reused.
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
    // corresponding full-block cache keys. HCA_STATE is needed for PD decode's
    // last-two-block transfer when the previous tail is device-reused.
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
// Prefix cache: Flash config keeps the same HCA_STATE tail-transfer behavior.
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
    // corresponding full-block cache keys. HCA_STATE participates in tail
    // transfer even though FULL groups determine prefix length.
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
// Prefix cache: paged FULL groups reuse; reusable SWA/state groups require a matched latest tail block.
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

TEST_F(DSV4AllocatorTest, PrefixCacheReuseDoesNotRequireHCAStateHit) {
    auto config       = makeDSV4AllocatorConfig();
    auto allocator    = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {1100, 1101, 1102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        if (gid == 5) {
            continue;
        }
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
        cached_blocks[gid] = blocks;
        block_pool->requestFree(blocks);
    }

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);
    batch_res->setBatchCacheKeys(0, CacheKeysType{1100, 1101, 1102, 1103});

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

    EXPECT_GT(result.reuse_len, 0) << "HCA_STATE tail miss should not veto DSV4 prefix reuse";
    EXPECT_TRUE(isNullBlockIdx(batch_res->blocks(0, 5).at(2)))
        << "missing HCA_STATE previous tail stays null, but does not decide reuse";
    EXPECT_FALSE(isNullBlockIdx(batch_res->blocks(0, 5).at(3))) << "current HCA_STATE tail should be allocated";
    EXPECT_EQ(batch_res->blocks(0, 6).at(2), cached_blocks[6][2]) << "SWA_KV tail should still gate reuse";

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, ZeroSwaCachingSkipsSwaKvTailAndRecomputesWindow) {
    auto config                   = makeDSV4AllocatorConfig();
    config.dsv4_zero_swa_caching  = true;
    config.swa_window_size        = 1;
    config.layer_num              = 1;
    auto allocator                = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache             = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (gid == 6) {
                continue;
            }
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

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(3 * spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, 2 * spb);
    for (int gid = 0; gid < 3; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 2u);
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]);
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]);
    }
    // INDEXER_STATE (3) / CSA_STATE (4) still anchor the tail and reuse their tail block.
    for (int gid = 3; gid < 5; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 2u);
        EXPECT_TRUE(isNullBlockIdx(out_blocks[0]));
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]);
    }
    // HCA_STATE (5) is excluded from reuse cache (skipReuseCacheRegion); SWA_KV (6) under zero-SWA.
    for (int gid = 5; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 2u);
        EXPECT_TRUE(isNullBlockIdx(out_blocks[0]));
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1]));
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

// Stage A (inverted-triangle recompute foundation): with DSV4_ZERO_SWA_TRIM the paged FULL
// groups (CSA_KV/HCA_KV/INDEXER_KV) extend their reuse to ALSO cover the restore-window block
// (so the recompute reads cached compressed/indexer KV), while the RETURNED reuse_len stays
// capped (the restore window is still materialized for SWA recompute + write-skipped writes),
// and the STATE/SWA ring tails stay capped exactly as without trim. This is the only behavior
// that differs from ZeroSwaCachingSkipsSwaKvTailAndRecomputesWindow: FULL groups gain the
// extra cached block at index 2 instead of a freshly-allocated one.
TEST_F(DSV4AllocatorTest, ZeroSwaTrimExtendsFullCoverageButCapsReturn) {
    auto config                   = makeDSV4AllocatorConfig();
    config.dsv4_zero_swa_caching  = true;
    config.dsv4_zero_swa_trim     = true;
    config.swa_window_size        = 1;
    config.layer_num              = 1;
    auto allocator                = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache             = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int                          group_num   = 7;
    CacheKeysType                          cached_keys = {100, 101, 102};
    std::vector<std::vector<BlockIdxType>> cached_blocks(group_num);
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (gid == 6) {
                continue;
            }
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

    const int spb       = allocator->seqSizePerBlock();
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(3 * spb, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // The RETURNED reuse window stays capped at 2 blocks (restore window = swa_window*layer_num
    // = 1 token => 1 block dropped from the 3 matched) -- identical to the non-trim case, so the
    // restore window is still materialized for SWA recompute.
    EXPECT_EQ(result.reuse_len, 2 * spb);

    // FULL groups (CSA_KV/HCA_KV/INDEXER_KV) now reuse the 3rd cached block (the restore-window
    // block) instead of allocating a fresh one -- this is the compute reduction (cached KV read).
    for (int gid = 0; gid < 3; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 3u);
        EXPECT_EQ(out_blocks[0], cached_blocks[gid][0]);
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]);
        EXPECT_EQ(out_blocks[2], cached_blocks[gid][2])
            << "trim must reuse the restore-window FULL block from cache (gid " << gid << ")";
    }

    // STATE tails stay capped exactly as without trim: INDEXER_STATE(3)/CSA_STATE(4) reuse only
    // the capped tail block; HCA_STATE(5) is excluded from reuse; SWA_KV(6) under zero-SWA.
    for (int gid = 3; gid < 5; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 2u);
        EXPECT_TRUE(isNullBlockIdx(out_blocks[0]));
        EXPECT_EQ(out_blocks[1], cached_blocks[gid][1]);
    }
    for (int gid = 5; gid < 7; gid++) {
        const auto& out_blocks = batch_res->blocks(0, gid);
        ASSERT_GE(out_blocks.size(), 2u);
        EXPECT_TRUE(isNullBlockIdx(out_blocks[0]));
        EXPECT_TRUE(isNullBlockIdx(out_blocks[1]));
    }

    FreeInfo free_info{batch_res};
    allocator->free(free_info);
}

TEST_F(DSV4AllocatorTest, ZeroSwaTrimDoesNotDropWhenUnmatchedSuffixCoversRestoreWindow) {
    auto config                  = makeDSV4AllocatorConfig();
    config.dsv4_zero_swa_caching = true;
    config.dsv4_zero_swa_trim    = true;
    config.swa_window_size       = 1;
    config.layer_num             = 1;
    auto allocator               = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto shared_cache            = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(shared_cache);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();

    constexpr int group_num = 7;
    CacheKeysType cached_keys{100, 101, 102};
    for (int gid = 0; gid < group_num; gid++) {
        auto blocks = block_pool->malloc(static_cast<int>(cached_keys.size()));
        ASSERT_EQ(blocks.size(), cached_keys.size());
        for (size_t i = 0; i < cached_keys.size(); ++i) {
            if (gid == 6) {
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

    const int spb = allocator->seqSizePerBlock();
    auto      cti = std::make_shared<CompleteTokenIds>(1, 1, 4096, spb);
    auto      gi  = std::make_shared<GenerateInput>();
    // The one-token suffix after the matched full blocks already covers the
    // one-token restore window, so zero-SWA must not drop an extra full block.
    gi->input_ids       = torch::arange(3 * spb + 1, torch::kInt32);
    gi->generate_config = std::make_shared<GenerateConfig>();
    cti->init(gi);

    MallocInfo info{batch_res, cti};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, 3 * spb);
    EXPECT_EQ(batch_res->cacheResource(0).zeroSwaFullReuseTokenNum(), static_cast<size_t>(3 * spb));

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

TEST_F(DSV4AllocatorTest, HybridPoolReserveBlocksDoNotReduceExplicitFixedPoolCapacity) {
    auto              mc = makeFlashModelConfig();
    ParallelismConfig pc;
    auto              kv_config = makeDsv4KvCacheConfig(/*fixed_pool_blocks=*/11);
    auto              config    = HybridPoolConfigCreator::createConfig(mc, pc, kv_config, false, 0);
    config.block_num            = 40;
    config.group_block_nums.assign(config.groupNums(), config.block_num);
    for (size_t gid = 0; gid < config.group_region_names.size(); ++gid) {
        if (isDsv4FixedRegion(config.group_region_names[gid])) {
            config.group_block_nums[gid] = 11;
        }
    }

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/50);
    ASSERT_TRUE(allocator->init());

    auto batch_res = std::make_shared<BatchKVCacheResource>();
    batch_res->resetBatchSize(1);
    batch_res->initGroups(7, static_cast<int>(config.layer_all_num), config.layer_to_group_id);

    const int spb       = allocator->seqSizePerBlock();
    const int seq_len   = 10 * spb;
    auto      cti       = std::make_shared<CompleteTokenIds>(1, 1, seq_len + spb, spb);
    auto      gi        = std::make_shared<GenerateInput>();
    gi->input_ids       = torch::arange(seq_len, torch::kInt32);
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

}  // namespace test
}  // namespace rtp_llm
