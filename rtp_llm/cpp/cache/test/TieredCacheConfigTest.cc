#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include <stdexcept>
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TieredCacheBudget budget() {
    TieredCacheBudget b;
    b.gpu_bytes            = 2000;
    b.host_bytes           = 1000;
    b.offload_block_bytes  = 100;
    b.device_block_bytes   = 10;
    b.metadata_block_bytes = 2;
    b.fixed_gpu_bytes      = 80;
    b.max_selected_rows    = 8;
    return b;
}

TEST(TieredCacheConfigTest, HostBudgetIsAnUpperBound) {
    const auto result = solveTieredCacheCapacity(budget());
    EXPECT_EQ(result.config.total_blocks, 26u);
    EXPECT_EQ(result.config.hbm_blocks, 16u);
    EXPECT_EQ(result.host_bytes, 1000u);
    EXPECT_EQ(result.gpu_bytes, 1992u);
}

TEST(TieredCacheConfigTest, IndexerCanLimitHostCapacity) {
    auto b               = budget();
    b.device_block_bytes = 500;
    const auto result    = solveTieredCacheCapacity(b);
    EXPECT_EQ(result.config.hbm_blocks, 2u);
    EXPECT_EQ(result.config.total_blocks, 3u);
    EXPECT_EQ(result.host_bytes, 100u);
    EXPECT_EQ(result.gpu_bytes, 1786u);
}

TEST(TieredCacheConfigTest, NoHostBlocksWhenBudgetIsTooSmall) {
    auto b = budget();
    for (size_t host : {0u, 99u}) {
        b.host_bytes      = host;
        const auto result = solveTieredCacheCapacity(b);
        EXPECT_EQ(result.config.total_blocks, 17u);
        EXPECT_EQ(result.config.hbm_blocks, 17u);
        EXPECT_EQ(result.host_bytes, 0u);
    }
}

TEST(TieredCacheConfigTest, ResidentCapacityIsDerivedFromSelection) {
    TieredCacheConfig config{0, 0, 8};
    EXPECT_EQ(config.residentRows(4), 8u);
    config.max_selected_rows = 9;
    EXPECT_EQ(config.residentRows(4), 12u);
    EXPECT_EQ(config.max_selected_rows, 9u);
    config.max_selected_rows = std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(config.residentRows(4), uint64_t{1} << 32);

    auto b              = budget();
    b.max_selected_rows = 0;
    EXPECT_THROW(solveTieredCacheCapacity(b), std::invalid_argument);
}

TEST(TieredCacheConfigTest, TwoHbmBlocksIncludeSentinel) {
    auto b            = budget();
    b.test_hbm_blocks = 1;
    EXPECT_THROW(solveTieredCacheCapacity(b), std::invalid_argument);
    b.test_hbm_blocks = 0;
    b.gpu_bytes       = 303;
    EXPECT_THROW(solveTieredCacheCapacity(b), std::invalid_argument);
    b.gpu_bytes       = 304;
    const auto result = solveTieredCacheCapacity(b);
    EXPECT_EQ(result.config.total_blocks, 2u);
    EXPECT_EQ(result.config.hbm_blocks, 2u);
}

TEST(TieredCacheConfigTest, TestHbmOverrideIsExact) {
    auto b            = budget();
    b.test_hbm_blocks = 3;
    const auto result = solveTieredCacheCapacity(b);
    EXPECT_EQ(result.config.hbm_blocks, 3u);
    EXPECT_EQ(result.config.total_blocks, 13u);
}

TEST(TieredCacheConfigTest, SwaUsesRoundedUpGlobalCapacity) {
    auto b            = budget();
    b.swa_block_bytes = 100;
    b.swa_step        = 4;
    const auto result = solveTieredCacheCapacity(b);
    EXPECT_EQ(result.config.hbm_blocks, 10u);
    EXPECT_EQ(result.config.total_blocks, 20u);
    EXPECT_EQ(result.swa_bytes, 500u);
    EXPECT_EQ(result.gpu_bytes, 1820u);
}

TEST(TieredCacheConfigTest, CountsAndProductsDoNotOverflow) {
    auto b      = budget();
    b.gpu_bytes = b.host_bytes = std::numeric_limits<size_t>::max();
    b.offload_block_bytes      = b.gpu_bytes / 2 + 1;
    EXPECT_THROW(solveTieredCacheCapacity(b), std::invalid_argument);
    b.offload_block_bytes = 1;
    b.max_blocks          = 25;
    const auto result     = solveTieredCacheCapacity(b);
    EXPECT_EQ(result.config.total_blocks, 25u);
    EXPECT_EQ(result.config.hbm_blocks, 2u);
}

TEST(TieredCacheConfigTest, OffloadPolicyParticipatesInGroupIdentity) {
    CacheGroupPolicy baseline;
    EXPECT_FALSE(baseline.offload.enabled);
    auto changed            = baseline;
    changed.offload.enabled = true;
    EXPECT_FALSE(CacheConfig::samePolicy(baseline, changed));
    changed                           = baseline;
    changed.offload.max_selected_rows = 8;
    EXPECT_FALSE(CacheConfig::samePolicy(baseline, changed));
}

TEST(TieredCacheConfigTest, MhaEntryViewsPreserveNativeHeadAndScaleStrides) {
    AttentionConfigs attention;
    attention.kv_head_num   = 2;
    attention.size_per_head = 8;
    ParallelismConfig parallel;
    SpecBuildContext  context;
    context.attn_config             = &attention;
    context.parallelism_config      = &parallel;
    context.seq_size_per_block      = 16;
    context.kernel_tokens_per_block = 8;
    KVCacheSpecDesc desc;
    desc.tag = "sparse_heads";
    for (auto dtype : {TYPE_BF16, TYPE_INT8}) {
        desc.dtype        = dtype;
        const auto spec   = MHAKVCacheSpec::build(desc, context);
        const auto layout = spec->entryLayout();
        EXPECT_NO_THROW(layout.validate(spec->block_size_bytes() / 2, spec->scale_block_size_bytes() / 2));
        EXPECT_EQ(layout.rows_per_page, 8u);
        ASSERT_EQ(layout.planes.size(), dtype == TYPE_INT8 ? 4u : 2u);
        EXPECT_EQ(layout.planes[0].shape, (std::vector<int64_t>{8, 2, 8}));
        EXPECT_EQ(layout.planes[0].strides, (std::vector<int64_t>{8, 64, 1}));
        EXPECT_EQ(layout.planes[1].offset_bytes, 128 * getTypeSize(dtype));
        if (dtype == TYPE_INT8) {
            EXPECT_TRUE(layout.planes[2].scale);
            EXPECT_EQ(layout.planes[2].dtype, TYPE_FP32);
            EXPECT_EQ(layout.planes[2].shape, (std::vector<int64_t>{8, 2}));
            EXPECT_EQ(layout.planes[2].strides, (std::vector<int64_t>{1, 8}));
            EXPECT_EQ(layout.planes[3].offset_bytes, 64u);
        }
    }
}

TEST(TieredCacheConfigTest, MlaAndCsaExposeNativeEntriesWithoutPaddingAsPayload) {
    AttentionConfigs attention;
    attention.kv_lora_rank  = 512;
    attention.rope_head_dim = 64;
    SpecBuildContext context;
    context.attn_config             = &attention;
    context.seq_size_per_block      = 256;
    context.kernel_tokens_per_block = 128;
    KVCacheSpecDesc desc;
    desc.dtype            = TYPE_FP8_E4M3;
    const auto mla        = MLAKVCacheSpec::build(desc, context);
    const auto mla_layout = mla->entryLayout();
    EXPECT_NO_THROW(mla_layout.validate(mla->block_size_bytes() / 2, 0));
    EXPECT_EQ(mla_layout.planes[0].shape, (std::vector<int64_t>{128, 656}));

    desc.tag                          = "csa_kv";
    desc.entry_dtype                  = TYPE_UINT8;
    desc.entry_elems                  = 584;
    desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.compression_ratio            = 4;
    desc.block_stride_bytes_alignment = 576;
    const auto csa                    = CompressedKVCacheSpec::build(desc, context);
    const auto csa_layout             = csa->entryLayout();
    EXPECT_NO_THROW(csa_layout.validate(csa->block_size_bytes() / 2, 0));
    EXPECT_EQ(csa_layout.rows_per_page, 32u);
    ASSERT_EQ(csa_layout.planes.size(), 2u);
    EXPECT_EQ(csa_layout.planes[0].shape, (std::vector<int64_t>{32, 576}));
    EXPECT_EQ(csa->block_size_bytes() / 2, 19008u);
    EXPECT_EQ(csa_layout.planes[0].strides, (std::vector<int64_t>{576, 1}));
    EXPECT_EQ(csa_layout.planes[1].shape, (std::vector<int64_t>{32, 8}));
    EXPECT_EQ(csa_layout.planes[1].strides, (std::vector<int64_t>{8, 1}));
    EXPECT_EQ(csa_layout.planes[1].offset_bytes, 32u * 576);
}

TEST(TieredCacheConfigTest, EntryLayoutsRejectMissingViewsAndOutOfBoundsPlanes) {
    EXPECT_ANY_THROW(CacheEntryLayout{}.validate(128, 0));
    CacheEntryLayout layout{8, {{false, TYPE_UINT8, 0, {8, 16}, {16, 1}}}};
    EXPECT_NO_THROW(layout.validate(128, 0));
    EXPECT_ANY_THROW(layout.validate(127, 0));
    layout.planes[0].offset_bytes = 128;
    EXPECT_ANY_THROW(layout.validate(128, 0));
    layout.planes[0].offset_bytes = 0;
    layout.planes[0].shape[0]     = 4;
    EXPECT_ANY_THROW(layout.validate(128, 0));
}

}  // namespace
}  // namespace rtp_llm
