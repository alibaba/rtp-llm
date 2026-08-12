#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace test {
namespace {

CacheConfig makeSparseMlaConfig(uint32_t layer_num,
                                uint32_t block_num,
                                size_t   tokens_per_block,
                                size_t   kernel_tokens_per_block,
                                size_t   scale_stride_bytes) {
    return makeSimpleMlaCacheConfig(static_cast<int>(layer_num),
                                    static_cast<int>(block_num),
                                    tokens_per_block,
                                    DataType::TYPE_BF16,
                                    /*sparse=*/true,
                                    scale_stride_bytes,
                                    kernel_tokens_per_block);
}

template<typename Fn>
void expectRuntimeErrorContains(Fn&& fn, const std::string& expected) {
    try {
        fn();
        FAIL() << "expected std::runtime_error containing: " << expected;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
    }
}

TEST(BlockPoolConfigHelperTest, MTPSparseIndexerUsesProposeTopologyAndScaleStride) {
    auto score_config   = makeSparseMlaConfig(/*layer_num=*/2,
                                            /*block_num=*/4,
                                            /*tokens_per_block=*/4,
                                            /*kernel_tokens_per_block=*/4,
                                            /*scale_stride_bytes=*/32);
    auto propose_config = std::make_shared<CacheConfig>(makeSparseMlaConfig(/*layer_num=*/1,
                                                                            /*block_num=*/4,
                                                                            /*tokens_per_block=*/4,
                                                                            /*kernel_tokens_per_block=*/2,
                                                                            /*scale_stride_bytes=*/128));
    score_config.mtp_sub_configs.push_back(propose_config);

    ASSERT_EQ(propose_config->specForGroup(0)->block_size_bytes(), 64u);
    ASSERT_EQ(propose_config->specForGroup(0)->scale_block_size_bytes(), 0u);

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);

    const auto& main_layout = pool_config.memory_layouts[0];
    EXPECT_EQ(main_layout.block_num, 4u);
    EXPECT_EQ(main_layout.kv_cache_offset_bytes, 0u);
    EXPECT_EQ(main_layout.kv_block_pool_size_bytes, 2u * 4u * 64u);
    EXPECT_EQ(main_layout.kv_scale_offset_bytes, 512u);
    EXPECT_EQ(main_layout.kv_scale_pool_size_bytes, 2u * 4u * 32u);

    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_TRUE(mtp_layout.is_mla);
    EXPECT_TRUE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.block_num, 4u);
    EXPECT_EQ(mtp_layout.layer_num, 1u);
    EXPECT_EQ(mtp_layout.seq_size_per_block, 4u);
    EXPECT_EQ(mtp_layout.kernel_blocks_per_kv_block, 2u);
    EXPECT_EQ(mtp_layout.kv_scale_stride_bytes, 128u);
    EXPECT_EQ(mtp_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 4u * 64u);
    EXPECT_EQ(mtp_layout.kv_scale_offset_bytes, 1024u);
    EXPECT_EQ(mtp_layout.kv_scale_pool_size_bytes, 1u * 4u * 128u);
    EXPECT_EQ(pool_config.total_size_bytes, 1536u);
}

TEST(BlockPoolConfigHelperTest, MTPNonSparseUsesNonzeroPhysicalScaleStride) {
    auto score_config   = makeSparseMlaConfig(/*layer_num=*/2,
                                            /*block_num=*/4,
                                            /*tokens_per_block=*/4,
                                            /*kernel_tokens_per_block=*/4,
                                            /*scale_stride_bytes=*/32);
    auto propose_config = std::make_shared<CacheConfig>(makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                                 /*block_num=*/4,
                                                                                 /*tokens_per_block=*/4,
                                                                                 DataType::TYPE_INT8,
                                                                                 /*local_head_num_kv=*/1,
                                                                                 /*size_per_head=*/2));
    score_config.mtp_sub_configs.push_back(propose_config);

    ASSERT_EQ(propose_config->specForGroup(0)->block_size_bytes(), 16u);
    ASSERT_EQ(propose_config->specForGroup(0)->scale_block_size_bytes(), 32u);
    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);

    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_FALSE(mtp_layout.is_mla);
    EXPECT_TRUE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.kv_scale_stride_bytes, 32u);
    EXPECT_EQ(mtp_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 4u * 16u);
    EXPECT_EQ(mtp_layout.kv_scale_offset_bytes, 832u);
    EXPECT_EQ(mtp_layout.kv_scale_pool_size_bytes, 1u * 4u * 32u);
    EXPECT_EQ(pool_config.total_size_bytes, 960u);
}

TEST(BlockPoolConfigHelperTest, MTPSelectsRealGroupAndAccumulatesMultipleOffsets) {
    auto score_config = makeSparseMlaConfig(/*layer_num=*/2,
                                            /*block_num=*/4,
                                            /*tokens_per_block=*/4,
                                            /*kernel_tokens_per_block=*/4,
                                            /*scale_stride_bytes=*/32);
    auto first        = std::make_shared<CacheConfig>(makeSparseMlaConfig(/*layer_num=*/1,
                                                                   /*block_num=*/4,
                                                                   /*tokens_per_block=*/4,
                                                                   /*kernel_tokens_per_block=*/2,
                                                                   /*scale_stride_bytes=*/128));

    auto second      = std::make_shared<CacheConfig>(makeSparseMlaConfig(/*layer_num=*/2,
                                                                    /*block_num=*/4,
                                                                    /*tokens_per_block=*/8,
                                                                    /*kernel_tokens_per_block=*/2,
                                                                    /*scale_stride_bytes=*/64));
    auto placeholder = makeMhaSpec("placeholder", 8, DataType::TYPE_BF16, 1, 1);
    auto real_spec   = makeMlaSpec("default", 8, DataType::TYPE_BF16, 4, 4);
    second->fromGroupedSpecs({placeholder, real_spec},
                             {{}, {0, 1}},
                             {CacheGroupType::FULL, CacheGroupType::FULL},
                             {"placeholder", "default"});
    const size_t placeholder_kv_stride = 256;
    const size_t real_kv_stride        = 192;
    second->setGroupBlockLayout({4, 4}, {placeholder_kv_stride, real_kv_stride}, {0, 64});

    score_config.mtp_sub_configs = {first, second};
    const auto pool_config       = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 3u);

    const auto& first_layout  = pool_config.memory_layouts[1];
    const auto& second_layout = pool_config.memory_layouts[2];
    EXPECT_EQ(first_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(first_layout.kv_scale_offset_bytes, 1024u);
    EXPECT_EQ(second_layout.kv_cache_offset_bytes, 1536u);
    EXPECT_GT(placeholder_kv_stride, real_kv_stride);
    EXPECT_NE(real_kv_stride, real_spec->block_size_bytes());
    EXPECT_EQ(second_layout.kv_block_stride_bytes, real_kv_stride);
    EXPECT_EQ(second_layout.local_head_num_kv, 1u);
    EXPECT_EQ(second_layout.layer_num, 2u);
    EXPECT_EQ(second_layout.seq_size_per_block, 8u);
    EXPECT_EQ(second_layout.kernel_blocks_per_kv_block, 4u);
    EXPECT_EQ(second_layout.kv_block_pool_size_bytes, 2u * 4u * 192u);
    EXPECT_EQ(second_layout.kv_scale_offset_bytes, 3072u);
    EXPECT_EQ(second_layout.kv_scale_pool_size_bytes, 2u * 4u * 64u);
    EXPECT_EQ(pool_config.total_size_bytes, 3584u);
}

TEST(BlockPoolConfigHelperTest, RejectsNullMTPSubConfig) {
    auto score_config = makeSparseMlaConfig(2, 4, 4, 4, 32);
    score_config.mtp_sub_configs.push_back(nullptr);
    expectRuntimeErrorContains([&score_config] { BlockPoolConfigHelper::createConfig(score_config); }, "is null");
}

TEST(BlockPoolConfigHelperTest, RejectsMTPSubConfigWithoutGroups) {
    auto score_config            = makeSparseMlaConfig(2, 4, 4, 4, 32);
    auto empty                   = std::make_shared<CacheConfig>();
    score_config.mtp_sub_configs = {empty};
    expectRuntimeErrorContains([&score_config] { BlockPoolConfigHelper::createConfig(score_config); },
                               "cache groups must not be empty");
}

TEST(BlockPoolConfigHelperTest, MTPLayoutUsesSynchronizedMainBlockNum) {
    auto score_config            = makeSparseMlaConfig(2, 4, 4, 4, 32);
    auto stale_mtp_config        = std::make_shared<CacheConfig>(makeSparseMlaConfig(1, 3, 4, 2, 128));
    score_config.mtp_sub_configs = {stale_mtp_config};

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);
    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_EQ(stale_mtp_config->block_num, 3u);
    EXPECT_EQ(mtp_layout.block_num, 4u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 4u * 64u);
    EXPECT_EQ(mtp_layout.kv_scale_pool_size_bytes, 1u * 4u * 128u);
}

TEST(BlockPoolConfigHelperTest, MTPWithoutScaleKeepsContiguousOffsets) {
    auto score_config            = makeSparseMlaConfig(2, 4, 4, 4, 32);
    auto no_scale                = std::make_shared<CacheConfig>(makeSimpleMlaCacheConfig(
        /*layer_num=*/1,
        /*block_num=*/4,
        /*tokens_per_block=*/4,
        DataType::TYPE_BF16,
        /*sparse=*/false));
    score_config.mtp_sub_configs = {no_scale};

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);
    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_FALSE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 4u * 64u);
    EXPECT_EQ(mtp_layout.kv_scale_offset_bytes, mtp_layout.kv_cache_offset_bytes + mtp_layout.kv_block_pool_size_bytes);
    EXPECT_EQ(pool_config.total_size_bytes, 1024u);
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
