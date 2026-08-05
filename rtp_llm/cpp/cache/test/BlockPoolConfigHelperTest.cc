#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace test {
namespace {

CacheConfig makeMlaCacheConfig(uint32_t layer_num,
                               uint32_t block_num,
                               uint32_t kernel_tokens_per_block,
                               bool     sparse,
                               size_t   config_scale_stride_bytes,
                               size_t   group_scale_stride_bytes) {
    constexpr uint32_t kSeqSizePerBlock = 4;
    auto spec = makeMlaSpec("default", kSeqSizePerBlock, DataType::TYPE_BF16, /*kv_lora_rank=*/4, /*rope_head_dim=*/4);

    CacheConfig config;
    config.dtype                     = DataType::TYPE_BF16;
    config.layer_num                 = layer_num;
    config.layer_all_num             = layer_num;
    config.use_mla                   = true;
    config.is_sparse                 = sparse;
    config.block_num                 = block_num;
    config.seq_size_per_block        = kSeqSizePerBlock;
    config.kernel_seq_size_per_block = kernel_tokens_per_block;
    config.kv_block_stride_bytes     = spec->block_size_bytes();
    config.kv_scale_stride_bytes     = config_scale_stride_bytes;

    GroupBase group;
    group.spec                  = spec;
    group.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.local_kv_head_num     = 1;
    group.kv_block_stride_bytes = config.kv_block_stride_bytes;
    group.kv_scale_stride_bytes = group_scale_stride_bytes;
    for (uint32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        group.layer_ids.push_back(static_cast<int>(layer_id));
    }
    config.groups.push_back(std::move(group));
    return config;
}

TEST(BlockPoolConfigHelperTest, MTPSparseIndexerUsesProposeConfigAndScaleStride) {
    auto score_config   = makeMlaCacheConfig(/*layer_num=*/2,
                                           /*block_num=*/4,
                                           /*kernel_tokens_per_block=*/4,
                                           /*sparse=*/true,
                                           /*config_scale_stride_bytes=*/32,
                                           /*group_scale_stride_bytes=*/16);
    auto propose_config = std::make_shared<CacheConfig>(makeMlaCacheConfig(/*layer_num=*/1,
                                                                           /*block_num=*/3,
                                                                           /*kernel_tokens_per_block=*/2,
                                                                           /*sparse=*/true,
                                                                           /*config_scale_stride_bytes=*/128,
                                                                           /*group_scale_stride_bytes=*/24));
    score_config.mtp_sub_configs.push_back(propose_config);

    ASSERT_EQ(propose_config->specForGroup(0)->block_size_bytes(), 64u);
    ASSERT_EQ(propose_config->specForGroup(0)->scale_block_size_bytes(), 0u);
    ASSERT_NE(propose_config->kv_scale_stride_bytes, propose_config->groups[0].kv_scale_stride_bytes);

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);

    const auto& main_layout = pool_config.memory_layouts[0];
    EXPECT_EQ(main_layout.block_num, 4u);
    EXPECT_EQ(main_layout.kv_cache_offset_bytes, 0u);
    EXPECT_EQ(main_layout.kv_block_pool_size_bytes, 2u * 4u * 64u);
    EXPECT_EQ(main_layout.kv_scale_offset_bytes, 2u * 4u * 64u);
    EXPECT_EQ(main_layout.kv_scale_pool_size_bytes, 2u * 4u * 32u);

    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_TRUE(mtp_layout.is_mla);
    EXPECT_TRUE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.block_num, 3u);
    EXPECT_EQ(mtp_layout.kernel_blocks_per_kv_block, 2u);
    EXPECT_EQ(mtp_layout.kv_scale_stride_bytes, propose_config->kv_scale_stride_bytes);
    EXPECT_EQ(mtp_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 3u * 64u);
    EXPECT_EQ(mtp_layout.kv_scale_offset_bytes, 960u);
    EXPECT_EQ(mtp_layout.kv_scale_pool_size_bytes, 1u * 3u * 128u);
    EXPECT_EQ(pool_config.total_size_bytes, 1344u);
}

TEST(BlockPoolConfigHelperTest, MTPNonSparseUsesSpecScaleAndNoScalePool) {
    auto score_config   = makeMlaCacheConfig(/*layer_num=*/2,
                                           /*block_num=*/4,
                                           /*kernel_tokens_per_block=*/4,
                                           /*sparse=*/true,
                                           /*config_scale_stride_bytes=*/32,
                                           /*group_scale_stride_bytes=*/16);
    auto propose_config = std::make_shared<CacheConfig>(makeMlaCacheConfig(/*layer_num=*/1,
                                                                           /*block_num=*/3,
                                                                           /*kernel_tokens_per_block=*/2,
                                                                           /*sparse=*/false,
                                                                           /*config_scale_stride_bytes=*/128,
                                                                           /*group_scale_stride_bytes=*/24));
    score_config.mtp_sub_configs.push_back(propose_config);

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);
    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_EQ(mtp_layout.kv_scale_stride_bytes, propose_config->specForGroup(0)->scale_block_size_bytes());
    EXPECT_FALSE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.kv_cache_offset_bytes, 768u);
    EXPECT_EQ(mtp_layout.kv_block_pool_size_bytes, 1u * 3u * 64u);
    EXPECT_EQ(mtp_layout.kv_scale_offset_bytes, mtp_layout.kv_cache_offset_bytes + mtp_layout.kv_block_pool_size_bytes);
    EXPECT_EQ(mtp_layout.kv_scale_pool_size_bytes, 0u);
    EXPECT_EQ(pool_config.total_size_bytes, 960u);
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
