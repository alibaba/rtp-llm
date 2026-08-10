#include "rtp_llm/cpp/model_rpc/CacheTransferBlockSelector.h"

#include <gtest/gtest.h>

#include <memory>

#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/LinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

TEST(CacheTransferBlockSelectorAllocationTest, ReserveAllocationDoesNotBecomeRemoteBackedCache) {
    auto block_pool = createBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->seq_size_per_block = 4;
    FullKVCacheGroup full_group({}, full_spec, block_pool, /*group_id=*/0);

    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->seq_size_per_block = 4;
    LinearKVCacheGroup linear_group({}, linear_spec, block_pool, /*group_id=*/1, /*linear_step=*/2);

    BlockIds full_blocks;
    ASSERT_TRUE(full_group.malloc(
        full_blocks, /*seq_len=*/12, /*enable_reuse_cache=*/false, /*reserve_step=*/4));
    ASSERT_EQ(full_blocks.blocksNum(), 4u);

    BlockIds linear_blocks;
    ASSERT_TRUE(linear_group.malloc(
        linear_blocks, /*seq_len=*/12, /*enable_reuse_cache=*/false, /*reserve_step=*/4));
    ASSERT_EQ(linear_blocks.blocksNum(), 6u);

    constexpr size_t cache_key_count = 3;
    auto full_positions = selectCacheTransferBlockPositions(
        CacheGroupType::FULL, full_blocks.blocks(), cache_key_count, /*reuse_block_count=*/0);
    auto linear_positions = selectCacheTransferBlockPositions(
        CacheGroupType::LINEAR, linear_blocks.blocks(), cache_key_count, /*reuse_block_count=*/0);

    ASSERT_TRUE(full_positions.ok()) << full_positions.status();
    EXPECT_EQ(*full_positions, (std::vector<size_t>{0, 1, 2}));
    ASSERT_TRUE(linear_positions.ok()) << linear_positions.status();
    EXPECT_EQ(*linear_positions, (std::vector<size_t>{2}));
    EXPECT_FALSE(isNullBlockIdx(linear_blocks.blocks()[2]));
    for (size_t position = cache_key_count; position < linear_blocks.blocksNum(); ++position) {
        EXPECT_FALSE(isNullBlockIdx(linear_blocks.blocks()[position]));
    }
}

}  // namespace test
}  // namespace rtp_llm
