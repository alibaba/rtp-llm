#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheBlockMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace test {
namespace {

CacheGroup makeGroup(std::string tag, size_t physical_span, CacheGroupType type, bool enable_prefix_reuse = true) {
    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block        = physical_span;
    spec->kernel_seq_size_per_block = physical_span;

    CacheGroup group;
    group.tag                        = std::move(tag);
    group.spec                       = std::move(spec);
    group.policy                     = defaultCacheGroupPolicy(type);
    group.policy.enable_prefix_reuse = enable_prefix_reuse;
    return group;
}

TEST(CacheBlockMapperTest, DerivesGroupSpanAndCheckedReuseAlignment) {
    CacheConfig config({makeGroup("two", 8, CacheGroupType::FULL),
                        makeGroup("three", 12, CacheGroupType::LINEAR),
                        makeGroup("ignored_swa", 20, CacheGroupType::SWA, false)},
                       {{"two", "three", "ignored_swa"}},
                       /*main_layer_num=*/1);
    config.seq_size_per_block = 4;

    EXPECT_EQ(CacheBlockMapper::cacheKeysPerPhysicalBlock(config, "two"), 2u);
    EXPECT_EQ(CacheBlockMapper::cacheKeysPerPhysicalBlock(config, "three"), 3u);
    EXPECT_EQ(CacheBlockMapper::reuseScanAlignmentKeyBlocks(config), 6u);
    EXPECT_EQ(CacheBlockMapper::physicalBlocksForCacheKeyPrefix(config, "three", 6), 2u);
    const auto plan = CacheBlockMapper::buildCacheKeyBlockPlan(config, "three", 7, 3);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan.back().key_index, 6);
    EXPECT_EQ(plan.back().offset_index, 2);
    EXPECT_EQ(CacheBlockMapper::checkedLeastCommonMultiple(6, 4, "four"), 12u);
    EXPECT_ANY_THROW(CacheBlockMapper::checkedLeastCommonMultiple(0, 1, "zero"));
    EXPECT_ANY_THROW(CacheBlockMapper::checkedLeastCommonMultiple(std::numeric_limits<size_t>::max(), 2, "overflow"));
}

TEST(CacheBlockMapperTest, RejectsInvalidGroupSpan) {
    CacheConfig config({makeGroup("invalid", 6, CacheGroupType::FULL)}, {{"invalid"}}, /*main_layer_num=*/1);
    config.seq_size_per_block = 4;
    EXPECT_ANY_THROW(CacheBlockMapper::cacheKeysPerPhysicalBlock(config, "invalid"));

    config.seq_size_per_block = 0;
    EXPECT_ANY_THROW(CacheBlockMapper::cacheKeysPerPhysicalBlock(config, "invalid"));
}

TEST(CacheBlockMapperTest, ConvertsCountsAndPositionsWithCheckedBoundaries) {
    EXPECT_EQ(CacheBlockMapper::physicalBlocksForCacheKeyPrefix(0, 3, "group"), 0u);
    EXPECT_EQ(CacheBlockMapper::physicalBlocksForCacheKeyPrefix(6, 3, "group"), 2u);
    EXPECT_ANY_THROW(CacheBlockMapper::physicalBlocksForCacheKeyPrefix(5, 3, "group"));
    EXPECT_ANY_THROW(CacheBlockMapper::physicalBlocksForCacheKeyPrefix(0, 0, "group"));

    EXPECT_EQ(CacheBlockMapper::physicalBlockCapacityForCacheKeys(0, 3), 0u);
    EXPECT_EQ(CacheBlockMapper::physicalBlockCapacityForCacheKeys(7, 3), 3u);
    EXPECT_EQ(CacheBlockMapper::cacheKeyCapacityForPhysicalBlocks(3, 3), 9u);
    EXPECT_ANY_THROW(CacheBlockMapper::cacheKeyCapacityForPhysicalBlocks(std::numeric_limits<size_t>::max(), 2));
    EXPECT_ANY_THROW(CacheBlockMapper::physicalBlockCapacityForCacheKeys(1, 0));

    EXPECT_EQ(CacheBlockMapper::physicalBlockPositionForCacheKeyPosition(7, 3), 2u);
    EXPECT_ANY_THROW(CacheBlockMapper::physicalBlockPositionForCacheKeyPosition(0, 0));
    EXPECT_EQ(CacheBlockMapper::representativeCacheKeyPosition(0, 7, 3), 2u);
    EXPECT_EQ(CacheBlockMapper::representativeCacheKeyPosition(2, 7, 3), 6u);
    EXPECT_ANY_THROW(CacheBlockMapper::representativeCacheKeyPosition(3, 7, 3));
    EXPECT_ANY_THROW(CacheBlockMapper::representativeCacheKeyPosition(0, 0, 3));
}

TEST(CacheBlockMapperTest, BuildsPartialTailPlanAndChecksCapacity) {
    const auto plan = CacheBlockMapper::buildCacheKeyBlockPlan(
        /*total_cache_key_blocks=*/7, /*physical_block_count=*/3, /*keys_per_physical_block=*/3, "group");
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].key_index, 2);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].key_index, 5);
    EXPECT_EQ(plan[1].offset_index, 1);
    EXPECT_EQ(plan[2].key_index, 6);
    EXPECT_EQ(plan[2].offset_index, 2);

    EXPECT_TRUE(CacheBlockMapper::buildCacheKeyBlockPlan(0, 3, 3, "group").empty());
    EXPECT_TRUE(CacheBlockMapper::buildCacheKeyBlockPlan(7, 0, 3, "group").empty());
    EXPECT_ANY_THROW(CacheBlockMapper::buildCacheKeyBlockPlan(7, 4, 3, "group"));
    EXPECT_ANY_THROW(CacheBlockMapper::buildCacheKeyBlockPlan(7, 1, 0, "group"));
    EXPECT_ANY_THROW(CacheBlockMapper::buildCacheKeyBlockPlan(
        static_cast<size_t>(std::numeric_limits<int>::max()) + 2, 1, std::numeric_limits<size_t>::max(), "group"));
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
