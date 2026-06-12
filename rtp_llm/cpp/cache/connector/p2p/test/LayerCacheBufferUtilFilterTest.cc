#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

namespace rtp_llm {
namespace {

class LayerCacheBufferUtilFilterTest: public ::testing::Test {
protected:
    KVCacheResource makeResource(int                                group_num,
                                 int                                layer_num,
                                 const std::vector<int>&            layer_to_group,
                                 const std::vector<CacheGroupType>& group_types,
                                 const std::vector<int32_t>&        block_ids_g0,
                                 const std::vector<int64_t>&        cache_keys_input) {
        KVCacheResource resource;
        resource.initGroups(group_num, layer_num, layer_to_group, 1, group_types);
        auto& blocks = resource.mutableBlockIds(0);
        blocks.assign(block_ids_g0);
        resource.cacheKeys() = cache_keys_input;
        return resource;
    }

    KVCacheResource makeTwoGroupResource(const std::vector<int32_t>& full_block_ids,
                                         const std::vector<int32_t>& linear_block_ids,
                                         const std::vector<int64_t>& cache_keys_input) {
        KVCacheResource resource;
        // 2 groups: group 0 = FULL, group 1 = LINEAR
        // 2 layers: layer 0 -> group 0, layer 1 -> group 1
        std::vector<int>            layer_to_group = {0, 1};
        std::vector<CacheGroupType> group_types    = {CacheGroupType::FULL, CacheGroupType::LINEAR};
        resource.initGroups(2, 2, layer_to_group, 1, group_types);
        resource.mutableBlockIds(0).assign(full_block_ids);
        resource.mutableBlockIds(1).assign(linear_block_ids);
        resource.cacheKeys() = cache_keys_input;
        return resource;
    }
};

TEST_F(LayerCacheBufferUtilFilterTest, FullGroupAllValidBlocks) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, 11, 12}, {100, 101, 102});

    auto result = LayerCacheBufferUtil::convertLayer(resource, 0, 0, 0, -1, CacheGroupType::FULL);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->blockIdMap().size(), 3u);
    EXPECT_EQ(result->getBlockId(100), 10);
    EXPECT_EQ(result->getBlockId(101), 11);
    EXPECT_EQ(result->getBlockId(102), 12);
}

TEST_F(LayerCacheBufferUtilFilterTest, FullGroupWithNullBlockSkipped) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, -1, 12}, {100, 101, 102});

    auto result = LayerCacheBufferUtil::convertLayer(resource, 0, 0, 0, -1, CacheGroupType::FULL);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->blockIdMap().size(), 2u);
    EXPECT_EQ(result->getBlockId(100), 10);
    EXPECT_EQ(result->getBlockId(102), 12);
    EXPECT_EQ(result->getBlockId(101), -1);  // not found
}

TEST_F(LayerCacheBufferUtilFilterTest, LinearGroupOnlyLastValidBlock) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {-1, -1, 25}, {100, 101, 102});

    auto result = LayerCacheBufferUtil::convertLayer(resource, 0, 0, 0, -1, CacheGroupType::LINEAR);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->blockIdMap().size(), 1u);
    EXPECT_EQ(result->getBlockId(102), 25);
}

TEST_F(LayerCacheBufferUtilFilterTest, LinearGroupMixedValidOnlyTakesLast) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {18, -1, 25}, {100, 101, 102});

    auto result = LayerCacheBufferUtil::convertLayer(resource, 0, 0, 0, -1, CacheGroupType::LINEAR);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->blockIdMap().size(), 1u);
    EXPECT_EQ(result->getBlockId(102), 25);
    EXPECT_EQ(result->getBlockId(100), -1);  // not included
}

TEST_F(LayerCacheBufferUtilFilterTest, AllNullBlocksReturnsNullptr) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {-1, -1, -1}, {100, 101, 102});

    auto result = LayerCacheBufferUtil::convertLayer(resource, 0, 0, 0, -1, CacheGroupType::LINEAR);
    EXPECT_EQ(result, nullptr);
}

TEST_F(LayerCacheBufferUtilFilterTest, ConvertWithLayerAttnTypes) {
    auto resource = makeTwoGroupResource({10, 11, 12},  // FULL group blocks
                                         {-1, -1, 25},  // LINEAR group blocks
                                         {100, 101, 102});

    std::vector<CacheGroupType> layer_attn_types = {CacheGroupType::FULL, CacheGroupType::LINEAR};
    auto                        results          = LayerCacheBufferUtil::convert(resource, 0, layer_attn_types, 0, -1);

    ASSERT_EQ(results.size(), 2u);

    // Layer 0 (FULL): all 3 blocks
    EXPECT_EQ(results[0]->getLayerId(), 0);
    EXPECT_EQ(results[0]->blockIdMap().size(), 3u);

    // Layer 1 (LINEAR): only last block
    EXPECT_EQ(results[1]->getLayerId(), 1);
    EXPECT_EQ(results[1]->blockIdMap().size(), 1u);
    EXPECT_EQ(results[1]->getBlockId(102), 25);
}

TEST_F(LayerCacheBufferUtilFilterTest, ConvertLinearAllNullSkipsLayer) {
    auto resource = makeTwoGroupResource({10, 11, 12},
                                         {-1, -1, -1},  // all null
                                         {100, 101, 102});

    std::vector<CacheGroupType> layer_attn_types = {CacheGroupType::FULL, CacheGroupType::LINEAR};
    auto                        results          = LayerCacheBufferUtil::convert(resource, 0, layer_attn_types, 0, -1);

    // Only FULL layer should appear (LINEAR layer returns nullptr and is skipped)
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0]->getLayerId(), 0);
}

}  // namespace
}  // namespace rtp_llm
