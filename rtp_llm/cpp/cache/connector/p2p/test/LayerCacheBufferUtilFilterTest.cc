#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"

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
        std::vector<std::vector<int>> layer_group_ids;
        for (int group_id : layer_to_group) {
            layer_group_ids.push_back({group_id});
        }
        topology_ = test::makeTestCacheTopology(group_num, layer_num, layer_group_ids, 1, group_types);
        resource.initGroups(topology_);
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
        topology_                                  = test::makeTestCacheTopology(2, 2, {{0}, {1}}, 1, group_types);
        resource.initGroups(topology_);
        resource.mutableBlockIds(0).assign(full_block_ids);
        resource.mutableBlockIds(1).assign(linear_block_ids);
        resource.cacheKeys() = cache_keys_input;
        return resource;
    }

    std::shared_ptr<const CacheTopology> topology_;
};

TEST_F(LayerCacheBufferUtilFilterTest, FullGroupAllValidBlocks) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, 11, 12}, {100, 101, 102});

    auto conversion = LayerCacheBufferUtil::convertLayer(resource, *topology_, 0, 0, -1);
    ASSERT_TRUE(conversion.ok()) << conversion.status().ToString();
    auto results = std::move(conversion.value());
    ASSERT_EQ(results.size(), 1u);
    auto result = results.front();
    EXPECT_EQ(result->blockIdMap().size(), 3u);
    EXPECT_EQ(result->getBlockId(100), 10);
    EXPECT_EQ(result->getBlockId(101), 11);
    EXPECT_EQ(result->getBlockId(102), 12);
}

TEST_F(LayerCacheBufferUtilFilterTest, FullGroupWithNullBlockRejected) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, -1, 12}, {100, 101, 102});

    auto results = LayerCacheBufferUtil::convertLayer(resource, *topology_, 0, 0, -1);
    EXPECT_FALSE(results.ok());
    EXPECT_TRUE(results.value().empty());
}

TEST_F(LayerCacheBufferUtilFilterTest, LinearGroupOnlyLastValidBlock) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {-1, -1, 25}, {100, 101, 102});

    auto conversion = LayerCacheBufferUtil::convertLayer(resource, *topology_, 0, 0, -1);
    ASSERT_TRUE(conversion.ok()) << conversion.status().ToString();
    auto results = std::move(conversion.value());
    ASSERT_EQ(results.size(), 1u);
    auto result = results.front();
    EXPECT_EQ(result->blockIdMap().size(), 1u);
    EXPECT_EQ(result->getBlockId(102), 25);
}

TEST_F(LayerCacheBufferUtilFilterTest, LinearGroupMixedValidOnlyTakesLast) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {18, -1, 25}, {100, 101, 102});

    auto conversion = LayerCacheBufferUtil::convertLayer(resource, *topology_, 0, 0, -1);
    ASSERT_TRUE(conversion.ok()) << conversion.status().ToString();
    auto results = std::move(conversion.value());
    ASSERT_EQ(results.size(), 1u);
    auto result = results.front();
    EXPECT_EQ(result->blockIdMap().size(), 1u);
    EXPECT_EQ(result->getBlockId(102), 25);
    EXPECT_EQ(result->getBlockId(100), -1);  // not included
}

TEST_F(LayerCacheBufferUtilFilterTest, AllNullBlocksReturnsError) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::LINEAR}, {-1, -1, -1}, {100, 101, 102});

    auto results = LayerCacheBufferUtil::convertLayer(resource, *topology_, 0, 0, -1);
    EXPECT_FALSE(results.ok());
    EXPECT_TRUE(results.value().empty());
}

TEST_F(LayerCacheBufferUtilFilterTest, RouteProjectionRejectsNullBlock) {
    auto resource = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, -1}, {100, 101});

    auto results = LayerCacheBufferUtil::convertTagForRoute(resource, *topology_, "group0", {0, 1}, 0, 1);

    EXPECT_TRUE(results.value().empty());
    ASSERT_FALSE(results.ok());
    EXPECT_NE(results.status().ToString().find("layer=0 tag=group0"), std::string::npos);
    EXPECT_NE(results.status().ToString().find("cache_key=101 block_id=-1"), std::string::npos);
}

TEST_F(LayerCacheBufferUtilFilterTest, ConvertWithLayerAttnTypes) {
    auto resource = makeTwoGroupResource({10, 11, 12},  // FULL group blocks
                                         {-1, -1, 25},  // LINEAR group blocks
                                         {100, 101, 102});

    auto conversion = LayerCacheBufferUtil::convert(resource, *topology_);
    ASSERT_TRUE(conversion.ok()) << conversion.status().ToString();
    auto results = std::move(conversion.value());

    ASSERT_EQ(results.size(), 2u);

    // Layer 0 (FULL): all 3 blocks
    EXPECT_EQ(results[0]->getLayerId(), 0);
    EXPECT_EQ(results[0]->blockIdMap().size(), 3u);

    // Layer 1 (LINEAR): only last block
    EXPECT_EQ(results[1]->getLayerId(), 1);
    EXPECT_EQ(results[1]->blockIdMap().size(), 1u);
    EXPECT_EQ(results[1]->getBlockId(102), 25);
}

TEST_F(LayerCacheBufferUtilFilterTest, ConvertLinearAllNullRejectsRequest) {
    auto resource = makeTwoGroupResource({10, 11, 12},
                                         {-1, -1, -1},  // all null
                                         {100, 101, 102});

    auto results = LayerCacheBufferUtil::convert(resource, *topology_);
    EXPECT_FALSE(results.ok());
    EXPECT_TRUE(results.value().empty());
}

TEST_F(LayerCacheBufferUtilFilterTest, NonEmptyRouteRejectsEmptyResourceAndDuplicateKeys) {
    auto empty = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {}, {});
    EXPECT_FALSE(LayerCacheBufferUtil::convertTagForRoute(empty, *topology_, "group0", {0}, 0, 1).ok());
    auto duplicate = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10, 11}, {100, 100});
    EXPECT_FALSE(LayerCacheBufferUtil::convertTagForRoute(duplicate, *topology_, "group0", {0, 1}, 0, 1).ok());
    auto missing = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10}, {100, 101});
    EXPECT_FALSE(LayerCacheBufferUtil::convertTagForRoute(missing, *topology_, "group0", {0, 1}, 0, 1).ok());
    auto no_work = LayerCacheBufferUtil::convertTagForRoute(missing, *topology_, "group0", {}, 0, 1);
    ASSERT_TRUE(no_work.ok());
    EXPECT_TRUE(no_work.value().empty());
}

TEST_F(LayerCacheBufferUtilFilterTest, CpOwnershipFilteringDoesNotHideRequiredMissingBlock) {
    auto resource           = makeResource(1, 1, {0}, {CacheGroupType::FULL}, {10}, {100, 101});
    auto group              = topology_->group("group0");
    group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    auto projected          = LayerCacheBufferUtil::convertLayerTag(resource, group, 0, 0, -1, 1, 2);
    ASSERT_TRUE(projected.ok()) << projected.status().ToString();
    ASSERT_NE(projected.value(), nullptr);
    EXPECT_EQ(projected.value()->blockIdMap().size(), 1u);
    EXPECT_EQ(projected.value()->getBlockId(101), 10);
    EXPECT_FALSE(LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, {0}, 1, 2).ok());
}

class CheckedBlockConverter: public LayerBlockConverter {
public:
    mutable char           storage[16]    = {};
    bool                   missing_second = false;
    std::vector<BlockInfo> convertIndexToBuffer(int, const std::string&, int block_id, int, int) const override {
        if (missing_second && block_id == 2) {
            return {};
        }
        BlockInfo info;
        info.addr       = storage;
        info.size_bytes = sizeof(storage);
        return {info};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

TEST(LayerCacheBufferValidationTest, PartialConversionFailsWithoutReturningEarlierKeys) {
    auto converter            = std::make_shared<CheckedBlockConverter>();
    converter->missing_second = true;
    auto buffer               = std::make_shared<LayerCacheBuffer>(0, "full");
    buffer->addBlockId(100, 1);
    buffer->addBlockId(101, 2);
    auto result = LayerCacheBufferUtil::buildKeyBlockInfos(converter, buffer);
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.value().empty());
    EXPECT_NE(result.status().ToString().find("cache_key=101 block_id=2"), std::string::npos);
}

TEST(LayerCacheBufferValidationTest, SliceValidationRejectsInvalidParametersAndPreservesValidRange) {
    auto converter = std::make_shared<CheckedBlockConverter>();
    auto buffer    = std::make_shared<LayerCacheBuffer>(0, "full");
    buffer->addBlockId(100, 1);
    for (auto slice : {SliceSpec{CpBlockSliceMode::EQUAL_BYTES, 0, 0},
                       SliceSpec{CpBlockSliceMode::EQUAL_BYTES, 2, -1},
                       SliceSpec{CpBlockSliceMode::EQUAL_BYTES, 2, 2},
                       SliceSpec{static_cast<CpBlockSliceMode>(99), 2, 0}}) {
        EXPECT_FALSE(LayerCacheBufferUtil::buildKeyBlockInfosSliced(converter, buffer, 1, 0, slice, 16).ok());
    }
    const SliceSpec slice{CpBlockSliceMode::PAYLOAD_BYTES, 2, 1};
    EXPECT_FALSE(LayerCacheBufferUtil::buildKeyBlockInfosSliced(converter, buffer, 2, 0, slice, 16).ok());
    EXPECT_FALSE(LayerCacheBufferUtil::buildKeyBlockInfosSliced(converter, buffer, 1, 0, slice, 15).ok());
    EXPECT_FALSE(LayerCacheBufferUtil::buildKeyBlockInfosSliced(converter, buffer, 1, 0, slice, 32).ok());
    auto result = LayerCacheBufferUtil::buildKeyBlockInfosSliced(converter, buffer, 1, 0, slice, 12);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result.value().size(), 1u);
    const auto& part = result.value().at(100)->blocks.front();
    EXPECT_EQ(part.addr, converter->storage + 6);
    EXPECT_EQ(part.size_bytes, 6u);
}

TEST(LayerCacheBufferValidationTest, PublicationErrorNotifiesOutsideBufferLockAndSurvivesLateSubscription) {
    ComputedLayerCacheBuffer buffer(1, nullptr, 1000);
    const ErrorInfo          error(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "required block missing");
    buffer.setError(error);
    int notifications = 0;
    buffer.setErrorHandler([&](const ErrorInfo& notified) {
        ++notifications;
        EXPECT_EQ(notified.ToString(), error.ToString());
        EXPECT_TRUE(buffer.error().hasError());
    });
    EXPECT_EQ(notifications, 1);
    buffer.setError(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "later error"));
    EXPECT_EQ(buffer.error().ToString(), error.ToString());
    EXPECT_EQ(notifications, 1);
}

}  // namespace
}  // namespace rtp_llm
