#include <gtest/gtest.h>
#include <numeric>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"

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
        auto& blocks = resource.mutableBlockIds(resource.groupTags().at(0));
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
        resource.mutableBlockIds(resource.groupTags().at(0)).assign(full_block_ids);
        resource.mutableBlockIds(resource.groupTags().at(1)).assign(linear_block_ids);
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

TEST_F(LayerCacheBufferUtilFilterTest, SwaPhysicalTailMatchesPublicationForCpFourToOne) {
    for (size_t key_count : {280u, 281u, 282u, 283u}) {
        SCOPED_TRACE(key_count);
        const size_t         slots = (key_count + 3) / 4;
        std::vector<int32_t> blocks(slots, -1);
        blocks[slots - 2] = 101;
        blocks[slots - 1] = 102;
        std::vector<int64_t> keys(key_count);
        std::iota(keys.begin(), keys.end(), 1000);
        topology_ = test::makeTestCacheTopology(1, 1, {{0}}, 64, {CacheGroupType::SWA});
        KVCacheResource resource;
        resource.initGroups(topology_);
        resource.mutableBlockIds(resource.groupTags().at(0)).assign(blocks);
        resource.cacheKeys()            = keys;
        const auto&               group = topology_->group("group0");
        const std::vector<size_t> positions =
            key_count == 280 ? std::vector<size_t>{275, 279} : std::vector<size_t>{279, key_count - 1};

        // Real publication policy: two physical tail slots, not the last two base keys.
        const auto published = buildCacheStorePlan(group.policy, key_count, 0, true, 0, 4, 1, key_count);
        ASSERT_EQ(published.size(), 2u);
        for (size_t i = 0; i < positions.size(); ++i) {
            EXPECT_EQ(published[i].key_index, positions[i]);
            EXPECT_EQ(published[i].offset_index, slots - 2 + i);
        }
        ParallelismConfig pc;
        pc.tp_size                            = 4;
        pc.prefill_cp_config.kv_cache_sharded = true;
        pc.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        const auto src                        = ShardLayoutFactory::fromTopology(*topology_, pc, RoleType::PREFILL, 16);
        pc.tp_size                            = 1;
        pc.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
        const auto dst                        = ShardLayoutFactory::fromTopology(*topology_, pc, RoleType::DECODE, 16);
        const auto plan                       = KVCacheTransferPlanner::plan(src, dst, {group.tag});
        ASSERT_TRUE(plan.ok()) << plan.error.ToString();
        ASSERT_EQ(plan.plan.routes.size(), 1u);
        const auto resolved = KVCacheTransferPlanner::resolveKeys(plan.plan.routes.front().src_keys, key_count);
        EXPECT_EQ(resolved, positions);
        auto decode  = LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, resolved, 0, 1, 16);
        auto prefill = LayerCacheBufferUtil::convertLayerTag(resource, group, 0, 0, -1, 0, 4, 16);
        ASSERT_TRUE(decode.ok()) << decode.status().ToString();
        ASSERT_TRUE(prefill.ok()) << prefill.status().ToString();
        ASSERT_NE(decode.value(), nullptr);
        ASSERT_NE(prefill.value(), nullptr);
        for (size_t i = 0; i < positions.size(); ++i) {
            EXPECT_EQ(decode.value()->getBlockId(keys[positions[i]]), 101 + i);
            EXPECT_EQ(prefill.value()->getBlockId(keys[positions[i]]), 101 + i);
        }
        EXPECT_EQ(decode.value()->blockIdMap().size(), 2u);
        EXPECT_EQ(prefill.value()->blockIdMap().size(), 2u);
        // Prefix clipping keeps the full-sequence projection and its physical slot.
        auto suffix = LayerCacheBufferUtil::convertLayerTag(resource, group, 0, key_count - 1, 1, 0, 1, 16);
        ASSERT_TRUE(suffix.ok()) << suffix.status().ToString();
        ASSERT_NE(suffix.value(), nullptr);
        EXPECT_EQ(suffix.value()->blockIdMap().size(), 1u);
        EXPECT_EQ(suffix.value()->getBlockId(keys.back()), 102);

        resource.mutableBlockIds(resource.groupTags().at(0)).setAt(slots - 2, -1);
        auto missing = LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, positions, 0, 1, 16);
        ASSERT_FALSE(missing.ok());
        EXPECT_NE(missing.status().ToString().find("null/invalid block"), std::string::npos);
    }
}

TEST_F(LayerCacheBufferUtilFilterTest, UnscaledSwaUsesDirectSlotsEvenOnFourPrefillRanks) {
    auto resource = makeResource(
        1, 1, {0}, {CacheGroupType::SWA}, {-1, -1, -1, -1, -1, -1, 31, 32}, {100, 101, 102, 103, 104, 105, 106, 107});
    const auto& group  = topology_->group("group0");
    auto        result = LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, {6, 7}, 0, 4, 1);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_NE(result.value(), nullptr);
    EXPECT_EQ(result.value()->getBlockId(106), 31);
    EXPECT_EQ(result.value()->getBlockId(107), 32);
    EXPECT_FALSE(LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, {5, 7}, 0, 4, 1).ok());
    EXPECT_FALSE(LayerCacheBufferUtil::convertLayerTagForRoute(resource, group, 0, {6, 7}, 0, 4, 0).ok());
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
