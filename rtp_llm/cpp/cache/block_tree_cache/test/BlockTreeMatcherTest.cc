#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeMatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

class BlockTreeMatcherTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto                     tree       = std::make_unique<BlockTree>(1);
        auto                     full_group = std::make_shared<FullGroupSet>();
        std::vector<GroupSetPtr> groups     = {full_group};
        cache_                              = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
    }

    BlockTreeMatcher makeMatcher(BlockTreeCache& cache) {
        return BlockTreeMatcher(
            cache.tree(), cache.group_sets_, cache.reusable_group_locations_, cache.evictor_);
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeMatcherTest, NoMatchReturnsEmptyResultAndPath) {
    BlockTreeMatcher matcher = makeMatcher(*cache_);
    std::lock_guard<std::mutex> lock(cache_->mutex_);
    auto [result, logical_matched_path] = matcher.matchLocked({100});

    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_TRUE(logical_matched_path.empty());
}

TEST_F(BlockTreeMatcherTest, MatchLifecycleBalancesRequestReferences) {
    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    cache_->insert(nullptr, {100, 200}, slots);

    const DeviceBlockPoolPtr& pool = cache_->groupSets()[0]->devicePools()[0];
    ASSERT_EQ(pool->refCount(42), 1u);
    ASSERT_EQ(pool->refCount(43), 1u);

    BlockTreeMatcher matcher(
        cache_->tree(), cache_->group_sets_, cache_->reusable_group_locations_, cache_->evictor_);
    std::lock_guard<std::mutex> lock(cache_->mutex_);
    auto [result, logical_matched_path] = matcher.matchLocked({100, 200});

    ASSERT_EQ(logical_matched_path.size(), 2u);
    EXPECT_EQ(logical_matched_path[0]->cache_key, 100);
    EXPECT_EQ(logical_matched_path[1]->cache_key, 200);
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(matcher.matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{42, 43}));
    EXPECT_TRUE(matcher.matchedBlocksForGroup(99, result.matched_resources).empty());
    EXPECT_EQ(pool->refCount(42), 2u);
    EXPECT_EQ(pool->refCount(43), 2u);

    matcher.releaseMatchedResourcesLocked(result.matched_resources);
    EXPECT_EQ(pool->refCount(42), 1u);
    EXPECT_EQ(pool->refCount(43), 1u);
}

TEST_F(BlockTreeMatcherTest, GroupPoliciesSelectTheirReadyReuseRanges) {
    auto full   = std::make_shared<FullGroupSet>();
    auto linear = std::make_shared<LinearGroupSet>();
    auto swa    = std::make_shared<SWAGroupSet>(128, 64);
    std::vector<GroupSetPtr> groups = {full, linear, swa};
    auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(3), std::move(groups));

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatcher matcher = makeMatcher(*cache);
    std::lock_guard<std::mutex> lock(cache->mutex_);
    auto [result, logical_matched_path] = matcher.matchLocked({100, 200, 300});

    EXPECT_EQ(logical_matched_path.size(), 3u);
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(matcher.matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(matcher.matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22}));
    EXPECT_EQ(matcher.matchedBlocksForGroup(2, result.matched_resources), (BlockIndicesType{31, 32}));
    matcher.releaseMatchedResourcesLocked(result.matched_resources);
}

TEST_F(BlockTreeMatcherTest, SwaGapRequiresACompleteWindowForLogicalMatch) {
    auto full = std::make_shared<FullGroupSet>();
    auto swa  = std::make_shared<SWAGroupSet>(128, 64);
    std::vector<GroupSetPtr> groups = {full, swa};
    auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));

    std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
    }
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatcher matcher = makeMatcher(*cache);
    std::lock_guard<std::mutex> lock(cache->mutex_);
    auto [partial_result, partial_path] = matcher.matchLocked({100, 200, 300});
    EXPECT_EQ(partial_path.size(), 1u);
    EXPECT_EQ(partial_result.matched_blocks, 1u);
    matcher.releaseMatchedResourcesLocked(partial_result.matched_resources);

    auto [full_result, full_path] = matcher.matchLocked({100, 200, 300, 400});
    EXPECT_EQ(full_path.size(), 4u);
    EXPECT_EQ(full_result.matched_blocks, 4u);
    EXPECT_EQ(matcher.matchedBlocksForGroup(1, full_result.matched_resources), (BlockIndicesType{22, 23}));
    matcher.releaseMatchedResourcesLocked(full_result.matched_resources);
}

TEST_F(BlockTreeMatcherTest, BusySwaSlotOutsideWindowDoesNotTruncateLogicalMatch) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto                            full   = std::make_shared<FullGroupSet>();
        auto                            swa    = std::make_shared<SWAGroupSet>(2, 1);
        std::vector<GroupSetPtr>        groups = {full, swa};
        std::unique_ptr<BlockTreeCache> cache =
            makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));
        ASSERT_NE(cache, nullptr);

        std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        cache->insert(nullptr, {100, 200, 300, 400}, slots);

        const std::vector<TreeNode*> path = cache->tree()->findNode({100, 200, 300, 400}).path;
        ASSERT_EQ(path.size(), 4u);
        path[1]->group_set_resources[1].transfer_state = state;

        BlockTreeMatcher matcher = makeMatcher(*cache);
        {
            std::lock_guard<std::mutex> lock(cache->mutex_);
            auto [result, logical_matched_path] = matcher.matchLocked({100, 200, 300, 400});

            EXPECT_EQ(logical_matched_path.size(), 4u);
            EXPECT_EQ(result.matched_blocks, 4u);
            EXPECT_EQ(matcher.matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12, 13}));
            EXPECT_EQ(matcher.matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22, 23}));

            matcher.releaseMatchedResourcesLocked(result.matched_resources);
            path[1]->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
        }
    }
}

TEST_F(BlockTreeMatcherTest, BusyLinearSlotDoesNotTruncateLaterPointState) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto                            full   = std::make_shared<FullGroupSet>();
        auto                            linear = std::make_shared<LinearGroupSet>();
        std::vector<GroupSetPtr>        groups = {full, linear};
        std::unique_ptr<BlockTreeCache> cache =
            makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));
        ASSERT_NE(cache, nullptr);

        std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        cache->insert(nullptr, {100, 200, 300}, slots);

        const std::vector<TreeNode*> path = cache->tree()->findNode({100, 200, 300}).path;
        ASSERT_EQ(path.size(), 3u);
        path[1]->group_set_resources[1].transfer_state = state;

        BlockTreeMatcher matcher = makeMatcher(*cache);
        {
            std::lock_guard<std::mutex> lock(cache->mutex_);
            auto [result, logical_matched_path] = matcher.matchLocked({100, 200, 300});

            EXPECT_EQ(logical_matched_path.size(), 3u);
            EXPECT_EQ(result.matched_blocks, 3u);
            EXPECT_EQ(matcher.matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12}));
            EXPECT_EQ(matcher.matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22}));

            matcher.releaseMatchedResourcesLocked(result.matched_resources);
            path[1]->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
        }
    }
}

TEST_F(BlockTreeMatcherTest, BusyFullSlotTruncatesLogicalMatch) {
    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
    }
    cache_->insert(nullptr, {100, 200, 300}, slots);

    const std::vector<TreeNode*> path = cache_->tree()->findNode({100, 200, 300}).path;
    ASSERT_EQ(path.size(), 3u);
    path[1]->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    BlockTreeMatcher matcher = makeMatcher(*cache_);
    {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        auto [result, logical_matched_path] = matcher.matchLocked({100, 200, 300});

        EXPECT_EQ(logical_matched_path.size(), 1u);
        if (!logical_matched_path.empty()) {
            EXPECT_EQ(logical_matched_path.front()->cache_key, 100);
        }
        EXPECT_EQ(result.matched_blocks, 1u);
        EXPECT_EQ(matcher.matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10}));

        matcher.releaseMatchedResourcesLocked(result.matched_resources);
        path[1]->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
    }
}

TEST_F(BlockTreeMatcherTest, MatchFailsFastForPartialDeviceResource) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    cache_->insert(nullptr, {100}, slots);

    TreeNode* node = cache_->tree()->root()->children.at(100);
    node->group_set_resources[0].device_blocks = {10, NULL_BLOCK_IDX};

    BlockTreeMatcher matcher = makeMatcher(*cache_);
    std::lock_guard<std::mutex> lock(cache_->mutex_);
    EXPECT_THROW(matcher.matchLocked({100}), std::runtime_error);

    node->group_set_resources[0].device_blocks = {10};
}

TEST_F(BlockTreeMatcherTest, ReleaseValidatesWholeBatchBeforeMutation) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    cache_->insert(nullptr, {100}, slots);

    const DeviceBlockPoolPtr& pool = cache_->groupSets()[0]->devicePools()[0];
    BlockTreeMatcher          matcher = makeMatcher(*cache_);
    std::lock_guard<std::mutex> lock(cache_->mutex_);
    auto match_output = matcher.matchLocked({100});
    BlockTreeMatchResult& result = match_output.first;
    ASSERT_EQ(result.matched_resources.size(), 1u);
    ASSERT_EQ(pool->refCount(10), 2u);

    MultiNodeResource invalid_group{1, Tier::DEVICE, {{10}}};
    EXPECT_THROW(matcher.releaseMatchedResourcesLocked({result.matched_resources[0], invalid_group}),
                 std::runtime_error);
    EXPECT_EQ(pool->refCount(10), 2u);

    matcher.releaseMatchedResourcesLocked(result.matched_resources);
    EXPECT_EQ(pool->refCount(10), 1u);
}

TEST_F(BlockTreeMatcherTest, ReleaseReadmitsEvictionCandidate) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    cache_->insert(nullptr, {100}, slots);

    BlockTreeMatcher matcher = makeMatcher(*cache_);
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        auto match_output = matcher.matchLocked({100});
        result = std::move(match_output.first);
    }

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 0);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        matcher.releaseMatchedResourcesLocked(result.matched_resources);
    }
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 1u);
}

}  // namespace
}  // namespace rtp_llm
