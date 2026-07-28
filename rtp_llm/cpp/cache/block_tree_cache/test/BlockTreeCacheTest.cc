#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm::block_tree_cache_test {
class LoadShutdownTestPeer {
public:
    static void setShutdownWaitObserver(LoadTicketRegistry& registry, const std::function<void()>& observer) {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.shutdown_wait_observer_for_test_ = observer;
    }

    static void setShutdownWaitObserver(BlockTreeCache& cache, const std::function<void()>& observer) {
        setShutdownWaitObserver(*cache.loader_.load_ticket_registry_, observer);
    }

    static void setPendingTaskWaitObserver(BlockTreeCache& cache, const std::function<void()>& observer) {
        std::lock_guard<std::mutex> lock(cache.task_pool_->wait_mutex_);
        cache.task_pool_->pending_task_wait_observer_for_test_ = observer;
    }
};
}  // namespace rtp_llm::block_tree_cache_test

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;
using PendingLoadItem = LoadTicket::PendingLoadItem;

std::vector<std::string> makeTestTags(size_t count, size_t first_tag = 0) {
    std::vector<std::string> tags;
    tags.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        tags.push_back("tag_" + std::to_string(first_tag + index));
    }
    return tags;
}

std::vector<DeviceBlockPoolPtr> makeStructuralDevicePools(size_t count, const std::string& pool_name_prefix) {
    static std::atomic<size_t>      next_pool_id{0};
    std::vector<DeviceBlockPoolPtr> pools;
    pools.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        constexpr size_t physical_block_count = 129;
        constexpr size_t block_bytes          = 1;

        MemoryLayoutConfig layout;
        layout.layer_num                  = 1;
        layout.block_num                  = static_cast<uint32_t>(physical_block_count);
        layout.dtype                      = TYPE_INT8;
        layout.kv_cache_offset_bytes      = 0;
        layout.kv_block_stride_bytes      = block_bytes;
        layout.kv_block_pool_size_bytes   = physical_block_count * block_bytes;
        layout.block_stride_bytes         = block_bytes;
        layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;

        auto config                     = std::make_shared<DeviceBlockPoolConfig>();
        config->pool_type               = BlockPoolType::DEVICE;
        config->pool_name               = pool_name_prefix + "_" + std::to_string(next_pool_id.fetch_add(1));
        config->physical_block_count    = physical_block_count;
        config->total_size_bytes        = layout.total_size_bytes;
        config->memory_layouts          = {layout};
        config->use_cuda_malloc_backing = false;

        auto device_pool = std::make_shared<DeviceBlockPool>(config);
        RTP_LLM_CHECK(device_pool->init());
        pools.push_back(std::move(device_pool));
    }
    return pools;
}

void initializeTestGroupSet(const GroupSetPtr&                     group_set,
                            const std::vector<DeviceBlockPoolPtr>& device_pools,
                            const std::vector<std::string>&        tags,
                            size_t                                 logical_layer_bytes = 1,
                            size_t                                 group_set_id        = 0) {
    RTP_LLM_CHECK(group_set != nullptr && !device_pools.empty() && device_pools.size() == tags.size());
    CacheGroupType type               = CacheGroupType::FULL;
    size_t         seq_size_per_block = 1;
    if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
        type               = CacheGroupType::SWA;
        seq_size_per_block = swa->seqSizePerBlock();
    } else if (dynamic_cast<LinearGroupSet*>(group_set.get()) != nullptr) {
        type = CacheGroupType::LINEAR;
    }
    auto policy                = defaultCacheGroupPolicy(type);
    policy.enable_prefix_reuse = true;
    if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
        policy.sliding_window_size = static_cast<int>(swa->slidingWindowSize());
    }

    std::vector<block_transfer_engine_test::TestGroupSpec> specs;
    std::vector<size_t>                                    group_ids;
    specs.reserve(tags.size());
    group_ids.reserve(tags.size());
    for (size_t group_id = 0; group_id < tags.size(); ++group_id) {
        specs.push_back({tags[group_id], policy, {0}, logical_layer_bytes, 0, 128, seq_size_per_block});
        group_ids.push_back(group_id);
    }
    group_set->initialize(group_set_id,
                          block_transfer_engine_test::makeTestTopology(std::move(specs)),
                          std::move(group_ids),
                          device_pools);
}

void initializeSingleMemberGroupSets(const std::vector<GroupSetPtr>&        group_sets,
                                     const std::vector<DeviceBlockPoolPtr>& device_pools,
                                     const std::vector<std::string>&        tags,
                                     size_t                                 logical_layer_bytes = 1) {
    RTP_LLM_CHECK(!group_sets.empty() && group_sets.size() == device_pools.size() && group_sets.size() == tags.size());
    std::vector<block_transfer_engine_test::TestGroupSpec> specs;
    specs.reserve(group_sets.size());
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets[group_set_id];
        RTP_LLM_CHECK(group_set != nullptr);
        CacheGroupType type               = CacheGroupType::FULL;
        size_t         seq_size_per_block = 1;
        if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
            type               = CacheGroupType::SWA;
            seq_size_per_block = swa->seqSizePerBlock();
        } else if (dynamic_cast<LinearGroupSet*>(group_set.get()) != nullptr) {
            type = CacheGroupType::LINEAR;
        }
        auto policy                = defaultCacheGroupPolicy(type);
        policy.enable_prefix_reuse = true;
        if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
            policy.sliding_window_size = static_cast<int>(swa->slidingWindowSize());
        }
        specs.push_back({tags[group_set_id],
                         policy,
                         {static_cast<int>(group_set_id)},
                         logical_layer_bytes,
                         0,
                         128,
                         seq_size_per_block});
    }
    auto topology = block_transfer_engine_test::makeTestTopology(std::move(specs));
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        group_sets[group_set_id]->initialize(group_set_id, topology, {group_set_id}, {device_pools[group_set_id]});
    }
}

class CallbackBarrier {
public:
    void enterAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    void waitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return entered_; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    entered_{false};
    bool                    released_{false};
};

class BarrierThrowingPerRankBlockTransferEngine final: public PerRankBlockTransferEngine {
public:
    BarrierThrowingPerRankBlockTransferEngine(const std::vector<GroupSetPtr>&  groups,
                                              std::shared_ptr<CallbackBarrier> barrier):
        PerRankBlockTransferEngine(groups), barrier_(std::move(barrier)) {}

    std::shared_ptr<AsyncContext> submit(const TransferDescriptor&) override {
        barrier_->enterAndWait();
        throw std::runtime_error("injected copy failure");
    }

private:
    std::shared_ptr<CallbackBarrier> barrier_;
};

class ThreadCompletion {
public:
    void markEntered() {
        std::lock_guard<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
    }

    void waitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return entered_; });
    }

    void markFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_;
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    entered_{false};
    bool                    finished_{false};
};

class CountedEvent {
public:
    void notify() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_all();
    }

    void waitUntilCount(size_t expected_count) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, expected_count] { return count_ >= expected_count; });
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  count_{0};
};

class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto                     tree       = std::make_unique<BlockTree>(1);
        auto                     full_group = std::make_shared<FullGroupSet>();
        std::vector<GroupSetPtr> groups     = {full_group};

        cache_ = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyThenFullAndPartialPath) {
    BlockTreeMatchResult empty_result = cache_->match({100, 200, 300});
    EXPECT_EQ(empty_result.matched_node, nullptr);
    EXPECT_EQ(empty_result.matched_blocks, 0u);
    EXPECT_TRUE(empty_result.matched_resources.empty());

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult full_result = cache_->match({100, 200, 300});
    ASSERT_NE(full_result.matched_node, nullptr);
    EXPECT_EQ(full_result.matched_node->cache_key, 300);
    EXPECT_EQ(full_result.matched_blocks, 3u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, full_result.matched_resources), (BlockIndicesType{42, 43, 44}));
    cache_->releaseMatchedResources(full_result.matched_resources);

    BlockTreeMatchResult partial_result = cache_->match({100, 200, 999});
    ASSERT_NE(partial_result.matched_node, nullptr);
    EXPECT_EQ(partial_result.matched_node->cache_key, 200);
    EXPECT_EQ(partial_result.matched_blocks, 2u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, partial_result.matched_resources), (BlockIndicesType{42, 43}));
    cache_->releaseMatchedResources(partial_result.matched_resources);
}

TEST_F(BlockTreeCacheTest, KeySnapshotTracksMutationVersionAndLimit) {
    const auto empty = cache_->getKeySnapshot(/*limit=*/10);
    EXPECT_EQ(empty.version, 0u);
    EXPECT_TRUE(empty.keys.empty());

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

    const auto version_only = cache_->getKeySnapshot(/*limit=*/0);
    EXPECT_GT(version_only.version, empty.version);
    EXPECT_TRUE(version_only.keys.empty());

    const auto limited = cache_->getKeySnapshot(/*limit=*/2);
    EXPECT_EQ(limited.version, version_only.version);
    EXPECT_EQ(limited.keys.size(), 2u);
    for (CacheKeyType key : limited.keys) {
        EXPECT_TRUE(key == 100 || key == 200 || key == 300);
    }
}

TEST_F(BlockTreeCacheTest, MatchPartialPath) {
    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    slots[2][0].device_blocks = {12};

    ASSERT_TRUE(insertGroupSetSlots(*cache_, nullptr, {100, 200, 300}, slots));

    BlockTreeMatchResult result = cache_->match({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 100);
    EXPECT_EQ(result.matched_blocks, 1u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10}));

    cache_->releaseMatchedResources(result.matched_resources);
}

TEST_F(BlockTreeCacheTest, MatchFailsFastAtPartialDeviceSlot) {
    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, slots);

    TreeNode* first_node                             = cache_->tree()->root()->children.at(100);
    first_node->group_set_resources[0].device_blocks = {10, NULL_BLOCK_IDX};

    EXPECT_THROW(cache_->match({100, 200}), std::runtime_error);

    // Restore the production slot-shape invariant before the fixture drains
    // synthetic tree holds during teardown.
    first_node->group_set_resources[0].device_blocks = {10};
}

TEST_F(BlockTreeCacheTest, MatchFailsFastAtIdleResourceWithMultipleServingTiers) {
    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, slots);

    TreeNode* first_node                          = cache_->tree()->root()->children.at(100);
    first_node->group_set_resources[0].host_block = 7;

    EXPECT_THROW(cache_->match({100, 200}), std::runtime_error);

    first_node->group_set_resources[0].host_block = NULL_BLOCK_IDX;
}

TEST_F(BlockTreeCacheTest, MatchDoesNotReuseBusyFullSlot) {
    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, slots);

    TreeNode* first_node                              = cache_->tree()->root()->children.at(100);
    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    const BlockTreeMatchResult result = cache_->match({100, 200});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.matched_resources.empty());

    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
}

TEST_F(BlockTreeCacheTest, MatchSkipsBusySwaSlotWithoutTruncatingFullPrefix) {
    // FULL + SWA(window=2 blocks): a busy SWA slot on a middle node is outside
    // the trailing window and must not truncate the FULL prefix match.
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto                     tree       = std::make_unique<BlockTree>(2);
        auto                     full_group = std::make_shared<FullGroupSet>();
        auto                     swa_group  = std::make_shared<SWAGroupSet>(2, 1);
        std::vector<GroupSetPtr> groups     = {full_group, swa_group};
        std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
        ASSERT_NE(multi_cache, nullptr);

        std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < 4; ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        multi_cache->insert(nullptr, {100, 200, 300, 400}, slots);

        TreeNode* busy_node = multi_cache->tree()->root()->children.at(100)->children.at(200);
        busy_node->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result = multi_cache->match({100, 200, 300, 400});
        ASSERT_NE(result.matched_node, nullptr);
        EXPECT_EQ(result.matched_node->cache_key, 400);
        EXPECT_EQ(result.matched_blocks, 4u);
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12, 13}));
        // SWA locks only the trailing window; the busy middle slot stays untouched.
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22, 23}));
        const auto& swa_pool = multi_cache->groupSets()[1]->devicePools()[0];
        EXPECT_EQ(swa_pool->refCount(21), 1u);  // cache hold only, no match reference
        EXPECT_EQ(swa_pool->refCount(22), 2u);

        multi_cache->releaseMatchedResources(result.matched_resources);
        busy_node->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    }
}

TEST_F(BlockTreeCacheTest, MatchStillTruncatesAtBusyFullSlot) {
    // The FULL prefix latch must keep truncating when the FULL slot itself is busy.
    auto                     tree       = std::make_unique<BlockTree>(2);
    auto                     full_group = std::make_shared<FullGroupSet>();
    auto                     swa_group  = std::make_shared<SWAGroupSet>(2, 1);
    std::vector<GroupSetPtr> groups     = {full_group, swa_group};
    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
    ASSERT_NE(multi_cache, nullptr);

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(2));
    for (size_t i = 0; i < 3; ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
    }
    multi_cache->insert(nullptr, {100, 200, 300}, slots);

    TreeNode* busy_node = multi_cache->tree()->root()->children.at(100)->children.at(200);
    busy_node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    BlockTreeMatchResult result = multi_cache->match({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 100);
    EXPECT_EQ(result.matched_blocks, 1u);
    EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10}));

    multi_cache->releaseMatchedResources(result.matched_resources);
    busy_node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
}

TEST_F(BlockTreeCacheTest, MatchSkipsBusyLinearSlotAndReusesTailState) {
    // LINEAR only consumes the deepest node's state; a busy middle slot must
    // not truncate the match nor be referenced.
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto                     tree         = std::make_unique<BlockTree>(2);
        auto                     full_group   = std::make_shared<FullGroupSet>();
        auto                     linear_group = std::make_shared<LinearGroupSet>();
        std::vector<GroupSetPtr> groups       = {full_group, linear_group};
        std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
        ASSERT_NE(multi_cache, nullptr);

        std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < 3; ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        multi_cache->insert(nullptr, {100, 200, 300}, slots);

        TreeNode* busy_node = multi_cache->tree()->root()->children.at(100)->children.at(200);
        busy_node->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result = multi_cache->match({100, 200, 300});
        ASSERT_NE(result.matched_node, nullptr);
        EXPECT_EQ(result.matched_node->cache_key, 300);
        EXPECT_EQ(result.matched_blocks, 3u);
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12}));
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22}));
        const auto& linear_pool = multi_cache->groupSets()[1]->devicePools()[0];
        EXPECT_EQ(linear_pool->refCount(21), 1u);  // busy middle slot not referenced
        EXPECT_EQ(linear_pool->refCount(22), 2u);

        multi_cache->releaseMatchedResources(result.matched_resources);
        busy_node->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    }
}

TEST_F(BlockTreeCacheTest, InsertFailsFastForNonIdleOrMultiTierResource) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks  = {10};
    slots[0][0].transfer_state = GroupSetTransferState::DEMOTING;
    EXPECT_THROW(cache_->insert(nullptr, {100}, slots), std::runtime_error);
    EXPECT_EQ(cache_->tree()->nodeCount(), 0u);

    slots[0][0].transfer_state = GroupSetTransferState::IDLE;
    slots[0][0].host_block     = 7;
    EXPECT_THROW(cache_->insert(nullptr, {100}, slots), std::runtime_error);
    EXPECT_EQ(cache_->tree()->nodeCount(), 0u);
}

TEST_F(BlockTreeCacheTest, ReleaseMatchedResourcesFailsFastForMalformedCanonicalResource) {
    MultiNodeResource wrong_width{0, Tier::DEVICE, {{10, 11}}};
    EXPECT_THROW(cache_->releaseMatchedResources({wrong_width}), std::runtime_error);

    MultiNodeResource null_block{0, Tier::DEVICE, {{NULL_BLOCK_IDX}}};
    EXPECT_THROW(cache_->releaseMatchedResources({null_block}), std::runtime_error);

    MultiNodeResource misaligned_nodes{0, Tier::DEVICE, {{10}}};
    misaligned_nodes.tree_nodes = {cache_->tree()->root(), cache_->tree()->root()};
    EXPECT_THROW(cache_->releaseMatchedResources({misaligned_nodes}), std::runtime_error);

    MultiNodeResource duplicate{0, Tier::DEVICE, {{10}}};
    EXPECT_THROW(cache_->releaseMatchedResources({duplicate, duplicate}), std::runtime_error);
}

TEST_F(BlockTreeCacheTest, ReleaseMatchedResourcesValidatesWholeBatchBeforeMutation) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {10};
    cache_->insert(nullptr, {100}, slots);

    BlockTreeMatchResult match = cache_->match({100});
    ASSERT_EQ(match.matched_resources.size(), 1u);
    const auto& pool = cache_->groupSets()[0]->devicePools()[0];
    ASSERT_EQ(pool->refCount(10), 2u);

    MultiNodeResource invalid_group{1, Tier::DEVICE, {{10}}};
    EXPECT_THROW(cache_->releaseMatchedResources({match.matched_resources[0], invalid_group}), std::runtime_error);
    EXPECT_EQ(pool->refCount(10), 2u);

    cache_->releaseMatchedResources(match.matched_resources);
    EXPECT_EQ(pool->refCount(10), 1u);
}

TEST_F(BlockTreeCacheTest, DuplicateInsertDoesNotCreateNodes) {
    CacheStats stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<std::vector<GroupSetResource>> original_slots(2, std::vector<GroupSetResource>(1));
    original_slots[0][0].device_blocks = {10};
    original_slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, original_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    std::vector<std::vector<GroupSetResource>> duplicate_slots(2, std::vector<GroupSetResource>(1));
    duplicate_slots[0][0].device_blocks = {20};
    duplicate_slots[1][0].device_blocks = {21};
    cache_->insert(nullptr, {100, 200}, duplicate_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    BlockTreeFindResult find_result = cache_->tree()->findNode({100, 200});
    ASSERT_EQ(find_result.path.size(), 2u);
    EXPECT_EQ(find_result.path[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(find_result.path[1]->group_set_resources[0].device_blocks, (BlockIndicesType{11}));
}

TEST_F(BlockTreeCacheTest, ReclaimCascadesToLowerPriorityGroup) {
    // Build a cache with Full + SWA groups
    auto tree = std::make_unique<BlockTree>(2);  // 2 group sets

    auto full_group = std::make_shared<FullGroupSet>();

    auto swa_group = std::make_shared<SWAGroupSet>(128, 64);

    std::vector<GroupSetPtr> groups = {full_group, swa_group};

    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(2));
    slots[0][0].device_blocks = {10};  // Full
    slots[0][1].device_blocks = {20};  // SWA

    multi_cache->insert(nullptr, {100}, slots);

    // Reclaim Full group at DEVICE → should cascade to SWA.
    int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*multi_cache, 1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);

    multi_cache->waitForPendingTasks();
}

TEST_F(BlockTreeCacheTest, MultiGroupConstruction) {
    auto tree = std::make_unique<BlockTree>(3);

    auto full = std::make_shared<FullGroupSet>();

    auto swa = std::make_shared<SWAGroupSet>(128, 64);

    auto linear = std::make_shared<LinearGroupSet>();

    std::vector<GroupSetPtr> groups = {full, swa, linear};

    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));

    EXPECT_EQ(multi_cache->groupSets().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSetResourceCount(), 3);
}

TEST(BlockTreeCacheConstructionTest, OutOfRangeGroupSetIdFailsInitializationWithoutThrowing) {
    auto tree = std::make_unique<BlockTree>(1);
    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(
        full, makeStructuralDevicePools(1, "out_of_range_group_set"), {"kv"}, /*logical_layer_bytes=*/1, 1);
    std::vector<GroupSetPtr> groups          = {full};
    auto                     per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(groups);
    auto transfer_dispatcher                 = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool                           = std::make_unique<BlockTreeTaskPool>(2, 1000, "BlockTreeEvictionPool");

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool));
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(full->groupSetId(), 1);
}

TEST(BlockTreeCacheConstructionTest, NullGroupSetFailsInitializationAndDestructionReturnsNormally) {
    auto                     tree            = std::make_unique<BlockTree>(1);
    std::vector<GroupSetPtr> groups          = {nullptr};
    auto                     per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(groups);
    auto transfer_dispatcher                 = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool                           = std::make_unique<BlockTreeTaskPool>(2, 1000, "BlockTreeEvictionPool");

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool));
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConstructionTest, MissingCollaboratorsFailInitializationAndDestructionReturnsNormally) {
    auto                     tree   = std::make_unique<BlockTree>(1);
    auto                     full   = std::make_shared<FullGroupSet>();
    std::vector<GroupSetPtr> groups = {full};

    auto cache = std::make_unique<BlockTreeCache>(
        std::move(tree), std::move(groups), BlockTreeCacheConfig{}, nullptr, nullptr, nullptr);
    EXPECT_FALSE(cache->init());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
}

TEST_F(BlockTreeCacheTest, EmptyKeysAreNoOps) {
    const CacheStats stats_before = cache_->getStats();
    cache_->insert(nullptr, {}, {});
    const CacheStats stats_after = cache_->getStats();
    EXPECT_EQ(stats_after.tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(stats_after.device_heap_total_size, stats_before.device_heap_total_size);

    BlockTreeMatchResult result = cache_->match({});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_EQ(result.load_ticket, nullptr);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
            slots[0][0].device_blocks = {static_cast<BlockIdxType>(i * 100 + 1)};
            CacheKeysType keys        = {static_cast<CacheKeyType>(i * 1000 + 1)};
            cache_->insert(nullptr, keys, slots);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 4u);
}

TEST_F(BlockTreeCacheTest, ConcurrentDoubleMatch_LastReleaseReadmitsExactlyOnce) {
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    cache_->insert(nullptr, {100}, slots);
    ASSERT_EQ(cache_->getStats().device_heap_total_size, 1u);

    std::mutex               mutex;
    std::condition_variable  cv;
    bool                     start{false};
    size_t                   matched_count{0};
    size_t                   released_count{0};
    std::array<bool, 2>      release_match{false, false};
    std::array<size_t, 2>    matched_blocks{0, 0};
    std::vector<std::thread> threads;
    threads.reserve(2);
    for (size_t thread_id = 0; thread_id < 2; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return start; });
            }
            BlockTreeMatchResult result = cache_->match({100});
            {
                std::unique_lock<std::mutex> lock(mutex);
                matched_blocks[thread_id] = result.matched_blocks;
                ++matched_count;
                cv.notify_all();
                cv.wait(lock, [&] { return release_match[thread_id]; });
            }
            cache_->releaseMatchedResources(result.matched_resources);
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++released_count;
                cv.notify_all();
            }
        });
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
        cv.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return matched_count == 2; });
    }
    EXPECT_EQ(matched_blocks, (std::array<size_t, 2>{1, 1}));

    // Selection lazily drops the now request-pinned candidate. It must stay out
    // after only one of the two concurrent holders releases it.
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 0);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_match[0] = true;
        cv.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return released_count == 1; });
    }
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_match[1] = true;
        cv.notify_all();
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(released_count, 2u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    cache_->waitForPendingTasks();
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ConcurrentMatchInsertSameAndForkedPrefixes) {
    constexpr size_t kThreadCount = 6;
    constexpr size_t kIterations  = 200;

    std::mutex               start_mutex;
    std::condition_variable  start_cv;
    bool                     start{false};
    std::atomic<bool>        consistent{true};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                start_cv.wait(lock, [&] { return start; });
            }
            const CacheKeyType fork_key   = static_cast<CacheKeyType>(1000 + thread_id);
            const BlockIdxType fork_block = static_cast<BlockIdxType>(20 + thread_id);
            for (size_t iteration = 0; iteration < kIterations; ++iteration) {
                std::vector<std::vector<GroupSetResource>> same_slots(2, std::vector<GroupSetResource>(1));
                same_slots[0][0].device_blocks = {10};
                same_slots[1][0].device_blocks = {11};
                cache_->insert(nullptr, {100, 200}, same_slots);

                std::vector<std::vector<GroupSetResource>> fork_slots(2, std::vector<GroupSetResource>(1));
                fork_slots[0][0].device_blocks = {10};
                fork_slots[1][0].device_blocks = {fork_block};
                cache_->insert(nullptr, {100, fork_key}, fork_slots);

                for (const CacheKeysType& keys : {CacheKeysType{100, 200}, CacheKeysType{100, fork_key}}) {
                    BlockTreeMatchResult match  = cache_->match(keys);
                    const auto           blocks = cache_->matchedBlocksForGroup(0, match.matched_resources);
                    if (match.matched_blocks != 2 || blocks.size() != 2 || blocks[0] != 10) {
                        consistent.store(false);
                    }
                    cache_->releaseMatchedResources(match.matched_resources);
                }
            }
        });
    }

    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start = true;
        start_cv.notify_all();
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_TRUE(consistent.load());
    const CacheStats stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, kThreadCount + 2u);         // shared parent + same leaf + fork leaves
    EXPECT_EQ(stats.device_heap_total_size, kThreadCount + 1u);  // every leaf appears exactly once

    const auto& pool = cache_->groupSets()[0]->devicePools()[0];
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->refCount(10), 1u);
    EXPECT_EQ(pool->refCount(11), 1u);
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        const CacheKeyType fork_key   = static_cast<CacheKeyType>(1000 + thread_id);
        const BlockIdxType fork_block = static_cast<BlockIdxType>(20 + thread_id);
        const auto         found      = cache_->tree()->findNode({100, fork_key});
        ASSERT_EQ(found.path.size(), 2u);
        EXPECT_EQ(found.path[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
        EXPECT_EQ(found.path[1]->group_set_resources[0].device_blocks, (BlockIndicesType{fork_block}));
        EXPECT_EQ(pool->refCount(fork_block), 1u);
    }

    // Final reclaim: drain leaves first, then the promoted shared parent, until
    // the tree is empty and every cache hold is released back to the pool.
    for (size_t attempt = 0; attempt < (kThreadCount + 2) * 2; ++attempt) {
        if (BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE) == 0) {
            break;
        }
        cache_->waitForPendingTasks();
    }
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    EXPECT_FALSE(pool->isAllocated(10));
    EXPECT_FALSE(pool->isAllocated(11));
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        EXPECT_FALSE(pool->isAllocated(static_cast<BlockIdxType>(20 + thread_id)));
    }
}

TEST(BlockTreeCacheFinalizationTest, CopyExceptionSettlesCreditsBeforePendingTaskCompletion) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length             = 1;
    options.usable_device_blocks    = 4;
    options.usable_host_blocks      = 4;
    options.enable_disk             = false;
    options.enable_reverse_eviction = false;
    auto environment                = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    ASSERT_NE(environment->cache, nullptr);

    auto barrier = std::make_shared<CallbackBarrier>();
    auto per_rank_transfer_engine =
        std::make_shared<BarrierThrowingPerRankBlockTransferEngine>(environment->groups, barrier);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, per_rank_transfer_engine);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    ASSERT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));

    std::vector<BlockIdxType> source_blocks;
    std::vector<size_t>       source_free_before;
    std::vector<size_t>       source_refs_before;
    source_blocks.reserve(environment->device_pools.size());
    source_free_before.reserve(environment->device_pools.size());
    source_refs_before.reserve(environment->device_pools.size());
    for (size_t tag_id = 0; tag_id < environment->device_pools.size(); ++tag_id) {
        const auto blocks = environment->blocksForTag(tag_id);
        ASSERT_EQ(blocks.size(), 1u);
        source_blocks.push_back(blocks.front());
        source_free_before.push_back(environment->device_pools[tag_id]->freeBlocksNum());
        source_refs_before.push_back(environment->device_pools[tag_id]->refCount(blocks.front()));
        ASSERT_EQ(source_refs_before.back(), 1u);
    }

    environment->cache->setTierWatermark(Tier::DEVICE, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);
    ASSERT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    barrier->waitUntilEntered();

    {
        std::lock_guard<std::mutex> lock(environment->cache->mutex_);
        EXPECT_FALSE(environment->cache->in_flight_device_release_credits_.empty());
        environment->cache->setTierWatermark(Tier::DEVICE, 0.0, 0);
    }
    barrier->release();
    environment->cache->waitForPendingTasks();

    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    {
        std::lock_guard<std::mutex> lock(environment->cache->mutex_);
        EXPECT_TRUE(environment->cache->in_flight_device_release_credits_.empty());
    }
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    for (size_t tag_id = 0; tag_id < environment->device_pools.size(); ++tag_id) {
        EXPECT_EQ(environment->device_pools[tag_id]->freeBlocksNum(), source_free_before[tag_id]);
        EXPECT_EQ(environment->device_pools[tag_id]->refCount(source_blocks[tag_id]), source_refs_before[tag_id]);
    }

    EXPECT_NO_THROW(environment->cache.reset());
}

TEST_F(BlockTreeCacheTest, FullMatch_PreservesPathAndPoolOrder) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 8;
    auto             pool0         = makeDevicePool({{64, 0}}, kUsableBlocks, "full_order_pool0");
    auto             pool1         = makeDevicePool({{64, 0}}, kUsableBlocks, "full_order_pool1");

    auto pool0_prefix = pool0->malloc(1);
    auto pool1_prefix = pool1->malloc(3);
    ASSERT_TRUE(pool0_prefix.has_value());
    ASSERT_TRUE(pool1_prefix.has_value());

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {pool0, pool1}, makeTestTags(2));
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));

    MultiNodeResource request_blocks = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 2u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    ASSERT_EQ(request_blocks.per_node[1].size(), 2u);

    const BlockIdxType a_pool0 = request_blocks.per_node[0][0];
    const BlockIdxType a_pool1 = request_blocks.per_node[0][1];
    const BlockIdxType b_pool0 = request_blocks.per_node[1][0];
    const BlockIdxType b_pool1 = request_blocks.per_node[1][1];
    EXPECT_NE(a_pool0, a_pool1);
    EXPECT_NE(b_pool0, b_pool1);

    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {a_pool0, a_pool1};
    slots[1][0].device_blocks = {b_pool0, b_pool1};
    cache->insert(nullptr, {100, 200}, slots);
    full->unreferenceBlocks(request_blocks, BlockRefType::REQUEST);
    EXPECT_TRUE(pool0->isAllocated(a_pool0));
    EXPECT_TRUE(pool0->isAllocated(b_pool0));
    EXPECT_TRUE(pool1->isAllocated(a_pool1));
    EXPECT_TRUE(pool1->isAllocated(b_pool1));

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{a_pool0, b_pool0}));
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{a_pool1, b_pool1}));
    cache->releaseMatchedResources(result.matched_resources);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 2, Tier::DEVICE), 2);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool0->isAllocated(a_pool0));
    EXPECT_FALSE(pool0->isAllocated(b_pool0));
    EXPECT_FALSE(pool1->isAllocated(a_pool1));
    EXPECT_FALSE(pool1->isAllocated(b_pool1));

    pool0->free(*pool0_prefix);
    pool1->free(*pool1_prefix);
    EXPECT_EQ(pool0->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(pool1->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, DuplicateInsert_KeepsExistingSlotAndCallerOwnsLoser) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "duplicate_insert_pool");

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {pool}, makeTestTags(1));
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));

    MultiNodeResource existing = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    MultiNodeResource loser    = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(existing.per_node.size(), 1u);
    ASSERT_EQ(loser.per_node.size(), 1u);
    ASSERT_EQ(existing.per_node[0].size(), 1u);
    ASSERT_EQ(loser.per_node[0].size(), 1u);
    const BlockIdxType existing_block = existing.per_node[0][0];
    const BlockIdxType loser_block    = loser.per_node[0][0];
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    std::vector<std::vector<GroupSetResource>> first_slots(1, std::vector<GroupSetResource>(1));
    first_slots[0][0].device_blocks = existing.per_node[0];
    cache->insert(nullptr, {100}, first_slots);
    EXPECT_EQ(pool->refCount(existing_block), 2u);
    BlockTreeFindResult initial_find = cache->tree()->findNode({100});
    ASSERT_NE(initial_find.matched_node, nullptr);
    MultiNodeResource released_existing = existing;
    released_existing.tree_nodes        = {initial_find.matched_node};
    cache->releaseMatchedResources({released_existing});
    EXPECT_EQ(pool->refCount(existing_block), 1u);

    std::vector<std::vector<GroupSetResource>> duplicate_slots(1, std::vector<GroupSetResource>(1));
    duplicate_slots[0][0].device_blocks = loser.per_node[0];
    cache->insert(nullptr, {100}, duplicate_slots);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(find.matched_node->group_set_resources[0].device_blocks, (std::vector<BlockIdxType>{existing_block}));
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    full->unreferenceBlocks(loser, BlockRefType::REQUEST);
    EXPECT_FALSE(pool->isAllocated(loser_block));
    EXPECT_TRUE(pool->isAllocated(existing_block));

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(existing_block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, DuplicateInsert_FillsExistingEmptyGroupAndAddsOneCacheHold) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "existing_group_fill_pool");

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {pool}, makeTestTags(1));
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));

    std::vector<std::vector<GroupSetResource>> empty_slots(1, std::vector<GroupSetResource>(1));
    empty_slots[0][0].device_blocks = {NULL_BLOCK_IDX};
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100}, empty_slots));
    TreeNode* existing_node = cache->tree()->nodes().front().get();
    ASSERT_NE(existing_node, nullptr);

    MultiNodeResource request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    ASSERT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(existing_node->group_set_resources[0].device_blocks, request_blocks.per_node[0]);
    EXPECT_EQ(pool->refCount(block), 2u);

    request_blocks.tree_nodes = {existing_node};
    cache->releaseMatchedResources({request_blocks});
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, InsertFailsFastForPartialMultiPoolGroupWithoutAddingCacheHold) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool0         = makeDevicePool({{64, 0}}, kUsableBlocks, "partial_group_pool_0");
    auto             pool1         = makeDevicePool({{64, 0}}, kUsableBlocks, "partial_group_pool_1");

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {pool0, pool1}, makeTestTags(2));
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));

    MultiNodeResource request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    const BlockIdxType block0 = request_blocks.per_node[0][0];
    const BlockIdxType block1 = request_blocks.per_node[0][1];

    std::vector<std::vector<GroupSetResource>> partial_slots(1, std::vector<GroupSetResource>(1));
    partial_slots[0][0].device_blocks = {block0, NULL_BLOCK_IDX};
    EXPECT_THROW(cache->insert(nullptr, {100}, partial_slots), std::runtime_error);
    EXPECT_EQ(cache->tree()->nodeCount(), 0u);
    EXPECT_EQ(pool0->refCount(block0), 1u);
    EXPECT_EQ(pool1->refCount(block1), 1u);

    full->unreferenceBlocks(request_blocks, BlockRefType::REQUEST);
    EXPECT_FALSE(pool0->isAllocated(block0));
    EXPECT_FALSE(pool1->isAllocated(block1));
}

TEST_F(BlockTreeCacheTest, InsertMatchReleaseReclaim_RefcountLifecycle) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "refcount_lifecycle_pool");

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {pool}, makeTestTags(1));
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));

    MultiNodeResource request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks - 1);
    EXPECT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);
    EXPECT_EQ(pool->refCount(block), 2u);

    full->unreferenceBlocks(request_blocks, BlockRefType::REQUEST);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{block}));
    ASSERT_EQ(result.matched_resources.size(), 1u);
    EXPECT_EQ(result.matched_resources[0].group_set_id, 0);
    EXPECT_EQ(result.matched_resources[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_resources[0].per_node, (std::vector<std::vector<BlockIdxType>>{{block}}));
    EXPECT_EQ(pool->refCount(block), 2u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 0);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    cache->releaseMatchedResources(result.matched_resources);
    result.matched_resources.clear();
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, SequentialReclaimDrainsChainWithoutHostBlocks) {
    auto                     tree   = std::make_unique<BlockTree>(1);
    auto                     full   = std::make_shared<FullGroupSet>();
    std::vector<GroupSetPtr> groups = {full};

    // No Host pool, Host disabled → direct release on reclaim.
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.eviction_thread_pool_size = 2;
    seq_cfg.enable_device_cache       = true;
    seq_cfg.enable_memory_cache       = false;

    std::unique_ptr<BlockTreeCache> ce_cache =
        makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::move(seq_cfg));

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    ce_cache->insert(nullptr, {100, 200, 300}, slots);

    // Reclaim all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*ce_cache, 1, Tier::DEVICE);
        EXPECT_EQ(reclaimed, 1) << "Reclaim " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = makeHostPool(256, 4);

    auto                     tree   = std::make_unique<BlockTree>(1);
    auto                     full   = std::make_shared<FullGroupSet>();
    std::vector<GroupSetPtr> groups = {full};

    // Host disabled (default): Device reclaim → direct release.
    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(
        std::move(tree), std::move(groups), BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto host_pool = makeHostPool(1, 2);
    auto disk_pool = makeDiskPool(1, 2, std::make_unique<MemoryDiskBlockIO>());

    auto tree = std::make_unique<BlockTree>(1);
    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, makeStructuralDevicePools(1, "tier_enable_queries"), {"kv"});
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {
    auto tree = std::make_unique<BlockTree>(1);

    auto full = std::make_shared<FullGroupSet>();

    std::vector<GroupSetPtr>        groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));

    // Insert
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Reclaim device data.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // Node should be deleted: group empty
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, SWABuildTransferSupportsHostToDisk) {
    auto swa = std::make_shared<SWAGroupSet>(128, 64);
    initializeTestGroupSet(swa, makeStructuralDevicePools(1, "swa_build_transfer"), {"swa"});

    // Create a mock tree node with host data
    auto                                       tree = std::make_unique<BlockTree>(1);
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    tree->insertNode(nullptr, {100}, slots);
    auto find = tree->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_set_resources[0].host_block = 7;

    // Verify HOST_TO_DISK transfer descriptor is correct
    TransferDescriptor desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    EXPECT_EQ(desc.host_block, 7);
}

TEST_F(BlockTreeCacheTest, MatchCollectsBlocksSelectedByGroupPolicy) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(3);

    std::shared_ptr<FullGroupSet>   full   = std::make_shared<FullGroupSet>();
    std::shared_ptr<LinearGroupSet> linear = std::make_shared<LinearGroupSet>();
    std::shared_ptr<SWAGroupSet>    swa    = std::make_shared<SWAGroupSet>(128, 64);

    std::vector<GroupSetPtr>        group_sets = {full, linear, swa};
    std::unique_ptr<BlockTreeCache> cache      = makeBlockTreeCacheForTest(std::move(tree), std::move(group_sets));

    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_resources), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_resources), (BlockIndicesType{22}));
    EXPECT_EQ(cache->matchedBlocksForGroup(2, result.matched_resources), (BlockIndicesType{31, 32}));
    cache->releaseMatchedResources(result.matched_resources);
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::unique_ptr<BlockTree>    tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>();

    std::vector<DeviceBlockPoolPtr> device_pools = makeStructuralDevicePools(2, "aggregated_device_pool");
    auto                            pool0_prefix = device_pools[0]->malloc(1);
    auto                            pool1_prefix = device_pools[1]->malloc(3);
    ASSERT_TRUE(pool0_prefix.has_value());
    ASSERT_TRUE(pool1_prefix.has_value());
    initializeTestGroupSet(full, device_pools, makeTestTags(2));

    std::vector<GroupSetPtr> group_sets = {full};
    auto                     cache      = makeBlockTreeCacheForTest(std::move(tree), std::move(group_sets));
    ASSERT_NE(cache, nullptr);

    MultiNodeResource request_holder = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
    ASSERT_EQ(request_holder.per_node.size(), 2u);
    ASSERT_EQ(request_holder.per_node[0].size(), 2u);
    ASSERT_EQ(request_holder.per_node[1].size(), 2u);
    const BlockIndicesType tag0_blocks = {request_holder.per_node[0][0], request_holder.per_node[1][0]};
    const BlockIndicesType tag1_blocks = {request_holder.per_node[0][1], request_holder.per_node[1][1]};
    EXPECT_NE(tag0_blocks, tag1_blocks);

    std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = request_holder.per_node[0];
    slots[1][0].device_blocks = request_holder.per_node[1];
    cache->insert(nullptr, {100, 200}, slots);
    full->unreferenceBlocks(request_holder, BlockRefType::REQUEST);
    device_pools[0]->free(*pool0_prefix);
    device_pools[1]->free(*pool1_prefix);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_resources), tag0_blocks);
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_resources), tag1_blocks);
    cache->releaseMatchedResources(result.matched_resources);
}

TEST_F(BlockTreeCacheTest, ReorderedPoolsPreserveTagAddressedMatchResults) {
    auto make_cache = [](std::vector<std::string> tags, const std::string& pool_name_prefix) {
        auto full = std::make_shared<FullGroupSet>();

        std::vector<DeviceBlockPoolPtr> device_pools = makeStructuralDevicePools(2, pool_name_prefix);
        std::vector<BlockIdList>        prefix_blocks;
        prefix_blocks.reserve(tags.size());
        for (size_t index = 0; index < tags.size(); ++index) {
            const size_t prefix_count = tags[index] == "hca_kv" ? 1 : 3;
            auto         prefix       = device_pools[index]->malloc(prefix_count);
            RTP_LLM_CHECK(prefix.has_value());
            prefix_blocks.push_back(std::move(*prefix));
        }
        initializeTestGroupSet(full, device_pools, tags);

        std::vector<GroupSetPtr> group_sets = {full};
        auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(group_sets));
        RTP_LLM_CHECK(cache != nullptr);

        MultiNodeResource request_holder = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
        RTP_LLM_CHECK(request_holder.per_node.size() == 2);
        RTP_LLM_CHECK(request_holder.per_node[0].size() == 2);
        RTP_LLM_CHECK(request_holder.per_node[1].size() == 2);
        std::vector<std::vector<GroupSetResource>> slots(2, std::vector<GroupSetResource>(1));
        slots[0][0].device_blocks = request_holder.per_node[0];
        slots[1][0].device_blocks = request_holder.per_node[1];
        cache->insert(nullptr, {100, 200}, slots);
        full->unreferenceBlocks(request_holder, BlockRefType::REQUEST);
        for (size_t index = 0; index < device_pools.size(); ++index) {
            device_pools[index]->free(prefix_blocks[index]);
        }
        return cache;
    };

    auto original  = make_cache({"hca_kv", "csa_kv"}, "reordered_pool_original");
    auto reordered = make_cache({"csa_kv", "hca_kv"}, "reordered_pool_swapped");

    const BlockTreeMatchResult original_result  = original->match({100, 200});
    const BlockTreeMatchResult reordered_result = reordered->match({100, 200});
    EXPECT_EQ(original_result.matched_blocks, 2u);
    EXPECT_EQ(reordered_result.matched_blocks, 2u);
    const size_t original_hca  = original->groupSets()[0]->topology()->groupIdForTag("hca_kv");
    const size_t original_csa  = original->groupSets()[0]->topology()->groupIdForTag("csa_kv");
    const size_t reordered_hca = reordered->groupSets()[0]->topology()->groupIdForTag("hca_kv");
    const size_t reordered_csa = reordered->groupSets()[0]->topology()->groupIdForTag("csa_kv");
    EXPECT_EQ(original->matchedBlocksForGroup(original_hca, original_result.matched_resources),
              reordered->matchedBlocksForGroup(reordered_hca, reordered_result.matched_resources));
    EXPECT_EQ(original->matchedBlocksForGroup(original_csa, original_result.matched_resources),
              reordered->matchedBlocksForGroup(reordered_csa, reordered_result.matched_resources));
    EXPECT_NE(original->matchedBlocksForGroup(original_hca, original_result.matched_resources),
              original->matchedBlocksForGroup(original_csa, original_result.matched_resources));
    original->releaseMatchedResources(original_result.matched_resources);
    reordered->releaseMatchedResources(reordered_result.matched_resources);
}

TEST_F(BlockTreeCacheTest, InvalidTopologyOrPoolCardinalityFailsBeforeGroupMutation) {
    auto pools = makeStructuralDevicePools(2, "invalid_membership");
    EXPECT_ANY_THROW(initializeTestGroupSet(std::make_shared<FullGroupSet>(), {pools[0]}, {""}));
    EXPECT_ANY_THROW(
        initializeTestGroupSet(std::make_shared<FullGroupSet>(), {pools[0], pools[1]}, {"duplicate", "duplicate"}));

    auto group    = std::make_shared<FullGroupSet>();
    auto topology = block_transfer_engine_test::makeTestTopology(
        {{"only_one", defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 1, 0, 128, 1}});
    EXPECT_ANY_THROW(group->initialize(0, std::move(topology), {0}, {pools[0], pools[1]}));
    EXPECT_EQ(group->topology(), nullptr);
}

TEST_F(BlockTreeCacheTest, EmptyDevicePoolsFailBeforeGroupMutation) {
    auto group    = std::make_shared<FullGroupSet>();
    auto topology = block_transfer_engine_test::makeTestTopology(
        {{"tag_0", defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 1, 0, 128, 1}});
    EXPECT_ANY_THROW(group->initialize(0, std::move(topology), {}, {}));
    EXPECT_TRUE(group->devicePools().empty());
    EXPECT_EQ(group->topology(), nullptr);
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>();

    std::shared_ptr<SWAGroupSet> swa = std::make_shared<SWAGroupSet>(128, 64);

    std::vector<GroupSetPtr>        groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));

    std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    slots[2][0].device_blocks = {12};
    slots[3][0].device_blocks = {13};
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};

    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_blocks, 1u);
    cache->releaseMatchedResources(partial.matched_resources);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_blocks, 4u);
    cache->releaseMatchedResources(restored.matched_resources);
}

TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto                     tree   = std::make_unique<BlockTree>(1);
    auto                     full   = std::make_shared<FullGroupSet>();
    std::vector<GroupSetPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSetResource>> slots(3, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache->insert(nullptr, {100, 200, 300}, slots);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim C -> B becomes DeviceLeaf -> enters heap.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim B -> A becomes DeviceLeaf -> enters heap.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

TEST(BlockTreeCacheConfigurationTest, RejectsHostLayoutPayloadMismatchAtInit) {
    auto host_pool = makeHostPool(65, 2);
    auto group     = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(
        group, makeStructuralDevicePools(1, "host_layout_payload_mismatch"), {"kv"}, /*logical_layer_bytes=*/64);
    group->setHostPool(host_pool);

    BlockTreeCacheConfig config;
    config.enable_memory_cache      = true;
    std::vector<GroupSetPtr> groups = {group};
    auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), std::move(config));

    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConfigurationTest, RejectsGroupSetsBackedByDifferentTopologies) {
    auto first  = std::make_shared<FullGroupSet>();
    auto second = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(first, makeStructuralDevicePools(1, "first_topology"), {"first"});
    initializeTestGroupSet(
        second, makeStructuralDevicePools(1, "second_topology"), {"second"}, /*logical_layer_bytes=*/1, 1);

    std::vector<GroupSetPtr> groups = {first, second};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));
    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConfigurationTest, RejectsTreeAndGroupSetRegistryCountMismatch) {
    auto group = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(group, makeStructuralDevicePools(1, "tree_registry_mismatch"), {"kv"});

    std::vector<GroupSetPtr> groups = {group};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));
    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConfigurationTest, RejectsMissingReusableTopologyGroup) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology              = block_transfer_engine_test::makeTestTopology(
        {{"first", policy, {0}, 1, 0, 128, 1}, {"second", policy, {1}, 1, 0, 128, 1}});
    auto pools = makeStructuralDevicePools(1, "missing_reusable_group");
    auto group = std::make_shared<FullGroupSet>();
    group->initialize(0, topology, {0}, {pools.front()});

    std::vector<GroupSetPtr> groups = {group};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));
    EXPECT_EQ(cache, nullptr);
}

TEST_F(BlockTreeCacheTest, LoadOnlyReloadsSWAWindow) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>();

    std::shared_ptr<SWAGroupSet> swa = std::make_shared<SWAGroupSet>(128, 64);

    std::vector<GroupSetPtr>        groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
    cache->setEnableLoad(true);

    std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].host_block    = static_cast<BlockIdxType>(100 + i);
    }

    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_EQ(result.host_load_blocks, 2u);
    EXPECT_EQ(result.load_blocks, 2u);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), 4u);
    ASSERT_EQ(result.load_ticket->itemCount(), 6u);
    const auto count_exact_item = [&ticket = *result.load_ticket](size_t       group_set_id,
                                                                  Tier         source_tier,
                                                                  size_t       path_index,
                                                                  BlockIdxType source_block) {
        size_t count = 0;
        for (size_t item_index = 0; item_index < ticket.itemCount(); ++item_index) {
            count += ticket.groupSetId(item_index) == group_set_id && ticket.sourceTier(item_index) == source_tier
                     && ticket.pathIndex(item_index) == path_index
                     && ticket.sourceBlocks(item_index) == std::vector<BlockIdxType>{source_block};
        }
        return count;
    };
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(
            count_exact_item(/*group_id=*/0, Tier::DEVICE, path_index, static_cast<BlockIdxType>(10 + path_index)), 1);
    }
    for (size_t path_index = 2; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_item(/*group_id=*/1, Tier::HOST, path_index, static_cast<BlockIdxType>(100 + path_index)),
                  1);
    }
}

TEST_F(BlockTreeCacheTest, LoadPlanningIgnoresBusySwaSlotOutsideTrailingWindow) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto                            full   = std::make_shared<FullGroupSet>();
        auto                            swa    = std::make_shared<SWAGroupSet>(2, 1);
        std::vector<GroupSetPtr>        groups = {full, swa};
        std::unique_ptr<BlockTreeCache> cache =
            makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups));
        ASSERT_NE(cache, nullptr);
        cache->setEnableLoad(true);

        std::vector<std::vector<GroupSetResource>> slots(4, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            slots[i][1].host_block    = static_cast<BlockIdxType>(100 + i);
        }
        if (state == GroupSetTransferState::DEMOTING) {
            slots[1][1].host_block    = NULL_BLOCK_IDX;
            slots[1][1].device_blocks = {21};
        }
        ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

        const std::vector<TreeNode*> path = cache->tree()->findNode({100, 200, 300, 400}).path;
        ASSERT_EQ(path.size(), 4u);
        path[1]->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
        path[1]->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
        EXPECT_EQ(result.matched_blocks, 0u);
        EXPECT_TRUE(result.matched_resources.empty());
        EXPECT_EQ(result.host_load_blocks, 2u);
        EXPECT_EQ(result.load_blocks, 2u);
        ASSERT_NE(result.load_ticket, nullptr);
        EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), 4u);
        ASSERT_EQ(result.load_ticket->itemCount(), 6u);

        size_t swa_item_count = 0;
        for (size_t item_index = 0; item_index < result.load_ticket->itemCount(); ++item_index) {
            if (result.load_ticket->groupSetId(item_index) != 1) {
                continue;
            }
            ++swa_item_count;
            EXPECT_GE(result.load_ticket->pathIndex(item_index), 2u);
            EXPECT_EQ(result.load_ticket->sourceTier(item_index), Tier::HOST);
        }
        EXPECT_EQ(swa_item_count, 2u);

        result.load_ticket.reset();
    }
}

// ---------------------------------------------------------------------------
// Test: enable_load — match detects Host/Disk data needing reload
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, LoadDetectsHostData) {
    auto                     tree   = std::make_unique<BlockTree>(1);
    auto                     full   = std::make_shared<FullGroupSet>();
    std::vector<GroupSetPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(tree), std::move(groups));
    cache->setEnableLoad(true);

    // Insert a node and manually set host data (simulating prior demotion).
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim without host demotion, then manually set up a host-only node.
    // Instead, manually set up a node with host_block but no device_blocks
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After reclaim without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<std::vector<GroupSetResource>> slots2(1, std::vector<GroupSetResource>(1));
    slots2[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots2);

    // Manually set host_block and clear device_blocks to simulate a demoted state.
    auto find = cache->tree()->findNode({200});
    ASSERT_NE(find.matched_node, nullptr);
    GroupSetResource& slot   = find.matched_node->group_set_resources[0];
    slot.host_block          = 7;
    const auto device_blocks = full->getBlocks(slot, Tier::DEVICE);
    ASSERT_EQ(device_blocks, (BlockIndicesType{55}));
    full->unreferenceBlocks(MultiNodeResource{full->groupSetId(), Tier::DEVICE, {device_blocks}},
                            BlockRefType::BLOCK_CACHE);
    slot.device_blocks.clear();

    // Match should detect load
    auto result = cache->match({200});
    EXPECT_EQ(result.host_load_blocks, 1u);
    EXPECT_EQ(result.load_blocks, 1u);
}

static std::unique_ptr<BlockTreeCache> makeHostOnlyLoadCache(std::vector<DeviceBlockPoolPtr> device_pools = {}) {
    if (device_pools.empty()) {
        device_pools.push_back(makeDevicePool({{1, 0}}, 1, "load_ticket_abort"));
    }
    for (const DeviceBlockPoolPtr& device_pool : device_pools) {
        RTP_LLM_CHECK(device_pool != nullptr);
    }
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/device_pools.size(), /*usable_count=*/1);
    RTP_LLM_CHECK(host_pool != nullptr);

    std::unique_ptr<BlockTree>    tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, device_pools, makeTestTags(device_pools.size()));
    full->setHostPool(host_pool);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_memory_cache = true;
    config.enable_load         = true;
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::move(config));
    RTP_LLM_CHECK(cache != nullptr);

    MultiNodeResource request_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    RTP_LLM_CHECK(request_holder.per_node.size() == 1);
    RTP_LLM_CHECK(request_holder.per_node[0].size() == device_pools.size());
    const std::vector<BlockIdxType> device_blocks = request_holder.per_node[0];

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = device_blocks;
    cache->insert(nullptr, {200}, slots);
    full->unreferenceBlocks(request_holder, BlockRefType::REQUEST);

    BlockTreeFindResult find = cache->tree()->findNode({200});
    RTP_LLM_CHECK(find.matched_node != nullptr);
    GroupSetResource& slot = find.matched_node->group_set_resources[0];
    slot.host_block        = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    RTP_LLM_CHECK(slot.host_block != NULL_BLOCK_IDX);
    RTP_LLM_CHECK(full->getBlocks(slot, Tier::DEVICE) == device_blocks);
    full->unreferenceBlocks(MultiNodeResource{full->groupSetId(), Tier::DEVICE, {device_blocks}},
                            BlockRefType::BLOCK_CACHE);
    slot.device_blocks.clear();
    return cache;
}

TEST_F(BlockTreeCacheTest, PendingLoadTicketHardStopsSecondMatchUntilAbort) {
    std::unique_ptr<BlockTreeCache> cache = makeHostOnlyLoadCache();
    ASSERT_NE(cache, nullptr);

    const GroupSetPtr& group       = cache->groupSets().front();
    const auto         host_pool   = group->hostPool();
    TreeNode*          source_node = cache->tree()->findNode({200}).matched_node;
    ASSERT_NE(source_node, nullptr);
    const BlockIdxType source_block = source_node->group_set_resources[0].host_block;
    ASSERT_NE(source_block, NULL_BLOCK_IDX);

    BlockTreeMatchResult first_match = cache->match({200});
    ASSERT_NE(first_match.load_ticket, nullptr);
    EXPECT_EQ(source_node->group_set_resources[0].transfer_state, GroupSetTransferState::LOAD_PENDING);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    BlockTreeMatchResult second_match = cache->match({200});
    EXPECT_EQ(second_match.matched_node, nullptr);
    EXPECT_EQ(second_match.matched_blocks, 0u);
    EXPECT_EQ(second_match.load_ticket, nullptr);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    first_match.load_ticket.reset();
    EXPECT_EQ(source_node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(host_pool->refCount(source_block), 1u);
}

TEST_F(BlockTreeCacheTest, LoadPreparedPrefixFailureRollsBackAllSourceAndTargetHolders) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr first_device_pool  = makeDevicePool({{1, 0}}, 1, "load_prepared_prefix_first");
    DeviceBlockPoolPtr second_device_pool = makeDevicePool({{1, 0}}, 1, "load_prepared_prefix_second");
    ASSERT_NE(first_device_pool, nullptr);
    ASSERT_NE(second_device_pool, nullptr);

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);
    ASSERT_NE(first_host_pool, nullptr);
    ASSERT_NE(second_host_pool, nullptr);

    auto first_group = std::make_shared<FullGroupSet>();
    first_group->setHostPool(first_host_pool);
    auto second_group = std::make_shared<FullGroupSet>();
    second_group->setHostPool(second_host_pool);
    initializeSingleMemberGroupSets(
        {first_group, second_group}, {first_device_pool, second_device_pool}, {"tag_0", "tag_1"});

    BlockTreeCacheConfig config;
    config.enable_memory_cache                 = true;
    config.enable_load                         = true;
    std::vector<GroupSetPtr>        group_sets = {first_group, second_group};
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(group_sets), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType first_source  = first_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType second_source = second_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(first_source, NULL_BLOCK_IDX);
    ASSERT_NE(second_source, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(2));
    slots[0][0].host_block = first_source;
    slots[0][1].host_block = second_source;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_ticket, nullptr);
    LoadTicket::PendingLoadItems& items = result.load_ticket->items_;
    ASSERT_EQ(items.size(), 2u);
    ASSERT_EQ(items[0].group_set_id, 0);
    ASSERT_EQ(items[1].group_set_id, 1);
    EXPECT_EQ(first_host_pool->refCount(first_source), 2u);
    EXPECT_EQ(second_host_pool->refCount(second_source), 2u);

    // Duplicate the first item immediately after itself. The complete batch passes
    // preflight while both slots are IDLE. Preparation then claims the first item
    // and takes its target holder; beginLoad for the duplicate observes the
    // same slot already LOADING and fails with one prepared item and one
    // untouched trailing item. Add the matching source planning hold explicitly
    // so every item in the synthetic ticket owns exactly one source hold.
    PendingLoadItem duplicate_first_item = items.front();
    items.insert(items.begin() + 1, std::move(duplicate_first_item));
    first_group->referenceBlocks(MultiNodeResource{0, Tier::HOST, {{first_source}}}, BlockRefType::REQUEST);
    ASSERT_EQ(items.size(), 3u);
    EXPECT_EQ(first_host_pool->refCount(first_source), 3u);
    EXPECT_EQ(second_host_pool->refCount(second_source), 2u);

    const BlockIdList first_request_targets  = first_device_pool->malloc(1).value();
    const BlockIdList second_request_targets = second_device_pool->malloc(1).value();
    ASSERT_EQ(first_request_targets.size(), 1u);
    ASSERT_EQ(second_request_targets.size(), 1u);
    first_device_pool->incRef(first_request_targets, BlockRefType::REQUEST);
    second_device_pool->incRef(second_request_targets, BlockRefType::REQUEST);
    const BlockIdxType first_target  = first_request_targets.front();
    const BlockIdxType second_target = second_request_targets.front();
    items[0].target_device_blocks    = {first_target};
    items[1].target_device_blocks    = {first_target};
    items[2].target_device_blocks    = {second_target};

    const size_t first_refs_before  = first_device_pool->refCount(first_target);
    const size_t second_refs_before = second_device_pool->refCount(second_target);
    ASSERT_EQ(first_refs_before, 1u);
    ASSERT_EQ(second_refs_before, 1u);
    ASSERT_TRUE(first_device_pool->isAllocated(first_target));
    ASSERT_TRUE(second_device_pool->isAllocated(second_target));

    EXPECT_EQ(result.load_ticket->commit(), nullptr);
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);

    // The first item's acquired target holder and both of its source planning
    // holds are gone; the unprepared trailing item's source hold is also gone.
    // Request ownership remains untouched for both target blocks.
    EXPECT_EQ(first_host_pool->refCount(first_source), 1u);
    EXPECT_EQ(second_host_pool->refCount(second_source), 1u);
    EXPECT_TRUE(first_device_pool->isAllocated(first_target));
    EXPECT_TRUE(second_device_pool->isAllocated(second_target));
    EXPECT_EQ(first_device_pool->refCount(first_target), first_refs_before);
    EXPECT_EQ(second_device_pool->refCount(second_target), second_refs_before);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(find.matched_node->group_set_resources.size(), 2u);
    EXPECT_EQ(find.matched_node->group_set_resources[0].host_block, first_source);
    EXPECT_EQ(find.matched_node->group_set_resources[1].host_block, second_source);
    EXPECT_TRUE(find.matched_node->group_set_resources[0].device_blocks.empty());
    EXPECT_TRUE(find.matched_node->group_set_resources[1].device_blocks.empty());
    EXPECT_EQ(find.matched_node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    result.load_ticket.reset();
    EXPECT_EQ(first_host_pool->refCount(first_source), 1u) << "committed ticket must not release source twice";
    EXPECT_EQ(second_host_pool->refCount(second_source), 1u) << "committed ticket must not release source twice";
    first_device_pool->decRef(first_request_targets, BlockRefType::REQUEST);
    second_device_pool->decRef(second_request_targets, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheTest, LoadQueueRejectionRollsBackCoreHoldersAndRetainsRequestTarget) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             device_pool = makeDevicePool({{1, 0}}, 2, "load_queue_rejection");
    std::shared_ptr<HostBlockPool> host_pool   = makeHostPool(1, 2);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {device_pool}, makeTestTags(1));
    full->setHostPool(host_pool);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_memory_cache = true;
    config.enable_load         = true;
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);
    const size_t source_ref_before = host_pool->refCount(source_block);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_EQ(result.load_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_ticket->groupSetId(0), 0);
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before + 1);

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets, BlockRefType::REQUEST);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    result.load_ticket->items_.front().target_device_blocks = {request_target};
    ASSERT_EQ(device_pool->refCount(request_target), 1u);

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    ASSERT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before);
    EXPECT_EQ(device_pool->refCount(request_target), 1u);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(find.matched_node->group_set_resources.size(), 1u);
    EXPECT_EQ(find.matched_node->group_set_resources[0].host_block, source_block);
    EXPECT_TRUE(find.matched_node->group_set_resources[0].device_blocks.empty());
    EXPECT_EQ(find.matched_node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    result.load_ticket.reset();
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before) << "committed ticket must not release source twice";
    cache->waitForPendingTasks();
    device_pool->decRef(request_targets, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheTest, LoadQueueRejectionRollsBackMixedDeviceAndHostItems) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             resident_device_pool = makeDevicePool({{1, 0}}, 1, "load_mixed_resident");
    DeviceBlockPoolPtr             target_device_pool   = makeDevicePool({{1, 0}}, 2, "load_mixed_target");
    std::shared_ptr<HostBlockPool> resident_host_pool   = makeHostPool(1, 1);
    std::shared_ptr<HostBlockPool> host_pool            = makeHostPool(1, 2);
    ASSERT_NE(resident_device_pool, nullptr);
    ASSERT_NE(target_device_pool, nullptr);
    ASSERT_NE(resident_host_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto resident_group = std::make_shared<FullGroupSet>();
    resident_group->setHostPool(resident_host_pool);
    auto loading_group = std::make_shared<FullGroupSet>();
    loading_group->setHostPool(host_pool);
    initializeSingleMemberGroupSets(
        {resident_group, loading_group}, {resident_device_pool, target_device_pool}, {"resident", "loading"});

    BlockTreeCacheConfig config;
    config.enable_memory_cache             = true;
    config.enable_load                     = true;
    std::vector<GroupSetPtr>        groups = {resident_group, loading_group};
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    MultiNodeResource resident_holder = resident_group->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(resident_holder.per_node.size(), 1u);
    ASSERT_EQ(resident_holder.per_node.front().size(), 1u);
    const BlockIdxType resident_block = resident_holder.per_node.front().front();
    const BlockIdxType host_block     = loading_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(2));
    slots[0][0].device_blocks = {resident_block};
    slots[0][1].host_block    = host_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);
    ASSERT_EQ(resident_device_pool->refCount(resident_block), 1u);
    ASSERT_EQ(host_pool->refCount(host_block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_EQ(result.load_ticket->itemCount(), 2u);
    EXPECT_EQ(result.load_ticket->sourceTier(0), Tier::DEVICE);
    EXPECT_EQ(result.load_ticket->sourceTier(1), Tier::HOST);
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 2u);
    EXPECT_EQ(host_pool->refCount(host_block), 2u);

    const BlockIdxType request_target = poolMalloc(*target_device_pool);
    ASSERT_NE(request_target, NULL_BLOCK_IDX);
    target_device_pool->incRef(request_target, BlockRefType::REQUEST);
    ASSERT_EQ(target_device_pool->refCount(request_target), 1u);
    ASSERT_TRUE(result.load_ticket->bindTargetDeviceBlocks(0, {resident_block}));
    ASSERT_TRUE(result.load_ticket->bindTargetDeviceBlocks(1, {request_target}));

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 1u);
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    EXPECT_EQ(target_device_pool->refCount(request_target), 1u);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(find.matched_node->group_set_resources.size(), 2u);
    EXPECT_EQ(find.matched_node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    result.load_ticket.reset();
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 1u);
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    target_device_pool->decRef(request_target, BlockRefType::REQUEST);
}

// Deferred load: match() plans (references the source blocks) but does NOT execute
// load. The result carries a LoadTicket; the allocator binds request-owned
// device targets before committing it. Dropping it uncommitted aborts (unreferences
// the source) without allocating or copying anything.

// Not committing the ticket: no device block is allocated and no async copy is submitted;
// the ticket destructor aborts safely.
TEST_F(BlockTreeCacheTest, LoadTicketAbortSkipsLoad) {
    auto cache = makeHostOnlyLoadCache();

    auto result = cache->match({200});
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_FALSE(result.load_ticket->empty());
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), 1u);
    // Counters reflect the planned load; match() submits nothing async and leaves
    // async_context null (the async context is produced only at commit).
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_EQ(result.host_load_blocks, 1u);
    EXPECT_EQ(result.load_blocks, 1u);
    EXPECT_EQ(result.async_context, nullptr);

    // Drop the ticket without committing => RAII abort (source unreferenced). No async
    // task was ever submitted, so waitForPendingTasks returns immediately.
    result.load_ticket.reset();
    cache->releaseMatchedResources(result.matched_resources);
    cache->waitForPendingTasks();
}

// Committing the ticket uses the allocator-owned device target and submits the async
// copy, yielding a non-null AsyncContext.
TEST_F(BlockTreeCacheTest, LoadTicketCommitTriggersLoad) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr device_pool = makeDevicePool({{1, 0}}, 1, "load_ticket_commit");
    ASSERT_NE(device_pool, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeHostOnlyLoadCache({device_pool});

    BlockTreeMatchResult result = cache->match({200});
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), 1u);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_EQ(result.host_load_blocks, 1u);
    EXPECT_EQ(result.load_blocks, 1u);

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets, BlockRefType::REQUEST);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    ASSERT_EQ(result.load_ticket->items().size(), 1u);
    result.load_ticket->items_[0].target_device_blocks = {request_target};

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    EXPECT_NE(context, nullptr);

    cache->releaseMatchedResources(result.matched_resources);
    cache->waitForPendingTasks();
    device_pool->decRef(request_targets, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheTest, MalformedLoadTargetFailsBeforeStateMutationAndAllowsRetry) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::vector<DeviceBlockPoolPtr> device_pools = {
        makeDevicePool({{1, 0}}, 1, "load_malformed_target_0"),
        makeDevicePool({{1, 0}}, 1, "load_malformed_target_1"),
    };
    ASSERT_NE(device_pools[0], nullptr);
    ASSERT_NE(device_pools[1], nullptr);
    std::unique_ptr<BlockTreeCache>      cache     = makeHostOnlyLoadCache(device_pools);
    const GroupSetPtr&                   group     = cache->groupSets().front();
    const std::shared_ptr<HostBlockPool> host_pool = group->hostPool();
    TreeNode*                            node      = cache->tree()->findNode({200}).matched_node;
    ASSERT_NE(node, nullptr);
    const BlockIdxType source_block = node->group_set_resources[0].host_block;
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    ASSERT_EQ(host_pool->refCount(source_block), 1u);

    BlockTreeMatchResult malformed = cache->match({200});
    ASSERT_NE(malformed.load_ticket, nullptr);
    const BlockIdList malformed_targets = device_pools[0]->malloc(1).value();
    ASSERT_EQ(malformed_targets.size(), 1u);
    device_pools[0]->incRef(malformed_targets, BlockRefType::REQUEST);
    const size_t malformed_target_ref_count = device_pools[0]->refCount(malformed_targets.front());
    ASSERT_TRUE(malformed.load_ticket->bindTargetDeviceBlocks(0, malformed_targets));
    EXPECT_EQ(malformed.load_ticket->commit(), nullptr);
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(node->group_set_resources[0].host_block, source_block);
    EXPECT_TRUE(node->group_set_resources[0].device_blocks.empty());
    EXPECT_EQ(host_pool->refCount(source_block), 1u);
    EXPECT_EQ(device_pools[0]->refCount(malformed_targets.front()), malformed_target_ref_count);
    malformed.load_ticket.reset();
    device_pools[0]->decRef(malformed_targets, BlockRefType::REQUEST);

    BlockTreeMatchResult retry = cache->match({200});
    ASSERT_NE(retry.load_ticket, nullptr);
    BlockIdList request_targets;
    for (const DeviceBlockPoolPtr& device_pool : device_pools) {
        const BlockIdxType target = device_pool->malloc().value();
        device_pool->incRef(target, BlockRefType::REQUEST);
        request_targets.push_back(target);
    }
    ASSERT_TRUE(retry.load_ticket->bindTargetDeviceBlocks(0, request_targets));

    const std::shared_ptr<AsyncContext> context = retry.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, request_targets);
    EXPECT_FALSE(host_pool->isAllocated(source_block));

    for (size_t pool_index = 0; pool_index < device_pools.size(); ++pool_index) {
        device_pools[pool_index]->decRef(request_targets[pool_index], BlockRefType::REQUEST);
    }
}

// C006-T01: destructor drains real root/live-node holds across Device, Host, and Disk.
TEST_F(BlockTreeCacheTest, ShutdownDrainsRootAndLiveTreeHoldsAcrossAllPhysicalTiers) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t                kBlockBytes  = 16;
    constexpr size_t                kPoolSize    = 4;
    std::vector<DeviceBlockPoolPtr> device_pools = {
        makeDevicePool({{kBlockBytes, 0}}, kPoolSize, "shutdown_drain_device_0"),
        makeDevicePool({{kBlockBytes, 0}}, kPoolSize, "shutdown_drain_device_1"),
        makeDevicePool({{kBlockBytes, 0}}, kPoolSize, "shutdown_drain_device_2"),
    };
    auto host_pool = makeHostPool(device_pools.size() * kBlockBytes, kPoolSize);
    auto disk_pool = makeDiskPool(device_pools.size() * kBlockBytes, kPoolSize, std::make_unique<MemoryDiskBlockIO>());

    const std::vector<size_t> device_free_before = {
        device_pools[0]->freeBlocksNum(),
        device_pools[1]->freeBlocksNum(),
        device_pools[2]->freeBlocksNum(),
    };
    const size_t host_free_before = host_pool->freeBlocksNum();
    const size_t disk_free_before = disk_pool->freeBlocksNum();

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, device_pools, makeTestTags(device_pools.size()), kBlockBytes);
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);

    BlockTreeCacheConfig config;
    config.enable_device_cache      = true;
    config.enable_memory_cache      = true;
    config.enable_disk_cache        = true;
    std::vector<GroupSetPtr> groups = {full};
    auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    MultiNodeResource root_device_holds = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(root_device_holds.per_node.size(), 1u);
    ASSERT_EQ(root_device_holds.per_node[0].size(), 3u);
    const BlockIdxType device_block_0 = root_device_holds.per_node[0][0];
    const BlockIdxType device_hole    = root_device_holds.per_node[0][1];
    const BlockIdxType device_block_2 = root_device_holds.per_node[0][2];
    ASSERT_NE(device_block_0, NULL_BLOCK_IDX);
    ASSERT_NE(device_hole, NULL_BLOCK_IDX);
    ASSERT_NE(device_block_2, NULL_BLOCK_IDX);

    MultiNodeResource hole_holder{0, Tier::DEVICE, {{NULL_BLOCK_IDX, device_hole, NULL_BLOCK_IDX}}};
    full->unreferenceBlocks(hole_holder, BlockRefType::BLOCK_CACHE);
    root_device_holds.per_node[0][1]                            = NULL_BLOCK_IDX;
    cache->tree()->root()->group_set_resources[0].device_blocks = root_device_holds.per_node[0];

    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block = full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> lower_tier_slots(2, std::vector<GroupSetResource>(1));
    lower_tier_slots[0][0].host_block = host_block;
    lower_tier_slots[1][0].disk_slot  = disk_block;
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100, 200}, lower_tier_slots));

    EXPECT_EQ(device_pools[0]->freeBlocksNum(), device_free_before[0] - 1);
    EXPECT_EQ(device_pools[1]->freeBlocksNum(), device_free_before[1]);
    EXPECT_EQ(device_pools[2]->freeBlocksNum(), device_free_before[2] - 1);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);
    EXPECT_EQ(device_pools[0]->refCount(device_block_0), 1u);
    EXPECT_EQ(device_pools[2]->refCount(device_block_2), 1u);
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    EXPECT_EQ(disk_pool->refCount(disk_block), 1u);

    cache.reset();

    EXPECT_EQ(device_pools[0]->freeBlocksNum(), device_free_before[0]);
    EXPECT_EQ(device_pools[1]->freeBlocksNum(), device_free_before[1]);
    EXPECT_EQ(device_pools[2]->freeBlocksNum(), device_free_before[2]);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
    EXPECT_FALSE(device_pools[0]->isAllocated(device_block_0));
    EXPECT_FALSE(device_pools[2]->isAllocated(device_block_2));
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_FALSE(disk_pool->isAllocated(disk_block));
}

// C006-T02: an external co-holder remains at refcount one after the tree hold drains.
TEST_F(BlockTreeCacheTest, ShutdownReleasesOnlyTreeHoldWhenExternalCoHolderSurvives) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kBlockBytes = 16;
    constexpr size_t kPoolSize   = 2;
    auto             device_pool = makeDevicePool({{kBlockBytes, 0}}, kPoolSize, "shutdown_external_coholder");
    const size_t     free_before = device_pool->freeBlocksNum();

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {device_pool}, makeTestTags(1), kBlockBytes);
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups));
    ASSERT_NE(cache, nullptr);

    MultiNodeResource tree_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(tree_holder.per_node.size(), 1u);
    ASSERT_EQ(tree_holder.per_node[0].size(), 1u);
    const BlockIdxType block = tree_holder.per_node[0][0];
    ASSERT_NE(block, NULL_BLOCK_IDX);
    MultiNodeResource external_holder = tree_holder;
    full->referenceBlocks(external_holder, BlockRefType::REQUEST);
    EXPECT_EQ(device_pool->refCount(block), 2u);

    std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
    slots[0][0].device_blocks = tree_holder.per_node[0];
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100}, slots));
    full->unreferenceBlocks(tree_holder, BlockRefType::BLOCK_CACHE);

    cache.reset();

    EXPECT_TRUE(device_pool->isAllocated(block));
    EXPECT_EQ(device_pool->refCount(block), 1u);
    EXPECT_EQ(device_pool->freeBlocksNum(), free_before - 1);

    full->unreferenceBlocks(external_holder, BlockRefType::REQUEST);
    EXPECT_FALSE(device_pool->isAllocated(block));
    EXPECT_EQ(device_pool->freeBlocksNum(), free_before);
}

// C006-T04: partial reclaim leaves only valid Host/Disk tree holds for shutdown to drain.
TEST_F(BlockTreeCacheTest, ShutdownDrainsOnlyHoldsRemainingAfterPartialMixedTierReclaim) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kBlockBytes        = 16;
    constexpr size_t kPoolSize          = 2;
    auto             device_pool        = makeDevicePool({{kBlockBytes, 0}}, kPoolSize, "shutdown_partial_device");
    auto             host_pool          = makeHostPool(kBlockBytes, kPoolSize);
    auto             disk_pool          = makeDiskPool(kBlockBytes, kPoolSize, std::make_unique<MemoryDiskBlockIO>());
    const size_t     device_free_before = device_pool->freeBlocksNum();
    const size_t     host_free_before   = host_pool->freeBlocksNum();
    const size_t     disk_free_before   = disk_pool->freeBlocksNum();

    auto full = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, {device_pool}, makeTestTags(1), kBlockBytes);
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    BlockTreeCacheConfig config;
    config.enable_device_cache      = true;
    config.enable_memory_cache      = true;
    config.enable_disk_cache        = true;
    std::vector<GroupSetPtr> groups = {full};
    auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);
    auto per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(std::vector<GroupSetPtr>{full});
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    MultiNodeResource device_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_holder.per_node.size(), 1u);
    ASSERT_EQ(device_holder.per_node[0].size(), 1u);
    const BlockIdxType device_block = device_holder.per_node[0][0];
    const BlockIdxType host_block   = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block   = full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSetResource>> device_slots(1, std::vector<GroupSetResource>(1));
    device_slots[0][0].device_blocks = device_holder.per_node[0];
    std::vector<std::vector<GroupSetResource>> host_slots(1, std::vector<GroupSetResource>(1));
    host_slots[0][0].host_block = host_block;
    std::vector<std::vector<GroupSetResource>> disk_slots(1, std::vector<GroupSetResource>(1));
    disk_slots[0][0].disk_slot = disk_block;
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100}, device_slots));
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {200}, host_slots));
    ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {300}, disk_slots));
    full->unreferenceBlocks(device_holder, BlockRefType::BLOCK_CACHE);
    cache->onBlocksReleased();

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_FALSE(device_pool->isAllocated(device_block));
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);

    cache.reset();

    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_FALSE(disk_pool->isAllocated(disk_block));
}

TEST_F(BlockTreeCacheTest, LoadTicketOutlivesHostAndDiskCacheShutdown) {
    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(source_tier));

        auto full      = std::make_shared<FullGroupSet>();
        auto host_pool = makeHostPool(1, 2);
        auto disk_pool = makeDiskPool(1, 2, std::make_unique<MemoryDiskBlockIO>());
        full->setHostPool(host_pool);
        full->setDiskPool(disk_pool);

        BlockTreeCacheConfig config;
        config.enable_memory_cache      = true;
        config.enable_disk_cache        = true;
        config.enable_load              = true;
        std::vector<GroupSetPtr> groups = {full};
        auto cache = makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), std::move(config));
        ASSERT_NE(cache, nullptr);

        const BlockIdxType source_block = full->allocateSingleBlock(source_tier, BlockRefType::BLOCK_CACHE);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        IBlockPool& source_pool =
            source_tier == Tier::HOST ? static_cast<IBlockPool&>(*host_pool) : static_cast<IBlockPool&>(*disk_pool);
        EXPECT_EQ(source_pool.refCount(source_block), 1u);

        std::vector<std::vector<GroupSetResource>> slots(1, std::vector<GroupSetResource>(1));
        if (source_tier == Tier::HOST) {
            slots[0][0].host_block = source_block;
        } else {
            slots[0][0].disk_slot = source_block;
        }
        ASSERT_TRUE(insertGroupSetSlots(*cache, nullptr, {100}, slots));

        BlockTreeMatchResult result = cache->match({100});
        ASSERT_NE(result.load_ticket, nullptr);
        ASSERT_FALSE(result.load_ticket->empty());
        ASSERT_EQ(result.load_ticket->items().size(), 1u);
        EXPECT_EQ(result.load_ticket->items()[0].source_tier, source_tier);
        EXPECT_EQ(result.load_ticket->items()[0].source_blocks, (BlockIndicesType{source_block}));
        EXPECT_EQ(source_pool.refCount(source_block), 2u);

        std::shared_ptr<LoadTicket>     outliving_ticket = std::move(result.load_ticket);
        ThreadCompletion                destruction;
        std::thread                     destroy_thread([cache = std::move(cache), &destruction]() mutable {
            destruction.markEntered();
            cache.reset();
            destruction.markFinished();
        });
        destruction.waitUntilEntered();
        destroy_thread.join();

        EXPECT_TRUE(destruction.finished());
        EXPECT_FALSE(source_pool.isAllocated(source_block));
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);
        EXPECT_EQ(outliving_ticket->commit(), nullptr);
        EXPECT_EQ(outliving_ticket->commit(), nullptr);
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);

        outliving_ticket.reset();
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);
    }
}

TEST_F(BlockTreeCacheTest, LoadTicketKeepsExplicitLogicalDepthIndependentOfItemPositions) {
    size_t abort_calls = 0;
    auto   registry =
        std::make_shared<LoadTicketRegistry>([](const LoadTicket&) { return std::shared_ptr<AsyncContext>{}; },
                                             [&](const LoadTicket& ticket) {
                                                 const auto& items = ticket.items();
                                                 ++abort_calls;
                                                 EXPECT_EQ(items.size(), 1u);
                                             });

    PendingLoadItem pending_item;
    pending_item.path_index = 1;
    std::shared_ptr<LoadTicket> ticket = registry->createTicket({pending_item}, /*logical_matched_blocks=*/7, nullptr);
    ASSERT_NE(ticket, nullptr);
    EXPECT_EQ(ticket->logicalMatchedBlocks(), 7u);
    ASSERT_EQ(ticket->items().size(), 1u);
    EXPECT_EQ(ticket->items().front().path_index, 1u);

    ticket.reset();
    EXPECT_EQ(abort_calls, 1u);
}

TEST_F(BlockTreeCacheTest, TicketRegistryShutdownWaitsForClaimedCommit) {
    CallbackBarrier  commit_callback;
    ThreadCompletion shutdown_detached_abort;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};

    auto registry = std::make_shared<LoadTicketRegistry>(
        [&](const LoadTicket&) {
            ++commit_calls;
            commit_callback.enterAndWait();
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_set_id, 1);
            }
            shutdown_detached_abort.markEntered();
        });
    PendingLoadItem pending_item;
    pending_item.group_set_id              = 0;
    std::shared_ptr<LoadTicket> ticket     = registry->createTicket({pending_item}, 0, nullptr);
    ASSERT_NE(ticket, nullptr);
    PendingLoadItem shutdown_pending_item;
    shutdown_pending_item.group_set_id = 1;
    std::shared_ptr<LoadTicket> shutdown_pending_ticket = registry->createTicket({shutdown_pending_item}, 0, nullptr);
    ASSERT_NE(shutdown_pending_ticket, nullptr);

    std::shared_ptr<AsyncContext> commit_result;
    std::thread                   commit_thread([&] { commit_result = ticket->commit(); });
    commit_callback.waitUntilEntered();
    EXPECT_EQ(commit_calls.load(), 1);
    EXPECT_EQ(abort_calls.load(), 0);

    std::thread shutdown_thread([&] {
        registry->shutdown();
        shutdown.markFinished();
    });
    shutdown_detached_abort.waitUntilEntered();
    EXPECT_FALSE(shutdown.finished());
    EXPECT_EQ(abort_calls.load(), 1);

    commit_callback.release();
    commit_thread.join();
    shutdown_thread.join();
    EXPECT_TRUE(shutdown.finished());
    EXPECT_EQ(commit_result, nullptr);
    EXPECT_EQ(commit_calls.load(), 1);
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(ticket->commit(), nullptr);
    EXPECT_EQ(ticket->commit(), nullptr);
    ticket.reset();
    EXPECT_EQ(shutdown_pending_ticket->commit(), nullptr);
    shutdown_pending_ticket.reset();
    EXPECT_EQ(commit_calls.load(), 1);
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(registry->createTicket({pending_item}, 0, nullptr), nullptr);
    registry->shutdown();
}

TEST_F(BlockTreeCacheTest, TicketRegistryCloseDetachesAndAbortsOnce) {
    auto host_pool = makeHostPool(1, 2);
    auto full      = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, makeStructuralDevicePools(1, "ticket_close"), {"full"});
    full->setHostPool(host_pool);
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    MultiNodeResource source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadTicketRegistry>(
        [&](const LoadTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            full->unreferenceBlocks(source_protection, BlockRefType::REQUEST);
            abort_callback.enterAndWait();
        });
    PendingLoadItem pending_item;
    pending_item.group_set_id              = 0;
    pending_item.source_tier               = Tier::HOST;
    pending_item.source_blocks             = {source_block};
    std::shared_ptr<LoadTicket> ticket     = registry->createTicket({pending_item}, 0, nullptr);
    ASSERT_NE(ticket, nullptr);

    std::thread shutdown_thread([&] {
        shutdown.markEntered();
        registry->shutdown();
        shutdown.markFinished();
    });
    abort_callback.waitUntilEntered();
    EXPECT_FALSE(shutdown.finished());
    EXPECT_EQ(host_pool->refCount(source_block), 1u);
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);
    EXPECT_EQ(ticket->commit(), nullptr);
    EXPECT_EQ(ticket->commit(), nullptr);
    ticket.reset();
    EXPECT_EQ(abort_calls.load(), 1);

    abort_callback.release();
    shutdown_thread.join();
    EXPECT_TRUE(shutdown.finished());
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);
    full->releaseSingleBlock(Tier::HOST, source_block, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
}

TEST_F(BlockTreeCacheTest, TicketRegistryConcurrentShutdownCallersShareDetachedAbortCompletion) {
    CallbackBarrier   abort_callback;
    CountedEvent      shutdown_waits;
    ThreadCompletion  first_shutdown;
    ThreadCompletion  second_shutdown;
    std::atomic<int>  commit_calls{0};
    std::atomic<int>  abort_calls{0};
    std::atomic<bool> abort_released{false};
    std::atomic<int>  shutdown_returns_before_release{0};
    auto              registry = std::make_shared<LoadTicketRegistry>(
        [&](const LoadTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_set_id, 7);
            }
            abort_callback.enterAndWait();
        });
    LoadShutdownTestPeer::setShutdownWaitObserver(*registry, [&shutdown_waits] { shutdown_waits.notify(); });
    PendingLoadItem pending_item;
    pending_item.group_set_id              = 7;
    std::shared_ptr<LoadTicket> ticket     = registry->createTicket({pending_item}, 0, nullptr);
    ASSERT_NE(ticket, nullptr);

    std::thread first_shutdown_thread([&] {
        registry->shutdown();
        if (!abort_released.load()) {
            ++shutdown_returns_before_release;
        }
        first_shutdown.markFinished();
    });
    abort_callback.waitUntilEntered();
    EXPECT_FALSE(first_shutdown.finished());

    std::thread second_shutdown_thread([&] {
        registry->shutdown();
        if (!abort_released.load()) {
            ++shutdown_returns_before_release;
        }
        second_shutdown.markFinished();
    });
    shutdown_waits.waitUntilCount(1);
    EXPECT_FALSE(first_shutdown.finished());
    EXPECT_FALSE(second_shutdown.finished());
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);

    abort_released.store(true);
    abort_callback.release();
    first_shutdown_thread.join();
    second_shutdown_thread.join();
    EXPECT_TRUE(first_shutdown.finished());
    EXPECT_TRUE(second_shutdown.finished());
    EXPECT_EQ(shutdown_returns_before_release.load(), 0);
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);

    registry->shutdown();
    EXPECT_EQ(ticket->commit(), nullptr);
    EXPECT_EQ(ticket->commit(), nullptr);
    ticket.reset();
    registry->shutdown();
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);
    EXPECT_EQ(registry->createTicket({pending_item}, 0, nullptr), nullptr);
    LoadShutdownTestPeer::setShutdownWaitObserver(*registry, std::function<void()>{});
}

TEST_F(BlockTreeCacheTest, TicketRegistryShutdownWaitsForAbortInFlight) {
    auto host_pool = makeHostPool(1, 2);
    auto full      = std::make_shared<FullGroupSet>();
    initializeTestGroupSet(full, makeStructuralDevicePools(1, "ticket_abort_inflight"), {"full"});
    full->setHostPool(host_pool);
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    MultiNodeResource source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown_detached_abort;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadTicketRegistry>(
        [&](const LoadTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u && items[0].group_set_id == 0) {
                full->unreferenceBlocks(source_protection, BlockRefType::REQUEST);
                abort_callback.enterAndWait();
                return;
            }
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_set_id, 1);
            }
            shutdown_detached_abort.markEntered();
        });
    PendingLoadItem pending_item;
    pending_item.group_set_id              = 0;
    pending_item.source_tier               = Tier::HOST;
    pending_item.source_blocks             = {source_block};
    std::shared_ptr<LoadTicket> ticket     = registry->createTicket({pending_item}, 0, nullptr);
    ASSERT_NE(ticket, nullptr);
    PendingLoadItem shutdown_pending_item;
    shutdown_pending_item.group_set_id = 1;
    std::shared_ptr<LoadTicket> shutdown_pending_ticket = registry->createTicket({shutdown_pending_item}, 0, nullptr);
    ASSERT_NE(shutdown_pending_ticket, nullptr);

    std::thread abort_thread([ticket = std::move(ticket)]() mutable { ticket.reset(); });
    abort_callback.waitUntilEntered();
    EXPECT_EQ(abort_calls.load(), 1);
    EXPECT_EQ(commit_calls.load(), 0);
    EXPECT_EQ(host_pool->refCount(source_block), 1u);

    std::thread shutdown_thread([&] {
        registry->shutdown();
        shutdown.markFinished();
    });
    shutdown_detached_abort.waitUntilEntered();
    EXPECT_FALSE(shutdown.finished());
    EXPECT_EQ(abort_calls.load(), 2);

    abort_callback.release();
    abort_thread.join();
    shutdown_thread.join();
    EXPECT_TRUE(shutdown.finished());
    EXPECT_EQ(abort_calls.load(), 2);
    EXPECT_EQ(commit_calls.load(), 0);
    EXPECT_EQ(shutdown_pending_ticket->commit(), nullptr);
    shutdown_pending_ticket.reset();
    EXPECT_EQ(abort_calls.load(), 2);
    EXPECT_EQ(registry->createTicket({pending_item}, 0, nullptr), nullptr);
    full->releaseSingleBlock(Tier::HOST, source_block, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
}

// A no-match match() plans nothing and returns a null ticket (never created).
TEST_F(BlockTreeCacheTest, EmptyMatchYieldsNoTicket) {
    auto result = cache_->match({100, 200, 300});  // empty tree => no match
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.load_ticket, nullptr);
}

}  // namespace
}  // namespace rtp_llm
