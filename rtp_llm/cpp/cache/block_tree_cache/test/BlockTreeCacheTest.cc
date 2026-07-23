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
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm::block_tree_cache_test {
class LoadBackShutdownTestPeer {
public:
    static void setShutdownWaitObserver(LoadBackTicketRegistry& registry, const std::function<void()>& observer) {
        std::lock_guard<std::mutex> lock(registry.mutex_);
        registry.shutdown_wait_observer_for_test_ = observer;
    }

    static void setShutdownWaitObserver(BlockTreeCache& cache, const std::function<void()>& observer) {
        setShutdownWaitObserver(*cache.load_back_ticket_registry_, observer);
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
using PendingLoadBackItem = LoadBackTicket::PendingLoadBackItem;

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
    BarrierThrowingPerRankBlockTransferEngine(const std::vector<ComponentGroupPtr>& groups,
                              const std::vector<Component>&         components,
                              std::shared_ptr<CallbackBarrier>      barrier):
        PerRankBlockTransferEngine(groups, std::make_shared<const std::vector<Component>>(components)),
        barrier_(std::move(barrier)) {}

    TransferHandle submit(const TransferDescriptor&) override {
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
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full_group                       = std::make_shared<FullComponentGroup>();
        full_group->component_group_id        = 0;
        std::vector<ComponentGroupPtr> groups = {full_group};
        std::vector<Component>         components;

        cache_ = BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyThenFullAndPartialPath) {
    BlockTreeMatchResult empty_result = cache_->match({100, 200, 300});
    EXPECT_EQ(empty_result.matched_node, nullptr);
    EXPECT_EQ(empty_result.matched_blocks, 0u);
    EXPECT_TRUE(empty_result.group_block_indices.empty());

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache_->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult full_result = cache_->match({100, 200, 300});
    ASSERT_NE(full_result.matched_node, nullptr);
    EXPECT_EQ(full_result.matched_node->cache_key, 300);
    EXPECT_EQ(full_result.matched_blocks, 3u);
    ASSERT_EQ(full_result.group_block_indices.count("tag_0"), 1u);
    EXPECT_EQ(full_result.group_block_indices.at("tag_0"), (BlockIndicesType{42, 43, 44}));
    cache_->releaseMatchedBlocks(full_result.matched_block_sets);

    BlockTreeMatchResult partial_result = cache_->match({100, 200, 999});
    ASSERT_NE(partial_result.matched_node, nullptr);
    EXPECT_EQ(partial_result.matched_node->cache_key, 200);
    EXPECT_EQ(partial_result.matched_blocks, 2u);
    ASSERT_EQ(partial_result.group_block_indices.count("tag_0"), 1u);
    EXPECT_EQ(partial_result.group_block_indices.at("tag_0"), (BlockIndicesType{42, 43}));
    cache_->releaseMatchedBlocks(partial_result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, KeySnapshotTracksMutationVersionAndLimit) {
    const auto empty = cache_->getKeySnapshot(/*limit=*/10);
    EXPECT_EQ(empty.version, 0u);
    EXPECT_TRUE(empty.keys.empty());

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
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
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[2][0].device_blocks = {12};

    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache_, nullptr, {100, 200, 300}, slots));

    BlockTreeMatchResult result = cache_->match({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 100);
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count("tag_0"), 1u);
    EXPECT_EQ(result.group_block_indices.at("tag_0"), (BlockIndicesType{10}));

    cache_->releaseMatchedBlocks(result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, MatchHardStopsAtPartialDeviceSlot) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, slots);

    TreeNode* first_node = cache_->tree()->root()->children.at(100);
    first_node->group_slots[0].device_blocks = {10, NULL_BLOCK_IDX};

    const BlockTreeMatchResult result = cache_->match({100, 200});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.group_block_indices.empty());

    // Restore the production slot-shape invariant before the fixture drains
    // synthetic tree holds during teardown.
    first_node->group_slots[0].device_blocks = {10};
}

TEST_F(BlockTreeCacheTest, MatchHardStopsAtSlotWithMultipleServingTiers) {
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, slots);

    TreeNode* first_node                  = cache_->tree()->root()->children.at(100);
    first_node->group_slots[0].host_block = 7;

    const BlockTreeMatchResult result = cache_->match({100, 200});
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_TRUE(result.group_block_indices.empty());
}

TEST_F(BlockTreeCacheTest, DuplicateInsertDoesNotCreateNodes) {
    CacheStats stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<std::vector<GroupSlot>> original_slots(2, std::vector<GroupSlot>(1));
    original_slots[0][0].device_blocks = {10};
    original_slots[1][0].device_blocks = {11};
    cache_->insert(nullptr, {100, 200}, original_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    std::vector<std::vector<GroupSlot>> duplicate_slots(2, std::vector<GroupSlot>(1));
    duplicate_slots[0][0].device_blocks = {20};
    duplicate_slots[1][0].device_blocks = {21};
    cache_->insert(nullptr, {100, 200}, duplicate_slots);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    BlockTreeFindResult find_result = cache_->tree()->findNode({100, 200});
    ASSERT_EQ(find_result.path.size(), 2u);
    EXPECT_EQ(find_result.path[0]->group_slots[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(find_result.path[1]->group_slots[0].device_blocks, (BlockIndicesType{11}));
}

TEST_F(BlockTreeCacheTest, ReclaimCascadesToLowerPriorityGroup) {
    // Build a cache with Full + SWA groups
    auto tree = std::make_unique<BlockTree>(2);  // 2 component groups

    auto full_group                = std::make_shared<FullComponentGroup>();
    full_group->component_group_id = 0;

    auto swa_group                = std::make_shared<SWAComponentGroup>(128, 64);
    swa_group->component_group_id = 1;

    std::vector<ComponentGroupPtr> groups = {full_group, swa_group};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 1;

    auto linear                = std::make_shared<LinearComponentGroup>();
    linear->component_group_id = 2;

    std::vector<ComponentGroupPtr> groups = {full, swa, linear};
    std::vector<Component>         components;

    std::unique_ptr<BlockTreeCache> multi_cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::move(components));

    EXPECT_EQ(multi_cache->componentGroups().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSlotCount(), 3);
}

TEST(BlockTreeCacheConstructionTest, OutOfRangeComponentGroupIdFailsInitializationWithoutThrowing) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 1;
    std::vector<ComponentGroupPtr> groups = {full};
    std::vector<Component>         components;
    auto components_ptr      = std::make_shared<const std::vector<Component>>(std::move(components));
    auto per_rank_engine     = std::make_shared<PerRankBlockTransferEngine>(groups, components_ptr);
    auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool           = std::make_unique<BlockCacheTaskPool>(2, 1000, "BlockTreeEvictionPool");

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components_ptr),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool),
                                                  std::vector<std::string>{"tag_0"},
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(full->component_group_id, 1);
}

TEST(BlockTreeCacheConstructionTest, NullComponentGroupFailsInitializationAndDestructionReturnsNormally) {
    auto                           tree   = std::make_unique<BlockTree>(1);
    std::vector<ComponentGroupPtr> groups = {nullptr};
    std::vector<Component>         components;
    auto components_ptr      = std::make_shared<const std::vector<Component>>(std::move(components));
    auto per_rank_engine     = std::make_shared<PerRankBlockTransferEngine>(groups, components_ptr);
    auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool           = std::make_unique<BlockCacheTaskPool>(2, 1000, "BlockTreeEvictionPool");

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components_ptr),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool),
                                                  std::vector<std::string>{"tag_0"},
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConstructionTest, MissingCollaboratorsFailInitializationAndDestructionReturnsNormally) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};
    auto components = std::make_shared<const std::vector<Component>>();

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  std::vector<std::string>{"tag_0"},
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
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
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.load_back_ticket, nullptr);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
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
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
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
            cache_->releaseMatchedBlocks(result.matched_block_sets);
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
                std::vector<std::vector<GroupSlot>> same_slots(2, std::vector<GroupSlot>(1));
                same_slots[0][0].device_blocks = {10};
                same_slots[1][0].device_blocks = {11};
                cache_->insert(nullptr, {100, 200}, same_slots);

                std::vector<std::vector<GroupSlot>> fork_slots(2, std::vector<GroupSlot>(1));
                fork_slots[0][0].device_blocks = {10};
                fork_slots[1][0].device_blocks = {fork_block};
                cache_->insert(nullptr, {100, fork_key}, fork_slots);

                for (const CacheKeysType& keys : {CacheKeysType{100, 200}, CacheKeysType{100, fork_key}}) {
                    BlockTreeMatchResult match = cache_->match(keys);
                    const auto           tag   = match.group_block_indices.find("tag_0");
                    if (match.matched_blocks != 2 || tag == match.group_block_indices.end() || tag->second.size() != 2
                        || tag->second[0] != 10) {
                        consistent.store(false);
                    }
                    cache_->releaseMatchedBlocks(match.matched_block_sets);
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

    const auto& pool = cache_->componentGroups()[0]->devicePools()[0];
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->refCount(10), 1u);
    EXPECT_EQ(pool->refCount(11), 1u);
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        const CacheKeyType fork_key   = static_cast<CacheKeyType>(1000 + thread_id);
        const BlockIdxType fork_block = static_cast<BlockIdxType>(20 + thread_id);
        const auto         found      = cache_->tree()->findNode({100, fork_key});
        ASSERT_EQ(found.path.size(), 2u);
        EXPECT_EQ(found.path[0]->group_slots[0].device_blocks, (BlockIndicesType{10}));
        EXPECT_EQ(found.path[1]->group_slots[0].device_blocks, (BlockIndicesType{fork_block}));
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
    auto per_rank_transfer_engine = std::make_shared<BarrierThrowingPerRankBlockTransferEngine>(
        environment->groups, environment->components, barrier);
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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool0, pool1}, makeTestTags(2));
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 2u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    ASSERT_EQ(request_blocks.per_node[1].size(), 2u);

    const BlockIdxType a_pool0 = request_blocks.per_node[0][0];
    const BlockIdxType a_pool1 = request_blocks.per_node[0][1];
    const BlockIdxType b_pool0 = request_blocks.per_node[1][0];
    const BlockIdxType b_pool1 = request_blocks.per_node[1][1];
    EXPECT_NE(a_pool0, a_pool1);
    EXPECT_NE(b_pool0, b_pool1);

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
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
    ASSERT_EQ(result.group_block_indices.count("tag_0"), 1u);
    ASSERT_EQ(result.group_block_indices.count("tag_1"), 1u);
    EXPECT_EQ(result.group_block_indices.at("tag_0"), (BlockIndicesType{a_pool0, b_pool0}));
    EXPECT_EQ(result.group_block_indices.at("tag_1"), (BlockIndicesType{a_pool1, b_pool1}));
    cache->releaseMatchedBlocks(result.matched_block_sets);

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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool}, makeTestTags(1));
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet existing = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    GroupBlockSet loser    = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(existing.per_node.size(), 1u);
    ASSERT_EQ(loser.per_node.size(), 1u);
    ASSERT_EQ(existing.per_node[0].size(), 1u);
    ASSERT_EQ(loser.per_node[0].size(), 1u);
    const BlockIdxType existing_block = existing.per_node[0][0];
    const BlockIdxType loser_block    = loser.per_node[0][0];
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    std::vector<std::vector<GroupSlot>> first_slots(1, std::vector<GroupSlot>(1));
    first_slots[0][0].device_blocks = existing.per_node[0];
    cache->insert(nullptr, {100}, first_slots);
    EXPECT_EQ(pool->refCount(existing_block), 2u);
    BlockTreeFindResult initial_find = cache->tree()->findNode({100});
    ASSERT_NE(initial_find.matched_node, nullptr);
    GroupBlockSet released_existing = existing;
    released_existing.nodes         = {initial_find.matched_node};
    cache->releaseMatchedBlocks({released_existing});
    EXPECT_EQ(pool->refCount(existing_block), 1u);

    std::vector<std::vector<GroupSlot>> duplicate_slots(1, std::vector<GroupSlot>(1));
    duplicate_slots[0][0].device_blocks = loser.per_node[0];
    cache->insert(nullptr, {100}, duplicate_slots);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(find.matched_node->group_slots[0].device_blocks, (std::vector<BlockIdxType>{existing_block}));
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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> empty_slots(1, std::vector<GroupSlot>(1));
    empty_slots[0][0].device_blocks = {NULL_BLOCK_IDX};
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, empty_slots));
    TreeNode* existing_node = cache->tree()->nodes().front().get();
    ASSERT_NE(existing_node, nullptr);

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    ASSERT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(existing_node->group_slots[0].device_blocks, request_blocks.per_node[0]);
    EXPECT_EQ(pool->refCount(block), 2u);

    request_blocks.nodes = {existing_node};
    cache->releaseMatchedBlocks({request_blocks});
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, InsertRejectsPartialMultiPoolGroupWithoutAddingCacheHold) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool0         = makeDevicePool({{64, 0}}, kUsableBlocks, "partial_group_pool_0");
    auto             pool1         = makeDevicePool({{64, 0}}, kUsableBlocks, "partial_group_pool_1");

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool0, pool1});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    const BlockIdxType block0 = request_blocks.per_node[0][0];
    const BlockIdxType block1 = request_blocks.per_node[0][1];

    std::vector<std::vector<GroupSlot>> partial_slots(1, std::vector<GroupSlot>(1));
    partial_slots[0][0].device_blocks = {block0, NULL_BLOCK_IDX};
    cache->insert(nullptr, {100}, partial_slots);

    ASSERT_EQ(cache->tree()->nodes().size(), 1u);
    const GroupSlot& cached_slot = cache->tree()->nodes().front()->group_slots[0];
    EXPECT_EQ(cached_slot.device_blocks, (std::vector<BlockIdxType>{NULL_BLOCK_IDX, NULL_BLOCK_IDX}));
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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool}, makeTestTags(1));
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks - 1);
    EXPECT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);
    EXPECT_EQ(pool->refCount(block), 2u);

    full->unreferenceBlocks(request_blocks, BlockRefType::REQUEST);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count("tag_0"), 1u);
    EXPECT_EQ(result.group_block_indices.at("tag_0"), (BlockIndicesType{block}));
    ASSERT_EQ(result.matched_block_sets.size(), 1u);
    EXPECT_EQ(result.matched_block_sets[0].component_group_id, 0);
    EXPECT_EQ(result.matched_block_sets[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[0].per_node, (std::vector<std::vector<BlockIdxType>>{{block}}));
    EXPECT_EQ(pool->refCount(block), 2u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 0);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    result.matched_block_sets.clear();
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, SequentialReclaimDrainsChainWithoutHostBlocks) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // No Host pool, Host disabled → direct release on reclaim.
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.eviction_thread_pool_size = 2;
    seq_cfg.enable_device_cache       = true;
    seq_cfg.enable_memory_cache       = false;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(seq_cfg));

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
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

    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Host disabled (default): Device reclaim → direct release.
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
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

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools(makeStructuralDevicePools(1, "tier_enable_queries"));
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    Component component;
    component.component_id            = 0;
    component.component_group_id      = 0;
    component.tag                     = "kv";
    component.model_layer_ids         = {0};
    component.layer_bytes             = {1};
    std::vector<Component> components = {component};
    ASSERT_TRUE(full->finalizeLayout({0}, components));

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::move(components), std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isMemoryCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {
    auto tree = std::make_unique<BlockTree>(1);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;

    std::vector<ComponentGroupPtr>  groups = {full};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
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
    auto swa                = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id = 0;

    // Create a mock tree node with host data
    auto                                tree = std::make_unique<BlockTree>(1);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    tree->insertNode(nullptr, {100}, slots);
    auto find = tree->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;

    // Verify HOST_TO_DISK transfer descriptor is correct
    TransferDescriptor desc = swa->buildTransfer(find.matched_node, TransferType::HOST_TO_DISK);
    EXPECT_EQ(desc.source_tier, Tier::HOST);
    EXPECT_EQ(desc.target_tier, Tier::DISK);
    EXPECT_EQ(desc.host_block, 7);
}

TEST_F(BlockTreeCacheTest, MatchCollectsBlocksSelectedByGroupPolicy) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(3);

    std::shared_ptr<FullComponentGroup> full     = std::make_shared<FullComponentGroup>();
    full->component_group_id                     = 0;
    std::shared_ptr<LinearComponentGroup> linear = std::make_shared<LinearComponentGroup>();
    linear->component_group_id                   = 1;
    std::shared_ptr<SWAComponentGroup> swa       = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                      = 2;

    std::vector<ComponentGroupPtr>  component_groups = {full, linear, swa};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(component_groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(result.group_block_indices.at("tag_0"), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(result.group_block_indices.at("tag_1"), (BlockIndicesType{22}));
    EXPECT_EQ(result.group_block_indices.at("tag_2"), (BlockIndicesType{31, 32}));
    cache->releaseMatchedBlocks(result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::vector<DeviceBlockPoolPtr> device_pools = makeStructuralDevicePools(2, "aggregated_device_pool");
    auto                            pool0_prefix = device_pools[0]->malloc(1);
    auto                            pool1_prefix = device_pools[1]->malloc(3);
    ASSERT_TRUE(pool0_prefix.has_value());
    ASSERT_TRUE(pool1_prefix.has_value());
    full->setDevicePools(device_pools, makeTestTags(2));

    Component first_component;
    first_component.component_id       = 0;
    first_component.component_group_id = 0;
    first_component.tag                = "tag_0";
    first_component.model_layer_ids    = {0};
    first_component.layer_bytes        = {1};
    Component second_component;
    second_component.component_id       = 1;
    second_component.component_group_id = 0;
    second_component.tag                = "tag_1";
    second_component.model_layer_ids    = {0};
    second_component.layer_bytes        = {1};
    std::vector<Component> components   = {first_component, second_component};
    ASSERT_TRUE(full->finalizeLayout({0, 1}, components));

    std::vector<ComponentGroupPtr>             component_groups = {full};
    std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping  = {{0, 0}, {0, 1}};
    auto components_ptr      = std::make_shared<const std::vector<Component>>(std::move(components));
    auto per_rank_engine     = std::make_shared<PerRankBlockTransferEngine>(component_groups, components_ptr);
    auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool           = std::make_unique<BlockCacheTaskPool>(2, 1000, "BlockTreeEvictionPool");
    std::unique_ptr<BlockTreeCache>            cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::move(components_ptr),
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::move(transfer_dispatcher),
                                         std::move(task_pool),
                                         makeTestTags(2),
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr, nullptr},
                                         std::move(per_tag_mapping));
    ASSERT_TRUE(cache->init());

    GroupBlockSet request_holder = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
    ASSERT_EQ(request_holder.per_node.size(), 2u);
    ASSERT_EQ(request_holder.per_node[0].size(), 2u);
    ASSERT_EQ(request_holder.per_node[1].size(), 2u);
    const BlockIndicesType tag0_blocks = {request_holder.per_node[0][0], request_holder.per_node[1][0]};
    const BlockIndicesType tag1_blocks = {request_holder.per_node[0][1], request_holder.per_node[1][1]};
    EXPECT_NE(tag0_blocks, tag1_blocks);

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = request_holder.per_node[0];
    slots[1][0].device_blocks = request_holder.per_node[1];
    cache->insert(nullptr, {100, 200}, slots);
    full->unreferenceBlocks(request_holder, BlockRefType::REQUEST);
    device_pools[0]->free(*pool0_prefix);
    device_pools[1]->free(*pool1_prefix);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(result.group_block_indices.at("tag_0"), tag0_blocks);
    EXPECT_EQ(result.group_block_indices.at("tag_1"), tag1_blocks);
    cache->releaseMatchedBlocks(result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, ReorderedPoolsPreserveTagAddressedMatchResults) {
    auto make_cache = [](std::vector<std::string> tags, const std::string& pool_name_prefix) {
        auto full                = std::make_shared<FullComponentGroup>();
        full->component_group_id = 0;

        std::vector<DeviceBlockPoolPtr> device_pools = makeStructuralDevicePools(2, pool_name_prefix);
        std::vector<BlockIdList>        prefix_blocks;
        prefix_blocks.reserve(tags.size());
        for (size_t index = 0; index < tags.size(); ++index) {
            const size_t prefix_count = tags[index] == "hca_kv" ? 1 : 3;
            auto         prefix       = device_pools[index]->malloc(prefix_count);
            RTP_LLM_CHECK(prefix.has_value());
            prefix_blocks.push_back(std::move(*prefix));
        }
        full->setDevicePools(device_pools, std::move(tags));

        std::vector<Component> components;
        std::vector<int>       membership;
        components.reserve(full->tags().size());
        membership.reserve(full->tags().size());
        for (size_t index = 0; index < full->tags().size(); ++index) {
            Component component;
            component.component_id       = static_cast<int>(index);
            component.component_group_id = 0;
            component.tag                = full->tags()[index];
            component.model_layer_ids    = {0};
            component.layer_bytes        = {1};
            components.push_back(std::move(component));
            membership.push_back(static_cast<int>(index));
        }
        RTP_LLM_CHECK(full->finalizeLayout(std::move(membership), components));

        std::vector<ComponentGroupPtr>             component_groups = {full};
        std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping  = {{0, 0}, {0, 1}};
        const std::vector<std::string>             per_tag_tags     = full->tags();
        auto components_ptr      = std::make_shared<const std::vector<Component>>(std::move(components));
        auto per_rank_engine     = std::make_shared<PerRankBlockTransferEngine>(component_groups, components_ptr);
        auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
        auto task_pool           = std::make_unique<BlockCacheTaskPool>(2, 1000, "BlockTreeEvictionPool");
        auto cache = std::make_unique<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                                      std::move(component_groups),
                                                      std::move(components_ptr),
                                                      BlockTreeCacheConfig{},
                                                      std::shared_ptr<StorageBackend>{},
                                                      std::move(transfer_dispatcher),
                                                      std::move(task_pool),
                                                      per_tag_tags,
                                                      std::vector<DeviceKVCacheGroupPtr>{nullptr, nullptr},
                                                      std::move(per_tag_mapping));
        RTP_LLM_CHECK(cache->init());

        GroupBlockSet request_holder = full->allocateBlocks(Tier::DEVICE, 2, BlockRefType::REQUEST);
        RTP_LLM_CHECK(request_holder.per_node.size() == 2);
        RTP_LLM_CHECK(request_holder.per_node[0].size() == 2);
        RTP_LLM_CHECK(request_holder.per_node[1].size() == 2);
        std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
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
    EXPECT_EQ(original_result.group_block_indices.at("hca_kv"), reordered_result.group_block_indices.at("hca_kv"));
    EXPECT_EQ(original_result.group_block_indices.at("csa_kv"), reordered_result.group_block_indices.at("csa_kv"));
    EXPECT_NE(original_result.group_block_indices.at("hca_kv"), original_result.group_block_indices.at("csa_kv"));
    original->releaseMatchedBlocks(original_result.matched_block_sets);
    reordered->releaseMatchedBlocks(reordered_result.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, InvalidExplicitTagsFailBeforeGroupMutation) {
    auto       pools          = makeStructuralDevicePools(2, "invalid_explicit_tags");
    const auto expect_invalid = [](std::vector<DeviceBlockPoolPtr> pools, std::vector<std::string> tags) {
        auto group = std::make_shared<FullComponentGroup>();
        EXPECT_ANY_THROW(group->setDevicePools(std::move(pools), std::move(tags)));
        EXPECT_TRUE(group->devicePools().empty());
        EXPECT_TRUE(group->tags().empty());
    };

    expect_invalid({pools[0]}, {""});
    expect_invalid({pools[0], pools[1]}, {"duplicate", "duplicate"});
    expect_invalid({pools[0], pools[1]}, {"only_one"});
}

TEST_F(BlockTreeCacheTest, EmptyDevicePoolsFailBeforeGroupMutation) {
    auto group = std::make_shared<FullComponentGroup>();
    EXPECT_ANY_THROW(group->setDevicePools({}, {}));
    EXPECT_TRUE(group->devicePools().empty());
    EXPECT_TRUE(group->tags().empty());
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 1;

    std::vector<ComponentGroupPtr>  groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    slots[2][0].device_blocks = {12};
    slots[3][0].device_blocks = {13};
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};

    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_blocks, 1u);
    cache->releaseMatchedBlocks(partial.matched_block_sets);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_blocks, 4u);
    cache->releaseMatchedBlocks(restored.matched_block_sets);
}

TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
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
    auto host_pool            = makeHostPool(65, 2);
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = 0;
    group->setDevicePools(makeStructuralDevicePools(1, "host_layout_payload_mismatch"));
    group->setHostPool(host_pool);

    Component component;
    component.component_id            = 0;
    component.component_group_id      = 0;
    component.tag                     = "kv";
    component.model_layer_ids         = {0};
    component.layer_bytes             = {64};
    std::vector<Component> components = {component};
    ASSERT_TRUE(group->finalizeLayout({0}, components));

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    std::vector<ComponentGroupPtr> groups = {group};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::move(components), std::move(config));

    EXPECT_EQ(cache, nullptr);
}

TEST(BlockTreeCacheConfigurationTest, RejectsComponentBindingDrift) {
    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = 0;
    group->setDevicePools(makeStructuralDevicePools(1, "component_binding_drift"));

    Component component;
    component.component_id            = 0;
    component.component_group_id      = 0;
    component.tag                     = "kv";
    component.model_layer_ids         = {0};
    component.layer_bytes             = {64};
    std::vector<Component> components = {component};
    ASSERT_TRUE(group->finalizeLayout({0}, components));
    // Drift the descriptor so component_id no longer matches its registry index.
    components[0].component_id = 1;

    std::vector<ComponentGroupPtr> groups = {group};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::move(components));

    EXPECT_EQ(cache, nullptr);
}

TEST_F(BlockTreeCacheTest, LoadBackOnlyReloadsSWAWindow) {
    std::unique_ptr<BlockTree> tree = std::make_unique<BlockTree>(2);

    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;

    std::shared_ptr<SWAComponentGroup> swa = std::make_shared<SWAComponentGroup>(128, 64);
    swa->component_group_id                = 1;

    std::vector<ComponentGroupPtr>  groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].host_block    = static_cast<BlockIdxType>(100 + i);
    }

    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100, 200, 300, 400}, slots));

    BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.host_load_back_blocks, 2u);
    EXPECT_EQ(result.load_back_blocks, 2u);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), 4u);
    ASSERT_EQ(result.load_back_ticket->itemCount(), 6u);
    const auto count_exact_item = [&ticket = *result.load_back_ticket](int                group_id,
                                                                      Tier               source_tier,
                                                                      size_t             path_index,
                                                                      BlockIdxType       source_block,
                                                                      const std::string& device_group_tag) {
        size_t count = 0;
        for (size_t item_index = 0; item_index < ticket.itemCount(); ++item_index) {
            count += ticket.groupId(item_index) == group_id && ticket.sourceTier(item_index) == source_tier
                     && ticket.pathIndex(item_index) == path_index
                     && ticket.sourceBlocks(item_index) == std::vector<BlockIdxType>{source_block}
                     && ticket.deviceGroupTags(item_index) == std::vector<std::string>{device_group_tag};
        }
        return count;
    };
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_item(/*group_id=*/0,
                                   Tier::DEVICE,
                                   path_index,
                                   static_cast<BlockIdxType>(10 + path_index),
                                   /*device_group_tag=*/"tag_0"),
                  1);
    }
    for (size_t path_index = 2; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_item(/*group_id=*/1,
                                   Tier::HOST,
                                   path_index,
                                   static_cast<BlockIdxType>(100 + path_index),
                                   /*device_group_tag=*/"tag_1"),
                  1);
    }
}

// ---------------------------------------------------------------------------
// Test: enable_load_back — match detects Host/Disk data needing reload
// ---------------------------------------------------------------------------
TEST_F(BlockTreeCacheTest, LoadBackDetectsHostData) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    // Insert a node and manually set host data (simulating prior demotion).
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim without host demotion, then manually set up a host-only node.
    // Instead, manually set up a node with host_block but no device_blocks
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After reclaim without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<std::vector<GroupSlot>> slots2(1, std::vector<GroupSlot>(1));
    slots2[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots2);

    // Manually set host_block and clear device_blocks to simulate a demoted state.
    auto find = cache->tree()->findNode({200});
    ASSERT_NE(find.matched_node, nullptr);
    GroupSlot& slot          = find.matched_node->group_slots[0];
    slot.host_block          = 7;
    const auto device_blocks = full->getBlocks(slot, Tier::DEVICE);
    ASSERT_EQ(device_blocks, (BlockIndicesType{55}));
    full->unreferenceBlocks(GroupBlockSet{full->component_group_id, Tier::DEVICE, {device_blocks}},
                            BlockRefType::BLOCK_CACHE);
    slot.device_blocks.clear();

    // Match should detect load_back
    auto result = cache->match({200});
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
}

static std::unique_ptr<BlockTreeCache> makeHostOnlyLoadBackCache(DeviceBlockPoolPtr device_pool = nullptr) {
    if (device_pool == nullptr) {
        device_pool = makeDevicePool({{1, 0}}, 1, "load_back_ticket_abort");
    }
    RTP_LLM_CHECK(device_pool != nullptr);
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/1);
    RTP_LLM_CHECK(host_pool != nullptr);

    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({device_pool}, makeTestTags(1));
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    Component component;
    component.component_id            = 0;
    component.component_group_id      = 0;
    component.tag                     = full->tags().front();
    component.model_layer_ids         = {0};
    component.layer_bytes             = {1};
    std::vector<Component> components = {component};
    RTP_LLM_CHECK(full->finalizeLayout({0}, components));

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::move(components), std::move(config));
    RTP_LLM_CHECK(cache != nullptr);

    GroupBlockSet request_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::REQUEST);
    RTP_LLM_CHECK(request_holder.per_node.size() == 1);
    RTP_LLM_CHECK(request_holder.per_node[0].size() == 1);
    const BlockIdxType device_block = request_holder.per_node[0][0];

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {device_block};
    cache->insert(nullptr, {200}, slots);
    full->unreferenceBlocks(request_holder, BlockRefType::REQUEST);

    BlockTreeFindResult find = cache->tree()->findNode({200});
    RTP_LLM_CHECK(find.matched_node != nullptr);
    GroupSlot& slot = find.matched_node->group_slots[0];
    slot.host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    RTP_LLM_CHECK(slot.host_block != NULL_BLOCK_IDX);
    const auto device_blocks = full->getBlocks(slot, Tier::DEVICE);
    RTP_LLM_CHECK(device_blocks == BlockIndicesType{device_block});
    full->unreferenceBlocks(GroupBlockSet{full->component_group_id, Tier::DEVICE, {device_blocks}},
                            BlockRefType::BLOCK_CACHE);
    slot.device_blocks.clear();
    return cache;
}

static std::unique_ptr<BlockTreeCache>
makeMappingValidationCache(std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping,
                           size_t                                     device_pool_count,
                           const std::shared_ptr<HostBlockPool>&      host_pool,
                           bool                                       initialize) {
    auto full                                   = std::make_shared<FullComponentGroup>();
    full->component_group_id                    = 0;
    const std::vector<std::string> per_tag_tags = makeTestTags(per_tag_mapping.size());
    std::vector<std::string>       local_tags(device_pool_count);
    bool                           complete_local_mapping = per_tag_mapping.size() == device_pool_count;
    for (size_t tag_index = 0; tag_index < per_tag_mapping.size() && complete_local_mapping; ++tag_index) {
        const auto& mapping = per_tag_mapping[tag_index];
        if (mapping.component_group_id != 0 || mapping.local_pool_index < 0
            || static_cast<size_t>(mapping.local_pool_index) >= local_tags.size()
            || !local_tags[static_cast<size_t>(mapping.local_pool_index)].empty()) {
            complete_local_mapping = false;
            break;
        }
        local_tags[static_cast<size_t>(mapping.local_pool_index)] = per_tag_tags[tag_index];
    }
    if (!complete_local_mapping
        || std::any_of(local_tags.begin(), local_tags.end(), [](const std::string& tag) { return tag.empty(); })) {
        local_tags = makeTestTags(device_pool_count);
    }
    full->setDevicePools(makeStructuralDevicePools(device_pool_count, "mapping_validation_device"),
                         std::move(local_tags));
    full->setHostPool(host_pool);

    const size_t payload_bytes = host_pool == nullptr ? device_pool_count : host_pool->payloadBytes();
    RTP_LLM_CHECK(payload_bytes >= device_pool_count);
    std::vector<Component> components;
    std::vector<int>       membership;
    size_t                 remaining_bytes = payload_bytes;
    for (size_t pool_index = 0; pool_index < device_pool_count; ++pool_index) {
        Component component;
        component.component_id       = static_cast<int>(pool_index);
        component.component_group_id = 0;
        component.tag                = full->tags()[pool_index];
        component.model_layer_ids    = {0};
        component.layer_bytes        = {pool_index + 1 == device_pool_count ? remaining_bytes : 1};
        remaining_bytes -= component.layer_bytes.front();
        components.push_back(std::move(component));
        membership.push_back(static_cast<int>(pool_index));
    }
    RTP_LLM_CHECK(full->finalizeLayout(std::move(membership), components));

    BlockTreeCacheConfig config;
    config.enable_memory_cache = host_pool != nullptr;
    config.enable_load_back    = host_pool != nullptr;

    std::vector<ComponentGroupPtr>     groups = {full};
    std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups(per_tag_mapping.size());
    auto components_ptr      = std::make_shared<const std::vector<Component>>(std::move(components));
    auto per_rank_engine     = std::make_shared<PerRankBlockTransferEngine>(groups, components_ptr);
    auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine));
    auto task_pool           = std::make_unique<BlockCacheTaskPool>(2, 1000, "BlockTreeEvictionPool");
    auto                               cache = std::make_unique<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                                  std::move(groups),
                                                  std::move(components_ptr),
                                                  std::move(config),
                                                  nullptr,
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool),
                                                  per_tag_tags,
                                                  std::move(per_tag_device_groups),
                                                  std::move(per_tag_mapping));
    if (initialize && !cache->init()) {
        return nullptr;
    }
    return cache;
}

TEST_F(BlockTreeCacheTest, LoadBackGroupMappingUsesLocalPoolIndexOrderAndLeavesTicketUntouched) {
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/2, /*usable_count=*/2);
    ASSERT_NE(host_pool, nullptr);

    // Global tag order is deliberately the reverse of the component group's local
    // device-pool order. local_pool_index, not gid iteration order, is authoritative.
    std::unique_ptr<BlockTreeCache> cache = makeMappingValidationCache(
        {{/*component_group_id=*/0, /*local_pool_index=*/1}, {/*component_group_id=*/0, /*local_pool_index=*/0}},
        /*device_pool_count=*/2,
        host_pool,
        /*initialize=*/true);
    ASSERT_NE(cache, nullptr);

    const ComponentGroupPtr& group        = cache->componentGroups().front();
    const BlockIdxType       source_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);

    const size_t     free_before      = host_pool->freeBlocksNum();
    const size_t     ref_before       = host_pool->refCount(source_block);
    const CacheStats stats_before     = cache->getStats();
    const auto       transfer_before  = slots[0][0].transfer_state;
    const auto       expected_tag_map = std::vector<std::string>{"tag_1", "tag_0"};

    EXPECT_TRUE(cache->validateDeviceGroupTagsForComponentGroup(/*component_group_id=*/0, expected_tag_map));
    EXPECT_FALSE(cache->validateDeviceGroupTagsForComponentGroup(/*component_group_id=*/0, {"tag_0", "tag_1"}));
    EXPECT_FALSE(cache->validateDeviceGroupTagsForComponentGroup(/*component_group_id=*/-1, expected_tag_map));
    EXPECT_FALSE(cache->validateDeviceGroupTagsForComponentGroup(/*component_group_id=*/1, expected_tag_map));

    EXPECT_EQ(host_pool->freeBlocksNum(), free_before);
    EXPECT_EQ(host_pool->refCount(source_block), ref_before);
    EXPECT_EQ(cache->getStats().tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(cache->tree()->findNode({100}).matched_node->group_slots[0].transfer_state, transfer_before);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_back_ticket->items().front().device_group_tags, expected_tag_map)
        << "the shared validator must not normalize or rewrite producer-owned metadata";
    EXPECT_EQ(host_pool->refCount(source_block), ref_before + 1);

    result.load_back_ticket.reset();
    EXPECT_EQ(host_pool->refCount(source_block), ref_before);
    EXPECT_EQ(host_pool->freeBlocksNum(), free_before);
}

TEST_F(BlockTreeCacheTest, PendingLoadBackTicketHardStopsSecondMatchUntilAbort) {
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/2);
    ASSERT_NE(host_pool, nullptr);

    std::unique_ptr<BlockTreeCache> cache = makeMappingValidationCache(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}}, /*device_pool_count=*/1, host_pool, /*initialize=*/true);
    ASSERT_NE(cache, nullptr);

    const ComponentGroupPtr& group        = cache->componentGroups().front();
    const BlockIdxType       source_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    TreeNode* source_node  = cache->tree()->insertNode(nullptr, {100}, slots).leaf;
    ASSERT_NE(source_node, nullptr);

    BlockTreeMatchResult first_match = cache->match({100});
    ASSERT_NE(first_match.load_back_ticket, nullptr);
    EXPECT_EQ(source_node->group_slots[0].transfer_state, SlotTransferState::LOAD_BACK_PENDING);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    BlockTreeMatchResult second_match = cache->match({100});
    EXPECT_EQ(second_match.matched_node, nullptr);
    EXPECT_EQ(second_match.matched_blocks, 0u);
    EXPECT_EQ(second_match.load_back_ticket, nullptr);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    first_match.load_back_ticket.reset();
    EXPECT_EQ(source_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(host_pool->refCount(source_block), 1u);
}

TEST_F(BlockTreeCacheTest, LoadBackGroupMappingInitRejectsOutOfRangeDuplicateAndHoleMetadata) {
    const auto make_uninitialized = [](std::vector<BlockTreeCache::PerTagMapping> mapping, size_t pool_count) {
        return makeMappingValidationCache(std::move(mapping), pool_count, nullptr, /*initialize=*/false);
    };

    std::unique_ptr<BlockTreeCache> out_of_range = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/2}},
        /*pool_count=*/2);
    ASSERT_NE(out_of_range, nullptr);
    EXPECT_FALSE(out_of_range->init());

    std::unique_ptr<BlockTreeCache> duplicate = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/0}},
        /*pool_count=*/2);
    ASSERT_NE(duplicate, nullptr);
    EXPECT_FALSE(duplicate->init());

    std::unique_ptr<BlockTreeCache> hole = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/2}},
        /*pool_count=*/3);
    ASSERT_NE(hole, nullptr);
    EXPECT_FALSE(hole->init());
}

TEST_F(BlockTreeCacheTest, InvalidProducerMappingFailsInitializationWithoutSourceProtection) {
    // Two device components require at least one payload byte each. Keep the
    // layout valid so initialization reaches the duplicate producer mapping.
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/2, /*usable_count=*/2);
    ASSERT_NE(host_pool, nullptr);

    std::unique_ptr<BlockTreeCache> cache = makeMappingValidationCache(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/0}},
        /*device_pool_count=*/2,
        host_pool,
        /*initialize=*/false);
    ASSERT_NE(cache, nullptr);
    const size_t free_before = host_pool->freeBlocksNum();
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
    EXPECT_EQ(cache->tree()->nodeCount(), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), free_before);
}

TEST_F(BlockTreeCacheTest, LoadBackPreparedPrefixFailureRollsBackAllSourceAndTargetHolders) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr first_device_pool  = makeDevicePool({{1, 0}}, 1, "load_back_prepared_prefix_first");
    DeviceBlockPoolPtr second_device_pool = makeDevicePool({{1, 0}}, 1, "load_back_prepared_prefix_second");
    ASSERT_NE(first_device_pool, nullptr);
    ASSERT_NE(second_device_pool, nullptr);

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);
    ASSERT_NE(first_host_pool, nullptr);
    ASSERT_NE(second_host_pool, nullptr);

    auto first_group                = std::make_shared<FullComponentGroup>();
    first_group->component_group_id = 0;
    first_group->setDevicePools({first_device_pool}, makeTestTags(1));
    first_group->setHostPool(first_host_pool);
    auto second_group                = std::make_shared<FullComponentGroup>();
    second_group->component_group_id = 1;
    second_group->setDevicePools({second_device_pool}, makeTestTags(1, 1));
    second_group->setHostPool(second_host_pool);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                       = true;
    config.enable_load_back                          = true;
    std::vector<ComponentGroupPtr>  component_groups = {first_group, second_group};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(2), std::move(component_groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType first_source  = first_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType second_source = second_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(first_source, NULL_BLOCK_IDX);
    ASSERT_NE(second_source, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].host_block = first_source;
    slots[0][1].host_block = second_source;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    std::vector<PendingLoadBackItem>& items = result.load_back_ticket->items();
    ASSERT_EQ(items.size(), 2u);
    ASSERT_EQ(items[0].group_id, 0);
    ASSERT_EQ(items[1].group_id, 1);
    EXPECT_EQ(first_host_pool->refCount(first_source), 2u);
    EXPECT_EQ(second_host_pool->refCount(second_source), 2u);

    // Duplicate the first item immediately after itself. The complete batch passes
    // preflight while both slots are IDLE. Preparation then claims the first item
    // and takes its target holder; beginLoadBack for the duplicate observes the
    // same slot already LOADING_BACK and fails with one prepared item and one
    // untouched trailing item. Add the matching source planning hold explicitly
    // so every item in the synthetic ticket owns exactly one source hold.
    PendingLoadBackItem duplicate_first_item = items.front();
    items.insert(items.begin() + 1, std::move(duplicate_first_item));
    first_group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{first_source}}}, BlockRefType::REQUEST);
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

    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
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
    ASSERT_EQ(find.matched_node->group_slots.size(), 2u);
    EXPECT_EQ(find.matched_node->group_slots[0].host_block, first_source);
    EXPECT_EQ(find.matched_node->group_slots[1].host_block, second_source);
    EXPECT_TRUE(find.matched_node->group_slots[0].device_blocks.empty());
    EXPECT_TRUE(find.matched_node->group_slots[1].device_blocks.empty());
    EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_slots[1].transfer_state, SlotTransferState::IDLE);

    result.load_back_ticket.reset();
    EXPECT_EQ(first_host_pool->refCount(first_source), 1u) << "committed ticket must not release source twice";
    EXPECT_EQ(second_host_pool->refCount(second_source), 1u) << "committed ticket must not release source twice";
    first_device_pool->decRef(first_request_targets, BlockRefType::REQUEST);
    second_device_pool->decRef(second_request_targets, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheTest, LoadBackQueueRejectionRollsBackCoreHoldersAndRetainsRequestTarget) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             device_pool = makeDevicePool({{1, 0}}, 2, "load_back_queue_rejection");
    std::shared_ptr<HostBlockPool> host_pool   = makeHostPool(1, 2);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({device_pool}, makeTestTags(1));
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);
    const size_t source_ref_before = host_pool->refCount(source_block);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_back_ticket->items().front().device_group_tags, (std::vector<std::string>{"tag_0"}));
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before + 1);

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets, BlockRefType::REQUEST);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    result.load_back_ticket->items().front().target_device_blocks = {request_target};
    ASSERT_EQ(device_pool->refCount(request_target), 1u);

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    ASSERT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before);
    EXPECT_EQ(device_pool->refCount(request_target), 1u);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(find.matched_node->group_slots.size(), 1u);
    EXPECT_EQ(find.matched_node->group_slots[0].host_block, source_block);
    EXPECT_TRUE(find.matched_node->group_slots[0].device_blocks.empty());
    EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    result.load_back_ticket.reset();
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before) << "committed ticket must not release source twice";
    cache->waitForPendingTasks();
    device_pool->decRef(request_targets, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheTest, LoadBackQueueRejectionRollsBackMixedDeviceAndHostItems) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr resident_device_pool = makeDevicePool({{1, 0}}, 1, "load_back_mixed_resident");
    DeviceBlockPoolPtr target_device_pool   = makeDevicePool({{1, 0}}, 2, "load_back_mixed_target");
    std::shared_ptr<HostBlockPool> resident_host_pool = makeHostPool(1, 1);
    std::shared_ptr<HostBlockPool> host_pool          = makeHostPool(1, 2);
    ASSERT_NE(resident_device_pool, nullptr);
    ASSERT_NE(target_device_pool, nullptr);
    ASSERT_NE(resident_host_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto resident_group                = std::make_shared<FullComponentGroup>();
    resident_group->component_group_id = 0;
    resident_group->setDevicePools({resident_device_pool});
    resident_group->setHostPool(resident_host_pool);
    auto loading_group                = std::make_shared<FullComponentGroup>();
    loading_group->component_group_id = 1;
    loading_group->setDevicePools({target_device_pool});
    loading_group->setHostPool(host_pool);

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::vector<ComponentGroupPtr> groups = {resident_group, loading_group};
    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(2), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    GroupBlockSet resident_holder = resident_group->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(resident_holder.per_node.size(), 1u);
    ASSERT_EQ(resident_holder.per_node.front().size(), 1u);
    const BlockIdxType resident_block = resident_holder.per_node.front().front();
    const BlockIdxType host_block     = loading_group->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {resident_block};
    slots[0][1].host_block    = host_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);
    ASSERT_EQ(resident_device_pool->refCount(resident_block), 1u);
    ASSERT_EQ(host_pool->refCount(host_block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->itemCount(), 2u);
    EXPECT_EQ(result.load_back_ticket->sourceTier(0), Tier::DEVICE);
    EXPECT_EQ(result.load_back_ticket->sourceTier(1), Tier::HOST);
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 2u);
    EXPECT_EQ(host_pool->refCount(host_block), 2u);

    const BlockIdxType request_target = poolMalloc(*target_device_pool);
    ASSERT_NE(request_target, NULL_BLOCK_IDX);
    target_device_pool->incRef(request_target, BlockRefType::REQUEST);
    ASSERT_EQ(target_device_pool->refCount(request_target), 1u);
    ASSERT_TRUE(result.load_back_ticket->bindTargetDeviceBlocks(0, {resident_block}));
    ASSERT_TRUE(result.load_back_ticket->bindTargetDeviceBlocks(1, {request_target}));

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 1u);
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    EXPECT_EQ(target_device_pool->refCount(request_target), 1u);

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(find.matched_node->group_slots.size(), 2u);
    EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_slots[1].transfer_state, SlotTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    result.load_back_ticket.reset();
    EXPECT_EQ(resident_device_pool->refCount(resident_block), 1u);
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    target_device_pool->decRef(request_target, BlockRefType::REQUEST);
}

// Deferred load_back: match() plans (references the source blocks) but does NOT execute
// load_back. The result carries a LoadBackTicket; the allocator binds request-owned
// device targets before committing it. Dropping it uncommitted aborts (unreferences
// the source) without allocating or copying anything.

// Not committing the ticket: no device block is allocated and no async copy is submitted;
// the ticket destructor aborts safely.
TEST_F(BlockTreeCacheTest, LoadBackTicketAbortSkipsLoadBack) {
    auto cache = makeHostOnlyLoadBackCache();

    auto result = cache->match({200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), 1u);
    // Counters reflect the planned load_back; match() submits nothing async and leaves
    // async_context null (the async context is produced only at commit).
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
    EXPECT_EQ(result.async_context, nullptr);

    // Drop the ticket without committing => RAII abort (source unreferenced). No async
    // task was ever submitted, so waitForPendingTasks returns immediately.
    result.load_back_ticket.reset();
    cache->releaseMatchedBlocks(result.matched_block_sets);
    cache->waitForPendingTasks();
}

// Committing the ticket uses the allocator-owned device target and submits the async
// copy, yielding a non-null AsyncContext.
TEST_F(BlockTreeCacheTest, LoadBackTicketCommitTriggersLoadBack) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr device_pool = makeDevicePool({{1, 0}}, 1, "load_back_ticket_commit");
    ASSERT_NE(device_pool, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeHostOnlyLoadBackCache(device_pool);

    BlockTreeMatchResult result = cache->match({200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), 1u);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets, BlockRefType::REQUEST);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    result.load_back_ticket->items()[0].target_device_blocks = {request_target};

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    EXPECT_NE(context, nullptr);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    cache->waitForPendingTasks();
    device_pool->decRef(request_targets, BlockRefType::REQUEST);
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
    auto host_pool = makeHostPool(kBlockBytes, kPoolSize);
    auto disk_pool = makeDiskPool(kBlockBytes, kPoolSize, std::make_unique<MemoryDiskBlockIO>());

    const std::vector<size_t> device_free_before = {
        device_pools[0]->freeBlocksNum(),
        device_pools[1]->freeBlocksNum(),
        device_pools[2]->freeBlocksNum(),
    };
    const size_t host_free_before = host_pool->freeBlocksNum();
    const size_t disk_free_before = disk_pool->freeBlocksNum();

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools(device_pools, makeTestTags(device_pools.size()));
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);

    BlockTreeCacheConfig config;
    config.enable_device_cache            = true;
    config.enable_memory_cache            = true;
    config.enable_disk_cache              = true;
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    GroupBlockSet root_device_holds = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(root_device_holds.per_node.size(), 1u);
    ASSERT_EQ(root_device_holds.per_node[0].size(), 3u);
    const BlockIdxType device_block_0 = root_device_holds.per_node[0][0];
    const BlockIdxType device_hole    = root_device_holds.per_node[0][1];
    const BlockIdxType device_block_2 = root_device_holds.per_node[0][2];
    ASSERT_NE(device_block_0, NULL_BLOCK_IDX);
    ASSERT_NE(device_hole, NULL_BLOCK_IDX);
    ASSERT_NE(device_block_2, NULL_BLOCK_IDX);

    GroupBlockSet hole_holder{0, Tier::DEVICE, {{NULL_BLOCK_IDX, device_hole, NULL_BLOCK_IDX}}};
    full->unreferenceBlocks(hole_holder, BlockRefType::BLOCK_CACHE);
    root_device_holds.per_node[0][1]                    = NULL_BLOCK_IDX;
    cache->tree()->root()->group_slots[0].device_blocks = root_device_holds.per_node[0];

    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block = full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> lower_tier_slots(2, std::vector<GroupSlot>(1));
    lower_tier_slots[0][0].host_block = host_block;
    lower_tier_slots[1][0].disk_slot  = disk_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100, 200}, lower_tier_slots));

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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({device_pool}, makeTestTags(1));
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});
    ASSERT_NE(cache, nullptr);

    GroupBlockSet tree_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(tree_holder.per_node.size(), 1u);
    ASSERT_EQ(tree_holder.per_node[0].size(), 1u);
    const BlockIdxType block = tree_holder.per_node[0][0];
    ASSERT_NE(block, NULL_BLOCK_IDX);
    GroupBlockSet external_holder = tree_holder;
    full->referenceBlocks(external_holder, BlockRefType::REQUEST);
    EXPECT_EQ(device_pool->refCount(block), 2u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = tree_holder.per_node[0];
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));
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

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({device_pool}, makeTestTags(1));
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    BlockTreeCacheConfig config;
    config.enable_device_cache            = true;
    config.enable_memory_cache            = true;
    config.enable_disk_cache              = true;
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);
    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(
        std::vector<ComponentGroupPtr>{full}, std::vector<Component>{});
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    GroupBlockSet device_holder = full->allocateBlocks(Tier::DEVICE, 1, BlockRefType::BLOCK_CACHE);
    ASSERT_EQ(device_holder.per_node.size(), 1u);
    ASSERT_EQ(device_holder.per_node[0].size(), 1u);
    const BlockIdxType device_block = device_holder.per_node[0][0];
    const BlockIdxType host_block   = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    const BlockIdxType disk_block   = full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSlot>> device_slots(1, std::vector<GroupSlot>(1));
    device_slots[0][0].device_blocks = device_holder.per_node[0];
    std::vector<std::vector<GroupSlot>> host_slots(1, std::vector<GroupSlot>(1));
    host_slots[0][0].host_block = host_block;
    std::vector<std::vector<GroupSlot>> disk_slots(1, std::vector<GroupSlot>(1));
    disk_slots[0][0].disk_slot = disk_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, device_slots));
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {200}, host_slots));
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {300}, disk_slots));
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

TEST_F(BlockTreeCacheTest, LoadBackTicketOutlivesHostAndDiskCacheShutdown) {
    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(source_tier));

        auto full                = std::make_shared<FullComponentGroup>();
        full->component_group_id = 0;
        auto host_pool           = makeHostPool(1, 2);
        auto disk_pool           = makeDiskPool(1, 2, std::make_unique<MemoryDiskBlockIO>());
        full->setHostPool(host_pool);
        full->setDiskPool(disk_pool);

        BlockTreeCacheConfig config;
        config.enable_memory_cache            = true;
        config.enable_disk_cache              = true;
        config.enable_load_back               = true;
        std::vector<ComponentGroupPtr> groups = {full};
        auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
            std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
        ASSERT_NE(cache, nullptr);

        const BlockIdxType source_block = full->allocateSingleBlock(source_tier, BlockRefType::BLOCK_CACHE);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        IBlockPool& source_pool =
            source_tier == Tier::HOST ? static_cast<IBlockPool&>(*host_pool) : static_cast<IBlockPool&>(*disk_pool);
        EXPECT_EQ(source_pool.refCount(source_block), 1u);

        std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
        if (source_tier == Tier::HOST) {
            slots[0][0].host_block = source_block;
        } else {
            slots[0][0].disk_slot = source_block;
        }
        ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

        BlockTreeMatchResult result = cache->match({100});
        ASSERT_NE(result.load_back_ticket, nullptr);
        ASSERT_FALSE(result.load_back_ticket->empty());
        ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
        EXPECT_EQ(result.load_back_ticket->items()[0].source_tier, source_tier);
        EXPECT_EQ(result.load_back_ticket->items()[0].source_blocks, (BlockIndicesType{source_block}));
        EXPECT_EQ(source_pool.refCount(source_block), 2u);

        std::shared_ptr<LoadBackTicket> outliving_ticket = std::move(result.load_back_ticket);
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

TEST_F(BlockTreeCacheTest, LoadBackTicketKeepsExplicitLogicalDepthIndependentOfItemPositions) {
    size_t abort_calls = 0;
    auto   registry =
        std::make_shared<LoadBackTicketRegistry>([](const LoadBackTicket&) { return std::shared_ptr<AsyncContext>{}; },
        [&](const LoadBackTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
        });

    PendingLoadBackItem pending_item;
    pending_item.path_index                = 1;
    std::shared_ptr<LoadBackTicket> ticket = registry->createTicket({pending_item}, /*logical_matched_blocks=*/7);
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

    auto registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const LoadBackTicket&) {
            ++commit_calls;
            commit_callback.enterAndWait();
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadBackTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_id, 1);
            }
            shutdown_detached_abort.markEntered();
        });
    PendingLoadBackItem pending_item;
    pending_item.group_id                  = 0;
    std::shared_ptr<LoadBackTicket> ticket = registry->createTicket({pending_item});
    ASSERT_NE(ticket, nullptr);
    PendingLoadBackItem shutdown_pending_item;
    shutdown_pending_item.group_id                          = 1;
    std::shared_ptr<LoadBackTicket> shutdown_pending_ticket = registry->createTicket({shutdown_pending_item});
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
    EXPECT_EQ(registry->createTicket({pending_item}), nullptr);
    registry->shutdown();
}

TEST_F(BlockTreeCacheTest, TicketRegistryCloseDetachesAndAbortsOnce) {
    auto host_pool           = makeHostPool(1, 2);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    GroupBlockSet source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const LoadBackTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadBackTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            full->unreferenceBlocks(source_protection, BlockRefType::REQUEST);
            abort_callback.enterAndWait();
        });
    PendingLoadBackItem pending_item;
    pending_item.group_id                  = 0;
    pending_item.source_tier               = Tier::HOST;
    pending_item.source_blocks             = {source_block};
    std::shared_ptr<LoadBackTicket> ticket = registry->createTicket({pending_item});
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
    auto              registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const LoadBackTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadBackTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_id, 7);
            }
            abort_callback.enterAndWait();
        });
    LoadBackShutdownTestPeer::setShutdownWaitObserver(*registry, [&shutdown_waits] { shutdown_waits.notify(); });
    PendingLoadBackItem pending_item;
    pending_item.group_id                  = 7;
    std::shared_ptr<LoadBackTicket> ticket = registry->createTicket({pending_item});
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
    EXPECT_EQ(registry->createTicket({pending_item}), nullptr);
    LoadBackShutdownTestPeer::setShutdownWaitObserver(*registry, std::function<void()>{});
}

TEST_F(BlockTreeCacheTest, TicketRegistryShutdownWaitsForAbortInFlight) {
    auto host_pool           = makeHostPool(1, 2);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    GroupBlockSet source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown_detached_abort;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const LoadBackTicket&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const LoadBackTicket& ticket) {
            const auto& items = ticket.items();
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u && items[0].group_id == 0) {
                full->unreferenceBlocks(source_protection, BlockRefType::REQUEST);
                abort_callback.enterAndWait();
                return;
            }
            if (items.size() == 1u) {
                EXPECT_EQ(items[0].group_id, 1);
            }
            shutdown_detached_abort.markEntered();
        });
    PendingLoadBackItem pending_item;
    pending_item.group_id                  = 0;
    pending_item.source_tier               = Tier::HOST;
    pending_item.source_blocks             = {source_block};
    std::shared_ptr<LoadBackTicket> ticket = registry->createTicket({pending_item});
    ASSERT_NE(ticket, nullptr);
    PendingLoadBackItem shutdown_pending_item;
    shutdown_pending_item.group_id                          = 1;
    std::shared_ptr<LoadBackTicket> shutdown_pending_ticket = registry->createTicket({shutdown_pending_item});
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
    EXPECT_EQ(registry->createTicket({pending_item}), nullptr);
    full->releaseSingleBlock(Tier::HOST, source_block, BlockRefType::REQUEST);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
}

// A no-match match() plans nothing and returns a null ticket (never created).
TEST_F(BlockTreeCacheTest, EmptyMatchYieldsNoTicket) {
    auto result = cache_->match({100, 200, 300});  // empty tree => no match
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_EQ(result.load_back_ticket, nullptr);
}

}  // namespace
}  // namespace rtp_llm
