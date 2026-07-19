#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceFullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"

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
        std::lock_guard<std::mutex> lock(cache.wait_mutex_);
        cache.pending_task_wait_observer_for_test_ = observer;
    }
};
}  // namespace rtp_llm::block_tree_cache_test

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

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

// A few legacy unit tests intentionally use literal block indices with null
// pools to exercise tree/match policy in isolation. They do not model physical
// ownership. Clear only those deliberate synthetic slots before the production
// destructor enforces and drains real tree holds.
void clearDeliberateNonPhysicalSlots(BlockTreeCache& cache) {
    const auto clear_node = [&cache](TreeNode* node) {
        ASSERT_NE(node, nullptr);
        node->group_slots.resize(cache.componentGroups().size());
        for (GroupSlot& slot : node->group_slots) {
            std::fill(slot.device_blocks.begin(), slot.device_blocks.end(), NULL_BLOCK_IDX);
            slot.host_block     = NULL_BLOCK_IDX;
            slot.disk_slot      = NULL_BLOCK_IDX;
            slot.transfer_state = SlotTransferState::IDLE;
        }
    };

    clear_node(cache.tree()->root());
    for (const std::unique_ptr<TreeNode>& node : cache.tree()->nodes()) {
        clear_node(node.get());
    }
}

class DeliberateNonPhysicalTopologyGuard {
public:
    explicit DeliberateNonPhysicalTopologyGuard(BlockTreeCache& cache): cache_(cache) {}
    ~DeliberateNonPhysicalTopologyGuard() {
        clearDeliberateNonPhysicalSlots(cache_);
    }

private:
    BlockTreeCache& cache_;
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

    void TearDown() override {
        clearDeliberateNonPhysicalSlots(*cache_);
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
    ASSERT_EQ(full_result.group_block_indices.count(0), 1u);
    EXPECT_EQ(full_result.group_block_indices.at(0), (BlockIndicesType{42, 43, 44}));
    cache_->releaseMatchedBlocks(full_result.matched_block_sets);

    BlockTreeMatchResult partial_result = cache_->match({100, 200, 999});
    ASSERT_NE(partial_result.matched_node, nullptr);
    EXPECT_EQ(partial_result.matched_node->cache_key, 200);
    EXPECT_EQ(partial_result.matched_blocks, 2u);
    ASSERT_EQ(partial_result.group_block_indices.count(0), 1u);
    EXPECT_EQ(partial_result.group_block_indices.at(0), (BlockIndicesType{42, 43}));
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

    cache_->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache_->match({100, 200, 300});
    ASSERT_NE(result.matched_node, nullptr);
    EXPECT_EQ(result.matched_node->cache_key, 100);
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10}));

    cache_->releaseMatchedBlocks(result.matched_block_sets);
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
    int reclaimed = multi_cache->reclaimBlocks(1, Tier::DEVICE);
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

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  nullptr,
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_EQ(cache->copyEngine(), nullptr);
    EXPECT_FALSE(cache->isInitialized());
    cache.reset();
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(full->component_group_id, 1);
}

TEST(BlockTreeCacheConstructionTest, NullComponentGroupFailsInitializationAndDestructionReturnsNormally) {
    auto                           tree   = std::make_unique<BlockTree>(1);
    std::vector<ComponentGroupPtr> groups = {nullptr};
    std::vector<Component>         components;

    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(groups),
                                                  std::move(components),
                                                  BlockTreeCacheConfig{},
                                                  nullptr,
                                                  nullptr,
                                                  std::vector<DeviceKVCacheGroupPtr>{nullptr},
                                                  std::vector<BlockTreeCache::PerTagMapping>{{0, 0}});
    EXPECT_FALSE(cache->init());
    EXPECT_EQ(cache->copyEngine(), nullptr);
    EXPECT_FALSE(cache->isInitialized());
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
            slots[0][0].device_blocks = {static_cast<BlockIdxType>(i * 100)};
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
    full->setDevicePools({pool0, pool1});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 2);
    ASSERT_EQ(request_blocks.per_node.size(), 2u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 2u);
    ASSERT_EQ(request_blocks.per_node[1].size(), 2u);

    const BlockIdxType a_pool0 = request_blocks.per_node[0][0];
    const BlockIdxType a_pool1 = request_blocks.per_node[0][1];
    const BlockIdxType b_pool0 = request_blocks.per_node[1][0];
    const BlockIdxType b_pool1 = request_blocks.per_node[1][1];
    EXPECT_NE(a_pool0, a_pool1);
    EXPECT_NE(b_pool0, b_pool1);

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {a_pool0};
    slots[0][1].device_blocks = {a_pool1};
    slots[1][0].device_blocks = {b_pool0};
    slots[1][1].device_blocks = {b_pool1};
    cache->insert(nullptr, {100, 200}, slots);
    full->unreferenceBlocks(request_blocks);
    EXPECT_TRUE(pool0->isAllocated(a_pool0));
    EXPECT_TRUE(pool0->isAllocated(b_pool0));
    EXPECT_TRUE(pool1->isAllocated(a_pool1));
    EXPECT_TRUE(pool1->isAllocated(b_pool1));

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    ASSERT_EQ(result.group_block_indices.count(1), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{a_pool0, b_pool0}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{a_pool1, b_pool1}));
    cache->releaseMatchedBlocks(result.matched_block_sets);

    EXPECT_EQ(cache->reclaimBlocks(2, Tier::DEVICE), 2);
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
    full->setDevicePools({pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet existing = full->allocateBlocks(Tier::DEVICE, 1);
    GroupBlockSet loser    = full->allocateBlocks(Tier::DEVICE, 1);
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

    full->unreferenceBlocks(loser);
    EXPECT_FALSE(pool->isAllocated(loser_block));
    EXPECT_TRUE(pool->isAllocated(existing_block));

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(existing_block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, InsertMatchReleaseReclaim_RefcountLifecycle) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "refcount_lifecycle_pool");

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});

    GroupBlockSet request_blocks = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(request_blocks.per_node.size(), 1u);
    ASSERT_EQ(request_blocks.per_node[0].size(), 1u);
    const BlockIdxType block = request_blocks.per_node[0][0];
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks - 1);
    EXPECT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = request_blocks.per_node[0];
    cache->insert(nullptr, {100}, slots);
    EXPECT_EQ(pool->refCount(block), 2u);

    full->unreferenceBlocks(request_blocks);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    EXPECT_EQ(result.matched_blocks, 1u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{block}));
    ASSERT_EQ(result.matched_block_sets.size(), 1u);
    EXPECT_EQ(result.matched_block_sets[0].component_group_id, 0);
    EXPECT_EQ(result.matched_block_sets[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[0].per_node, (std::vector<std::vector<BlockIdxType>>{{block}}));
    EXPECT_EQ(pool->refCount(block), 2u);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 0);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 2u);
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    result.matched_block_sets.clear();
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotAllocateHostBlock) {
    auto host_pool           = makeHostPool(256, 4);
    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig ce_cfg;
    ce_cfg.eviction_thread_pool_size = 2;
    ce_cfg.enable_device_cache       = true;
    ce_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(ce_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    ce_cache->waitForPendingTasks();

    // Direct reclaim bypasses demotion/copy, so host capacity is unchanged.
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
    auto find = ce_cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
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
        int reclaimed = ce_cache->reclaimBlocks(1, Tier::DEVICE);
        EXPECT_EQ(reclaimed, 1) << "Reclaim " << i << " should succeed";
        ce_cache->waitForPendingTasks();
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ReusableReclaimDoesNotAllocateHostBlock) {
    auto host_pool = makeHostPool(256, 4);

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig reuse_cfg;
    reuse_cfg.eviction_thread_pool_size = 2;
    reuse_cfg.enable_device_cache       = true;
    reuse_cfg.enable_memory_cache       = true;

    std::unique_ptr<BlockTreeCache> ce_cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(reuse_cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    ce_cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(ce_cache->reclaimBlocks(1, Tier::DEVICE), 1);
    // Synchronous, no wait needed

    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, ReclaimDisabledTierReturnsZero) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    // Device enabled, Host disabled (default)
    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                   std::move(groups),
                                                   std::vector<Component>{},
                                                   BlockTreeCacheConfig{.eviction_thread_pool_size = 2});

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim HOST tier — disabled → returns 0
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::HOST), 0);
    // Reclaim DEVICE tier — enabled → returns 1
    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
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

    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

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
    cache->reclaimBlocks(1, Tier::DEVICE);
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
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);

    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(3));
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        slots[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        slots[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert(nullptr, {100, 200, 300}, slots);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_blocks, 3u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{22}));
    EXPECT_EQ(result.group_block_indices.at(2), (BlockIndicesType{31, 32}));
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    full->setDevicePools({DeviceBlockPoolPtr{}, DeviceBlockPoolPtr{}});

    std::vector<ComponentGroupPtr>             component_groups = {full};
    std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping  = {{0, 0}, {0, 1}};
    std::unique_ptr<BlockTreeCache>            cache =
        std::make_unique<BlockTreeCache>(std::move(tree),
                                         std::move(component_groups),
                                         std::vector<Component>{},
                                         BlockTreeCacheConfig{},
                                         std::shared_ptr<StorageBackend>{},
                                         std::shared_ptr<BroadcastManager>{},
                                         std::vector<DeviceKVCacheGroupPtr>{nullptr, nullptr},
                                         std::move(per_tag_mapping));
    ASSERT_TRUE(cache->init());
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);

    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[0][1].device_blocks = {20};
    slots[1][0].device_blocks = {11};
    slots[1][1].device_blocks = {21};
    cache->insert(nullptr, {100, 200}, slots);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_EQ(result.group_block_indices.at(0), (BlockIndicesType{10, 11}));
    EXPECT_EQ(result.group_block_indices.at(1), (BlockIndicesType{20, 21}));
}

static DeviceKVCacheGroupPtr makeInjectedDeviceGroupMarker(int group_id) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 1;
    return std::make_shared<DeviceFullKVCacheGroup>(LayerIdsType{2}, std::move(spec), DeviceBlockPoolPtr{}, group_id);
}

static std::unique_ptr<BlockTreeCache>
makeCopyProjectionInitializationCache(std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping,
                                      std::vector<DeviceKVCacheGroupPtr>         per_tag_device_groups,
                                      size_t                                     device_pool_count) {
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->component_indices  = {0};
    full->setDevicePools(std::vector<DeviceBlockPoolPtr>(device_pool_count));

    Component component;
    component.component_id                 = 0;
    component.component_group_id           = 0;
    component.memory_block_layer_tag_slots = {{2, "kv", 1}};
    component.device_pool_index            = 0;

    return std::make_unique<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                            std::vector<ComponentGroupPtr>{std::move(full)},
                                            std::vector<Component>{std::move(component)},
                                            BlockTreeCacheConfig{},
                                            std::shared_ptr<StorageBackend>{},
                                            std::shared_ptr<BroadcastManager>{},
                                            std::move(per_tag_device_groups),
                                            std::move(per_tag_mapping));
}

TEST_F(BlockTreeCacheTest, InjectedDeviceRegistryRejectsComponentWithMissingMappingTuple) {
    std::unique_ptr<BlockTreeCache> cache =
        makeCopyProjectionInitializationCache({{/*component_group_id=*/0, /*local_pool_index=*/1}},
                                              {makeInjectedDeviceGroupMarker(/*group_id=*/1)},
                                              /*device_pool_count=*/2);
    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
}

TEST_F(BlockTreeCacheTest, InjectedDeviceRegistryRejectsMatchedNullDeviceGroup) {
    std::unique_ptr<BlockTreeCache> cache = makeCopyProjectionInitializationCache(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/1}},
        {nullptr, makeInjectedDeviceGroupMarker(/*group_id=*/1)},
        /*device_pool_count=*/2);
    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->init());
    EXPECT_FALSE(cache->isInitialized());
}

TEST_F(BlockTreeCacheTest, FullyAbsentDeviceRegistriesPreserveDirectConstructionInitialization) {
    std::unique_ptr<BlockTreeCache> cache = makeCopyProjectionInitializationCache({}, {}, /*device_pool_count=*/1);
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->init());
    EXPECT_TRUE(cache->isInitialized());
}

TEST_F(BlockTreeCacheTest, NonemptyAllNullDeviceRegistryPreservesLegacyDirectConstructionInitialization) {
    std::unique_ptr<BlockTreeCache> cache =
        makeCopyProjectionInitializationCache({{/*component_group_id=*/0, /*local_pool_index=*/0}},
                                              {nullptr},
                                              /*device_pool_count=*/1);
    ASSERT_NE(cache, nullptr);
    EXPECT_TRUE(cache->init());
    EXPECT_TRUE(cache->isInitialized());
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
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);

    std::vector<std::vector<GroupSlot>> slots(4, std::vector<GroupSlot>(2));
    slots[0][0].device_blocks = {10};
    slots[1][0].device_blocks = {11};
    slots[2][0].device_blocks = {12};
    slots[3][0].device_blocks = {13};
    slots[0][1].device_blocks = {20};
    slots[2][1].device_blocks = {22};
    slots[3][1].device_blocks = {23};

    cache->insert(nullptr, {100, 200, 300, 400}, slots);

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_blocks, 1u);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_blocks, 4u);
}

TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto tree                             = std::make_unique<BlockTree>(1);
    auto full                             = std::make_shared<FullComponentGroup>();
    full->component_group_id              = 0;
    std::vector<ComponentGroupPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSlot>> slots(3, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    slots[1][0].device_blocks = {43};
    slots[2][0].device_blocks = {44};
    cache->insert(nullptr, {100, 200, 300}, slots);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim C -> B becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim B -> A becomes DeviceLeaf -> enters heap.
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheTest, ReclaimBlocksDoesNotUpdateHostSlot) {
    auto host_pool = makeHostPool(256, 4);

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->component_indices  = {0};
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    // Create a component with MemoryBlockLayerTagSlot so deviceToHost attempts real copy
    Component comp;
    comp.component_id                 = 0;
    comp.component_group_id           = 0;
    comp.memory_block_layer_tag_slots = {{0, "kv", 128}};
    std::vector<Component> components = {comp};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_memory_cache = true;

    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::move(tree), std::move(groups), std::move(components), std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    EXPECT_EQ(find.matched_node, nullptr);
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
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
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);
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
    const auto& items = result.load_back_ticket->items();
    ASSERT_EQ(items.size(), 6u);
    const auto count_exact_item =
        [&items](int group_id, Tier source_tier, size_t path_index, BlockIdxType source_block, int device_group_id) {
            return std::count_if(items.begin(), items.end(), [&](const PendingLoadBackItem& item) {
                return item.group_id == group_id && item.source_tier == source_tier && item.path_index == path_index
                       && item.source_blocks == std::vector<BlockIdxType>{source_block}
                       && item.device_group_ids == std::vector<int>{device_group_id};
            });
        };
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_item(/*group_id=*/0,
                                   Tier::DEVICE,
                                   path_index,
                                   static_cast<BlockIdxType>(10 + path_index),
                                   /*device_group_id=*/0),
                  1);
    }
    for (size_t path_index = 2; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_item(/*group_id=*/1,
                                   Tier::HOST,
                                   path_index,
                                   static_cast<BlockIdxType>(100 + path_index),
                                   /*device_group_id=*/1),
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
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);
    cache->setEnableLoadBack(true);

    // Insert a node and manually set host data (simulating prior demotion).
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {42};
    cache->insert(nullptr, {100}, slots);

    // Reclaim without host demotion, then manually set up a host-only node.
    // Instead, manually set up a node with host_block but no device_blocks
    cache->reclaimBlocks(1, Tier::DEVICE);
    cache->waitForPendingTasks();

    // After reclaim without host enabled, node is deleted.
    // Let's insert again and manually simulate host-only state
    std::vector<std::vector<GroupSlot>> slots2(1, std::vector<GroupSlot>(1));
    slots2[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots2);

    // Manually set host_block and clear device_blocks to simulate a demoted state.
    auto find = cache->tree()->findNode({200});
    ASSERT_NE(find.matched_node, nullptr);
    find.matched_node->group_slots[0].host_block = 7;
    for (BlockIdxType& device_block_index : find.matched_node->group_slots[0].device_blocks) {
        device_block_index = NULL_BLOCK_IDX;
    }

    // Match should detect load_back
    auto result = cache->match({200});
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);
}

static std::unique_ptr<BlockTreeCache> makeHostOnlyLoadBackCache() {
    std::unique_ptr<BlockTree>          tree = std::make_unique<BlockTree>(1);
    std::shared_ptr<FullComponentGroup> full = std::make_shared<FullComponentGroup>();
    full->component_group_id                 = 0;
    std::vector<ComponentGroupPtr> groups    = {full};

    std::unique_ptr<BlockTreeCache> cache =
        BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree), std::move(groups), std::vector<Component>{});
    cache->setEnableLoadBack(true);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = {55};
    cache->insert(nullptr, {200}, slots);

    BlockTreeFindResult find = cache->tree()->findNode({200});
    RTP_LLM_CHECK(find.matched_node != nullptr);
    find.matched_node->group_slots[0].host_block = 7;
    for (BlockIdxType& device_block_index : find.matched_node->group_slots[0].device_blocks) {
        device_block_index = NULL_BLOCK_IDX;
    }
    return cache;
}

static std::unique_ptr<BlockTreeCache>
makeMappingValidationCache(std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping,
                           size_t                                     device_pool_count,
                           const std::shared_ptr<HostBlockPool>&      host_pool,
                           bool                                       initialize) {
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools(std::vector<DeviceBlockPoolPtr>(device_pool_count));
    full->setHostPool(host_pool);

    BlockTreeCacheConfig config;
    config.enable_memory_cache = host_pool != nullptr;
    config.enable_load_back    = host_pool != nullptr;

    std::vector<ComponentGroupPtr>     groups = {full};
    std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups(per_tag_mapping.size());
    auto                               cache = std::make_unique<BlockTreeCache>(std::make_unique<BlockTree>(1),
                                                  std::move(groups),
                                                  std::vector<Component>{},
                                                  std::move(config),
                                                  nullptr,
                                                  nullptr,
                                                  std::move(per_tag_device_groups),
                                                  std::move(per_tag_mapping));
    if (initialize && !cache->init()) {
        return nullptr;
    }
    return cache;
}

TEST_F(BlockTreeCacheTest, LoadBackGroupMappingUsesLocalPoolIndexOrderAndLeavesTicketUntouched) {
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/2);
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
    const BlockIdxType       source_block = group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);

    const size_t     free_before      = host_pool->freeBlocksNum();
    const size_t     ref_before       = host_pool->refCount(source_block);
    const CacheStats stats_before     = cache->getStats();
    const auto       transfer_before  = slots[0][0].transfer_state;
    const auto       expected_gid_map = std::vector<int>{1, 0};

    EXPECT_TRUE(cache->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/0, expected_gid_map));
    EXPECT_FALSE(cache->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/0, {0, 1}));
    EXPECT_FALSE(cache->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/-1, expected_gid_map));
    EXPECT_FALSE(cache->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/1, expected_gid_map));

    EXPECT_EQ(host_pool->freeBlocksNum(), free_before);
    EXPECT_EQ(host_pool->refCount(source_block), ref_before);
    EXPECT_EQ(cache->getStats().tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(cache->tree()->findNode({100}).matched_node->group_slots[0].transfer_state, transfer_before);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_back_ticket->items().front().device_group_ids, expected_gid_map)
        << "the shared validator must not normalize or rewrite producer-owned metadata";
    EXPECT_EQ(host_pool->refCount(source_block), ref_before + 1);

    result.load_back_ticket.reset();
    EXPECT_EQ(host_pool->refCount(source_block), ref_before);
    EXPECT_EQ(host_pool->freeBlocksNum(), free_before);
}

TEST_F(BlockTreeCacheTest, LoadBackGroupMappingValidatorRejectsOutOfRangeDuplicateAndHoleMetadata) {
    const auto make_uninitialized = [](std::vector<BlockTreeCache::PerTagMapping> mapping, size_t pool_count) {
        return makeMappingValidationCache(std::move(mapping), pool_count, nullptr, /*initialize=*/false);
    };

    std::unique_ptr<BlockTreeCache> out_of_range = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/2}},
        /*pool_count=*/2);
    ASSERT_NE(out_of_range, nullptr);
    EXPECT_FALSE(out_of_range->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/0, {0, 1}));

    std::unique_ptr<BlockTreeCache> duplicate = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/0}},
        /*pool_count=*/2);
    ASSERT_NE(duplicate, nullptr);
    EXPECT_FALSE(duplicate->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/0, {0, 1}));

    std::unique_ptr<BlockTreeCache> hole = make_uninitialized(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/2}},
        /*pool_count=*/3);
    ASSERT_NE(hole, nullptr);
    EXPECT_FALSE(hole->validateDeviceGroupIdsForComponentGroup(/*component_group_id=*/0, {0, 1}));
}

TEST_F(BlockTreeCacheTest, InvalidProducerMappingCreatesNoPendingItemOrSourceProtection) {
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/2);
    ASSERT_NE(host_pool, nullptr);

    std::unique_ptr<BlockTreeCache> cache = makeMappingValidationCache(
        {{/*component_group_id=*/0, /*local_pool_index=*/0}, {/*component_group_id=*/0, /*local_pool_index=*/0}},
        /*device_pool_count=*/2,
        host_pool,
        /*initialize=*/true);
    ASSERT_NE(cache, nullptr);
    auto copy_engine = std::make_shared<ScriptedCopyEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, copy_engine);

    const ComponentGroupPtr& group        = cache->componentGroups().front();
    const BlockIdxType       source_block = group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);

    const size_t         source_ref_before = host_pool->refCount(source_block);
    BlockTreeMatchResult result            = cache->match({100});

    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.load_back_blocks, 0u);
    EXPECT_TRUE(result.load_back_ticket == nullptr || result.load_back_ticket->empty());
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before);
    EXPECT_EQ(copy_engine->submitCount(), 0u);
    EXPECT_EQ(cache->tree()->findNode({100}).matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
}

TEST_F(BlockTreeCacheTest, LoadBackWholeBatchMappingPreflightIsAtomicForLaterInvalidItem) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             device_pool = makeDevicePool({{1, 0}}, 4, "load_back_mapping_preflight");
    std::shared_ptr<HostBlockPool> host_pool   = makeHostPool(1, 4);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({device_pool});
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    auto copy_engine = std::make_shared<ScriptedCopyEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, copy_engine);

    const BlockIdxType first_source  = full->allocateSingleBlock(Tier::HOST);
    const BlockIdxType second_source = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(first_source, NULL_BLOCK_IDX);
    ASSERT_NE(second_source, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> slots(2, std::vector<GroupSlot>(1));
    slots[0][0].host_block = first_source;
    slots[1][0].host_block = second_source;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100, 200}, slots).leaf, nullptr);

    BlockTreeMatchResult result = cache->match({100, 200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->items().size(), 2u);
    ASSERT_EQ(host_pool->refCount(first_source), 2u);
    ASSERT_EQ(host_pool->refCount(second_source), 2u);

    const BlockIdxType first_target  = poolMalloc(*device_pool);
    const BlockIdxType second_target = poolMalloc(*device_pool);
    ASSERT_NE(first_target, NULL_BLOCK_IDX);
    ASSERT_NE(second_target, NULL_BLOCK_IDX);
    device_pool->incRef(first_target);
    device_pool->incRef(second_target);
    result.load_back_ticket->items()[0].target_device_blocks = {first_target};
    result.load_back_ticket->items()[1].target_device_blocks = {second_target};
    result.load_back_ticket->items()[1].device_group_ids     = {1};  // invalid for component group 0

    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    EXPECT_EQ(copy_engine->submitCount(), 0u);
    EXPECT_EQ(host_pool->refCount(first_source), 1u);
    EXPECT_EQ(host_pool->refCount(second_source), 1u);
    EXPECT_EQ(device_pool->refCount(first_target), 1u);
    EXPECT_EQ(device_pool->refCount(second_target), 1u);

    BlockTreeFindResult find = cache->tree()->findNode({100, 200});
    ASSERT_EQ(find.path.size(), 2u);
    EXPECT_EQ(find.path[0]->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(find.path[1]->group_slots[0].transfer_state, SlotTransferState::IDLE);

    result.load_back_ticket.reset();
    EXPECT_EQ(host_pool->refCount(first_source), 1u) << "committed ticket cleanup must execute exactly once";
    EXPECT_EQ(host_pool->refCount(second_source), 1u) << "committed ticket cleanup must execute exactly once";
    device_pool->decRef(first_target);
    device_pool->decRef(second_target);
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
    full->setDevicePools({device_pool});
    full->setHostPool(host_pool);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::unique_ptr<BlockTreeCache> cache = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    auto copy_engine = std::make_shared<ScriptedCopyEngine>(cache->componentGroups(), cache->components());
    BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, copy_engine);

    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = source_block;
    ASSERT_NE(cache->tree()->insertNode(nullptr, {100}, slots).leaf, nullptr);
    const size_t source_ref_before = host_pool->refCount(source_block);

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_back_ticket->items().front().device_group_ids, (std::vector<int>{0}));
    EXPECT_EQ(host_pool->refCount(source_block), source_ref_before + 1);

    const BlockIdxType request_target = poolMalloc(*device_pool);
    ASSERT_NE(request_target, NULL_BLOCK_IDX);
    device_pool->incRef(request_target);
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
    EXPECT_EQ(copy_engine->submitCount(), 0u);
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
    device_pool->decRef(request_target);
}

TEST_F(BlockTreeCacheTest, LoadBackTargetValidationFailureRollsBackAllTreeHolders) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr first_device_pool     = makeDevicePool({{1, 0}}, 1, "load_back_atomic_first");
    DeviceBlockPoolPtr exhausted_device_pool = makeDevicePool({{1, 0}}, 1, "load_back_atomic_exhausted");

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);

    std::shared_ptr<FullComponentGroup> first_group = std::make_shared<FullComponentGroup>();
    first_group->component_group_id                 = 0;
    first_group->setDevicePools({first_device_pool});
    first_group->setHostPool(first_host_pool);
    std::shared_ptr<FullComponentGroup> second_group = std::make_shared<FullComponentGroup>();
    second_group->component_group_id                 = 1;
    second_group->setDevicePools({exhausted_device_pool});
    second_group->setHostPool(second_host_pool);

    GroupBlockSet exhausted_holder = second_group->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(exhausted_holder.per_node.size(), 1u);
    ASSERT_EQ(exhausted_holder.per_node[0].size(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);

    const BlockIdxType first_host_block  = first_group->allocateSingleBlock(Tier::HOST);
    const BlockIdxType second_host_block = second_group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    BlockTreeCacheConfig config;
    config.enable_memory_cache                       = true;
    config.enable_load_back                          = true;
    std::vector<ComponentGroupPtr>  component_groups = {first_group, second_group};
    std::unique_ptr<BlockTreeCache> cache            = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(2), std::move(component_groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].host_block = first_host_block;
    slots[0][1].host_block = second_host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    BlockTreeMatchResult match_result = cache->match({100});
    ASSERT_NE(match_result.load_back_ticket, nullptr);
    ASSERT_EQ(match_result.load_back_ticket->items().size(), 2u);
    EXPECT_EQ(first_host_pool->refCount(first_host_block), 2u);
    EXPECT_EQ(second_host_pool->refCount(second_host_block), 2u);

    GroupBlockSet first_target_holder = first_group->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(first_target_holder.per_node.size(), 1u);
    ASSERT_EQ(first_target_holder.per_node[0].size(), 1u);
    const BlockIdxType first_target = first_target_holder.per_node[0][0];
    for (PendingLoadBackItem& item : match_result.load_back_ticket->items()) {
        if (item.group_id == 0) {
            item.target_device_blocks = {first_target};
        }
    }

    std::shared_ptr<AsyncContext> context = match_result.load_back_ticket->commit();
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(first_device_pool->refCount(first_target), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(first_host_pool->refCount(first_host_block), 1u);
    EXPECT_EQ(second_host_pool->refCount(second_host_block), 1u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 1u);

    EXPECT_EQ(cache->reclaimBlocks(2, Tier::HOST), 2);
    cache->waitForPendingTasks();
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 2u);

    first_group->unreferenceBlocks(first_target_holder);
    second_group->unreferenceBlocks(exhausted_holder);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 1u);
}

// Deferred load_back: match() plans (references the source blocks) but does NOT execute
// load_back. The result carries a LoadBackTicket; the allocator binds request-owned
// device targets before committing it. Dropping it uncommitted aborts (unreferences
// the source) without allocating or copying anything.

// Not committing the ticket: no device block is allocated and no async copy is submitted;
// the ticket destructor aborts safely.
TEST_F(BlockTreeCacheTest, LoadBackTicketAbortSkipsLoadBack) {
    auto                               cache = makeHostOnlyLoadBackCache();
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);

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
    std::unique_ptr<BlockTreeCache>    cache = makeHostOnlyLoadBackCache();
    DeliberateNonPhysicalTopologyGuard synthetic_topology(*cache);
    DeviceBlockPoolPtr                 device_pool = makeDevicePool({{1, 0}}, 1, "load_back_ticket_commit");
    cache->component_groups_[0]->setDevicePools({device_pool});

    BlockTreeMatchResult result = cache->match({200});
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), 1u);
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.host_load_back_blocks, 1u);
    EXPECT_EQ(result.load_back_blocks, 1u);

    const BlockIdxType request_target = poolMalloc(*device_pool);
    ASSERT_NE(request_target, NULL_BLOCK_IDX);
    device_pool->incRef(request_target);
    ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
    result.load_back_ticket->items()[0].target_device_blocks = {request_target};

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    EXPECT_NE(context, nullptr);

    cache->releaseMatchedBlocks(result.matched_block_sets);
    cache->waitForPendingTasks();
    device_pool->decRef(request_target);
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
    full->setDevicePools(device_pools);
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

    GroupBlockSet root_device_holds = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(root_device_holds.per_node.size(), 1u);
    ASSERT_EQ(root_device_holds.per_node[0].size(), 3u);
    const BlockIdxType device_block_0 = root_device_holds.per_node[0][0];
    const BlockIdxType device_hole    = root_device_holds.per_node[0][1];
    const BlockIdxType device_block_2 = root_device_holds.per_node[0][2];
    ASSERT_NE(device_block_0, NULL_BLOCK_IDX);
    ASSERT_NE(device_hole, NULL_BLOCK_IDX);
    ASSERT_NE(device_block_2, NULL_BLOCK_IDX);

    GroupBlockSet hole_holder{0, Tier::DEVICE, {{NULL_BLOCK_IDX, device_hole, NULL_BLOCK_IDX}}};
    full->unreferenceBlocks(hole_holder);
    root_device_holds.per_node[0][1]                    = NULL_BLOCK_IDX;
    cache->tree()->root()->group_slots[0].device_blocks = root_device_holds.per_node[0];

    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST);
    const BlockIdxType disk_block = full->allocateSingleBlock(Tier::DISK);
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
    full->setDevicePools({device_pool});
    std::vector<ComponentGroupPtr> groups = {full};
    auto                           cache  = BlockTreeCacheTestUtil::makeBlockTreeCache(
        std::make_unique<BlockTree>(1), std::move(groups), std::vector<Component>{});
    ASSERT_NE(cache, nullptr);

    GroupBlockSet tree_holder = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(tree_holder.per_node.size(), 1u);
    ASSERT_EQ(tree_holder.per_node[0].size(), 1u);
    const BlockIdxType block = tree_holder.per_node[0][0];
    ASSERT_NE(block, NULL_BLOCK_IDX);
    GroupBlockSet external_holder = tree_holder;
    full->referenceBlocks(external_holder);
    EXPECT_EQ(device_pool->refCount(block), 2u);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].device_blocks = tree_holder.per_node[0];
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    cache.reset();

    EXPECT_TRUE(device_pool->isAllocated(block));
    EXPECT_EQ(device_pool->refCount(block), 1u);
    EXPECT_EQ(device_pool->freeBlocksNum(), free_before - 1);

    full->unreferenceBlocks(external_holder);
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
    full->setDevicePools({device_pool});
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
    auto copy_engine =
        std::make_shared<ScriptedCopyEngine>(std::vector<ComponentGroupPtr>{full}, std::vector<Component>{});
    BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, copy_engine);

    GroupBlockSet device_holder = full->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(device_holder.per_node.size(), 1u);
    ASSERT_EQ(device_holder.per_node[0].size(), 1u);
    const BlockIdxType device_block = device_holder.per_node[0][0];
    const BlockIdxType host_block   = full->allocateSingleBlock(Tier::HOST);
    const BlockIdxType disk_block   = full->allocateSingleBlock(Tier::DISK);
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

    EXPECT_EQ(cache->reclaimBlocks(1, Tier::DEVICE), 1);
    cache->waitForPendingTasks();
    EXPECT_EQ(copy_engine->submitCount(), 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_FALSE(device_pool->isAllocated(device_block));
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);

    cache.reset();

    EXPECT_EQ(copy_engine->submitCount(), 0u);
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

        const BlockIdxType source_block = full->allocateSingleBlock(source_tier);
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
    auto   registry    = std::make_shared<LoadBackTicketRegistry>(
        [](const std::vector<PendingLoadBackItem>&) { return std::shared_ptr<AsyncContext>{}; },
        [&](const std::vector<PendingLoadBackItem>& items) {
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
        [&](const std::vector<PendingLoadBackItem>&) {
            ++commit_calls;
            commit_callback.enterAndWait();
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const std::vector<PendingLoadBackItem>& items) {
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
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    GroupBlockSet source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const std::vector<PendingLoadBackItem>&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const std::vector<PendingLoadBackItem>& items) {
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            full->unreferenceBlocks(source_protection);
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
    full->releaseSingleBlock(Tier::HOST, source_block);
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
        [&](const std::vector<PendingLoadBackItem>&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const std::vector<PendingLoadBackItem>& items) {
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
    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    GroupBlockSet source_protection{0, Tier::HOST, {{source_block}}};
    full->referenceBlocks(source_protection);
    EXPECT_EQ(host_pool->refCount(source_block), 2u);

    CallbackBarrier  abort_callback;
    ThreadCompletion shutdown_detached_abort;
    ThreadCompletion shutdown;
    std::atomic<int> commit_calls{0};
    std::atomic<int> abort_calls{0};
    auto             registry = std::make_shared<LoadBackTicketRegistry>(
        [&](const std::vector<PendingLoadBackItem>&) {
            ++commit_calls;
            return std::shared_ptr<AsyncContext>{};
        },
        [&](const std::vector<PendingLoadBackItem>& items) {
            ++abort_calls;
            EXPECT_EQ(items.size(), 1u);
            if (items.size() == 1u && items[0].group_id == 0) {
                full->unreferenceBlocks(source_protection);
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
    full->releaseSingleBlock(Tier::HOST, source_block);
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
