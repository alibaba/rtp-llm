#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"

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

class PausableCopyEngine: public CopyEngine {
public:
    PausableCopyEngine(const std::vector<ComponentGroupPtr>& groups,
                       const std::vector<Component>&         components,
                       CopyStatus                            result,
                       bool                                  pause_enabled = true):
        CopyEngine(groups, std::make_shared<const std::vector<Component>>(components)),
        pause_enabled_(pause_enabled),
        result_(result) {}

    TransferHandle submit(const TransferDescriptor& descriptor) override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!pause_enabled_) {
                lock.unlock();
                return CopyEngine::submit(descriptor);
            }
            ++submit_count_;
            descriptors_.push_back(descriptor);
            entered_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return released_; });
        }
        if (result_ != CopyStatus::OK) {
            return TransferHandle::completed(result_);
        }
        return CopyEngine::submit(descriptor);
    }

    void enablePause() {
        std::lock_guard<std::mutex> lock(mutex_);
        ASSERT_FALSE(pause_enabled_);
        ASSERT_FALSE(entered_);
        ASSERT_FALSE(released_);
        ASSERT_EQ(submit_count_, 0u);
        ASSERT_TRUE(descriptors_.empty());
        pause_enabled_ = true;
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

    size_t submitCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return submit_count_;
    }

    std::vector<TransferDescriptor> descriptors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptors_;
    }

private:
    mutable std::mutex              mutex_;
    std::condition_variable         cv_;
    bool                            pause_enabled_{true};
    bool                            entered_{false};
    bool                            released_{false};
    size_t                          submit_count_{0};
    std::vector<TransferDescriptor> descriptors_;
    CopyStatus                      result_{CopyStatus::OK};
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

class BlockTreeCacheIntegrationTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full_group                       = std::make_shared<FullComponentGroup>();
        full_group->component_group_id        = 0;
        std::vector<ComponentGroupPtr> groups = {full_group};
        cache_ = makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::vector<Component>{});
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

constexpr size_t kPathLength = 4;
constexpr size_t kPoolSize   = 16;

DeviceBlockPoolPtr devicePoolForTag(const FullSWAEnvironment& environment, const std::string& tag) {
    for (const ComponentGroupPtr& group : environment.groups) {
        for (size_t index = 0; index < group->tags().size(); ++index) {
            if (group->tags()[index] == tag) {
                return group->devicePools()[index];
            }
        }
    }
    return nullptr;
}

enum class DemotionFailureStage {
    D2H,
    H2DISK,
};

std::string tierParamName(const ::testing::TestParamInfo<Tier>& info) {
    return info.param == Tier::HOST ? "Host" : "Disk";
}

std::string demotionFailureParamName(const ::testing::TestParamInfo<DemotionFailureStage>& info) {
    return info.param == DemotionFailureStage::D2H ? "D2H" : "H2Disk";
}

void demoteTo(FullSWAEnvironment& environment, Tier target_tier) {
    environment.demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment.allSlotsAtTier(Tier::HOST));
    if (target_tier == Tier::DISK) {
        environment.demoteAll(Tier::HOST);
        ASSERT_TRUE(environment.allSlotsAtTier(Tier::DISK));
    }
}

void expectUnpublishedResult(const BlockTreeMatchResult& result) {
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.async_context, nullptr);
}

void expectAggregatedReadyResult(const BlockTreeMatchResult& result, size_t full_blocks, size_t swa_blocks) {
    ASSERT_EQ(result.group_block_indices.size(), 3u);
    ASSERT_EQ(result.group_block_indices.count("tag_0"), 1u);
    ASSERT_EQ(result.group_block_indices.count("tag_1"), 1u);
    ASSERT_EQ(result.group_block_indices.count("tag_2"), 1u);
    EXPECT_EQ(result.group_block_indices.at("tag_0").size(), full_blocks);
    EXPECT_EQ(result.group_block_indices.at("tag_1").size(), full_blocks);
    EXPECT_EQ(result.group_block_indices.at("tag_2").size(), swa_blocks);

    ASSERT_EQ(result.matched_block_sets.size(), 2u);
    EXPECT_EQ(result.matched_block_sets[0].component_group_id, 0);
    EXPECT_EQ(result.matched_block_sets[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[0].per_node.size(), full_blocks);
    EXPECT_EQ(result.matched_block_sets[1].component_group_id, 1);
    EXPECT_EQ(result.matched_block_sets[1].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[1].per_node.size(), swa_blocks);
}

void expectPlanningSourceRefCounts(const FullSWAEnvironment& environment, Tier tier) {
    for (size_t path_index = 0; path_index < environment.keys.size(); ++path_index) {
        const std::vector<GroupSlot> slots = environment.slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        for (size_t group_id = 0; group_id < slots.size(); ++group_id) {
            const BlockIdxType block = tier == Tier::HOST ? slots[group_id].host_block : slots[group_id].disk_slot;
            const IBlockPool&  pool  = tier == Tier::HOST ?
                                           static_cast<const IBlockPool&>(*environment.host_pools[group_id]) :
                                           static_cast<const IBlockPool&>(*environment.disk_pools[group_id]);
            const bool         in_swa_window = group_id == 0 || path_index + 2 >= environment.keys.size();
            const uint32_t     expected      = in_swa_window ? 2u : 1u;
            EXPECT_EQ(pool.refCount(block), expected);
        }
    }
}

size_t ticketItemCountForGroup(const std::shared_ptr<LoadBackTicket>& ticket, int group_id) {
    if (ticket == nullptr) {
        return 0;
    }
    return static_cast<size_t>(
        std::count_if(ticket->items().begin(), ticket->items().end(), [group_id](const PendingLoadBackItem& item) {
            return item.group_id == group_id;
        }));
}

std::vector<BlockIdxType> allocatedBlocksSnapshot(const IBlockPool& pool) {
    std::vector<BlockIdxType> blocks;
    for (BlockIdxType block = 1; block <= static_cast<BlockIdxType>(pool.totalBlocksNum()); ++block) {
        if (pool.isAllocated(block)) {
            blocks.push_back(block);
        }
    }
    return blocks;
}

void runSingleMaintenance(FullSWAEnvironment& environment, Tier tier, double ratio) {
    environment.cache->setTierWatermark(tier, ratio, 0);
    environment.runMaintenance();
    environment.cache->setTierWatermark(tier, 0.0, 0);
}

BlockTreeMatchResult makePartialReadyDeviceTicket(FullSWAEnvironment& environment) {
    environment.insertRequestPath();
    BlockTreeMatchResult prefix_hold = environment.cache->match({environment.keys[0], environment.keys[1]});
    EXPECT_EQ(prefix_hold.matched_blocks, 2u);
    environment.releaseRequestRefsForGroup(1);
    runSingleMaintenance(environment, Tier::DEVICE, 0.125);
    environment.releaseMatch(prefix_hold);

    environment.scripted_copy_engine->clear();
    BlockTreeMatchResult result = environment.cache->match(environment.keys);
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_NE(result.load_back_ticket, nullptr);
    if (result.load_back_ticket != nullptr) {
        EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 2u);
        EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    }
    return result;
}

class OneShotWatermarkTestPeer {
public:
    static void runDevicePass(BlockTreeCache& cache, double ratio) {
        ASSERT_EQ(cache.pending_tasks_.load(), 0);
        {
            std::lock_guard<std::mutex> lock(cache.mutex_);
            cache.setTierWatermark(Tier::HOST, 0.0, 0);
            cache.setTierWatermark(Tier::DISK, 0.0, 0);
            cache.setTierWatermark(Tier::DEVICE, ratio, 0);
            cache.checkWatermark();
            cache.setTierWatermark(Tier::DEVICE, 0.0, 0);
        }
        cache.waitForPendingTasks();
        EXPECT_EQ(cache.pending_tasks_.load(), 0);
    }
};

TEST_F(BlockTreeCacheIntegrationTest, WatermarkDemotionCopiesHostBlockToDisk) {
    auto host_pool = makeHostPool(256, 8);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    full->setDevicePools({makeDevicePool({{256, 0}}, 8, "watermark_host_to_disk")}, {"watermark_kv"});
    std::vector<Component> layout_components = {copy_engine_test::makeSchemaComponent(0, 0, "watermark_kv", {256})};
    ASSERT_TRUE(full->finalizeLayout({0}, layout_components));
    auto host_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = false;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;

    auto cache =
        makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::move(layout_components), std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    auto before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    const auto& slot = find.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_TRUE(slot.has_value(Tier::DISK));
    EXPECT_NE(slot.disk_slot, NULL_BLOCK_IDX);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_TRUE(disk_pool->isAllocated(slot.disk_slot));
    EXPECT_EQ(disk_pool->refCount(slot.disk_slot), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheIntegrationTest, OneShotCascadeFailureRollsBackSWAAndRetriesOnce) {
    ASSERT_TRUE(cudaAvailable()) << "C002-T05 requires CUDA";
    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    ASSERT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));

    const std::vector<GroupSlot> initial_slots = environment->slotsForPathNode(0);
    ASSERT_EQ(initial_slots.size(), 2u);
    ASSERT_EQ(initial_slots[0].device_blocks.size(), 2u);
    ASSERT_EQ(initial_slots[1].device_blocks.size(), 1u);
    const std::vector<BlockIdxType> full_sources = initial_slots[0].device_blocks;
    const BlockIdxType              swa_source   = initial_slots[1].device_blocks[0];

    environment->scripted_copy_engine->clear();
    environment->scripted_copy_engine->enqueue(CopyStatus::OK);
    environment->scripted_copy_engine->enqueue(CopyStatus::DEVICE_IO_ERROR);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> first_descriptors = environment->scripted_copy_engine->descriptors();
    ASSERT_EQ(first_descriptors.size(), 2u);
    EXPECT_EQ(first_descriptors[0].component_group_id, 0);
    EXPECT_EQ(first_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(first_descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(first_descriptors[0].device_blocks, full_sources);
    EXPECT_EQ(first_descriptors[1].component_group_id, 1);
    EXPECT_EQ(first_descriptors[1].source_tier, Tier::DEVICE);
    EXPECT_EQ(first_descriptors[1].target_tier, Tier::HOST);
    EXPECT_EQ(first_descriptors[1].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 2u);

    const std::vector<GroupSlot> after_failure = environment->slotsForPathNode(0);
    ASSERT_EQ(after_failure.size(), 2u);
    EXPECT_EQ(after_failure[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(after_failure[0].has_value(Tier::DEVICE));
    EXPECT_TRUE(after_failure[0].has_value(Tier::HOST));
    EXPECT_EQ(environment->host_pools[0]->refCount(after_failure[0].host_block), 1u);
    EXPECT_EQ(after_failure[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(after_failure[1].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_FALSE(after_failure[1].has_value(Tier::HOST));
    EXPECT_EQ(environment->device_pools[2]->refCount(swa_source), 1u);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), 16u);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment->device_pools[2]->activeTreeCachedBlocksNum(), 0u);
    environment->expectPoolFreeCounts({16, 16, 15}, {15, 16}, {16, 16});
    environment->expectPayloads();

    environment->scripted_copy_engine->clear();
    environment->scripted_copy_engine->enqueue(CopyStatus::OK);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> retry_descriptors = environment->scripted_copy_engine->descriptors();
    ASSERT_EQ(retry_descriptors.size(), 1u);
    EXPECT_EQ(retry_descriptors[0].component_group_id, 1);
    EXPECT_EQ(retry_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(retry_descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(retry_descriptors[0].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 1u);

    const std::vector<GroupSlot> after_retry = environment->slotsForPathNode(0);
    ASSERT_EQ(after_retry.size(), 2u);
    EXPECT_EQ(after_retry[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_TRUE(after_retry[0].has_value(Tier::HOST));
    EXPECT_EQ(after_retry[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(after_retry[1].has_value(Tier::DEVICE));
    EXPECT_TRUE(after_retry[1].has_value(Tier::HOST));
    EXPECT_FALSE(environment->device_pools[2]->isAllocated(swa_source));
    EXPECT_EQ(environment->host_pools[1]->refCount(after_retry[1].host_block), 1u);
    environment->expectPoolFreeCounts({16, 16, 16}, {15, 15}, {16, 16});
    environment->expectPayloads();

    environment->scripted_copy_engine->clear();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, LoadBackDeviceAllocationFailureRollsBackAllItems) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr first_device_pool     = makeDevicePool({{1, 0}}, 1, "load_back_atomic_first");
    DeviceBlockPoolPtr exhausted_device_pool = makeDevicePool({{1, 0}}, 1, "load_back_atomic_exhausted");

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);

    std::vector<Component> components = {
        copy_engine_test::makeSchemaComponent(0, 0, "tag_0", {1}),
        copy_engine_test::makeSchemaComponent(1, 1, "tag_1", {1}),
    };

    std::shared_ptr<FullComponentGroup> first_group = std::make_shared<FullComponentGroup>();
    first_group->component_group_id                 = 0;
    first_group->setDevicePools({first_device_pool}, {"tag_0"});
    first_group->setHostPool(first_host_pool);
    ASSERT_TRUE(first_group->finalizeLayout({0}, components));
    std::shared_ptr<FullComponentGroup> second_group = std::make_shared<FullComponentGroup>();
    second_group->component_group_id                 = 1;
    second_group->setDevicePools({exhausted_device_pool}, {"tag_1"});
    second_group->setHostPool(second_host_pool);
    ASSERT_TRUE(second_group->finalizeLayout({1}, components));

    GroupBlockSet exhausted_holder = second_group->allocateBlocks(Tier::DEVICE, 1);
    ASSERT_EQ(exhausted_holder.per_node.size(), 1u);
    ASSERT_EQ(exhausted_holder.per_node[0].size(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);

    const BlockIdxType first_host_block  = first_group->allocateSingleBlock(Tier::HOST);
    const BlockIdxType second_host_block = second_group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    std::vector<ComponentGroupPtr> component_groups = {first_group, second_group};
    BlockTreeCacheConfig           config;
    config.enable_memory_cache            = true;
    config.enable_load_back               = true;
    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(
        std::make_unique<BlockTree>(2), std::move(component_groups), std::move(components), std::move(config));
    ASSERT_NE(cache, nullptr);

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(2));
    slots[0][0].host_block = first_host_block;
    slots[0][1].host_block = second_host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    BlockTreeFindResult find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 2u);

    BlockTreeMatchResult match_result = cache->match({100});
    ASSERT_NE(match_result.load_back_ticket, nullptr);
    ASSERT_EQ(match_result.load_back_ticket->items().size(), 2u);
    EXPECT_EQ(first_host_pool->refCount(first_host_block), 2u);
    EXPECT_EQ(second_host_pool->refCount(second_host_block), 2u);

    std::shared_ptr<AsyncContext> context = match_result.load_back_ticket->commit();
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(first_host_pool->refCount(first_host_block), 1u);
    EXPECT_EQ(second_host_pool->refCount(second_host_block), 1u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_slots[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(find.matched_node->group_slots[0].host_block, first_host_block);
    EXPECT_EQ(find.matched_node->group_slots[1].host_block, second_host_block);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 2u);

    EXPECT_EQ(cache->reclaimBlocks(2, Tier::HOST), 2);
    cache->waitForPendingTasks();
    const CacheStats stats = cache->getStats();
    EXPECT_EQ(stats.device_heap_total_size, 0u);
    EXPECT_EQ(stats.host_heap_total_size, 0u);
    EXPECT_EQ(stats.disk_heap_total_size, 0u);
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 2u);
    EXPECT_FALSE(first_host_pool->isAllocated(first_host_block));
    EXPECT_FALSE(second_host_pool->isAllocated(second_host_block));

    second_group->unreferenceBlocks(exhausted_holder);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeCacheIntegrationTest, UncommittedLoadBackTicketReleasesSourceReferences) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::unique_ptr<FullSWAEnvironment> environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    expectPlanningSourceRefCounts(*environment, Tier::HOST);

    result.load_back_ticket.reset();
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const std::vector<GroupSlot> slots = environment->slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        for (size_t group_id = 0; group_id < slots.size(); ++group_id) {
            EXPECT_EQ(environment->host_pools[group_id]->refCount(slots[group_id].host_block), 1u);
        }
    }

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

// C006-T03: shutdown waits for committed copy settlement before draining every tree hold.
TEST_F(BlockTreeCacheIntegrationTest, CacheShutdownWaitsForCommittedLoadBackSettlement) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    for (CopyStatus copy_result : {CopyStatus::OK, CopyStatus::DEVICE_IO_ERROR}) {
        SCOPED_TRACE(copy_result == CopyStatus::OK ? "copy_success" : "copy_failure");

        constexpr size_t  kBlockBytes = 16;
        constexpr size_t  kPoolSize   = 2;
        const std::string pool_name =
            copy_result == CopyStatus::OK ? "shutdown_load_back_success" : "shutdown_load_back_failure";
        auto         device_pool        = makeDevicePool({{kBlockBytes, 0}}, kPoolSize, pool_name);
        auto         host_pool          = makeHostPool(kBlockBytes, kPoolSize);
        auto         disk_pool          = makeDiskPool(kBlockBytes, kPoolSize, std::make_unique<MemoryDiskBlockIO>());
        const size_t device_free_before = device_pool->freeBlocksNum();
        const size_t host_free_before   = host_pool->freeBlocksNum();
        const size_t disk_free_before   = disk_pool->freeBlocksNum();

        auto full                = std::make_shared<FullComponentGroup>();
        full->component_group_id = 0;
        full->setDevicePools({device_pool}, {"shutdown_kv"});
        full->setHostPool(host_pool);
        full->setDiskPool(disk_pool);

        std::vector<Component> components = {
            copy_engine_test::makeSchemaComponent(0, 0, "shutdown_kv", {kBlockBytes}),
        };
        ASSERT_TRUE(full->finalizeLayout({0}, components));
        std::vector<ComponentGroupPtr> groups = {full};
        BlockTreeCacheConfig           config;
        config.enable_device_cache = true;
        config.enable_memory_cache = true;
        config.enable_disk_cache   = true;
        config.enable_load_back    = true;
        auto cache =
            makeBlockTreeCacheForTest(std::make_unique<BlockTree>(1), std::move(groups), components, std::move(config));
        ASSERT_NE(cache, nullptr);

        auto pausable_copy_engine =
            std::make_shared<PausableCopyEngine>(std::vector<ComponentGroupPtr>{full}, components, copy_result);
        BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, pausable_copy_engine);

        const BlockIdxType source_block = full->allocateSingleBlock(Tier::DISK);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        std::vector<std::vector<GroupSlot>> source_slots(1, std::vector<GroupSlot>(1));
        source_slots[0][0].disk_slot = source_block;
        ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, source_slots));

        BlockTreeMatchResult result = cache->match({100});
        ASSERT_NE(result.load_back_ticket, nullptr);
        ASSERT_EQ(result.load_back_ticket->items().size(), 1u);
        EXPECT_EQ(result.load_back_ticket->items()[0].source_tier, Tier::DISK);
        EXPECT_EQ(result.load_back_ticket->items()[0].source_blocks, (std::vector<BlockIdxType>{source_block}));
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);

        const BlockIdList request_targets = device_pool->malloc(1).value();
        ASSERT_EQ(request_targets.size(), 1u);
        device_pool->incRef(request_targets);
        const BlockIdxType target_block = request_targets.front();
        EXPECT_EQ(device_pool->refCount(target_block), 1u);
        result.load_back_ticket->items()[0].target_device_blocks = {target_block};
        std::shared_ptr<LoadBackTicket> outliving_ticket         = std::move(result.load_back_ticket);

        std::shared_ptr<AsyncContext> context = outliving_ticket->commit();
        ASSERT_NE(context, nullptr);
        pausable_copy_engine->waitUntilEntered();
        EXPECT_EQ(pausable_copy_engine->submitCount(), 1u);
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);
        // The request owns one reference and the committed pending copy protects the target with another.
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);
        EXPECT_EQ(outliving_ticket->commit(), nullptr);

        ThreadCompletion destruction;
        LoadBackShutdownTestPeer::setPendingTaskWaitObserver(*cache, [&destruction] { destruction.markEntered(); });
        std::thread destroy_thread([cache = std::move(cache), &destruction]() mutable {
            cache.reset();
            destruction.markFinished();
        });
        destruction.waitUntilEntered();
        EXPECT_FALSE(destruction.finished());
        const std::vector<BlockIdxType> device_blocks_after_wait = allocatedBlocksSnapshot(*device_pool);
        const std::vector<BlockIdxType> host_blocks_after_wait   = allocatedBlocksSnapshot(*host_pool);
        const std::vector<BlockIdxType> disk_blocks_after_wait   = allocatedBlocksSnapshot(*disk_pool);
        ASSERT_EQ(device_blocks_after_wait, (std::vector<BlockIdxType>{target_block}));
        ASSERT_EQ(host_blocks_after_wait.size(), 1u);
        const BlockIdxType staging_block = host_blocks_after_wait.front();
        ASSERT_EQ(disk_blocks_after_wait, (std::vector<BlockIdxType>{source_block}));
        const std::vector<TransferDescriptor> descriptors_after_wait = pausable_copy_engine->descriptors();
        ASSERT_EQ(descriptors_after_wait.size(), 1u);
        EXPECT_EQ(descriptors_after_wait[0].component_group_id, 0);
        EXPECT_EQ(descriptors_after_wait[0].source_tier, Tier::DISK);
        EXPECT_EQ(descriptors_after_wait[0].target_tier, Tier::HOST);
        EXPECT_EQ(descriptors_after_wait[0].disk_block, source_block);
        EXPECT_EQ(descriptors_after_wait[0].host_block, staging_block);
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        EXPECT_TRUE(host_pool->isAllocated(staging_block));
        EXPECT_TRUE(disk_pool->isAllocated(source_block));
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(host_pool->refCount(staging_block), 1u);
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);

        pausable_copy_engine->release();
        destroy_thread.join();
        EXPECT_TRUE(destruction.finished());
        context->waitDone();
        EXPECT_TRUE(context->done());
        EXPECT_EQ(context->success(), copy_result == CopyStatus::OK);
        EXPECT_EQ(pausable_copy_engine->submitCount(), copy_result == CopyStatus::OK ? 2u : 1u);
        EXPECT_FALSE(host_pool->isAllocated(staging_block));
        EXPECT_FALSE(disk_pool->isAllocated(source_block));
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        // Settlement releases the pending-copy hold while the request still owns the target.
        EXPECT_EQ(device_pool->refCount(target_block), 1u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);

        EXPECT_EQ(outliving_ticket->commit(), nullptr);
        outliving_ticket.reset();
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        EXPECT_EQ(device_pool->refCount(target_block), 1u);

        device_pool->decRef(request_targets);
        EXPECT_FALSE(device_pool->isAllocated(target_block));
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    }
}

TEST_F(BlockTreeCacheIntegrationTest, Evictor_SkipsRequestPinnedBlock) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    ASSERT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->scripted_copy_engine->clear();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    environment->expectPayloads();
    environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

class BlockTreeCacheDemotionFailureTest: public ::testing::TestWithParam<DemotionFailureStage> {};

TEST_P(BlockTreeCacheDemotionFailureTest, Evictor_DemotionFailure_RestoresSourceAndHeap) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    const Tier source_tier = GetParam() == DemotionFailureStage::D2H ? Tier::DEVICE : Tier::HOST;
    const Tier target_tier = GetParam() == DemotionFailureStage::D2H ? Tier::HOST : Tier::DISK;
    if (source_tier == Tier::HOST) {
        environment->demoteAll(Tier::DEVICE);
        ASSERT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    }

    environment->scripted_copy_engine->clear();
    for (size_t attempt = 0; attempt < 128; ++attempt) {
        environment->scripted_copy_engine->enqueue(
            GetParam() == DemotionFailureStage::D2H ? CopyStatus::DEVICE_IO_ERROR : CopyStatus::DISK_IO_ERROR);
    }
    environment->demoteAll(source_tier);

    EXPECT_TRUE(environment->allSlotsAtTier(source_tier));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    for (const TransferDescriptor& descriptor : environment->scripted_copy_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    environment->expectPayloads();
    if (source_tier == Tier::DEVICE) {
        environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});
        EXPECT_GT(environment->cache->getStats().device_heap_total_size, 0u);
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
        EXPECT_GT(environment->cache->getStats().host_heap_total_size, 0u);
    }

    environment->scripted_copy_engine->clear();
    environment->demoteAll(source_tier);
    EXPECT_TRUE(environment->allSlotsAtTier(target_tier));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    for (const TransferDescriptor& descriptor : environment->scripted_copy_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    environment->expectPayloads();
    if (target_tier == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(DemotionFailure,
                         BlockTreeCacheDemotionFailureTest,
                         ::testing::Values(DemotionFailureStage::D2H, DemotionFailureStage::H2DISK),
                         demotionFailureParamName);

class BlockTreeCacheLowerTierTest: public ::testing::TestWithParam<Tier> {};

TEST_P(BlockTreeCacheLowerTierTest, FullSWA_MatchLowerTierOnlyReturnsTicketWithoutPublishing) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    environment->scripted_copy_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(result.load_back_blocks, 6u);
    EXPECT_EQ(result.host_load_back_blocks, GetParam() == Tier::HOST ? 6u : 0u);
    EXPECT_EQ(result.disk_load_back_blocks, GetParam() == Tier::DISK ? 6u : 0u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 4u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    expectPlanningSourceRefCounts(*environment, GetParam());
    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (PendingLoadBackItem& item : result.load_back_ticket->items()) {
        item.target_device_blocks.clear();
        for (const std::string& tag : item.device_group_tags) {
            DeviceBlockPoolPtr pool = devicePoolForTag(*environment, tag);
            ASSERT_NE(pool, nullptr);
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            request_targets.emplace_back(std::move(pool), target);
        }
        ASSERT_EQ(item.target_device_blocks.size(), item.device_group_tags.size());
        ASSERT_NE(item.node, nullptr);
    }

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    expectUnpublishedResult(result);
    EXPECT_EQ(result.async_context, nullptr);

    const size_t submits_after_commit = environment->scripted_copy_engine->submitCount();
    EXPECT_GT(submits_after_commit, 0u);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), submits_after_commit);

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    EXPECT_EQ(rematch.matched_blocks, kPathLength);
    ASSERT_NE(rematch.matched_node, nullptr);
    expectAggregatedReadyResult(rematch, /*full_blocks=*/4, /*swa_blocks=*/2);
    environment->expectPayloads();
    environment->releaseMatch(rematch);

    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 14}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 16}, {16, 14});
    }
    environment->reclaimAll();
    environment->cache->waitForPendingTasks();
    for (const auto& [pool, block] : request_targets) {
        pool->decRef(block);
    }
    environment->cache->onBlocksReleased();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, CancelPausedLoadBackPreservesSourceAndDiscardsTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment          = FullSWAEnvironment::create();
    auto pausable_copy_engine = std::make_shared<PausableCopyEngine>(
        environment->groups, environment->components, CopyStatus::OK, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setCopyEngineForTest(*environment->cache, pausable_copy_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    std::vector<std::vector<GroupSlot>> slots_before;
    slots_before.reserve(environment->keys.size());
    const auto snapshot_before = environment->cache->getKeySnapshot(/*limit=*/32);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_FALSE(result.load_back_ticket->empty());
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), kPathLength);
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        slots_before.push_back(environment->slotsForPathNode(path_index));
    }

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (PendingLoadBackItem& item : result.load_back_ticket->items()) {
        ASSERT_EQ(item.source_tier, GetParam());
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[item.group_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[item.group_id].get());
        for (const BlockIdxType block : item.source_blocks) {
            ASSERT_NE(block, NULL_BLOCK_IDX);
            EXPECT_EQ(source_pool->refCount(block), 2u);
            source_refs.push_back(SourceRef{source_pool, block});
        }

        item.target_device_blocks.clear();
        for (const std::string& tag : item.device_group_tags) {
            DeviceBlockPoolPtr pool = devicePoolForTag(*environment, tag);
            ASSERT_NE(pool, nullptr);
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            target_blocks.emplace_back(std::move(pool), target);
        }
    }
    ASSERT_FALSE(source_refs.empty());
    ASSERT_FALSE(target_blocks.empty());

    pausable_copy_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    pausable_copy_engine->waitUntilEntered();
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(environment->cache->cancelLoadBack(context));
    pausable_copy_engine->release();
    context->waitDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(environment->cache->cancelLoadBack(context));
    const auto snapshot_after = environment->cache->getKeySnapshot(/*limit=*/32);
    EXPECT_EQ(snapshot_after.version, snapshot_before.version);
    EXPECT_EQ(snapshot_after.keys, snapshot_before.keys);

    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const auto slots_after = environment->slotsForPathNode(path_index);
        ASSERT_EQ(slots_after.size(), slots_before[path_index].size());
        for (size_t group_id = 0; group_id < slots_after.size(); ++group_id) {
            const GroupSlot& before = slots_before[path_index][group_id];
            const GroupSlot& after  = slots_after[group_id];
            EXPECT_EQ(after.device_blocks, before.device_blocks);
            EXPECT_EQ(after.host_block, before.host_block);
            EXPECT_EQ(after.disk_slot, before.disk_slot);
            EXPECT_EQ(after.transfer_state, SlotTransferState::IDLE);
            EXPECT_EQ(after.candidate_meta.last_access_seq, before.candidate_meta.last_access_seq);
            EXPECT_EQ(after.candidate_meta.admission_seq, before.candidate_meta.admission_seq);
            EXPECT_EQ(after.candidate_meta.hit_count, before.candidate_meta.hit_count);
        }
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->refCount(source.block), 1u);
    }
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    result.load_back_ticket.reset();
    environment->cache->waitForPendingTasks();
    for (const auto& [pool, block] : target_blocks) {
        pool->decRef(block);
    }
    environment->cache->onBlocksReleased();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, LoadBackDisabled_DoesNotReportLowerTierHit) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.enable_load_back = false;
    auto environment         = FullSWAEnvironment::create(options);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);
    environment->scripted_copy_engine->clear();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    EXPECT_EQ(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_blocks, 0u);
    EXPECT_EQ(result.host_load_back_blocks, 0u);
    EXPECT_EQ(result.disk_load_back_blocks, 0u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    environment->expectPayloads();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(HostAndDisk,
                         BlockTreeCacheLowerTierTest,
                         ::testing::Values(Tier::HOST, Tier::DISK),
                         tierParamName);

TEST_F(BlockTreeCacheIntegrationTest, FullSWA_MatchPublishesOnlyReadyBoundary) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    BlockTreeMatchResult prefix_hold = environment->cache->match({environment->keys[0], environment->keys[1]});
    ASSERT_EQ(prefix_hold.matched_blocks, 2u);
    environment->releaseRequestRefsForGroup(1);
    runSingleMaintenance(*environment, Tier::DEVICE, 0.125);
    environment->releaseMatch(prefix_hold);

    for (size_t path_index = 0; path_index < kPathLength; ++path_index) {
        const std::vector<GroupSlot> slots = environment->slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        EXPECT_TRUE(slots[0].has_value(Tier::DEVICE));
        if (path_index < 2) {
            EXPECT_TRUE(slots[1].has_value(Tier::DEVICE));
        } else {
            EXPECT_TRUE(slots[1].has_value(Tier::HOST));
            EXPECT_FALSE(slots[1].has_value(Tier::DEVICE));
        }
    }

    environment->scripted_copy_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_blocks, 2u);
    ASSERT_EQ(result.matched_node->cache_key, environment->keys[1]);
    expectAggregatedReadyResult(result, /*full_blocks=*/2, /*swa_blocks=*/2);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 2u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    for (const PendingLoadBackItem& item : result.load_back_ticket->items()) {
        if (item.group_id == 0) {
            EXPECT_EQ(item.source_tier, Tier::DEVICE);
            EXPECT_GE(item.path_index, 2u);
        } else {
            EXPECT_EQ(item.source_tier, Tier::HOST);
        }
    }
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    environment->releaseMatch(result);
    result.load_back_ticket.reset();
    environment->releaseRequestRefs();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadBackPreflightRejectionImmediatelyRestoresCandidates) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    BlockTreeMatchResult result = makePartialReadyDeviceTicket(*environment);
    ASSERT_NE(result.load_back_ticket, nullptr);
    ASSERT_FALSE(result.load_back_ticket->items().empty());
    ASSERT_EQ(result.load_back_ticket->items().front().source_tier, Tier::DEVICE);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    for (PendingLoadBackItem& item : result.load_back_ticket->items()) {
        if (item.source_tier != Tier::DEVICE) {
            continue;
        }
        ASSERT_EQ(item.device_group_tags.size(), item.source_blocks.size());
        item.node                 = nullptr;
        item.target_device_blocks = item.source_blocks;
        for (size_t local_pool_index = 0; local_pool_index < item.source_blocks.size(); ++local_pool_index) {
            DeviceBlockPoolPtr pool = devicePoolForTag(*environment, item.device_group_tags[local_pool_index]);
            ASSERT_NE(pool, nullptr);
            device_sources.emplace_back(std::move(pool), item.source_blocks[local_pool_index]);
        }
    }
    ASSERT_EQ(device_sources.size(), 4u);
    result.load_back_ticket->items().front().target_device_blocks.pop_back();

    environment->releaseRequestRefs();
    const size_t                           tree_nodes_before = environment->cache->getStats().tree_node_count;
    std::vector<std::vector<BlockIdxType>> allocations_before;
    for (const DeviceBlockPoolPtr& pool : environment->device_pools) {
        allocations_before.push_back(allocatedBlocksSnapshot(*pool));
    }
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    environment->expectPayloads();

    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    for (size_t pool_index = 0; pool_index < environment->device_pools.size(); ++pool_index) {
        EXPECT_EQ(allocatedBlocksSnapshot(*environment->device_pools[pool_index]), allocations_before[pool_index]);
    }
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    result.load_back_ticket.reset();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment->cache->evictForTag("tag_0", 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    environment->releaseMatch(result);
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadBackExplicitAbortImmediatelyRestoresCandidatesOnce) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    BlockTreeMatchResult result = makePartialReadyDeviceTicket(*environment);
    ASSERT_NE(result.load_back_ticket, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    for (PendingLoadBackItem& item : result.load_back_ticket->items()) {
        if (item.source_tier != Tier::DEVICE) {
            continue;
        }
        ASSERT_EQ(item.device_group_tags.size(), item.source_blocks.size());
        item.node = nullptr;
        for (size_t local_pool_index = 0; local_pool_index < item.source_blocks.size(); ++local_pool_index) {
            DeviceBlockPoolPtr pool = devicePoolForTag(*environment, item.device_group_tags[local_pool_index]);
            ASSERT_NE(pool, nullptr);
            device_sources.emplace_back(std::move(pool), item.source_blocks[local_pool_index]);
        }
    }
    ASSERT_EQ(device_sources.size(), 4u);

    environment->releaseRequestRefs();
    const size_t tree_nodes_before = environment->cache->getStats().tree_node_count;
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    environment->expectPayloads();

    std::shared_ptr<LoadBackTicket> ticket = std::move(result.load_back_ticket);
    ticket.reset();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    ticket.reset();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment->cache->evictForTag("tag_0", 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    environment->releaseMatch(result);
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadBackAsyncCompletionRefreshesBeforeTerminalPublication) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    auto pausable_copy_engine = std::make_shared<PausableCopyEngine>(
        environment->groups, environment->components, CopyStatus::OK, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setCopyEngineForTest(*environment->cache, pausable_copy_engine);
    BlockTreeMatchResult result = makePartialReadyDeviceTicket(*environment);
    ASSERT_NE(result.load_back_ticket, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_target_blocks;
    for (PendingLoadBackItem& item : result.load_back_ticket->items()) {
        item.target_device_blocks.clear();
        if (item.source_tier == Tier::DEVICE) {
            ASSERT_EQ(item.device_group_tags.size(), item.source_blocks.size());
            item.node                 = nullptr;
            item.target_device_blocks = item.source_blocks;
            for (size_t local_pool_index = 0; local_pool_index < item.source_blocks.size(); ++local_pool_index) {
                DeviceBlockPoolPtr pool = devicePoolForTag(*environment, item.device_group_tags[local_pool_index]);
                ASSERT_NE(pool, nullptr);
                device_sources.emplace_back(std::move(pool), item.source_blocks[local_pool_index]);
            }
            continue;
        }

        for (const std::string& tag : item.device_group_tags) {
            DeviceBlockPoolPtr pool = devicePoolForTag(*environment, tag);
            ASSERT_NE(pool, nullptr);
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            request_target_blocks.emplace_back(std::move(pool), target);
        }
        ASSERT_EQ(item.target_device_blocks.size(), item.device_group_tags.size());
        ASSERT_NE(item.node, nullptr);
    }
    ASSERT_EQ(device_sources.size(), 4u);
    ASSERT_EQ(request_target_blocks.size(), 2u);

    environment->releaseMatch(result);
    environment->releaseRequestRefs();
    const size_t tree_nodes_before        = environment->cache->getStats().tree_node_count;
    const size_t device_candidates_before = environment->cache->getStats().device_heap_total_size;

    pausable_copy_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    pausable_copy_engine->waitUntilEntered();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_before);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    environment->expectPayloads();

    pausable_copy_engine->release();
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(environment->cache->cancelLoadBack(context));
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_before + 1u);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();
    for (const TransferDescriptor& descriptor : pausable_copy_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
    }
    EXPECT_EQ(pausable_copy_engine->submitCount(), 2u);

    EXPECT_EQ(environment->cache->evictForTag("tag_0", 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    result.load_back_ticket.reset();
    context->waitDone();
    EXPECT_TRUE(context->success());
    environment->cache->waitForPendingTasks();
    for (const auto& [pool, block] : request_target_blocks) {
        pool->decRef(block);
    }
    environment->cache->onBlocksReleased();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, SparseDisconnectedSWADoesNotPublishVacuousReadyPrefix) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.enable_reverse_eviction = false;
    auto environment                = FullSWAEnvironment::create(options);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    BlockTreeFindResult find = environment->cache->tree()->findNode(environment->keys);
    ASSERT_EQ(find.path.size(), kPathLength);
    std::vector<BlockIdxType> swa_host_blocks;
    for (size_t path_index = 0; path_index < kPathLength; ++path_index) {
        GroupSlot& swa_slot = find.path[path_index]->group_slots[1];
        ASSERT_TRUE(swa_slot.has_value(Tier::DEVICE));
        const std::vector<BlockIdxType> old_device_blocks = environment->groups[1]->getBlocks(swa_slot, Tier::DEVICE);
        environment->groups[1]->unreferenceBlocks(GroupBlockSet{1, Tier::DEVICE, {old_device_blocks}});
        environment->groups[1]->setBlocks(swa_slot, Tier::DEVICE, {});
        if (path_index >= 2) {
            const BlockIdxType host_block = environment->groups[1]->allocateSingleBlock(Tier::HOST);
            ASSERT_NE(host_block, NULL_BLOCK_IDX);
            environment->groups[1]->setBlocks(swa_slot, Tier::HOST, {host_block});
            swa_host_blocks.push_back(host_block);
        }
    }
    environment->cache->onBlocksReleased();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 4u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    EXPECT_EQ(result.load_back_blocks, 2u);
    EXPECT_EQ(result.host_load_back_blocks, 2u);
    EXPECT_EQ(result.disk_load_back_blocks, 0u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);

    for (const PendingLoadBackItem& item : result.load_back_ticket->items()) {
        if (item.group_id == 0) {
            EXPECT_EQ(item.source_tier, Tier::DEVICE);
            ASSERT_EQ(item.source_blocks.size(), 2u);
            EXPECT_EQ(environment->device_pools[0]->refCount(item.source_blocks[0]), 2u);
            EXPECT_EQ(environment->device_pools[1]->refCount(item.source_blocks[1]), 2u);
        } else {
            EXPECT_EQ(item.source_tier, Tier::HOST);
            ASSERT_EQ(item.source_blocks.size(), 1u);
            EXPECT_GE(item.path_index, 2u);
            EXPECT_EQ(environment->host_pools[1]->refCount(item.source_blocks[0]), 2u);
        }
    }

    result.load_back_ticket.reset();
    for (const auto& blocks : environment->request_blocks[0].per_node) {
        EXPECT_EQ(environment->device_pools[0]->refCount(blocks[0]), 1u);
        EXPECT_EQ(environment->device_pools[1]->refCount(blocks[1]), 1u);
    }
    for (BlockIdxType host_block : swa_host_blocks) {
        EXPECT_EQ(environment->host_pools[1]->refCount(host_block), 1u);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}
}  // namespace
}  // namespace rtp_llm
