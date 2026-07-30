#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

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

class PausablePerRankBlockTransferEngine: public PerRankBlockTransferEngine {
public:
    PausablePerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups,
                                       bool                            succeed,
                                       bool                            pause_enabled   = true,
                                       size_t                          throw_on_submit = 0):
        PerRankBlockTransferEngine(groups),
        pause_enabled_(pause_enabled),
        throw_on_submit_(throw_on_submit),
        succeed_(succeed) {}

    std::shared_ptr<AsyncContext> submit(const TransferDescriptor& descriptor) override {
        size_t submit_index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!pause_enabled_) {
                lock.unlock();
                return PerRankBlockTransferEngine::submit(descriptor);
            }
            ++submit_count_;
            submit_index = submit_count_;
            descriptors_.push_back(descriptor);
            entered_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return released_; });
        }
        if (submit_index == throw_on_submit_) {
            throw std::runtime_error("injected transfer failure");
        }
        if (!succeed_) {
            return std::make_shared<CompletedAsyncContext>(
                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
        }
        return PerRankBlockTransferEngine::submit(descriptor);
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

    void setThrowOnSubmit(size_t submit_index) {
        std::lock_guard<std::mutex> lock(mutex_);
        ASSERT_FALSE(pause_enabled_);
        ASSERT_FALSE(entered_);
        ASSERT_FALSE(released_);
        ASSERT_EQ(submit_count_, 0u);
        throw_on_submit_ = submit_index;
    }

    void waitUntilEntered() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return entered_; });
    }

    bool waitUntilEnteredFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return entered_; });
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
    size_t                          throw_on_submit_{0};
    bool                            entered_{false};
    bool                            released_{false};
    size_t                          submit_count_{0};
    std::vector<TransferDescriptor> descriptors_;
    bool                            succeed_{true};
};

class PausableTransferReleaseGuard {
public:
    explicit PausableTransferReleaseGuard(std::shared_ptr<PausablePerRankBlockTransferEngine> transfer_engine):
        transfer_engine_(std::move(transfer_engine)) {}

    ~PausableTransferReleaseGuard() {
        transfer_engine_->release();
    }

private:
    std::shared_ptr<PausablePerRankBlockTransferEngine> transfer_engine_;
};

class ThrowingPerRankBlockTransferEngine final: public PerRankBlockTransferEngine {
public:
    explicit ThrowingPerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups):
        PerRankBlockTransferEngine(groups) {}

    std::shared_ptr<AsyncContext> submit(const TransferDescriptor& descriptor) override {
        if (!throw_enabled_) {
            return PerRankBlockTransferEngine::submit(descriptor);
        }
        throw std::runtime_error("injected load copy failure");
    }

    void enableThrow() {
        throw_enabled_ = true;
    }

private:
    bool throw_enabled_{false};
};

// Upper bound for every synchronization wait in racing tests. A regression
// fails the test after this deadline instead of hanging until the Bazel
// global timeout.
constexpr std::chrono::seconds kRaceWaitTimeout{30};

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
        auto                     full_group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups     = {full_group};
        cache_                              = makeBlockTreeCacheForTest(std::move(groups));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

constexpr size_t kPathLength = 4;
constexpr size_t kPoolSize   = 16;

enum class DemotionFailureStage {
    D2H,
    H2DISK,
};

enum class DisabledLoadTierLayout {
    HOST_ONLY,
    DISK_ONLY,
    HOST_AND_DISK,
};

std::string tierParamName(const ::testing::TestParamInfo<Tier>& info) {
    return info.param == Tier::HOST ? "Host" : "Disk";
}

std::string demotionFailureParamName(const ::testing::TestParamInfo<DemotionFailureStage>& info) {
    return info.param == DemotionFailureStage::D2H ? "D2H" : "H2Disk";
}

std::string disabledLoadTierLayoutName(const ::testing::TestParamInfo<DisabledLoadTierLayout>& info) {
    switch (info.param) {
        case DisabledLoadTierLayout::HOST_ONLY:
            return "HostOnly";
        case DisabledLoadTierLayout::DISK_ONLY:
            return "DiskOnly";
        case DisabledLoadTierLayout::HOST_AND_DISK:
            return "HostAndDisk";
    }
    return "Unknown";
}

void demoteTo(FullSWAEnvironment& environment, Tier target_tier) {
    environment.demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment.allResourcesAtTier(Tier::HOST));
    if (target_tier == Tier::DISK) {
        environment.demoteAll(Tier::HOST);
        ASSERT_TRUE(environment.allResourcesAtTier(Tier::DISK));
    }
}

void expectUnpublishedResult(const BlockTreeMatchResult& result) {
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.matched_resources.empty());
    EXPECT_EQ(result.async_context, nullptr);
}

void expectAggregatedReadyResult(const BlockTreeCache&       cache,
                                 const BlockTreeMatchResult& result,
                                 size_t                      full_blocks,
                                 size_t                      swa_blocks) {
    EXPECT_EQ(cache.matchedBlocksForGroup(0, result.matched_resources).size(), full_blocks);
    EXPECT_EQ(cache.matchedBlocksForGroup(1, result.matched_resources).size(), full_blocks);
    EXPECT_EQ(cache.matchedBlocksForGroup(2, result.matched_resources).size(), swa_blocks);
    ASSERT_EQ(result.matched_resources.size(), 2u);
    EXPECT_EQ(result.matched_resources[0].group_set_id, 0);
    EXPECT_EQ(result.matched_resources[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_resources[0].node_blocks.size(), full_blocks);
    EXPECT_EQ(result.matched_resources[1].group_set_id, 1);
    EXPECT_EQ(result.matched_resources[1].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_resources[1].node_blocks.size(), swa_blocks);
}

void expectPlanningSourceRefCounts(const FullSWAEnvironment& environment, Tier tier) {
    for (size_t path_index = 0; path_index < environment.keys.size(); ++path_index) {
        const std::vector<GroupSetResource> resources = environment.resourcesForPathNode(path_index);
        ASSERT_EQ(resources.size(), 2u);
        for (size_t group_id = 0; group_id < resources.size(); ++group_id) {
            const BlockIdxType block =
                tier == Tier::HOST ? resources[group_id].host_block : resources[group_id].disk_slot;
            const IBlockPool& pool          = tier == Tier::HOST ?
                                                  static_cast<const IBlockPool&>(*environment.host_pools[group_id]) :
                                                  static_cast<const IBlockPool&>(*environment.disk_pools[group_id]);
            const bool        in_swa_window = group_id == 0 || path_index + 2 >= environment.keys.size();
            const uint32_t    expected      = in_swa_window ? 2u : 1u;
            EXPECT_EQ(pool.refCount(block), expected);
        }
    }
}

size_t ticketItemCountForGroupSet(const std::shared_ptr<LoadTicket>& ticket, size_t group_set_id) {
    if (ticket == nullptr) {
        return 0;
    }
    return static_cast<size_t>(
        std::count_if(ticket->items().begin(), ticket->items().end(), [group_set_id](const PendingLoadItem& item) {
            return item.group_set_id == group_set_id;
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

    environment.scripted_per_rank_transfer_engine->clear();
    BlockTreeMatchResult result = environment.cache->match(environment.keys);
    EXPECT_EQ(result.matched_blocks, 2u);
    EXPECT_NE(result.load_ticket, nullptr);
    if (result.load_ticket != nullptr) {
        EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 0), 2u);
        EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 1), 2u);
    }
    return result;
}

class OneShotWatermarkTestPeer {
public:
    static void runDevicePass(BlockTreeCache& cache, double ratio) {
        ASSERT_EQ(cache.task_pool_->pending_tasks_.load(), 0);
        {
            std::lock_guard<std::mutex> lock(cache.mutex_);
            cache.setTierWatermark(Tier::HOST, 0.0, 0);
            cache.setTierWatermark(Tier::DISK, 0.0, 0);
            cache.setTierWatermark(Tier::DEVICE, ratio, 0);
            cache.checkWatermark();
            cache.setTierWatermark(Tier::DEVICE, 0.0, 0);
        }
        cache.waitForPendingTasks();
        EXPECT_EQ(cache.task_pool_->pending_tasks_.load(), 0);
    }
};

TEST_F(BlockTreeCacheIntegrationTest, HostDiskOnlyLifecycle) {
    auto host_pool = makeHostPool(256, 8);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());

    auto device_pool = makeDevicePool({{256, 0}}, 8, "watermark_host_to_disk");
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    auto topology    = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 256)});
    full->initialize(0, topology, {0});
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = false;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_load         = false;

    auto cache = makeBlockTreeCacheForTest(std::move(groups), std::move(cfg));
    ASSERT_NE(cache, nullptr);
    auto scripted_copy = std::make_shared<ScriptedPerRankBlockTransferEngine>(std::vector<GroupSetPtr>{full});
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, scripted_copy);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));

    auto before = cache->tree()->findNode({100});
    ASSERT_FALSE(before.empty());
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);
    const CandidateMeta before_meta     = before.back()->group_set_resources[0].candidate_meta;
    const auto          snapshot_before = cache->getKeySnapshot(/*limit=*/8);
    EXPECT_EQ(snapshot_before.keys, (CacheKeysType{100}));

    BlockTreeMatchResult host_match = cache->match({100});
    expectUnpublishedResult(host_match);
    EXPECT_EQ(host_match.load_ticket, nullptr);

    scripted_copy->enqueue(/*success=*/false);
    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();

    auto after_failure = cache->tree()->findNode({100});
    ASSERT_FALSE(after_failure.empty());
    const auto& failed_resource = after_failure.back()->group_set_resources[0];
    EXPECT_EQ(failed_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(failed_resource.host_block, host_block);
    EXPECT_FALSE(failed_resource.hasTier(Tier::DISK));
    EXPECT_TRUE(host_pool->isAllocated(host_block));
    EXPECT_EQ(host_pool->refCount(host_block), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(failed_resource.candidate_meta.last_access_seq, before_meta.last_access_seq);
    EXPECT_EQ(failed_resource.candidate_meta.admission_seq, before_meta.admission_seq);
    EXPECT_EQ(failed_resource.candidate_meta.hit_count, before_meta.hit_count);
    const auto snapshot_after_failure = cache->getKeySnapshot(/*limit=*/8);
    EXPECT_EQ(snapshot_after_failure.version, snapshot_before.version);
    EXPECT_EQ(snapshot_after_failure.keys, snapshot_before.keys);

    scripted_copy->clear();
    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    const auto& resource = find.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_TRUE(resource.hasTier(Tier::DISK));
    EXPECT_NE(resource.disk_slot, NULL_BLOCK_IDX);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_TRUE(disk_pool->isAllocated(resource.disk_slot));
    EXPECT_EQ(disk_pool->refCount(resource.disk_slot), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);

    const auto snapshot_after_success = cache->getKeySnapshot(/*limit=*/8);
    EXPECT_GT(snapshot_after_success.version, snapshot_after_failure.version);
    EXPECT_EQ(snapshot_after_success.keys, snapshot_before.keys);
    BlockTreeMatchResult disk_match = cache->match({100});
    expectUnpublishedResult(disk_match);
    EXPECT_EQ(disk_match.load_ticket, nullptr);
    EXPECT_EQ(scripted_copy->submitCount(), 1u);

    cache->setTierWatermark(Tier::HOST, 0.0, 0);
    cache.reset();
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
}

TEST_F(BlockTreeCacheIntegrationTest, OneShotCascadeFailureRollsBackSWAAndRetriesOnce) {
    ASSERT_TRUE(cudaAvailable()) << "C002-T05 requires CUDA";
    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));

    const std::vector<GroupSetResource> initial_resources = environment->resourcesForPathNode(0);
    ASSERT_EQ(initial_resources.size(), 2u);
    ASSERT_EQ(initial_resources[0].device_blocks.size(), 2u);
    ASSERT_EQ(initial_resources[1].device_blocks.size(), 1u);
    const std::vector<BlockIdxType> full_sources = initial_resources[0].device_blocks;
    const BlockIdxType              swa_source   = initial_resources[1].device_blocks[0];

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/true);
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/false);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> first_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(first_descriptors.size(), 2u);
    EXPECT_EQ(first_descriptors[0].group_set_id, 0);
    EXPECT_EQ(first_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(first_descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(first_descriptors[0].device_blocks, full_sources);
    EXPECT_EQ(first_descriptors[1].group_set_id, 1);
    EXPECT_EQ(first_descriptors[1].source_tier, Tier::DEVICE);
    EXPECT_EQ(first_descriptors[1].target_tier, Tier::HOST);
    EXPECT_EQ(first_descriptors[1].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 2u);

    const std::vector<GroupSetResource> after_failure = environment->resourcesForPathNode(0);
    ASSERT_EQ(after_failure.size(), 2u);
    EXPECT_EQ(after_failure[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(after_failure[0].hasTier(Tier::DEVICE));
    EXPECT_TRUE(after_failure[0].hasTier(Tier::HOST));
    EXPECT_EQ(environment->host_pools[0]->refCount(after_failure[0].host_block), 1u);
    EXPECT_EQ(after_failure[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(after_failure[1].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_FALSE(after_failure[1].hasTier(Tier::HOST));
    EXPECT_EQ(environment->device_pools[2]->refCount(swa_source), 1u);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), 16u);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 1u);
    EXPECT_EQ(environment->device_pools[2]->activeTreeCachedBlocksNum(), 0u);
    environment->expectPoolFreeCounts({16, 16, 15}, {15, 16}, {16, 16});
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/true);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> retry_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(retry_descriptors.size(), 1u);
    EXPECT_EQ(retry_descriptors[0].group_set_id, 1);
    EXPECT_EQ(retry_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(retry_descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(retry_descriptors[0].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 1u);

    const std::vector<GroupSetResource> after_retry = environment->resourcesForPathNode(0);
    ASSERT_EQ(after_retry.size(), 2u);
    EXPECT_EQ(after_retry[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(after_retry[0].hasTier(Tier::HOST));
    EXPECT_EQ(after_retry[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(after_retry[1].hasTier(Tier::DEVICE));
    EXPECT_TRUE(after_retry[1].hasTier(Tier::HOST));
    EXPECT_FALSE(environment->device_pools[2]->isAllocated(swa_source));
    EXPECT_EQ(environment->host_pools[1]->refCount(after_retry[1].host_block), 1u);
    environment->expectPoolFreeCounts({16, 16, 16}, {15, 15}, {16, 16});
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, UncommittedLoadTicketReleasesSourceReferences) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::unique_ptr<FullSWAEnvironment> environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_FALSE(result.load_ticket->empty());
    expectPlanningSourceRefCounts(*environment, Tier::HOST);

    result.load_ticket.reset();
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const std::vector<GroupSetResource> resources = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources.size(), 2u);
        for (size_t group_id = 0; group_id < resources.size(); ++group_id) {
            EXPECT_EQ(environment->host_pools[group_id]->refCount(resources[group_id].host_block), 1u);
        }
    }

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

// C006-T03: shutdown waits for committed copy settlement before draining every tree hold.
TEST_F(BlockTreeCacheIntegrationTest, CacheShutdownWaitsForCommittedLoadSettlement) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    for (bool copy_success : {true, false}) {
        SCOPED_TRACE(copy_success ? "copy_success" : "copy_failure");

        constexpr size_t  kBlockBytes = 16;
        constexpr size_t  kPoolSize   = 2;
        const std::string pool_name   = copy_success ? "shutdown_load_success" : "shutdown_load_failure";
        auto              device_pool = makeDevicePool({{kBlockBytes, 0}}, kPoolSize, pool_name);
        auto              host_pool   = makeHostPool(kBlockBytes, kPoolSize);
        auto              disk_pool   = makeDiskPool(kBlockBytes, kPoolSize, std::make_unique<MemoryDiskBlockIO>());
        const size_t      device_free_before = device_pool->freeBlocksNum();
        const size_t      host_free_before   = host_pool->freeBlocksNum();
        const size_t      disk_free_before   = disk_pool->freeBlocksNum();

        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
        auto topology = block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(
            defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, kBlockBytes)});
        full->initialize(0, topology, {0});
        std::vector<GroupSetPtr> groups = {full};
        BlockTreeCacheConfig     config;
        config.enable_device_cache = true;
        config.enable_memory_cache = true;
        config.enable_disk_cache   = true;
        config.enable_load         = true;
        auto cache = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
        ASSERT_NE(cache, nullptr);

        auto pausable_per_rank_transfer_engine =
            std::make_shared<PausablePerRankBlockTransferEngine>(std::vector<GroupSetPtr>{full}, copy_success);
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_per_rank_transfer_engine);

        const BlockIdxType source_block = full->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        std::vector<std::vector<GroupSetResource>> source_resources(1, std::vector<GroupSetResource>(1));
        source_resources[0][0].disk_slot = source_block;
        ASSERT_TRUE(insertGroupSetResources(*cache, {100}, source_resources));

        BlockTreeMatchResult result = cache->match({100});
        ASSERT_NE(result.load_ticket, nullptr);
        ASSERT_EQ(result.load_ticket->items().size(), 1u);
        EXPECT_EQ(result.load_ticket->items()[0].source_tier, Tier::DISK);
        EXPECT_EQ(result.load_ticket->items()[0].source_blocks, (std::vector<BlockIdxType>{source_block}));
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);

        const BlockIdList request_targets = device_pool->malloc(1).value();
        ASSERT_EQ(request_targets.size(), 1u);
        device_pool->incRef(request_targets, BlockRefType::REQUEST);
        const BlockIdxType target_block = request_targets.front();
        EXPECT_EQ(device_pool->refCount(target_block), 1u);
        result.load_ticket->items_[0].target_device_blocks = {target_block};
        std::shared_ptr<LoadTicket> outliving_ticket       = std::move(result.load_ticket);

        std::shared_ptr<AsyncContext> context = outliving_ticket->commit();
        ASSERT_NE(context, nullptr);
        pausable_per_rank_transfer_engine->waitUntilEntered();
        EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 1u);
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);
        // The request owns one reference and the committed pending copy protects the target with another.
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);
        EXPECT_EQ(outliving_ticket->commit(), nullptr);

        ThreadCompletion destruction;
        LoadShutdownTestPeer::setPendingTaskWaitObserver(*cache, [&destruction] { destruction.markEntered(); });
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
        EXPECT_TRUE(host_blocks_after_wait.empty());
        ASSERT_EQ(disk_blocks_after_wait, (std::vector<BlockIdxType>{source_block}));
        const std::vector<TransferDescriptor> descriptors_after_wait = pausable_per_rank_transfer_engine->descriptors();
        ASSERT_EQ(descriptors_after_wait.size(), 1u);
        EXPECT_EQ(descriptors_after_wait[0].group_set_id, 0);
        EXPECT_EQ(descriptors_after_wait[0].source_tier, Tier::DISK);
        EXPECT_EQ(descriptors_after_wait[0].target_tier, Tier::DEVICE);
        EXPECT_EQ(descriptors_after_wait[0].disk_block, source_block);
        EXPECT_TRUE(isNullBlockIdx(descriptors_after_wait[0].host_block));
        EXPECT_EQ(descriptors_after_wait[0].device_blocks, (std::vector<BlockIdxType>{target_block}));
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        EXPECT_TRUE(disk_pool->isAllocated(source_block));
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(disk_pool->refCount(source_block), 2u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);

        pausable_per_rank_transfer_engine->release();
        destroy_thread.join();
        EXPECT_TRUE(destruction.finished());
        context->waitDone();
        EXPECT_TRUE(context->done());
        EXPECT_EQ(context->success(), copy_success);
        EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 1u);
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

        device_pool->decRef(request_targets, BlockRefType::REQUEST);
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
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->scripted_per_rank_transfer_engine->clear();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
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
        ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    }

    std::vector<std::vector<GroupSetResource>> resources_before_failure;
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        resources_before_failure.push_back(environment->resourcesForPathNode(path_index));
    }

    environment->scripted_per_rank_transfer_engine->clear();
    for (size_t attempt = 0; attempt < 128; ++attempt) {
        environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/false);
    }
    environment->demoteAll(source_tier);

    EXPECT_TRUE(environment->allResourcesAtTier(source_tier));
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    const std::vector<TransferDescriptor> failure_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    for (const TransferDescriptor& descriptor : failure_descriptors) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const auto resources_after_failure = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources_after_failure.size(), resources_before_failure[path_index].size());
        for (size_t group_id = 0; group_id < resources_after_failure.size(); ++group_id) {
            const CandidateMeta& before = resources_before_failure[path_index][group_id].candidate_meta;
            const CandidateMeta& after  = resources_after_failure[group_id].candidate_meta;
            EXPECT_EQ(after.last_access_seq, before.last_access_seq);
            EXPECT_EQ(after.admission_seq, before.admission_seq);
            EXPECT_EQ(after.hit_count, before.hit_count);
        }
    }
    environment->expectPayloads();
    if (source_tier == Tier::DEVICE) {
        environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});
        EXPECT_GT(environment->cache->getStats().device_heap_total_size, 0u);
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
        EXPECT_GT(environment->cache->getStats().host_heap_total_size, 0u);
    }

    environment->scripted_per_rank_transfer_engine->clear();
    environment->demoteAll(source_tier);
    EXPECT_TRUE(environment->allResourcesAtTier(target_tier));
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    const std::vector<TransferDescriptor> retry_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    for (const TransferDescriptor& descriptor : retry_descriptors) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    ASSERT_FALSE(failure_descriptors.empty());
    ASSERT_FALSE(retry_descriptors.empty());
    EXPECT_EQ(retry_descriptors.front().group_set_id, failure_descriptors.front().group_set_id);
    if (source_tier == Tier::DEVICE) {
        EXPECT_EQ(retry_descriptors.front().device_blocks, failure_descriptors.front().device_blocks);
    } else {
        EXPECT_EQ(retry_descriptors.front().host_block, failure_descriptors.front().host_block);
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

TEST_F(BlockTreeCacheIntegrationTest, MatchHardStopsDuringDemotionAndJoinsLoad) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // --- D2H demotion: a match racing the paused DEMOTING copy returns nothing;
    //     Host residency becomes visible only after the copy commits. ---
    {
        FullSWAEnvironmentOptions options;
        options.path_length = 1;
        options.enable_disk = false;
        auto environment    = FullSWAEnvironment::create(options);
        ASSERT_NE(environment, nullptr);
        auto pausable_copy =
            std::make_shared<PausablePerRankBlockTransferEngine>(environment->groups, /*succeed=*/true);
        PausableTransferReleaseGuard release_guard(pausable_copy);
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, pausable_copy);
        environment->insertRequestPath();
        environment->releaseRequestRefs();

        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
        for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
            const auto blocks = environment->blocksForDevicePool(pool_id);
            ASSERT_EQ(blocks.size(), 1u);
            device_sources.emplace_back(environment->device_pools[pool_id], blocks.front());
        }

        environment->cache->setTierWatermark(Tier::DEVICE, 0.01, 0);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);
        environment->cache->setTierWatermark(Tier::DEVICE, 0.0, 0);
        const bool d2h_copy_entered = pausable_copy->waitUntilEnteredFor(kRaceWaitTimeout);
        if (!d2h_copy_entered) {
            pausable_copy->release();  // never leave a paused engine blocking teardown
        }
        ASSERT_TRUE(d2h_copy_entered);
        const std::vector<GroupSetResource> demoting_resources = environment->resourcesForPathNode(0);
        ASSERT_EQ(demoting_resources.size(), environment->groups.size());
        for (const GroupSetResource& resource : demoting_resources) {
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
        }

        // Racing match while the node is DEMOTING: hard-stop, no reusable prefix,
        // no ticket. The Device source keeps only its single cache hold.
        BlockTreeMatchResult during = environment->cache->match(environment->keys);
        EXPECT_EQ(during.matched_blocks, 0u);
        EXPECT_EQ(during.matched_node, nullptr);
        EXPECT_EQ(during.load_ticket, nullptr);
        EXPECT_TRUE(during.matched_resources.empty());
        for (const auto& [pool, block] : device_sources) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 1u);
        }

        pausable_copy->release();
        environment->cache->waitForPendingTasks();
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
        for (const auto& [pool, block] : device_sources) {
            EXPECT_FALSE(pool->isAllocated(block));  // source freed exactly once on commit
        }

        // The node is now Host-resident: a fresh match surfaces it only as a Host
        // load source (no device-ready prefix yet).
        BlockTreeMatchResult after = environment->cache->match(environment->keys);
        EXPECT_EQ(after.matched_blocks, 0u);
        ASSERT_NE(after.load_ticket, nullptr);
        for (const PendingLoadItem& item : after.load_ticket->items()) {
            EXPECT_EQ(item.source_tier, Tier::HOST);
        }
        after.load_ticket.reset();  // abort the uncommitted ticket -> resource back to IDLE
        environment->cache->releaseMatchedResources(after.matched_resources);

        environment->reclaimAll();
        environment->expectFullyReclaimed();
    }

    // --- H2D load: a second match joins the in-flight copy and reuses its
    //     target blocks without submitting another transfer. ---
    {
        FullSWAEnvironmentOptions options;
        options.path_length = 1;
        options.enable_disk = false;
        auto environment    = FullSWAEnvironment::create(options);
        ASSERT_NE(environment, nullptr);
        auto pausable_copy = std::make_shared<PausablePerRankBlockTransferEngine>(
            environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
        PausableTransferReleaseGuard release_guard(pausable_copy);
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, pausable_copy);
        environment->insertRequestPath();
        environment->releaseRequestRefs();
        demoteTo(*environment, Tier::HOST);

        BlockTreeMatchResult first = environment->cache->match(environment->keys);
        ASSERT_NE(first.load_ticket, nullptr);
        std::vector<std::pair<IBlockPool*, BlockIdxType>>        host_sources;
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
        for (LoadTicket::PendingLoadItem& item : first.load_ticket->items_) {
            ASSERT_EQ(item.source_tier, Tier::HOST);
            ASSERT_EQ(item.source_blocks.size(), 1u);
            host_sources.emplace_back(environment->host_pools[static_cast<size_t>(item.group_set_id)].get(),
                                      item.source_blocks.front());
            item.target_device_blocks.clear();
            for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
                BlockIdList blocks = pool->malloc(1).value();
                ASSERT_EQ(blocks.size(), 1u);
                pool->incRef(blocks, BlockRefType::REQUEST);
                item.target_device_blocks.push_back(blocks.front());
                request_targets.emplace_back(pool, blocks.front());
            }
        }

        pausable_copy->enablePause();
        std::shared_ptr<AsyncContext> context = first.load_ticket->commit();
        ASSERT_NE(context, nullptr);
        const bool h2d_copy_entered = pausable_copy->waitUntilEnteredFor(kRaceWaitTimeout);
        if (!h2d_copy_entered) {
            pausable_copy->release();  // never leave a paused engine blocking teardown
        }
        ASSERT_TRUE(h2d_copy_entered);
        const std::vector<GroupSetResource> loading_resources = environment->resourcesForPathNode(0);
        ASSERT_EQ(loading_resources.size(), environment->groups.size());
        for (const GroupSetResource& resource : loading_resources) {
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
        }
        for (const auto& [pool, block] : host_sources) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 2u);  // tree residency + first load source hold
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 2u);  // request hold + in-flight target hold
        }

        const size_t         submits_before_join = pausable_copy->submitCount();
        BlockTreeMatchResult second              = environment->cache->match(environment->keys);
        EXPECT_EQ(second.matched_blocks, 0u);
        EXPECT_EQ(second.matched_node, nullptr);
        ASSERT_NE(second.load_ticket, nullptr);
        EXPECT_TRUE(second.matched_resources.empty());
        for (size_t item_index = 0; item_index < second.load_ticket->itemCount(); ++item_index) {
            EXPECT_TRUE(second.load_ticket->joinedLoad(item_index));
            const size_t                     group_set_id   = second.load_ticket->groupSetId(item_index);
            const std::vector<BlockIdxType>& joined_targets = second.load_ticket->targetDeviceBlocks(item_index);
            ASSERT_EQ(joined_targets.size(), environment->groups[group_set_id]->devicePools().size());
            for (size_t pool_index = 0; pool_index < joined_targets.size(); ++pool_index) {
                DeviceBlockPoolPtr pool = environment->groups[group_set_id]->devicePools()[pool_index];
                ASSERT_NE(pool, nullptr);
                pool->incRef(joined_targets[pool_index], BlockRefType::REQUEST);
            }
        }
        std::shared_ptr<AsyncContext> joined_context = second.load_ticket->commit();
        ASSERT_NE(joined_context, nullptr);
        EXPECT_FALSE(joined_context->done());
        EXPECT_EQ(pausable_copy->submitCount(), submits_before_join);
        for (const auto& [pool, block] : host_sources) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 2u);
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 3u);
        }
        environment->cache->releaseMatchedResources(second.matched_resources);

        BlockTreeMatchResult abandoned = environment->cache->match(environment->keys);
        ASSERT_NE(abandoned.load_ticket, nullptr);
        for (size_t item_index = 0; item_index < abandoned.load_ticket->itemCount(); ++item_index) {
            EXPECT_TRUE(abandoned.load_ticket->joinedLoad(item_index));
        }
        abandoned.load_ticket.reset();
        for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target : request_targets) {
            EXPECT_TRUE(target.first->isAllocated(target.second));
            EXPECT_EQ(target.first->refCount(target.second), 3u);
        }

        pausable_copy->release();
        context->waitDone();
        joined_context->waitDone();
        ASSERT_TRUE(context->done());
        EXPECT_TRUE(context->success());
        ASSERT_TRUE(joined_context->done());
        EXPECT_TRUE(joined_context->success());
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
        for (const auto& [pool, block] : host_sources) {
            EXPECT_FALSE(pool->isAllocated(block));  // stale Host source released once on completion
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 3u);  // tree residency + two request holds
        }

        // The node is now Device-resident: a fresh match reuses it directly with
        // no load.
        BlockTreeMatchResult after = environment->cache->match(environment->keys);
        EXPECT_EQ(after.matched_blocks, 1u);
        EXPECT_EQ(after.load_ticket, nullptr);
        environment->cache->releaseMatchedResources(after.matched_resources);

        first.load_ticket.reset();
        second.load_ticket.reset();
        environment->reclaimAll();
        for (const auto& [pool, block] : request_targets) {
            releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
            releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
        }
        environment->reclaimAll();
        environment->expectFullyReclaimed();
    }
}

TEST_F(BlockTreeCacheIntegrationTest, ReverseEvictionTieredEndToEnd) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length             = 1;
    options.enable_disk             = true;
    options.enable_load             = true;
    options.enable_reverse_eviction = true;
    auto environment                = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    // Start from the lower-priority SWA group. A failed primary copy rolls back
    // both SWA and its reverse-selected FULL sibling without publishing Host.
    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/false);
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE));
    environment->cache->waitForPendingTasks();
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    const auto failed_descriptors = environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(failed_descriptors.size(), 1u);
    EXPECT_EQ(failed_descriptors[0].group_set_id, 1);
    EXPECT_EQ(failed_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(failed_descriptors[0].target_tier, Tier::HOST);

    environment->scripted_per_rank_transfer_engine->clear();
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE));
    environment->cache->waitForPendingTasks();
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    const auto host_descriptors = environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(host_descriptors.size(), 2u);
    EXPECT_EQ(host_descriptors[0].group_set_id, 1);
    EXPECT_EQ(host_descriptors[1].group_set_id, 0);
    for (const TransferDescriptor& descriptor : host_descriptors) {
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
    }
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::HOST));
    environment->cache->waitForPendingTasks();
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DISK));
    const auto disk_descriptors = environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(disk_descriptors.size(), 2u);
    EXPECT_EQ(disk_descriptors[0].group_set_id, 1);
    EXPECT_EQ(disk_descriptors[1].group_set_id, 0);
    for (const TransferDescriptor& descriptor : disk_descriptors) {
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DISK);
    }
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_ticket, nullptr);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        ASSERT_EQ(item.source_tier, Tier::DISK);
        item.target_device_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks, BlockRefType::REQUEST);
            item.target_device_blocks.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
    }
    auto context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    environment->expectPayloads();

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    ASSERT_EQ(rematch.matched_blocks, 1u);
    expectAggregatedReadyResult(*environment->cache, rematch, /*full_blocks=*/1, /*swa_blocks=*/1);
    environment->releaseMatch(rematch);

    result.load_ticket.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, EvictionRejectsNonCanonicalTargetBeforeCopy) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    environment->scripted_per_rank_transfer_engine->clear();
    EXPECT_FALSE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE, Tier::DISK));
    environment->cache->waitForPendingTasks();
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
}

TEST_F(BlockTreeCacheIntegrationTest, EvictionExplicitNoneIsNotNormalized) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length             = 1;
    options.enable_reverse_eviction = false;
    auto environment                = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    environment->scripted_per_rank_transfer_engine->clear();
    EXPECT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE, Tier::NONE));
    environment->cache->waitForPendingTasks();
    auto result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result.back()->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(result.back()->group_set_resources[1].getTopTier(), Tier::NONE);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
}

TEST_F(BlockTreeCacheIntegrationTest, DiskLoadRequestOnlyKeepsDiskResidency) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t payload_bytes = 256;
    auto             device_pool   = makeDevicePool({{payload_bytes, 0}}, 4, "request_only_load_device");
    auto             disk_pool     = makeDiskPool(payload_bytes, 4, std::make_unique<MemoryDiskBlockIO>());
    auto             topology      = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_bytes)});
    auto group = block_transfer_engine_test::makeTestGroupSet(0, topology, {0}, {device_pool}, nullptr, disk_pool);

    BlockTreeCacheConfig config;
    config.enable_device_cache      = false;
    config.enable_memory_cache      = false;
    config.enable_disk_cache        = true;
    config.enable_load              = true;
    std::vector<GroupSetPtr> groups = {group};
    auto cache = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    const size_t       disk_free_before = disk_pool->freeBlocksNum();
    const BlockIdxType source_block     = group->allocateSingleBlock(Tier::DISK, BlockRefType::BLOCK_CACHE);
    ASSERT_FALSE(isNullBlockIdx(source_block));
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].disk_slot = source_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));

    BlockTreeMatchResult result = cache->match({100});
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_EQ(result.load_ticket->items().size(), 1u);
    EXPECT_EQ(result.load_ticket->items()[0].source_tier, Tier::DISK);
    EXPECT_EQ(result.load_ticket->items()[0].source_blocks, (std::vector<BlockIdxType>{source_block}));

    const size_t      device_free_before = device_pool->freeBlocksNum();
    const BlockIdList request_targets    = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets, BlockRefType::REQUEST);
    result.load_ticket->items_[0].target_device_blocks = request_targets;

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success());
    auto find_result = cache->tree()->findNode({100});
    ASSERT_FALSE(find_result.empty());
    const GroupSetResource& resource = find_result.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.disk_slot, source_block);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_EQ(disk_pool->refCount(source_block), 1u);
    EXPECT_EQ(device_pool->refCount(request_targets.front()), 1u);

    BlockTreeMatchResult rematch = cache->match({100});
    ASSERT_NE(rematch.load_ticket, nullptr);
    for (const LoadTicket::PendingLoadItem& item : rematch.load_ticket->items()) {
        EXPECT_EQ(item.source_tier, Tier::DISK);
    }
    rematch.load_ticket.reset();
    result.load_ticket.reset();
    device_pool->decRef(request_targets, BlockRefType::REQUEST);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    cache.reset();
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
}

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

    environment->scripted_per_rank_transfer_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_FALSE(result.load_ticket->empty());
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(GetParam()), kPathLength);
    EXPECT_EQ(result.load_blocks, 6u);
    EXPECT_EQ(result.host_load_blocks, GetParam() == Tier::HOST ? 6u : 0u);
    EXPECT_EQ(result.disk_load_blocks, GetParam() == Tier::DISK ? 6u : 0u);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 0), 4u);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 1), 2u);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    expectPlanningSourceRefCounts(*environment, GetParam());
    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        item.target_device_blocks.clear();
        const auto& device_pools = environment->groups.at(item.group_set_id)->devicePools();
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets, BlockRefType::REQUEST);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            request_targets.emplace_back(pool, target);
        }
        ASSERT_EQ(item.target_device_blocks.size(), device_pools.size());
        ASSERT_NE(item.node, nullptr);
    }

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(result.load_ticket->commit(), nullptr);
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    expectUnpublishedResult(result);
    EXPECT_EQ(result.async_context, nullptr);

    const size_t submits_after_commit = environment->scripted_per_rank_transfer_engine->submitCount();
    EXPECT_GT(submits_after_commit, 0u);
    EXPECT_EQ(result.load_ticket->commit(), nullptr);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), submits_after_commit);

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    EXPECT_EQ(rematch.matched_blocks, kPathLength);
    ASSERT_NE(rematch.matched_node, nullptr);
    expectAggregatedReadyResult(*environment->cache, rematch, /*full_blocks=*/4, /*swa_blocks=*/2);
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
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, TransferExceptionSettlesTicketAndReleasesAllWorkerHolds) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length          = 1;
    options.usable_device_blocks = 4;
    options.usable_host_blocks   = 4;
    options.usable_disk_blocks   = 4;
    auto environment             = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    ASSERT_NE(environment->cache, nullptr);

    auto throwing_engine = std::make_shared<ThrowingPerRankBlockTransferEngine>(environment->groups);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, throwing_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    ASSERT_TRUE(environment->allResourcesAtTier(GetParam()));
    throwing_engine->enableThrow();

    std::vector<size_t> device_free_before;
    std::vector<size_t> host_free_before;
    std::vector<size_t> disk_free_before;
    for (const DeviceBlockPoolPtr& pool : environment->device_pools) {
        device_free_before.push_back(pool->freeBlocksNum());
    }
    for (const std::shared_ptr<HostBlockPool>& pool : environment->host_pools) {
        host_free_before.push_back(pool->freeBlocksNum());
    }
    for (const BlockTreeDiskBlockPoolPtr& pool : environment->disk_pools) {
        disk_free_before.push_back(pool->freeBlocksNum());
    }

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_FALSE(result.load_ticket->empty());

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t item_index = 0; item_index < result.load_ticket->itemCount(); ++item_index) {
        const size_t                           group_set_id = result.load_ticket->groupSetId(item_index);
        const std::vector<DeviceBlockPoolPtr>& device_pools = environment->groups.at(group_set_id)->devicePools();
        std::vector<BlockIdxType>              item_targets;
        item_targets.reserve(device_pools.size());
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            const BlockIdxType block = blocks.front();
            pool->incRef(block, BlockRefType::REQUEST);
            item_targets.push_back(block);
            request_targets.emplace_back(pool, block);
        }
        ASSERT_TRUE(result.load_ticket->bindTargetDeviceBlocks(item_index, std::move(item_targets)));
    }

    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    environment->cache->waitForPendingTasks();

    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    ASSERT_TRUE(environment->allResourcesAtTier(GetParam()));
    for (const GroupSetResource& resource : environment->resourcesForPathNode(0)) {
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (size_t item_index = 0; item_index < result.load_ticket->itemCount(); ++item_index) {
        const size_t group_set_id = result.load_ticket->groupSetId(item_index);
        ASSERT_EQ(result.load_ticket->sourceBlocks(item_index).size(), 1u);
        const BlockIdxType source_block = result.load_ticket->sourceBlocks(item_index).front();
        const IBlockPool&  source_pool  = GetParam() == Tier::HOST ?
                                              static_cast<const IBlockPool&>(*environment->host_pools.at(group_set_id)) :
                                              static_cast<const IBlockPool&>(*environment->disk_pools.at(group_set_id));
        EXPECT_EQ(source_pool.refCount(source_block), 1u);
    }
    for (size_t pool_index = 0; pool_index < environment->host_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->host_pools[pool_index]->freeBlocksNum(), host_free_before[pool_index]);
    }
    for (size_t pool_index = 0; pool_index < environment->disk_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->disk_pools[pool_index]->freeBlocksNum(), disk_free_before[pool_index]);
    }

    for (const auto& [pool, block] : request_targets) {
        EXPECT_EQ(pool->refCount(block), 1u);
        pool->decRef(block, BlockRefType::REQUEST);
    }
    for (size_t pool_index = 0; pool_index < environment->device_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->device_pools[pool_index]->freeBlocksNum(), device_free_before[pool_index]);
    }
    EXPECT_NO_THROW(environment->cache.reset());
}

TEST(LoadTicketMetricsTest, DeduplicatesPathAndPrefersLowerTier) {
    std::shared_ptr<LoadTicketRegistry> registry =
        std::make_shared<LoadTicketRegistry>(LoadTicketRegistry::CommitCallback{}, LoadTicketRegistry::AbortCallback{});
    std::vector<PendingLoadItem> items(5);
    items[0].path_index  = 0;
    items[0].source_tier = Tier::DEVICE;
    items[1].path_index  = 0;
    items[1].source_tier = Tier::HOST;
    items[2].path_index  = 0;
    items[2].source_tier = Tier::DISK;
    items[3].path_index  = 1;
    items[3].source_tier = Tier::DEVICE;
    items[4].path_index  = 1;
    items[4].source_tier = Tier::HOST;

    std::shared_ptr<LoadTicket> ticket = registry->createTicket(items, 2, nullptr);
    ASSERT_NE(ticket, nullptr);
    EXPECT_EQ(ticket->logicalMatchedBlocks(Tier::DEVICE), 0u);
    EXPECT_EQ(ticket->logicalMatchedBlocks(Tier::HOST), 1u);
    EXPECT_EQ(ticket->logicalMatchedBlocks(Tier::DISK), 1u);
}

TEST_P(BlockTreeCacheLowerTierTest, CancelPausedLoadStillInstallsTransferredTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment                       = FullSWAEnvironment::create();
    auto pausable_per_rank_transfer_engine = std::make_shared<PausablePerRankBlockTransferEngine>(
        environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    const auto snapshot_before = environment->cache->getKeySnapshot(/*limit=*/32);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_FALSE(result.load_ticket->empty());
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), kPathLength);
    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        ASSERT_EQ(item.source_tier, GetParam());
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[item.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[item.group_set_id].get());
        for (const BlockIdxType block : item.source_blocks) {
            ASSERT_NE(block, NULL_BLOCK_IDX);
            EXPECT_EQ(source_pool->refCount(block), 2u);
            source_refs.push_back(SourceRef{source_pool, block});
        }

        item.target_device_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets, BlockRefType::REQUEST);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            target_blocks.emplace_back(pool, target);
        }
    }
    ASSERT_FALSE(source_refs.empty());
    ASSERT_FALSE(target_blocks.empty());

    pausable_per_rank_transfer_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    pausable_per_rank_transfer_engine->waitUntilEntered();
    EXPECT_FALSE(context->done());
    EXPECT_TRUE(environment->cache->cancelLoad(context));
    pausable_per_rank_transfer_engine->release();
    context->waitDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(environment->cache->cancelLoad(context));
    const auto snapshot_after = environment->cache->getKeySnapshot(/*limit=*/32);
    EXPECT_GT(snapshot_after.version, snapshot_before.version);
    EXPECT_EQ(snapshot_after.keys, snapshot_before.keys);

    for (const LoadTicket::PendingLoadItem& item : result.load_ticket->items()) {
        ASSERT_NE(item.node, nullptr);
        ASSERT_LT(item.group_set_id, item.node->group_set_resources.size());
        const GroupSetResource& resource = item.node->group_set_resources[item.group_set_id];
        EXPECT_EQ(resource.device_blocks, item.target_device_blocks);
        EXPECT_FALSE(resource.hasTier(Tier::HOST));
        EXPECT_FALSE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_FALSE(source.pool->isAllocated(source.block));
    }
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 2u);
    }
    environment->expectPayloads();

    result.load_ticket.reset();
    environment->cache->waitForPendingTasks();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, CancelCompletionRaceSettlesExactlyOnce) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    auto pausable_per_rank_transfer_engine = std::make_shared<PausablePerRankBlockTransferEngine>(
        environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_FALSE(result.load_ticket->empty());

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[item.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[item.group_set_id].get());
        ASSERT_EQ(item.source_blocks.size(), 1u);
        EXPECT_EQ(source_pool->refCount(item.source_blocks.front()), 2u);
        source_refs.push_back({source_pool, item.source_blocks.front()});

        item.target_device_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks, BlockRefType::REQUEST);
            item.target_device_blocks.push_back(blocks.front());
            target_blocks.emplace_back(pool, blocks.front());
        }
    }

    pausable_per_rank_transfer_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    pausable_per_rank_transfer_engine->waitUntilEntered();

    ThreadCompletion race_start;
    bool             cancellation_won = false;
    std::thread      cancel_thread([&] {
        race_start.waitUntilEntered();
        cancellation_won = environment->cache->cancelLoad(context);
    });
    race_start.markEntered();
    pausable_per_rank_transfer_engine->release();

    context->waitDone();
    cancel_thread.join();
    environment->cache->waitForPendingTasks();

    ASSERT_TRUE(context->done());
    EXPECT_EQ(cancellation_won, !context->success());
    EXPECT_FALSE(environment->cache->cancelLoad(context));
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    for (const SourceRef& source : source_refs) {
        EXPECT_FALSE(source.pool->isAllocated(source.block));
    }
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 2u);
    }

    result.load_ticket.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, TransferExceptionSettlesLoadAndRestoresCandidates) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    auto pausable_per_rank_transfer_engine =
        std::make_shared<PausablePerRankBlockTransferEngine>(environment->groups,
                                                             /*succeed=*/true,
                                                             /*pause_enabled=*/false,
                                                             /*throw_on_submit=*/1);
    PausableTransferReleaseGuard release_guard(pausable_per_rank_transfer_engine);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    std::vector<std::vector<GroupSetResource>> resources_before;
    resources_before.reserve(environment->keys.size());
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        resources_before.push_back(environment->resourcesForPathNode(path_index));
    }
    const CacheStats          candidates_before = environment->cache->getStats();
    const std::vector<size_t> host_free_before  = {environment->host_pools[0]->freeBlocksNum(),
                                                   environment->host_pools[1]->freeBlocksNum()};

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_FALSE(result.load_ticket->empty());

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        ASSERT_EQ(item.source_tier, GetParam());
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[item.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[item.group_set_id].get());
        for (const BlockIdxType block : item.source_blocks) {
            ASSERT_NE(block, NULL_BLOCK_IDX);
            EXPECT_EQ(source_pool->refCount(block), 2u);
            source_refs.push_back(SourceRef{source_pool, block});
        }

        item.target_device_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets, BlockRefType::REQUEST);
            item.target_device_blocks.push_back(targets.front());
            target_blocks.emplace_back(pool, targets.front());
        }
    }
    ASSERT_FALSE(source_refs.empty());
    ASSERT_FALSE(target_blocks.empty());

    pausable_per_rank_transfer_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(pausable_per_rank_transfer_engine->waitUntilEnteredFor(kRaceWaitTimeout));
    EXPECT_FALSE(context->done());
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 2u);
    }

    const size_t         submits_before_join = pausable_per_rank_transfer_engine->submitCount();
    BlockTreeMatchResult joined_result       = environment->cache->match(environment->keys);
    ASSERT_NE(joined_result.load_ticket, nullptr);
    ASSERT_FALSE(joined_result.load_ticket->empty());
    for (size_t item_index = 0; item_index < joined_result.load_ticket->itemCount(); ++item_index) {
        EXPECT_TRUE(joined_result.load_ticket->joinedLoad(item_index));
        const size_t group_set_id = joined_result.load_ticket->groupSetId(item_index);
        ASSERT_LT(group_set_id, environment->groups.size());
        const std::vector<BlockIdxType>& joined_targets     = joined_result.load_ticket->targetDeviceBlocks(item_index);
        const std::vector<DeviceBlockPoolPtr>& device_pools = environment->groups[group_set_id]->devicePools();
        ASSERT_EQ(joined_targets.size(), device_pools.size());
        for (size_t pool_index = 0; pool_index < joined_targets.size(); ++pool_index) {
            ASSERT_NE(device_pools[pool_index], nullptr);
            device_pools[pool_index]->incRef(joined_targets[pool_index], BlockRefType::REQUEST);
        }
    }
    std::shared_ptr<AsyncContext> joined_context = joined_result.load_ticket->commit();
    ASSERT_NE(joined_context, nullptr);
    EXPECT_FALSE(joined_context->done());
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), submits_before_join);
    environment->cache->releaseMatchedResources(joined_result.matched_resources);
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 3u);
    }

    pausable_per_rank_transfer_engine->release();
    environment->cache->waitForPendingTasks();

    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    ASSERT_TRUE(joined_context->done());
    EXPECT_FALSE(joined_context->success());
    EXPECT_FALSE(environment->cache->cancelLoad(context));
    EXPECT_FALSE(environment->cache->cancelLoad(joined_context));
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), submits_before_join);
    EXPECT_EQ(environment->host_pools[0]->freeBlocksNum(), host_free_before[0]);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), host_free_before[1]);
    if (GetParam() == Tier::HOST) {
        EXPECT_EQ(environment->cache->getStats().host_heap_total_size, candidates_before.host_heap_total_size);
    } else {
        EXPECT_EQ(environment->cache->getStats().disk_heap_total_size, candidates_before.disk_heap_total_size);
    }
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const std::vector<GroupSetResource> resources_after = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources_after.size(), resources_before[path_index].size());
        for (size_t group_id = 0; group_id < resources_after.size(); ++group_id) {
            EXPECT_EQ(resources_after[group_id].device_blocks, resources_before[path_index][group_id].device_blocks);
            EXPECT_EQ(resources_after[group_id].host_block, resources_before[path_index][group_id].host_block);
            EXPECT_EQ(resources_after[group_id].disk_slot, resources_before[path_index][group_id].disk_slot);
            EXPECT_EQ(resources_after[group_id].transfer_state, GroupSetTransferState::IDLE);
        }
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->refCount(source.block), 1u);
    }
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 2u);
    }
    environment->expectPayloads();

    BlockTreeMatchResult retry = environment->cache->match(environment->keys);
    ASSERT_NE(retry.load_ticket, nullptr);
    retry.load_ticket.reset();

    result.load_ticket.reset();
    joined_result.load_ticket.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DiskLoadDirectTransferExceptionRestoresSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    auto pausable_per_rank_transfer_engine = std::make_shared<PausablePerRankBlockTransferEngine>(
        environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::DISK);
    environment->expectPayloads();

    const std::vector<GroupSetResource> resources_before = environment->resourcesForPathNode(0);
    const std::vector<size_t>           host_free_before = {environment->host_pools[0]->freeBlocksNum(),
                                                            environment->host_pools[1]->freeBlocksNum()};
    BlockTreeMatchResult                result           = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_ticket, nullptr);
    ASSERT_FALSE(result.load_ticket->empty());
    const size_t disk_submit_count = result.load_ticket->items().size();
    ASSERT_GT(disk_submit_count, 0u);

    pausable_per_rank_transfer_engine->setThrowOnSubmit(1);

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        ASSERT_EQ(item.source_tier, Tier::DISK);
        ASSERT_EQ(item.source_blocks.size(), 1u);
        IBlockPool* source_pool = environment->disk_pools[item.group_set_id].get();
        EXPECT_EQ(source_pool->refCount(item.source_blocks.front()), 2u);
        source_refs.push_back({source_pool, item.source_blocks.front()});

        item.target_device_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(item.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks, BlockRefType::REQUEST);
            item.target_device_blocks.push_back(blocks.front());
            target_blocks.emplace_back(pool, blocks.front());
        }
    }

    pausable_per_rank_transfer_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    pausable_per_rank_transfer_engine->waitUntilEntered();
    pausable_per_rank_transfer_engine->release();
    environment->cache->waitForPendingTasks();

    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 1u);
    EXPECT_EQ(environment->host_pools[0]->freeBlocksNum(), host_free_before[0]);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), host_free_before[1]);
    const std::vector<GroupSetResource> resources_after = environment->resourcesForPathNode(0);
    ASSERT_EQ(resources_after.size(), resources_before.size());
    for (size_t group_id = 0; group_id < resources_after.size(); ++group_id) {
        EXPECT_EQ(resources_after[group_id].device_blocks, resources_before[group_id].device_blocks);
        EXPECT_EQ(resources_after[group_id].host_block, resources_before[group_id].host_block);
        EXPECT_EQ(resources_after[group_id].disk_slot, resources_before[group_id].disk_slot);
        EXPECT_EQ(resources_after[group_id].transfer_state, GroupSetTransferState::IDLE);
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->refCount(source.block), 1u);
    }
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    BlockTreeMatchResult retry = environment->cache->match(environment->keys);
    ASSERT_NE(retry.load_ticket, nullptr);
    retry.load_ticket.reset();
    result.load_ticket.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

class BlockTreeCacheLoadDisabledTest: public ::testing::TestWithParam<DisabledLoadTierLayout> {};

TEST_P(BlockTreeCacheLoadDisabledTest, LoadDisabled_DoesNotReportLowerTierHit) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.enable_load             = false;
    options.enable_reverse_eviction = false;
    auto environment                = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    // The M6 matrix is Load=false and Reverse=false. Assert the effective
    // config so a fixture default change cannot silently alter the matrix.
    ASSERT_FALSE(environment->cache->config().enable_load);
    ASSERT_FALSE(environment->cache->config().enable_reverse_eviction);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    if (GetParam() == DisabledLoadTierLayout::HOST_ONLY) {
        demoteTo(*environment, Tier::HOST);
    } else if (GetParam() == DisabledLoadTierLayout::DISK_ONLY) {
        demoteTo(*environment, Tier::DISK);
    } else {
        demoteTo(*environment, Tier::HOST);
        runSingleMaintenance(*environment, Tier::HOST, 0.125);
    }
    environment->scripted_per_rank_transfer_engine->clear();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    EXPECT_EQ(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_blocks, 0u);
    EXPECT_EQ(result.host_load_blocks, 0u);
    EXPECT_EQ(result.disk_load_blocks, 0u);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    if (GetParam() == DisabledLoadTierLayout::HOST_ONLY) {
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    } else if (GetParam() == DisabledLoadTierLayout::DISK_ONLY) {
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::DISK));
    } else {
        bool saw_host = false;
        bool saw_disk = false;
        for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
            for (const GroupSetResource& resource : environment->resourcesForPathNode(path_index)) {
                EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
                saw_host = saw_host || resource.hasTier(Tier::HOST);
                saw_disk = saw_disk || resource.hasTier(Tier::DISK);
            }
        }
        EXPECT_TRUE(saw_host);
        EXPECT_TRUE(saw_disk);
    }
    environment->expectPayloads();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(HostDiskAndMixed,
                         BlockTreeCacheLoadDisabledTest,
                         ::testing::Values(DisabledLoadTierLayout::HOST_ONLY,
                                           DisabledLoadTierLayout::DISK_ONLY,
                                           DisabledLoadTierLayout::HOST_AND_DISK),
                         disabledLoadTierLayoutName);

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
        const std::vector<GroupSetResource> resources = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources.size(), 2u);
        EXPECT_TRUE(resources[0].hasTier(Tier::DEVICE));
        if (path_index < 2) {
            EXPECT_TRUE(resources[1].hasTier(Tier::DEVICE));
        } else {
            EXPECT_TRUE(resources[1].hasTier(Tier::HOST));
            EXPECT_FALSE(resources[1].hasTier(Tier::DEVICE));
        }
    }

    environment->scripted_per_rank_transfer_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_blocks, 2u);
    ASSERT_EQ(result.matched_node->cache_key, environment->keys[1]);
    expectAggregatedReadyResult(*environment->cache, result, /*full_blocks=*/2, /*swa_blocks=*/2);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 0), 2u);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 1), 2u);
    for (const PendingLoadItem& item : result.load_ticket->items()) {
        if (item.group_set_id == 0) {
            EXPECT_EQ(item.source_tier, Tier::DEVICE);
            EXPECT_GE(item.path_index, 2u);
        } else {
            EXPECT_EQ(item.source_tier, Tier::HOST);
        }
    }
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    environment->releaseMatch(result);
    result.load_ticket.reset();
    environment->releaseRequestRefs();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadExplicitAbortImmediatelyRestoresCandidatesOnce) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    BlockTreeMatchResult result = makePartialReadyDeviceTicket(*environment);
    ASSERT_NE(result.load_ticket, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        if (item.source_tier != Tier::DEVICE) {
            continue;
        }
        const auto& device_pools = environment->groups.at(item.group_set_id)->devicePools();
        ASSERT_EQ(device_pools.size(), item.source_blocks.size());
        for (size_t member_group_id = 0; member_group_id < item.source_blocks.size(); ++member_group_id) {
            device_sources.emplace_back(device_pools[member_group_id], item.source_blocks[member_group_id]);
        }
    }
    ASSERT_EQ(device_sources.size(), 4u);

    environment->releaseRequestRefs();
    const size_t tree_nodes_before = environment->cache->getStats().tree_node_count;
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    environment->expectPayloads();
    const size_t device_candidates_before_abort = environment->cache->getStats().device_heap_total_size;

    std::shared_ptr<LoadTicket> ticket = std::move(result.load_ticket);
    ticket.reset();
    const size_t device_candidates_after_abort = environment->cache->getStats().device_heap_total_size;
    EXPECT_GT(device_candidates_after_abort, device_candidates_before_abort);
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    ticket.reset();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_after_abort);
    EXPECT_EQ(environment->cache->evictForGroup(0, 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    environment->releaseMatch(result);
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadAsyncCompletionRefreshesBeforeTerminalPublication) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    auto pausable_per_rank_transfer_engine = std::make_shared<PausablePerRankBlockTransferEngine>(
        environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    BlockTreeMatchResult result = makePartialReadyDeviceTicket(*environment);
    ASSERT_NE(result.load_ticket, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_target_blocks;
    for (LoadTicket::PendingLoadItem& item : result.load_ticket->items_) {
        item.target_device_blocks.clear();
        const auto& device_pools = environment->groups.at(item.group_set_id)->devicePools();
        if (item.source_tier == Tier::DEVICE) {
            ASSERT_EQ(device_pools.size(), item.source_blocks.size());
            item.target_device_blocks = item.source_blocks;
            for (size_t member_group_id = 0; member_group_id < item.source_blocks.size(); ++member_group_id) {
                device_sources.emplace_back(device_pools[member_group_id], item.source_blocks[member_group_id]);
            }
            continue;
        }

        for (const DeviceBlockPoolPtr& pool : device_pools) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets, BlockRefType::REQUEST);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            item.target_device_blocks.push_back(target);
            request_target_blocks.emplace_back(pool, target);
        }
        ASSERT_EQ(item.target_device_blocks.size(), device_pools.size());
        ASSERT_NE(item.node, nullptr);
    }
    ASSERT_EQ(device_sources.size(), 4u);
    ASSERT_EQ(request_target_blocks.size(), 2u);

    environment->releaseMatch(result);
    environment->releaseRequestRefs();
    const size_t tree_nodes_before        = environment->cache->getStats().tree_node_count;
    const size_t device_candidates_before = environment->cache->getStats().device_heap_total_size;

    pausable_per_rank_transfer_engine->enablePause();
    std::shared_ptr<AsyncContext> context = result.load_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(result.load_ticket->commit(), nullptr);
    pausable_per_rank_transfer_engine->waitUntilEntered();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_before);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    environment->expectPayloads();

    pausable_per_rank_transfer_engine->release();
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(environment->cache->cancelLoad(context));
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_before + 1u);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();
    for (const TransferDescriptor& descriptor : pausable_per_rank_transfer_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
    }
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 2u);

    EXPECT_EQ(environment->cache->evictForGroup(0, 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    result.load_ticket.reset();
    context->waitDone();
    EXPECT_TRUE(context->success());
    environment->cache->waitForPendingTasks();
    for (const auto& [pool, block] : request_target_blocks) {
        releaseDeviceBlocksAndNotify(*environment->cache, pool, {block}, BlockRefType::REQUEST);
    }
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

    auto find = environment->cache->tree()->findNode(environment->keys);
    ASSERT_EQ(find.size(), kPathLength);
    std::vector<BlockIdxType> swa_host_blocks;
    for (size_t path_index = 0; path_index < kPathLength; ++path_index) {
        GroupSetResource& swa_resource = find[path_index]->group_set_resources[1];
        ASSERT_TRUE(swa_resource.hasTier(Tier::DEVICE));
        const std::vector<BlockIdxType> old_device_blocks =
            swa_resource.getBlocks(Tier::DEVICE);
        const MultiNodeResource device_resource{
            1, Tier::DEVICE, {{find[path_index], old_device_blocks}}};
        environment->groups[1]->unmapDeviceBlocksFromTreeNode(device_resource);
        swa_resource.evictFromTier(Tier::DEVICE);
        environment->groups[1]->unreferenceBlocks(device_resource, BlockRefType::BLOCK_CACHE);
        if (path_index >= 2) {
            const BlockIdxType host_block =
                environment->groups[1]->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
            ASSERT_NE(host_block, NULL_BLOCK_IDX);
            swa_resource.setBlocks(Tier::HOST, {host_block});
            swa_host_blocks.push_back(host_block);
        }
    }

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_ticket, nullptr);
    EXPECT_EQ(result.load_ticket->logicalMatchedBlocks(), kPathLength);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 0), 4u);
    EXPECT_EQ(ticketItemCountForGroupSet(result.load_ticket, 1), 2u);
    EXPECT_EQ(result.load_blocks, 2u);
    EXPECT_EQ(result.host_load_blocks, 2u);
    EXPECT_EQ(result.disk_load_blocks, 0u);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submitCount(), 0u);

    for (const PendingLoadItem& item : result.load_ticket->items()) {
        if (item.group_set_id == 0) {
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

    result.load_ticket.reset();
    for (const auto& blocks : environment->request_blocks[0]) {
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
