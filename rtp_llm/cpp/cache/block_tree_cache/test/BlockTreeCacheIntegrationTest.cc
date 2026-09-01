#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"

namespace rtp_llm::block_tree_cache_test {
class LoadShutdownTestPeer {
public:
    static void setPendingTaskWaitObserver(BlockTreeCache& cache, const std::function<void()>& observer) {
        std::lock_guard<std::mutex> lock(cache.task_pool_->wait_mutex_);
        cache.task_pool_->pending_task_wait_observer_for_test_ = observer;
    }
};
}  // namespace rtp_llm::block_tree_cache_test

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

std::shared_ptr<LoadAsyncContext> getLoadContext(const BlockTreeMatchResult& result) {
    return std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
}

std::shared_ptr<LoadAsyncContext> takeLoadContext(BlockTreeMatchResult& result) {
    std::shared_ptr<LoadAsyncContext> context = getLoadContext(result);
    if (context != nullptr) {
        const auto& descs  = context->loadDescs();
        const auto& joined = context->joinedLoads();
        for (size_t desc_index = 0; desc_index < descs.size(); ++desc_index) {
            const TransferDescriptor& desc = descs[desc_index];
            if (joined[desc_index]) {
                result.matched_device_resources.push_back(
                    MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.target_blocks}}});
            } else if (desc.source_tier == Tier::DEVICE) {
                result.matched_device_resources.push_back(
                    MultiNodeResource{desc.group_set_id, Tier::DEVICE, {{desc.node, desc.source_blocks}}});
            }
        }
    }
    result.async_context.reset();
    return context;
}

size_t transferBatchCount(const std::vector<TransferDescriptor>& descriptors, const BlockTreeCacheConfig& config) {
    std::vector<std::pair<std::tuple<Tier, Tier, size_t>, size_t>> groups;
    for (const auto& descriptor : descriptors) {
        const auto key = std::make_tuple(descriptor.source_tier, descriptor.target_tier, descriptor.group_set_id);
        const auto group =
            std::find_if(groups.begin(), groups.end(), [&key](const auto& item) { return item.first == key; });
        if (group == groups.end()) {
            groups.emplace_back(key, 1);
        } else {
            ++group->second;
        }
    }

    size_t batch_count = 0;
    for (const auto& [key, descriptor_count] : groups) {
        const Tier source = std::get<0>(key);
        const Tier target = std::get<1>(key);
        const bool device_host_direction =
            (source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE);
        const size_t batch_limit = device_host_direction ? config.max_descriptors_per_transfer_batch :
                                                           config.max_descriptors_per_non_device_host_transfer_batch;
        batch_count += (descriptor_count + batch_limit - 1) / batch_limit;
    }
    return batch_count;
}

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

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        size_t submit_index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!pause_enabled_) {
                lock.unlock();
                return PerRankBlockTransferEngine::submit(descriptors);
            }
            ++submit_count_;
            submit_index = submit_count_;
            descriptors_.insert(descriptors_.end(), descriptors.begin(), descriptors.end());
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
        return PerRankBlockTransferEngine::submit(descriptors);
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

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        if (!throw_enabled_) {
            return PerRankBlockTransferEngine::submit(descriptors);
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
        auto full_group = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups = {full_group};
        cache_                          = makeBlockTreeCacheForTest(std::move(groups));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

constexpr size_t kPathLength = 4;
constexpr size_t kPoolSize   = 16;

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
    ASSERT_TRUE(environment.allResourcesAtTier(Tier::HOST));
    if (target_tier == Tier::DISK) {
        environment.demoteAll(Tier::HOST);
        ASSERT_TRUE(environment.allResourcesAtTier(Tier::DISK));
    }
}

void expectUnpublishedResult(const BlockTreeMatchResult& result) {
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
}

void expectAggregatedReadyResult(const BlockTreeCache&       cache,
                                 const BlockTreeMatchResult& result,
                                 size_t                      full_blocks,
                                 size_t                      swa_blocks) {
    EXPECT_EQ(cache.matchedBlocksForGroup(0, result.matched_device_resources).size(), full_blocks);
    EXPECT_EQ(cache.matchedBlocksForGroup(1, result.matched_device_resources).size(), full_blocks);
    EXPECT_EQ(cache.matchedBlocksForGroup(2, result.matched_device_resources).size(), swa_blocks);
    ASSERT_EQ(result.matched_device_resources.size(), 2u);
    EXPECT_EQ(result.matched_device_resources[0].group_set_id, 0);
    EXPECT_EQ(result.matched_device_resources[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_device_resources[0].node_blocks.size(), full_blocks);
    EXPECT_EQ(result.matched_device_resources[1].group_set_id, 1);
    EXPECT_EQ(result.matched_device_resources[1].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_device_resources[1].node_blocks.size(), swa_blocks);
}

void expectPlanningSourceRefCounts(const FullSWAEnvironment& environment, Tier tier) {
    for (size_t path_index = 0; path_index < environment.keys.size(); ++path_index) {
        const std::vector<GroupSetResource> resources = environment.resourcesForPathNode(path_index);
        ASSERT_EQ(resources.size(), 2u);
        for (size_t group_id = 0; group_id < resources.size(); ++group_id) {
            const BlockIdxType block =
                tier == Tier::HOST ? resources[group_id].host_block : resources[group_id].disk_block;
            const IBlockPool& pool          = tier == Tier::HOST ?
                                                  static_cast<const IBlockPool&>(*environment.host_pools[group_id]) :
                                                  static_cast<const IBlockPool&>(*environment.disk_pools[group_id]);
            const bool        in_swa_window = group_id == 0 || path_index + 2 >= environment.keys.size();
            const uint32_t    expected      = in_swa_window ? 2u : 1u;
            EXPECT_EQ(pool.treeRefCount(block), expected);
        }
    }
}

size_t contextDescCountForGroupSet(const std::shared_ptr<LoadAsyncContext>& context, size_t group_set_id) {
    if (context == nullptr) {
        return 0;
    }
    return static_cast<size_t>(std::count_if(
        context->loadDescs().begin(), context->loadDescs().end(), [group_set_id](const TransferDescriptor& desc) {
            return desc.group_set_id == group_set_id;
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
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment.cache, tier, ratio);
    environment.runMaintenance();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment.cache, tier, 0.0);
}

void demoteSwaSuffixKeepingFullDevice(FullSWAEnvironment& environment) {
    // REQUEST references no longer mask candidates. Temporarily make the FULL
    // resources non-evictable so two targeted SWA demotions create the partial
    // ready boundary exercised by these load tests.
    const std::vector<TreeNode*> path = environment.cache->tree()->findNode(environment.keys);
    EXPECT_EQ(path.size(), kPathLength);
    for (TreeNode* node : path) {
        node->group_set_resources[0].transfer_state = GroupSetTransferState::LOADING;
        BlockTreeCacheTestPeer::refreshCandidateForTest(*environment.cache, node, /*group_set_id=*/0);
    }
    const bool first_demoted =
        BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment.cache, /*group_set_id=*/1, Tier::DEVICE);
    BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment.cache);
    const bool second_demoted =
        BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment.cache, /*group_set_id=*/1, Tier::DEVICE);
    BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment.cache);
    for (TreeNode* node : path) {
        node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
        BlockTreeCacheTestPeer::refreshCandidateForTest(*environment.cache, node, /*group_set_id=*/0);
    }
    EXPECT_TRUE(first_demoted);
    EXPECT_TRUE(second_demoted);
}

BlockTreeMatchResult makePartialReadyDeviceContext(FullSWAEnvironment& environment) {
    environment.insertRequestPath();
    BlockTreeMatchResult prefix_hold = environment.cache->match({environment.keys[0], environment.keys[1]});
    EXPECT_EQ(prefix_hold.matched_device_blocks, 2u);

    demoteSwaSuffixKeepingFullDevice(environment);
    environment.releaseMatch(prefix_hold);

    environment.scripted_per_rank_transfer_engine->clear();
    BlockTreeMatchResult result = environment.cache->match(environment.keys);
    EXPECT_EQ(result.matched_device_blocks, 2u);
    const std::shared_ptr<LoadAsyncContext> context = getLoadContext(result);
    EXPECT_NE(context, nullptr);
    if (context != nullptr) {
        EXPECT_EQ(contextDescCountForGroupSet(context, 0), 2u);
        EXPECT_EQ(contextDescCountForGroupSet(context, 1), 2u);
    }
    return result;
}

class OneShotWatermarkTestPeer {
public:
    static void runDevicePass(BlockTreeCache& cache, double ratio) {
        ASSERT_EQ(cache.task_pool_->pending_tasks_.load(), 0);
        {
            std::lock_guard<std::mutex> lock(cache.mutex_);
            cache.config_.watermark_host.ratio   = 0.0;
            cache.config_.watermark_disk.ratio   = 0.0;
            cache.config_.watermark_device.ratio = ratio;
            cache.checkWatermark();
            cache.config_.watermark_device.ratio = 0.0;
        }
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(cache);
        EXPECT_EQ(cache.task_pool_->pending_tasks_.load(), 0);
    }
};

TEST_F(BlockTreeCacheIntegrationTest, HostDiskOnlyLifecycle) {
    auto host_pool = makeHostPool(256, 8);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());

    auto device_pool = makeDevicePool({{256, 0}}, 8, "watermark_host_to_disk");
    auto full     = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    auto topology = block_transfer_engine_test::makeTestTopology(
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 256)});
    full->initialize(0, topology, {0});
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = false;
    cfg.enable_host_cache   = true;
    cfg.enable_disk_cache   = true;

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
    std::shared_ptr<LoadAsyncContext> host_context = takeLoadContext(host_match);
    ASSERT_NE(host_context, nullptr);
    host_context.reset();
    const CandidateMeta matched_meta = before.back()->group_set_resources[0].candidate_meta;
    EXPECT_GT(matched_meta.last_access_seq, before_meta.last_access_seq);
    EXPECT_EQ(matched_meta.admission_seq, before_meta.admission_seq);
    EXPECT_EQ(matched_meta.hit_count, before_meta.hit_count + 1);
    EXPECT_GE(matched_meta.last_access_time_us, before_meta.last_access_time_us);

    scripted_copy->enqueue(/*success=*/false);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto after_failure = cache->tree()->findNode({100});
    ASSERT_FALSE(after_failure.empty());
    const auto& failed_resource = after_failure.back()->group_set_resources[0];
    EXPECT_EQ(failed_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(failed_resource.host_block, host_block);
    EXPECT_FALSE(failed_resource.hasTier(Tier::DISK));
    EXPECT_TRUE(host_pool->isAllocated(host_block));
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 1u);
    EXPECT_EQ(failed_resource.candidate_meta.last_access_seq, matched_meta.last_access_seq);
    EXPECT_EQ(failed_resource.candidate_meta.admission_seq, matched_meta.admission_seq);
    EXPECT_EQ(failed_resource.candidate_meta.hit_count, matched_meta.hit_count);
    EXPECT_EQ(failed_resource.candidate_meta.last_access_time_us, matched_meta.last_access_time_us);
    const auto snapshot_after_failure = cache->getKeySnapshot(/*limit=*/8);
    EXPECT_EQ(snapshot_after_failure.version, snapshot_before.version);
    EXPECT_EQ(snapshot_after_failure.keys, snapshot_before.keys);

    scripted_copy->clear();
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    const auto& resource = find.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_TRUE(resource.hasTier(Tier::DISK));
    EXPECT_NE(resource.disk_block, NULL_BLOCK_IDX);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_TRUE(disk_pool->isAllocated(resource.disk_block));
    EXPECT_EQ(disk_pool->treeRefCount(resource.disk_block), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);

    const auto snapshot_after_success = cache->getKeySnapshot(/*limit=*/8);
    EXPECT_GT(snapshot_after_success.version, snapshot_after_failure.version);
    EXPECT_EQ(snapshot_after_success.keys, snapshot_before.keys);
    BlockTreeMatchResult disk_match = cache->match({100});
    expectUnpublishedResult(disk_match);
    std::shared_ptr<LoadAsyncContext> disk_context = takeLoadContext(disk_match);
    ASSERT_NE(disk_context, nullptr);
    disk_context.reset();
    EXPECT_EQ(scripted_copy->submittedDescriptorCount(), 1u);

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
    cache.reset();
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 8u);
}

TEST_F(BlockTreeCacheIntegrationTest, OneShotPrimaryFailureSkipsCascadeAndRetriesOnce) {
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
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/false);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> first_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(first_descriptors.size(), 1u);
    EXPECT_EQ(first_descriptors[0].group_set_id, 0);
    EXPECT_EQ(first_descriptors[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(first_descriptors[0].target_tier, Tier::HOST);
    EXPECT_EQ(first_descriptors[0].source_blocks, full_sources);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 1u);

    const std::vector<GroupSetResource> after_failure = environment->resourcesForPathNode(0);
    ASSERT_EQ(after_failure.size(), 2u);
    EXPECT_EQ(after_failure[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(after_failure[0].device_blocks, full_sources);
    EXPECT_FALSE(after_failure[0].hasTier(Tier::HOST));
    EXPECT_EQ(after_failure[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(after_failure[1].device_blocks, (std::vector<BlockIdxType>{swa_source}));
    EXPECT_FALSE(after_failure[1].hasTier(Tier::HOST));
    EXPECT_EQ(environment->device_pools[2]->refCount(swa_source), 1u);
    EXPECT_EQ(environment->host_pools[0]->freeBlocksNum(), 16u);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), 16u);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, 2u);
    environment->expectPoolFreeCounts({15, 15, 15}, {16, 16}, {16, 16});
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/true);
    OneShotWatermarkTestPeer::runDevicePass(*environment->cache, 0.01);

    const std::vector<TransferDescriptor> retry_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(retry_descriptors.size(), 2u);
    EXPECT_EQ(retry_descriptors[0].group_set_id, 0);
    EXPECT_EQ(retry_descriptors[0].source_blocks, full_sources);
    EXPECT_EQ(retry_descriptors[1].group_set_id, 1);
    EXPECT_EQ(retry_descriptors[1].source_blocks, (std::vector<BlockIdxType>{swa_source}));
    for (const TransferDescriptor& descriptor : retry_descriptors) {
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
    }
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedBatchCount(), 2u);

    const std::vector<GroupSetResource> after_retry = environment->resourcesForPathNode(0);
    ASSERT_EQ(after_retry.size(), 2u);
    EXPECT_EQ(after_retry[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(after_retry[0].hasTier(Tier::HOST));
    EXPECT_EQ(after_retry[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(after_retry[1].hasTier(Tier::DEVICE));
    EXPECT_TRUE(after_retry[1].hasTier(Tier::HOST));
    EXPECT_FALSE(environment->device_pools[2]->isAllocated(swa_source));
    EXPECT_EQ(environment->host_pools[1]->treeRefCount(after_retry[1].host_block), 1u);
    environment->expectPoolFreeCounts({16, 16, 16}, {15, 15}, {16, 16});
    environment->expectPayloads();

    environment->scripted_per_rank_transfer_engine->clear();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, UncommittedLoadContextReleasesSourceReferences) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::unique_ptr<FullSWAEnvironment> environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);

    BlockTreeMatchResult              result       = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_FALSE(load_context->empty());
    expectPlanningSourceRefCounts(*environment, Tier::HOST);

    load_context.reset();
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const std::vector<GroupSetResource> resources = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources.size(), 2u);
        for (size_t group_id = 0; group_id < resources.size(); ++group_id) {
            EXPECT_EQ(environment->host_pools[group_id]->treeRefCount(resources[group_id].host_block), 1u);
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

        auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
        auto topology = block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(
            defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, kBlockBytes)});
        full->initialize(0, topology, {0});
        std::vector<GroupSetPtr> groups = {full};
        BlockTreeCacheConfig     config;
        config.enable_device_cache = true;
        config.enable_host_cache   = true;
        config.enable_disk_cache   = true;
        auto cache                 = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
        ASSERT_NE(cache, nullptr);

        auto pausable_per_rank_transfer_engine =
            std::make_shared<PausablePerRankBlockTransferEngine>(std::vector<GroupSetPtr>{full}, copy_success);
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_per_rank_transfer_engine);

        const BlockIdxType source_block = full->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        std::vector<std::vector<GroupSetResource>> source_resources(1, std::vector<GroupSetResource>(1));
        source_resources[0][0].disk_block = source_block;
        ASSERT_TRUE(insertGroupSetResources(*cache, {100}, source_resources));

        BlockTreeMatchResult              result  = cache->match({100});
        std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->loadDescs().size(), 1u);
        EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::DISK);
        EXPECT_EQ(context->loadDescs()[0].source_blocks, (std::vector<BlockIdxType>{source_block}));
        EXPECT_EQ(disk_pool->treeRefCount(source_block), 2u);

        const BlockIdList request_targets = device_pool->malloc(1).value();
        ASSERT_EQ(request_targets.size(), 1u);
        device_pool->incRef(request_targets);
        const BlockIdxType target_block = request_targets.front();
        EXPECT_EQ(device_pool->refCount(target_block), 1u);
        context->setTargetBlocks(0, {target_block});

        ASSERT_TRUE(context->commit());
        pausable_per_rank_transfer_engine->waitUntilEntered();
        EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 1u);
        EXPECT_EQ(disk_pool->treeRefCount(source_block), 2u);
        // The request and the tree umbrella keep two outer references while LOAD protects the target internally.
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(device_pool->treeRefCount(target_block), 1u);
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before - 1);
        EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
        EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);
        EXPECT_FALSE(context->commit());

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
        EXPECT_EQ(descriptors_after_wait[0].singleBlockAt(Tier::DISK), source_block);
        EXPECT_EQ(descriptors_after_wait[0].target_blocks, (std::vector<BlockIdxType>{target_block}));
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        EXPECT_TRUE(disk_pool->isAllocated(source_block));
        EXPECT_EQ(device_pool->refCount(target_block), 2u);
        EXPECT_EQ(disk_pool->treeRefCount(source_block), 2u);
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

        EXPECT_FALSE(context->commit());
        context.reset();
        EXPECT_TRUE(device_pool->isAllocated(target_block));
        EXPECT_EQ(device_pool->refCount(target_block), 1u);

        device_pool->decRef(request_targets);
        EXPECT_FALSE(device_pool->isAllocated(target_block));
        EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    }
}

TEST_F(BlockTreeCacheIntegrationTest, DirectDropDetachesInFlightDemotionAndDiscardsItsTarget) {
    auto device_pool = makeStructuralDevicePool(0);
    auto host_pool   = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/4);
    auto disk_pool   = makeDiskPool(
        /*payload_bytes=*/1, /*usable_count=*/4, std::make_unique<MemoryDiskBlockIO>());
    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    config.enable_disk_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full}, config);
    ASSERT_NE(cache, nullptr);

    auto pausable_copy = std::make_shared<PausablePerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{full}, /*succeed=*/true, /*pause_enabled=*/false);
    PausableTransferReleaseGuard release_guard(pausable_copy);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_copy);

    const BlockIdxType host_source = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_source));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].host_block    = host_source;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200}, resources));

    pausable_copy->enablePause();
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*cache, 0, Tier::HOST));
    ASSERT_TRUE(pausable_copy->waitUntilEnteredFor(kRaceWaitTimeout));
    const std::vector<TransferDescriptor> descriptors = pausable_copy->descriptors();
    ASSERT_EQ(descriptors.size(), 1u);
    ASSERT_EQ(descriptors[0].source_tier, Tier::HOST);
    ASSERT_EQ(descriptors[0].target_tier, Tier::DISK);
    const BlockIdxType disk_target = descriptors[0].singleBlockAt(Tier::DISK);

    auto path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);
    EXPECT_TRUE(path[0]->group_set_resources[0].is_empty());
    EXPECT_TRUE(path[1]->group_set_resources[0].transfer_detached);
    EXPECT_TRUE(host_pool->isAllocated(host_source));
    EXPECT_TRUE(disk_pool->isAllocated(disk_target));

    pausable_copy->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    EXPECT_FALSE(host_pool->isAllocated(host_source));
    EXPECT_FALSE(disk_pool->isAllocated(disk_target));
    EXPECT_TRUE(cache->tree()->findNode({100, 200}).empty());
}

TEST_F(BlockTreeCacheIntegrationTest, DirectDropDetachesPendingLoadAndRejectsCommit) {
    auto device_pool = makeStructuralDevicePool(0);
    auto host_pool   = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/4);
    auto full        = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full}, config);
    ASSERT_NE(cache, nullptr);

    const BlockIdxType host_source = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_source));
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].host_block    = host_source;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200}, resources));

    BlockTreeMatchResult              result  = cache->match({100, 200});
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);

    auto path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::LOAD_PENDING);
    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);
    EXPECT_TRUE(path[1]->group_set_resources[0].transfer_detached);
    EXPECT_TRUE(host_pool->isAllocated(host_source));

    const BlockIdList target_blocks{20};
    device_pool->incRef(target_blocks);
    context->setTargetBlocks(0, target_blocks);
    EXPECT_FALSE(context->commit());
    EXPECT_TRUE(context->done());

    EXPECT_FALSE(host_pool->isAllocated(host_source));
    path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_TRUE(path[0]->group_set_resources[0].is_empty());
    EXPECT_TRUE(path[1]->group_set_resources[0].is_empty());
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(path[1]->group_set_resources[0].transfer_detached);

    context.reset();
    releaseDeviceBlocks(*cache, device_pool, target_blocks);
}

TEST_F(BlockTreeCacheIntegrationTest, DirectDropDetachesInFlightLoadAndDiscardsItsTarget) {
    auto device_pool = makeStructuralDevicePool(0);
    auto host_pool   = makeHostPool(/*payload_bytes=*/1, /*usable_count=*/4);
    auto full        = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);

    BlockTreeCacheConfig config;
    config.enable_host_cache = true;
    auto cache               = makeBlockTreeCacheForTest(std::vector<GroupSetPtr>{full}, config);
    ASSERT_NE(cache, nullptr);

    auto pausable_copy = std::make_shared<PausablePerRankBlockTransferEngine>(
        std::vector<GroupSetPtr>{full}, /*succeed=*/false, /*pause_enabled=*/false);
    PausableTransferReleaseGuard release_guard(pausable_copy);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_copy);

    const size_t       device_free_before = device_pool->freeBlocksNum();
    const size_t       host_free_before   = host_pool->freeBlocksNum();
    const BlockIdList  prefix_blocks{10};
    const BlockIdxType host_source = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_source));

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = prefix_blocks;
    resources[1][0].host_block    = host_source;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200}, resources));

    BlockTreeMatchResult              result  = cache->match({100, 200});
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    const BlockIdList target_blocks{20};
    device_pool->incRef(target_blocks);
    context->setTargetBlocks(0, target_blocks);

    pausable_copy->enablePause();
    ASSERT_TRUE(context->commit());
    ASSERT_TRUE(pausable_copy->waitUntilEnteredFor(kRaceWaitTimeout));
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);

    auto path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(cache->evictForGroup(/*group_id=*/0, /*num_blocks=*/1), 1);
    EXPECT_TRUE(path[0]->group_set_resources[0].is_empty());
    EXPECT_TRUE(path[1]->group_set_resources[0].transfer_detached);
    EXPECT_TRUE(host_pool->isAllocated(host_source));

    pausable_copy->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    context->waitDone();

    EXPECT_FALSE(context->success());
    EXPECT_FALSE(host_pool->isAllocated(host_source));
    path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_TRUE(path[0]->group_set_resources[0].is_empty());
    EXPECT_TRUE(path[1]->group_set_resources[0].is_empty());
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(path[1]->group_set_resources[0].transfer_detached);
    EXPECT_EQ(device_pool->refCount(target_blocks.front()), 1u);

    context.reset();
    releaseDeviceBlocks(*cache, device_pool, target_blocks);
    EXPECT_FALSE(device_pool->isAllocated(prefix_blocks.front()));
    EXPECT_FALSE(device_pool->isAllocated(target_blocks.front()));
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before + 2);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
}

TEST_F(BlockTreeCacheIntegrationTest, EvictorDemotesRequestReferencedBlock) {
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
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {12, 12}, {16, 16});

    environment->releaseRequestRefs();
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
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
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
    EXPECT_GT(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    const std::vector<TransferDescriptor> retry_descriptors =
        environment->scripted_per_rank_transfer_engine->descriptors();
    for (const TransferDescriptor& descriptor : retry_descriptors) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    ASSERT_FALSE(failure_descriptors.empty());
    ASSERT_FALSE(retry_descriptors.empty());
    const TransferDescriptor& failed = failure_descriptors.front();
    EXPECT_NE(std::find_if(retry_descriptors.begin(),
                           retry_descriptors.end(),
                           [&failed](const TransferDescriptor& retried) {
                               return retried.group_set_id == failed.group_set_id
                                      && retried.source_blocks == failed.source_blocks;
                           }),
              retry_descriptors.end());
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

        BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment->cache, Tier::DEVICE, 0.01);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment->cache, Tier::DEVICE, 0.0);
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

        // Racing match while the node is DEMOTING returns no reusable prefix or context.
        BlockTreeMatchResult during = environment->cache->match(environment->keys);
        EXPECT_EQ(during.matched_device_blocks, 0u);
        EXPECT_EQ(during.async_context, nullptr);
        EXPECT_TRUE(during.matched_device_resources.empty());
        for (const auto& [pool, block] : device_sources) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 1u);
        }

        pausable_copy->release();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::HOST));
        for (const auto& [pool, block] : device_sources) {
            EXPECT_FALSE(pool->isAllocated(block));  // source freed exactly once on commit
        }

        // The node is now Host-resident: a fresh match surfaces it only as a Host
        // load source (no device-ready prefix yet).
        BlockTreeMatchResult after = environment->cache->match(environment->keys);
        EXPECT_EQ(after.matched_device_blocks, 0u);
        std::shared_ptr<LoadAsyncContext> after_context = takeLoadContext(after);
        ASSERT_NE(after_context, nullptr);
        for (const TransferDescriptor& desc : after_context->loadDescs()) {
            EXPECT_EQ(desc.source_tier, Tier::HOST);
        }
        after_context.reset();
        block_tree_cache_test::releaseRequestRefsForTest(*environment->cache, after.matched_device_resources);

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

        BlockTreeMatchResult              first   = environment->cache->match(environment->keys);
        std::shared_ptr<LoadAsyncContext> context = takeLoadContext(first);
        ASSERT_NE(context, nullptr);
        std::vector<std::pair<IBlockPool*, BlockIdxType>>        host_sources;
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
        for (TransferDescriptor& desc : context->load_descs_) {
            ASSERT_EQ(desc.source_tier, Tier::HOST);
            ASSERT_EQ(desc.source_blocks.size(), 1u);
            host_sources.emplace_back(environment->host_pools[static_cast<size_t>(desc.group_set_id)].get(),
                                      desc.source_blocks.front());
            desc.target_blocks.clear();
            for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
                BlockIdList blocks = pool->malloc(1).value();
                ASSERT_EQ(blocks.size(), 1u);
                pool->incRef(blocks);
                desc.target_blocks.push_back(blocks.front());
                request_targets.emplace_back(pool, blocks.front());
            }
        }

        pausable_copy->enablePause();
        ASSERT_TRUE(context->commit());
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
            EXPECT_EQ(pool->treeRefCount(block), 2u);  // CACHE + first LOAD source hold
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 2u);  // request hold + in-flight target hold
        }

        const size_t         submits_before_join = pausable_copy->submitCount();
        BlockTreeMatchResult second              = environment->cache->match(environment->keys);
        EXPECT_EQ(second.matched_device_blocks, 0u);
        std::shared_ptr<LoadAsyncContext> joined_context = takeLoadContext(second);
        ASSERT_NE(joined_context, nullptr);
        EXPECT_EQ(second.matched_device_resources.size(), joined_context->loadDescs().size());
        ASSERT_EQ(joined_context->loadDescs().size(), joined_context->joinedLoads().size());
        for (size_t desc_index = 0; desc_index < joined_context->loadDescs().size(); ++desc_index) {
            EXPECT_TRUE(joined_context->joinedLoads()[desc_index]);
            const TransferDescriptor&        desc           = joined_context->loadDescs()[desc_index];
            const size_t                     group_set_id   = desc.group_set_id;
            const std::vector<BlockIdxType>& joined_targets = desc.target_blocks;
            ASSERT_EQ(joined_targets.size(), environment->groups[group_set_id]->devicePools().size());
            for (size_t pool_index = 0; pool_index < joined_targets.size(); ++pool_index) {
                ASSERT_NE(environment->groups[group_set_id]->devicePools()[pool_index], nullptr);
            }
        }
        ASSERT_TRUE(joined_context->commit());
        EXPECT_FALSE(joined_context->done());
        EXPECT_EQ(pausable_copy->submitCount(), submits_before_join);
        for (const auto& [pool, block] : host_sources) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->treeRefCount(block), 2u);
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_TRUE(pool->isAllocated(block));
            EXPECT_EQ(pool->refCount(block), 3u);
        }
        BlockTreeMatchResult              abandoned         = environment->cache->match(environment->keys);
        std::shared_ptr<LoadAsyncContext> abandoned_context = takeLoadContext(abandoned);
        ASSERT_NE(abandoned_context, nullptr);
        for (size_t desc_index = 0; desc_index < abandoned_context->loadDescs().size(); ++desc_index) {
            EXPECT_TRUE(abandoned_context->joinedLoads()[desc_index]);
        }
        abandoned_context.reset();
        for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target : request_targets) {
            EXPECT_TRUE(target.first->isAllocated(target.second));
            EXPECT_EQ(target.first->refCount(target.second), 4u);
        }
        environment->releaseMatch(abandoned);
        for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target : request_targets) {
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
        EXPECT_EQ(after.matched_device_blocks, 1u);
        EXPECT_EQ(after.async_context, nullptr);
        block_tree_cache_test::releaseRequestRefsForTest(*environment->cache, after.matched_device_resources);

        context.reset();
        joined_context.reset();
        environment->releaseMatch(second);
        for (const auto& [pool, block] : request_targets) {
            releaseDeviceBlocks(*environment->cache, pool, {block});
        }
        environment->reclaimAll();
        environment->expectFullyReclaimed();
    }
}

TEST_F(BlockTreeCacheIntegrationTest, MixedRequestTierJoinAlwaysPromotesWhenAnyParticipantEnablesDevice) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    for (const bool device_enabled_leader : {false, true}) {
        SCOPED_TRACE(device_enabled_leader ? "device-enabled leader" : "device-disabled leader");
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

        const BlockTreeMatchPolicy  device_enabled_policy{/*enable_device=*/true,
                                                         /*enable_host=*/true,
                                                         /*enable_disk=*/false,
                                                         /*enable_remote=*/false};
        const BlockTreeMatchPolicy  host_only_policy{/*enable_device=*/false,
                                                    /*enable_host=*/true,
                                                    /*enable_disk=*/false,
                                                    /*enable_remote=*/false};
        const BlockTreeMatchPolicy& leader_policy = device_enabled_leader ? device_enabled_policy : host_only_policy;
        const BlockTreeMatchPolicy& joiner_policy = device_enabled_leader ? host_only_policy : device_enabled_policy;

        BlockTreeMatchResult              first   = environment->cache->match(environment->keys, leader_policy);
        std::shared_ptr<LoadAsyncContext> context = takeLoadContext(first);
        ASSERT_NE(context, nullptr);
        std::vector<std::pair<IBlockPool*, BlockIdxType>>        host_sources;
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
        for (size_t desc_index = 0; desc_index < context->load_descs_.size(); ++desc_index) {
            TransferDescriptor& desc = context->load_descs_[desc_index];
            EXPECT_EQ(desc.source_tier, Tier::HOST);
            EXPECT_EQ(desc.install_target_in_cache, device_enabled_leader);
            ASSERT_EQ(desc.source_blocks.size(), 1u);
            host_sources.emplace_back(environment->host_pools[desc.group_set_id].get(), desc.source_blocks.front());
            desc.target_blocks.clear();
            for (const DeviceBlockPoolPtr& pool : environment->groups[desc.group_set_id]->devicePools()) {
                const BlockIdList blocks = pool->malloc(1).value();
                ASSERT_EQ(blocks.size(), 1u);
                pool->incRef(blocks);
                desc.target_blocks.push_back(blocks.front());
                request_targets.emplace_back(pool, blocks.front());
            }
        }

        pausable_copy->enablePause();
        ASSERT_TRUE(context->commit());
        ASSERT_TRUE(pausable_copy->waitUntilEnteredFor(kRaceWaitTimeout));

        BlockTreeMatchResult              second         = environment->cache->match(environment->keys, joiner_policy);
        std::shared_ptr<LoadAsyncContext> joined_context = takeLoadContext(second);
        ASSERT_NE(joined_context, nullptr);
        ASSERT_EQ(joined_context->loadDescs().size(), joined_context->joinedLoads().size());
        for (size_t desc_index = 0; desc_index < joined_context->loadDescs().size(); ++desc_index) {
            EXPECT_TRUE(joined_context->joinedLoads()[desc_index]);
            EXPECT_EQ(joined_context->loadDescs()[desc_index].install_target_in_cache, !device_enabled_leader);
        }
        ASSERT_TRUE(joined_context->commit());

        pausable_copy->release();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
        ASSERT_TRUE(context->done());
        ASSERT_TRUE(context->success());
        ASSERT_TRUE(joined_context->done());
        ASSERT_TRUE(joined_context->success());
        EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
        for (const auto& [pool, block] : host_sources) {
            EXPECT_FALSE(pool->isAllocated(block));
        }
        for (const auto& [pool, block] : request_targets) {
            EXPECT_EQ(pool->refCount(block), 3u);  // tree + leader request + joined request
        }

        context.reset();
        joined_context.reset();
        environment->releaseMatch(second);
        for (const auto& [pool, block] : request_targets) {
            releaseDeviceBlocks(*environment->cache, pool, {block});
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
    options.path_length = 1;
    options.enable_disk = true;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    // Start from the lower-priority SWA group. A failed primary copy skips the
    // reverse-selected FULL sibling and rolls back both resources.
    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(/*success=*/false);
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE));
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    const auto failed_descriptors = environment->scripted_per_rank_transfer_engine->descriptors();
    ASSERT_EQ(failed_descriptors.size(), 1u);
    EXPECT_EQ(failed_descriptors[0].group_set_id, 1);
    for (const TransferDescriptor& descriptor : failed_descriptors) {
        EXPECT_EQ(descriptor.source_tier, Tier::DEVICE);
        EXPECT_EQ(descriptor.target_tier, Tier::HOST);
    }

    environment->scripted_per_rank_transfer_engine->clear();
    ASSERT_TRUE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE));
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
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
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
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
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (TransferDescriptor& desc : context->load_descs_) {
        ASSERT_EQ(desc.source_tier, Tier::DISK);
        desc.target_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            desc.target_blocks.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
    }
    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    environment->expectPayloads();

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    ASSERT_EQ(rematch.matched_device_blocks, 1u);
    expectAggregatedReadyResult(*environment->cache, rematch, /*full_blocks=*/1, /*swa_blocks=*/1);
    environment->releaseMatch(rematch);

    context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, EvictionExplicitNoneCascadesAtLeafWithoutCopy) {
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
    EXPECT_TRUE(
        BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE, /*force_drop=*/true));
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    auto result = environment->cache->tree()->findNode(environment->keys);
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
}

TEST_F(BlockTreeCacheIntegrationTest, DiskLoadRequestOnlyKeepsDiskResidency) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t payload_bytes = 256;
    auto             device_pool   = makeDevicePool({{payload_bytes, 0}}, 4, "request_only_load_device");
    auto             disk_pool     = makeDiskPool(payload_bytes, 4, std::make_unique<MemoryDiskBlockIO>());
    auto topology = block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(
        defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, payload_bytes)});
    auto group    = block_transfer_engine_test::makeTestGroupSet(0, topology, {0}, {device_pool}, nullptr, disk_pool);

    BlockTreeCacheConfig config;
    config.enable_device_cache      = false;
    config.enable_host_cache        = false;
    config.enable_disk_cache        = true;
    std::vector<GroupSetPtr> groups = {group};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    const size_t       disk_free_before = disk_pool->freeBlocksNum();
    const BlockIdxType source_block     = group->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source_block));
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].disk_block = source_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));

    BlockTreeMatchResult              result  = cache->match({100});
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::DISK);
    EXPECT_EQ(context->loadDescs()[0].source_blocks, (std::vector<BlockIdxType>{source_block}));

    const size_t      device_free_before = device_pool->freeBlocksNum();
    const BlockIdList request_targets    = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets);
    context->load_descs_[0].target_blocks = request_targets;

    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->success());
    auto find_result = cache->tree()->findNode({100});
    ASSERT_FALSE(find_result.empty());
    const GroupSetResource& resource = find_result.back()->group_set_resources[0];
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.disk_block, source_block);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_EQ(disk_pool->treeRefCount(source_block), 1u);
    EXPECT_EQ(device_pool->refCount(request_targets.front()), 1u);

    BlockTreeMatchResult              rematch         = cache->match({100});
    std::shared_ptr<LoadAsyncContext> rematch_context = takeLoadContext(rematch);
    ASSERT_NE(rematch_context, nullptr);
    for (const TransferDescriptor& desc : rematch_context->loadDescs()) {
        EXPECT_EQ(desc.source_tier, Tier::DISK);
    }
    rematch_context.reset();
    context.reset();
    device_pool->decRef(request_targets);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    cache.reset();
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
}

class BlockTreeCacheLowerTierTest: public ::testing::TestWithParam<Tier> {};

TEST_P(BlockTreeCacheLowerTierTest, FullSWA_MatchLowerTierOnlyReturnsContextWithoutPublishing) {
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
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    EXPECT_FALSE(context->empty());
    EXPECT_EQ(context->matchedBlocks(), kPathLength);
    EXPECT_EQ(context->matchedBlocks(GetParam()), kPathLength);
    EXPECT_EQ(contextDescCountForGroupSet(context, 0), 4u);
    EXPECT_EQ(contextDescCountForGroupSet(context, 1), 2u);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    expectPlanningSourceRefCounts(*environment, GetParam());
    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (TransferDescriptor& desc : context->load_descs_) {
        desc.target_blocks.clear();
        const auto& device_pools = environment->groups.at(desc.group_set_id)->devicePools();
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            desc.target_blocks.push_back(target);
            request_targets.emplace_back(pool, target);
        }
        ASSERT_EQ(desc.target_blocks.size(), device_pools.size());
        ASSERT_NE(desc.node, nullptr);
    }

    ASSERT_TRUE(context->commit());
    EXPECT_FALSE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    expectUnpublishedResult(result);

    const size_t submits_after_commit = environment->scripted_per_rank_transfer_engine->submittedDescriptorCount();
    EXPECT_GT(submits_after_commit, 0u);
    EXPECT_FALSE(context->commit());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), submits_after_commit);

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    EXPECT_EQ(rematch.matched_device_blocks, kPathLength);
    expectAggregatedReadyResult(*environment->cache, rematch, /*full_blocks=*/4, /*swa_blocks=*/2);
    environment->expectPayloads();
    environment->releaseMatch(rematch);

    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 14}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 16}, {16, 14});
    }
    environment->reclaimAll();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, TransferExceptionSettlesContextAndReleasesAllWorkerHolds) {
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

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_FALSE(context->empty());

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < context->loadDescs().size(); ++desc_index) {
        const size_t                           group_set_id = context->loadDescs()[desc_index].group_set_id;
        const std::vector<DeviceBlockPoolPtr>& device_pools = environment->groups.at(group_set_id)->devicePools();
        std::vector<BlockIdxType>              desc_targets;
        desc_targets.reserve(device_pools.size());
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            const BlockIdxType block = blocks.front();
            pool->incRef(block);
            desc_targets.push_back(block);
            request_targets.emplace_back(pool, block);
        }
        context->setTargetBlocks(desc_index, std::move(desc_targets));
    }

    ASSERT_TRUE(context->commit());
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);

    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    ASSERT_TRUE(environment->allResourcesAtTier(GetParam()));
    for (const GroupSetResource& resource : environment->resourcesForPathNode(0)) {
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (size_t desc_index = 0; desc_index < context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& desc         = context->loadDescs()[desc_index];
        const size_t              group_set_id = desc.group_set_id;
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        const BlockIdxType source_block = desc.source_blocks.front();
        const IBlockPool&  source_pool  = GetParam() == Tier::HOST ?
                                              static_cast<const IBlockPool&>(*environment->host_pools.at(group_set_id)) :
                                              static_cast<const IBlockPool&>(*environment->disk_pools.at(group_set_id));
        EXPECT_EQ(source_pool.treeRefCount(source_block), 1u);
    }
    for (size_t pool_index = 0; pool_index < environment->host_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->host_pools[pool_index]->freeBlocksNum(), host_free_before[pool_index]);
    }
    for (size_t pool_index = 0; pool_index < environment->disk_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->disk_pools[pool_index]->freeBlocksNum(), disk_free_before[pool_index]);
    }

    for (const auto& [pool, block] : request_targets) {
        EXPECT_EQ(pool->refCount(block), 1u);
        pool->decRef(block);
    }
    for (size_t pool_index = 0; pool_index < environment->device_pools.size(); ++pool_index) {
        EXPECT_EQ(environment->device_pools[pool_index]->freeBlocksNum(), device_free_before[pool_index]);
    }
    EXPECT_NO_THROW(environment->cache.reset());
}

TEST_P(BlockTreeCacheLowerTierTest, SettlementStateMismatchRollsBackWholeBatch) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length                             = 1;
    options.usable_device_blocks                    = 4;
    options.usable_host_blocks                      = 4;
    options.usable_disk_blocks                      = 4;
    std::unique_ptr<FullSWAEnvironment> environment = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    std::shared_ptr<PausablePerRankBlockTransferEngine> pausable_transfer_engine =
        std::make_shared<PausablePerRankBlockTransferEngine>(
            environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, pausable_transfer_engine);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 2u);

    struct SourceBlockRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    struct TargetBlockRef {
        GroupSetPtr        group_set;
        DeviceBlockPoolPtr pool;
        size_t             member_group_id;
        BlockIdxType       block;
    };
    std::vector<SourceBlockRef> source_refs;
    std::vector<TargetBlockRef> target_refs;
    for (size_t desc_index = 0; desc_index < context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& desc        = context->loadDescs()[desc_index];
        IBlockPool*               source_pool = GetParam() == Tier::HOST ?
                                                    static_cast<IBlockPool*>(environment->host_pools[desc.group_set_id].get()) :
                                                    static_cast<IBlockPool*>(environment->disk_pools[desc.group_set_id].get());
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        EXPECT_EQ(source_pool->treeRefCount(desc.source_blocks.front()), 2u);
        source_refs.push_back(SourceBlockRef{source_pool, desc.source_blocks.front()});

        const GroupSetPtr&        group_set = environment->groups[desc.group_set_id];
        std::vector<BlockIdxType> target_blocks;
        for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
            const DeviceBlockPoolPtr& pool   = group_set->devicePools()[member_group_id];
            const BlockIdList         blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            target_blocks.push_back(blocks.front());
            target_refs.push_back(TargetBlockRef{group_set, pool, member_group_id, blocks.front()});
        }
        context->setTargetBlocks(desc_index, std::move(target_blocks));
    }

    pausable_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    ASSERT_TRUE(pausable_transfer_engine->waitUntilEnteredFor(kRaceWaitTimeout));
    for (const TargetBlockRef& target : target_refs) {
        EXPECT_EQ(target.pool->refCount(target.block), 2u);
    }

    const TransferDescriptor& invalid_desc                                           = context->loadDescs().back();
    invalid_desc.node->group_set_resources[invalid_desc.group_set_id].transfer_state = GroupSetTransferState::IDLE;
    pausable_transfer_engine->release();
    context->waitDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    for (const TransferDescriptor& desc : context->loadDescs()) {
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_TRUE(resource.hasTier(GetParam()));
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    }
    for (const SourceBlockRef& source : source_refs) {
        EXPECT_EQ(source.pool->treeRefCount(source.block), 1u);
    }
    for (const TargetBlockRef& target : target_refs) {
        EXPECT_EQ(target.pool->refCount(target.block), 1u);
    }
    environment->expectPayloads();

    context.reset();
    for (const TargetBlockRef& target : target_refs) {
        releaseDeviceBlocks(*environment->cache, target.pool, {target.block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, AbortCommittedLoadReturnsFalseAndTransferCompletes) {
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

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_FALSE(context->empty());
    EXPECT_EQ(context->matchedBlocks(), kPathLength);
    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (TransferDescriptor& desc : context->load_descs_) {
        ASSERT_EQ(desc.source_tier, GetParam());
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[desc.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[desc.group_set_id].get());
        for (const BlockIdxType block : desc.source_blocks) {
            ASSERT_NE(block, NULL_BLOCK_IDX);
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
            source_refs.push_back(SourceRef{source_pool, block});
        }

        desc.target_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            desc.target_blocks.push_back(target);
            target_blocks.emplace_back(pool, target);
        }
    }
    ASSERT_FALSE(source_refs.empty());
    ASSERT_FALSE(target_blocks.empty());

    pausable_per_rank_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    pausable_per_rank_transfer_engine->waitUntilEntered();
    EXPECT_FALSE(context->done());
    EXPECT_FALSE(environment->cache->abortPendingLoad(context));
    pausable_per_rank_transfer_engine->release();
    context->waitDone();

    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(environment->cache->abortPendingLoad(context));
    const auto snapshot_after = environment->cache->getKeySnapshot(/*limit=*/32);
    EXPECT_GT(snapshot_after.version, snapshot_before.version);
    EXPECT_EQ(snapshot_after.keys, snapshot_before.keys);

    for (const TransferDescriptor& desc : context->loadDescs()) {
        ASSERT_NE(desc.node, nullptr);
        ASSERT_LT(desc.group_set_id, desc.node->group_set_resources.size());
        const GroupSetResource& resource = desc.node->group_set_resources[desc.group_set_id];
        EXPECT_EQ(resource.device_blocks, desc.target_blocks);
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

    context.reset();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_P(BlockTreeCacheLowerTierTest, AbortCompletionRaceDoesNotChangeCommittedResult) {
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

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_FALSE(context->empty());

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (TransferDescriptor& desc : context->load_descs_) {
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[desc.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[desc.group_set_id].get());
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        EXPECT_EQ(source_pool->treeRefCount(desc.source_blocks.front()), 2u);
        source_refs.push_back({source_pool, desc.source_blocks.front()});

        desc.target_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            desc.target_blocks.push_back(blocks.front());
            target_blocks.emplace_back(pool, blocks.front());
        }
    }

    pausable_per_rank_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    pausable_per_rank_transfer_engine->waitUntilEntered();

    ThreadCompletion race_start;
    bool             abort_succeeded = false;
    std::thread      abort_thread([&] {
        race_start.waitUntilEntered();
        abort_succeeded = environment->cache->abortPendingLoad(context);
    });
    race_start.markEntered();
    pausable_per_rank_transfer_engine->release();

    context->waitDone();
    abort_thread.join();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);

    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_FALSE(abort_succeeded);
    EXPECT_FALSE(environment->cache->abortPendingLoad(context));
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    for (const SourceRef& source : source_refs) {
        EXPECT_FALSE(source.pool->isAllocated(source.block));
    }
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 2u);
    }

    context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
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

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_FALSE(context->empty());

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (TransferDescriptor& desc : context->load_descs_) {
        ASSERT_EQ(desc.source_tier, GetParam());
        IBlockPool* source_pool = GetParam() == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools[desc.group_set_id].get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools[desc.group_set_id].get());
        for (const BlockIdxType block : desc.source_blocks) {
            ASSERT_NE(block, NULL_BLOCK_IDX);
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
            source_refs.push_back(SourceRef{source_pool, block});
        }

        desc.target_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            desc.target_blocks.push_back(targets.front());
            target_blocks.emplace_back(pool, targets.front());
        }
    }
    ASSERT_FALSE(source_refs.empty());
    ASSERT_FALSE(target_blocks.empty());

    pausable_per_rank_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    ASSERT_TRUE(pausable_per_rank_transfer_engine->waitUntilEnteredFor(kRaceWaitTimeout));
    EXPECT_FALSE(context->done());
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 2u);
    }

    const size_t                      submits_before_join = pausable_per_rank_transfer_engine->submitCount();
    BlockTreeMatchResult              joined_result       = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> joined_context      = takeLoadContext(joined_result);
    ASSERT_NE(joined_context, nullptr);
    ASSERT_FALSE(joined_context->empty());
    for (size_t desc_index = 0; desc_index < joined_context->loadDescs().size(); ++desc_index) {
        EXPECT_TRUE(joined_context->joinedLoads()[desc_index]);
        const TransferDescriptor& desc         = joined_context->loadDescs()[desc_index];
        const size_t              group_set_id = desc.group_set_id;
        ASSERT_LT(group_set_id, environment->groups.size());
        const std::vector<BlockIdxType>&       joined_targets = desc.target_blocks;
        const std::vector<DeviceBlockPoolPtr>& device_pools   = environment->groups[group_set_id]->devicePools();
        ASSERT_EQ(joined_targets.size(), device_pools.size());
        for (size_t pool_index = 0; pool_index < joined_targets.size(); ++pool_index) {
            ASSERT_NE(device_pools[pool_index], nullptr);
        }
    }
    ASSERT_TRUE(joined_context->commit());
    EXPECT_FALSE(joined_context->done());
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), submits_before_join);
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 3u);
    }

    pausable_per_rank_transfer_engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);

    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    ASSERT_TRUE(joined_context->done());
    EXPECT_FALSE(joined_context->success());
    EXPECT_FALSE(environment->cache->abortPendingLoad(context));
    EXPECT_FALSE(environment->cache->abortPendingLoad(joined_context));
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(),
              transferBatchCount(context->loadDescs(), environment->cache->config()));
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
            EXPECT_EQ(resources_after[group_id].disk_block, resources_before[path_index][group_id].disk_block);
            EXPECT_EQ(resources_after[group_id].transfer_state, GroupSetTransferState::IDLE);
        }
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->treeRefCount(source.block), 1u);
    }
    for (const std::pair<DeviceBlockPoolPtr, BlockIdxType>& target_block : target_blocks) {
        EXPECT_EQ(target_block.first->refCount(target_block.second), 2u);
    }
    environment->expectPayloads();

    BlockTreeMatchResult              retry         = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> retry_context = takeLoadContext(retry);
    ASSERT_NE(retry_context, nullptr);
    retry_context.reset();

    context.reset();
    joined_context.reset();
    environment->releaseMatch(joined_result);
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
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
    std::shared_ptr<LoadAsyncContext>   context          = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_FALSE(context->empty());
    const size_t disk_submit_count = context->loadDescs().size();
    ASSERT_GT(disk_submit_count, 0u);

    pausable_per_rank_transfer_engine->setThrowOnSubmit(1);

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (TransferDescriptor& desc : context->load_descs_) {
        ASSERT_EQ(desc.source_tier, Tier::DISK);
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        IBlockPool* source_pool = environment->disk_pools[desc.group_set_id].get();
        EXPECT_EQ(source_pool->treeRefCount(desc.source_blocks.front()), 2u);
        source_refs.push_back({source_pool, desc.source_blocks.front()});

        desc.target_blocks.clear();
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            desc.target_blocks.push_back(blocks.front());
            target_blocks.emplace_back(pool, blocks.front());
        }
    }

    pausable_per_rank_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    pausable_per_rank_transfer_engine->waitUntilEntered();
    pausable_per_rank_transfer_engine->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);

    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(),
              transferBatchCount(context->loadDescs(), environment->cache->config()));
    EXPECT_EQ(environment->host_pools[0]->freeBlocksNum(), host_free_before[0]);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), host_free_before[1]);
    const std::vector<GroupSetResource> resources_after = environment->resourcesForPathNode(0);
    ASSERT_EQ(resources_after.size(), resources_before.size());
    for (size_t group_id = 0; group_id < resources_after.size(); ++group_id) {
        EXPECT_EQ(resources_after[group_id].device_blocks, resources_before[group_id].device_blocks);
        EXPECT_EQ(resources_after[group_id].host_block, resources_before[group_id].host_block);
        EXPECT_EQ(resources_after[group_id].disk_block, resources_before[group_id].disk_block);
        EXPECT_EQ(resources_after[group_id].transfer_state, GroupSetTransferState::IDLE);
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->treeRefCount(source.block), 1u);
    }
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    BlockTreeMatchResult              retry         = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> retry_context = takeLoadContext(retry);
    ASSERT_NE(retry_context, nullptr);
    retry_context.reset();
    context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, MixedHostDiskContextAbortRestoresSources) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);
    runSingleMaintenance(*environment, Tier::HOST, 0.125);
    environment->scripted_per_rank_transfer_engine->clear();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), kPathLength);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);

    bool                                              saw_host = false;
    bool                                              saw_disk = false;
    std::vector<std::pair<IBlockPool*, BlockIdxType>> source_refs;
    for (const TransferDescriptor& desc : context->loadDescs()) {
        ASSERT_TRUE(desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK);
        ASSERT_EQ(desc.source_blocks.size(), 1u);
        saw_host                        = saw_host || desc.source_tier == Tier::HOST;
        saw_disk                        = saw_disk || desc.source_tier == Tier::DISK;
        IBlockPool*        source_pool  = desc.source_tier == Tier::HOST ?
                                              static_cast<IBlockPool*>(environment->host_pools.at(desc.group_set_id).get()) :
                                              static_cast<IBlockPool*>(environment->disk_pools.at(desc.group_set_id).get());
        const BlockIdxType source_block = desc.source_blocks.front();
        EXPECT_EQ(source_pool->treeRefCount(source_block), 2u);
        ASSERT_NE(desc.node, nullptr);
        EXPECT_EQ(desc.node->group_set_resources[desc.group_set_id].transfer_state,
                  GroupSetTransferState::LOAD_PENDING);
        source_refs.emplace_back(source_pool, source_block);
    }
    EXPECT_TRUE(saw_host);
    EXPECT_TRUE(saw_disk);

    context.reset();
    for (const auto& [source_pool, source_block] : source_refs) {
        EXPECT_EQ(source_pool->treeRefCount(source_block), 1u);
    }
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        for (const GroupSetResource& resource : environment->resourcesForPathNode(path_index)) {
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        }
    }
    environment->expectPayloads();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, MixedHostDiskFailureInstallsNoTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);
    runSingleMaintenance(*environment, Tier::HOST, 0.125);
    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(true);
    environment->scripted_per_rank_transfer_engine->enqueue(false);

    std::vector<std::vector<GroupSetResource>> resources_before;
    resources_before.reserve(environment->keys.size());
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        resources_before.push_back(environment->resourcesForPathNode(path_index));
    }

    BlockTreeMatchResult              result  = environment->cache->match(environment->keys);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);

    struct SourceRef {
        IBlockPool*  pool;
        BlockIdxType block;
    };
    bool                                                     saw_host = false;
    bool                                                     saw_disk = false;
    std::vector<SourceRef>                                   source_refs;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> target_blocks;
    for (size_t desc_index = 0; desc_index < context->loadDescs().size(); ++desc_index) {
        const TransferDescriptor& desc = context->loadDescs()[desc_index];
        ASSERT_TRUE(desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK);
        saw_host                = saw_host || desc.source_tier == Tier::HOST;
        saw_disk                = saw_disk || desc.source_tier == Tier::DISK;
        IBlockPool* source_pool = desc.source_tier == Tier::HOST ?
                                      static_cast<IBlockPool*>(environment->host_pools.at(desc.group_set_id).get()) :
                                      static_cast<IBlockPool*>(environment->disk_pools.at(desc.group_set_id).get());
        for (const BlockIdxType block : desc.source_blocks) {
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
            source_refs.push_back({source_pool, block});
        }

        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(desc.group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            target_blocks.emplace_back(pool, blocks.front());
        }
        context->setTargetBlocks(desc_index, std::move(targets));
    }
    ASSERT_TRUE(saw_host);
    ASSERT_TRUE(saw_disk);

    ASSERT_TRUE(context->commit());
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedBatchCount(),
              transferBatchCount(context->loadDescs(), environment->cache->config()));

    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const auto resources_after = environment->resourcesForPathNode(path_index);
        ASSERT_EQ(resources_after.size(), resources_before[path_index].size());
        for (size_t group_id = 0; group_id < resources_after.size(); ++group_id) {
            EXPECT_EQ(resources_after[group_id].device_blocks, resources_before[path_index][group_id].device_blocks);
            EXPECT_EQ(resources_after[group_id].host_block, resources_before[path_index][group_id].host_block);
            EXPECT_EQ(resources_after[group_id].disk_block, resources_before[path_index][group_id].disk_block);
            EXPECT_EQ(resources_after[group_id].transfer_state, GroupSetTransferState::IDLE);
        }
    }
    for (const SourceRef& source : source_refs) {
        EXPECT_EQ(source.pool->treeRefCount(source.block), 1u);
    }
    for (const auto& [pool, block] : target_blocks) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    result.async_context.reset();
    context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
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
    ASSERT_EQ(prefix_hold.matched_device_blocks, 2u);
    demoteSwaSuffixKeepingFullDevice(*environment);
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
    EXPECT_EQ(result.matched_device_blocks, 2u);
    expectAggregatedReadyResult(*environment->cache, result, /*full_blocks=*/2, /*swa_blocks=*/2);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), kPathLength);
    EXPECT_EQ(contextDescCountForGroupSet(context, 0), 2u);
    EXPECT_EQ(contextDescCountForGroupSet(context, 1), 2u);
    for (const TransferDescriptor& desc : context->loadDescs()) {
        if (desc.group_set_id == 0) {
            EXPECT_EQ(desc.source_tier, Tier::DEVICE);
            EXPECT_GE(desc.path_index, 2u);
        } else {
            EXPECT_EQ(desc.source_tier, Tier::HOST);
        }
    }
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    environment->releaseMatch(result);
    context.reset();
    environment->releaseRequestRefs();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, LoadingHostChildSkipsDeviceParentUntilSettlement) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             device_pool = makeDevicePool({{16, 0}}, 3, "loading_child_device_parent");
    std::shared_ptr<HostBlockPool> host_pool   = makeHostPool(16, 1);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    std::vector<GroupSetPtr> groups = {full};
    BlockTreeCacheConfig     config;
    config.enable_device_cache = true;
    config.enable_host_cache   = true;
    config.enable_disk_cache   = false;
    auto cache                 = makeBlockTreeCacheForTest(groups, config);
    ASSERT_NE(cache, nullptr);

    auto pausable_per_rank_transfer_engine =
        std::make_shared<PausablePerRankBlockTransferEngine>(groups, /*succeed=*/true);
    PausableTransferReleaseGuard release_guard(pausable_per_rank_transfer_engine);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_per_rank_transfer_engine);

    MultiNodeBlocks parent_blocks = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(parent_blocks.size(), 1u);
    ASSERT_EQ(parent_blocks.front().size(), 1u);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = parent_blocks.front();
    resources[1][0].host_block    = host_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200}, resources));
    unreferenceDeviceBlocksForTest(*full, parent_blocks);

    std::vector<TreeNode*> path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    TreeNode* parent = path[0];
    TreeNode* child  = path[1];
    ASSERT_TRUE(parent->group_set_resources[0].hasTier(Tier::DEVICE));
    ASSERT_TRUE(child->group_set_resources[0].hasTier(Tier::HOST));
    ASSERT_EQ(cache->getStats().device_heap_total_size, 1u);

    BlockTreeMatchResult              result  = cache->match({100, 200});
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->loadDescs().size(), 1u);
    EXPECT_EQ(context->loadDescs().front().node, child);
    EXPECT_EQ(context->loadDescs().front().source_tier, Tier::HOST);

    BlockIdList target_blocks = device_pool->malloc(1).value();
    ASSERT_EQ(target_blocks.size(), 1u);
    device_pool->incRef(target_blocks);
    context->setTargetBlocks(0, target_blocks);

    ASSERT_TRUE(context->commit());
    pausable_per_rank_transfer_engine->waitUntilEntered();
    ASSERT_EQ(child->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
    path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[0], parent);
    EXPECT_EQ(path[1], child);
    EXPECT_TRUE(parent->group_set_resources[0].hasTier(Tier::DEVICE));

    pausable_per_rank_transfer_engine->release();
    context->waitDone();
    BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    ASSERT_TRUE(context->success());
    path = cache->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    EXPECT_TRUE(path[0]->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_TRUE(path[1]->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_EQ(path[1]->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    releaseRequestRefsForTest(*cache, result.matched_device_resources);
    result.matched_device_resources.clear();
    context.reset();
    cache.reset();
    device_pool->decRef(target_blocks);
    EXPECT_EQ(device_pool->freeBlocksNum(), 3u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadRequestReleaseKeepsCandidateMembership) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    BlockTreeMatchResult              result  = makePartialReadyDeviceContext(*environment);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    for (TransferDescriptor& desc : context->load_descs_) {
        if (desc.source_tier != Tier::DEVICE) {
            continue;
        }
        const auto& device_pools = environment->groups.at(desc.group_set_id)->devicePools();
        ASSERT_EQ(device_pools.size(), desc.source_blocks.size());
        for (size_t member_index = 0; member_index < desc.source_blocks.size(); ++member_index) {
            device_sources.emplace_back(device_pools[member_index], desc.source_blocks[member_index]);
        }
    }
    ASSERT_EQ(device_sources.size(), 4u);

    environment->releaseRequestRefs();
    const size_t tree_nodes_before = environment->cache->getStats().tree_node_count;
    for (const auto& [pool, block] : device_sources) {
        EXPECT_TRUE(pool->isAllocated(block));
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    environment->expectPayloads();
    const size_t device_candidates_before_abort = environment->cache->getStats().device_heap_total_size;

    context.reset();
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_before_abort);
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 2u);
    }

    environment->releaseMatch(result);
    const size_t device_candidates_after_release = environment->cache->getStats().device_heap_total_size;
    EXPECT_EQ(device_candidates_after_release, device_candidates_before_abort);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    environment->expectPayloads();

    context.reset();
    environment->releaseMatch(result);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_after_release);
    EXPECT_EQ(environment->cache->evictForGroup(0, 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, DeviceLoadAsyncCompletionKeepsRequestSourcesCandidates) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    ASSERT_NE(environment, nullptr);
    auto pausable_per_rank_transfer_engine = std::make_shared<PausablePerRankBlockTransferEngine>(
        environment->groups, /*succeed=*/true, /*pause_enabled=*/false);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 pausable_per_rank_transfer_engine);
    BlockTreeMatchResult              result  = makePartialReadyDeviceContext(*environment);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> device_sources;
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_target_blocks;
    for (TransferDescriptor& desc : context->load_descs_) {
        desc.target_blocks.clear();
        const auto& device_pools = environment->groups.at(desc.group_set_id)->devicePools();
        if (desc.source_tier == Tier::DEVICE) {
            ASSERT_EQ(device_pools.size(), desc.source_blocks.size());
            desc.target_blocks = desc.source_blocks;
            for (size_t member_index = 0; member_index < desc.source_blocks.size(); ++member_index) {
                device_sources.emplace_back(device_pools[member_index], desc.source_blocks[member_index]);
            }
            continue;
        }

        for (const DeviceBlockPoolPtr& pool : device_pools) {
            BlockIdList targets = pool->malloc(1).value();
            ASSERT_EQ(targets.size(), 1u);
            pool->incRef(targets);
            const BlockIdxType target = targets.front();
            EXPECT_EQ(pool->refCount(target), 1u);
            desc.target_blocks.push_back(target);
            request_target_blocks.emplace_back(pool, target);
        }
        ASSERT_EQ(desc.target_blocks.size(), device_pools.size());
        ASSERT_NE(desc.node, nullptr);
    }
    ASSERT_EQ(device_sources.size(), 4u);
    ASSERT_EQ(request_target_blocks.size(), 2u);

    environment->releaseRequestRefs();
    const size_t tree_nodes_before        = environment->cache->getStats().tree_node_count;
    const size_t device_candidates_before = environment->cache->getStats().device_heap_total_size;

    pausable_per_rank_transfer_engine->enablePause();
    ASSERT_TRUE(context->commit());
    EXPECT_FALSE(context->commit());
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
    EXPECT_FALSE(environment->cache->abortPendingLoad(context));
    EXPECT_EQ(environment->cache->getStats().tree_node_count, tree_nodes_before);
    const size_t device_candidates_after_load = environment->cache->getStats().device_heap_total_size;
    EXPECT_EQ(device_candidates_after_load, device_candidates_before + request_target_blocks.size());
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 2u);
    }
    environment->expectPayloads();
    for (const TransferDescriptor& descriptor : pausable_per_rank_transfer_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, Tier::HOST);
        EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
    }
    EXPECT_EQ(pausable_per_rank_transfer_engine->submitCount(), 1u);
    EXPECT_EQ(pausable_per_rank_transfer_engine->descriptors().size(), 2u);

    environment->releaseMatch(result);
    EXPECT_EQ(environment->cache->getStats().device_heap_total_size, device_candidates_after_load);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_EQ(pool->refCount(block), 1u);
    }
    EXPECT_EQ(environment->cache->evictForGroup(0, 2), 2);
    for (const auto& [pool, block] : device_sources) {
        EXPECT_FALSE(pool->isAllocated(block));
    }

    context->waitDone();
    EXPECT_TRUE(context->success());
    context.reset();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);
    for (const auto& [pool, block] : request_target_blocks) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, SparseDisconnectedSWADoesNotPublishVacuousReadyPrefix) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    auto                      environment = FullSWAEnvironment::create(options);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    auto find = environment->cache->tree()->findNode(environment->keys);
    ASSERT_EQ(find.size(), kPathLength);
    std::vector<BlockIdxType> swa_host_blocks;
    for (size_t path_index = 0; path_index < kPathLength; ++path_index) {
        GroupSetResource& swa_resource = find[path_index]->group_set_resources[1];
        ASSERT_TRUE(swa_resource.hasTier(Tier::DEVICE));
        const std::vector<BlockIdxType> old_device_blocks = swa_resource.getBlocks(Tier::DEVICE);
        const MultiNodeResource         device_resource{1, Tier::DEVICE, {{find[path_index], old_device_blocks}}};
        environment->cache->evictor_.suspendCandidate(find[path_index], /*group_set_id=*/1, Tier::DEVICE);
        swa_resource.evictFromTier(Tier::DEVICE);
        environment->groups[1]->unreferenceBlocks(device_resource, BlockTreeRefType::CACHE);
        if (path_index >= 2) {
            const BlockIdxType host_block =
                environment->groups[1]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_NE(host_block, NULL_BLOCK_IDX);
            swa_resource.setBlocks(Tier::HOST, {host_block});
            swa_host_blocks.push_back(host_block);
        }
    }

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->matchedBlocks(), kPathLength);
    EXPECT_EQ(contextDescCountForGroupSet(context, 0), 4u);
    EXPECT_EQ(contextDescCountForGroupSet(context, 1), 2u);
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedDescriptorCount(), 0u);

    for (const TransferDescriptor& desc : context->loadDescs()) {
        if (desc.group_set_id == 0) {
            EXPECT_EQ(desc.source_tier, Tier::DEVICE);
            ASSERT_EQ(desc.source_blocks.size(), 2u);
            EXPECT_EQ(environment->device_pools[0]->refCount(desc.source_blocks[0]), 2u);
            EXPECT_EQ(environment->device_pools[1]->refCount(desc.source_blocks[1]), 2u);
        } else {
            EXPECT_EQ(desc.source_tier, Tier::HOST);
            ASSERT_EQ(desc.source_blocks.size(), 1u);
            EXPECT_GE(desc.path_index, 2u);
            EXPECT_EQ(environment->host_pools[1]->treeRefCount(desc.source_blocks[0]), 2u);
        }
    }

    context.reset();
    environment->releaseMatch(result);
    for (const auto& blocks : environment->request_blocks[0]) {
        EXPECT_EQ(environment->device_pools[0]->refCount(blocks[0]), 1u);
        EXPECT_EQ(environment->device_pools[1]->refCount(blocks[1]), 1u);
    }
    for (BlockIdxType host_block : swa_host_blocks) {
        EXPECT_EQ(environment->host_pools[1]->treeRefCount(host_block), 1u);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}
}  // namespace
}  // namespace rtp_llm
