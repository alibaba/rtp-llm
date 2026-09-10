#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

struct StoreEnvironment {
    std::vector<DeviceBlockPoolPtr>                      device_pools;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools;
    std::vector<GroupSetPtr>                             groups;
    std::unique_ptr<BlockTreeCache>                      cache;

    IBlockPool& poolFor(Tier tier, size_t group_set_id = 0) const {
        return tier == Tier::HOST ? static_cast<IBlockPool&>(*host_pools[group_set_id]) :
                                    static_cast<IBlockPool&>(*disk_pools[group_set_id]);
    }

    size_t storeRefCount() const {
        size_t store_refs = 0;
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            store_refs += pool->referencedBlocksNum(BlockTreeRefType::STORE);
        }
        for (const std::shared_ptr<HostBlockPool>& pool : host_pools) {
            store_refs += pool == nullptr ? 0 : pool->referencedBlocksNum(BlockTreeRefType::STORE);
        }
        for (const std::shared_ptr<BlockTreeDiskBlockPool>& pool : disk_pools) {
            store_refs += pool == nullptr ? 0 : pool->referencedBlocksNum(BlockTreeRefType::STORE);
        }
        return store_refs;
    }
};

constexpr size_t kStoreDeviceBlocks = 4;

class CoreDumpGuard {
public:
    CoreDumpGuard(): old_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }
    ~CoreDumpGuard() {
        StaticConfig::user_ft_core_dump_on_exception = old_;
    }

private:
    bool old_;
};

StoreEnvironment makeStoreEnvironment(const std::string&              name,
                                      bool                            device_cache_on,
                                      bool                            host_cache_on,
                                      bool                            disk_cache_on,
                                      const std::vector<size_t>&      lower_tier_blocks = {2},
                                      int                             task_pool_size    = 4,
                                      std::shared_ptr<StorageBackend> storage_backend   = nullptr) {
    StoreEnvironment env;
    for (size_t group_set_id = 0; group_set_id < lower_tier_blocks.size(); ++group_set_id) {
        env.device_pools.push_back(
            makeDevicePool({{1, 0}}, kStoreDeviceBlocks, name + "_" + std::to_string(group_set_id)));
        RTP_LLM_CHECK(env.device_pools.back() != nullptr);
        env.host_pools.push_back(host_cache_on ? makeHostPool(1, lower_tier_blocks[group_set_id]) : nullptr);
        env.disk_pools.push_back(
            disk_cache_on ? makeDiskPool(1, lower_tier_blocks[group_set_id], std::make_unique<MemoryDiskBlockIO>()) :
                            nullptr);
        env.groups.push_back(std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{env.device_pools.back()}, env.host_pools.back(), env.disk_pools.back()));
    }

    BlockTreeCacheConfig config;
    config.enable_device_cache      = device_cache_on;
    config.enable_host_cache        = host_cache_on;
    config.enable_disk_cache        = disk_cache_on;
    config.enable_remote_cache      = storage_backend != nullptr;
    config.task_pool_size           = task_pool_size;
    std::vector<GroupSetPtr> groups = env.groups;
    env.cache = makeBlockTreeCacheForTest(std::move(groups), std::move(config), std::move(storage_backend));
    RTP_LLM_CHECK(env.cache != nullptr);
    return env;
}

class PendingWriteExecutor: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(std::move(task));
            ++submitted_count_;
        }
        cv_.notify_all();
        return true;
    }
    void shutdown() noexcept override {
        runAll();
    }
    void runAll() {
        std::deque<Task> tasks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks.swap(tasks_);
        }
        for (auto& task : tasks) {
            task();
        }
    }
    size_t submittedCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return submitted_count_;
    }
    bool waitForPendingCount(size_t expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return tasks_.size() >= expected; });
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::deque<Task>        tasks_;
    size_t                  submitted_count_{0};
};

class PendingWriteBackend: public StorageBackend {
public:
    PendingWriteBackend(): PendingWriteBackend(std::make_shared<PendingWriteExecutor>()) {}
    ~PendingWriteBackend() override {
        shutdown();
    }
    void setCache(BlockTreeCache* cache) {
        cache_ = cache;
    }
    void finishWrite() {
        executor_->runAll();
    }
    size_t submittedCount() const {
        return executor_->submittedCount();
    }
    CacheKeysType keys() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return keys_;
    }
    bool waitForPendingWrite() {
        return executor_->waitForPendingCount(1);
    }
    std::vector<BlockIdxType> blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocks_;
    }
    std::vector<std::string> groupTags() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return group_tags_;
    }
    std::vector<size_t> keyHandleCounts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return key_handle_counts_;
    }
    std::vector<void*> addresses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return addresses_;
    }
    bool submittedOutsideTreeLock() const {
        return submitted_outside_tree_lock_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        return {request.handles.size(), nullptr};
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {}
    void writeImpl(const StorageRequest& request) override {
        submitted_outside_tree_lock_ = cache_ != nullptr && cache_->mutex_.try_lock();
        if (submitted_outside_tree_lock_) {
            cache_->mutex_.unlock();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        keys_.insert(keys_.end(), request.keys->begin(), request.keys->end());
        for (const auto& key_handles : request.handles) {
            key_handle_counts_.push_back(key_handles.size());
            for (const auto& handle : key_handles) {
                const auto& group = topology().groupById(handle.group_id);
                group_tags_.push_back(group.tag);
                blocks_.push_back(handle.block);
                const auto buffers = convertIndexToBuffer(group.layer_ids.front(), handle.group_id, handle.block);
                addresses_.push_back(buffers.front().addr);
            }
        }
    }

private:
    explicit PendingWriteBackend(std::shared_ptr<PendingWriteExecutor> executor):
        StorageBackend(executor), executor_(std::move(executor)) {}

    std::shared_ptr<PendingWriteExecutor> executor_;
    BlockTreeCache*                       cache_{nullptr};
    mutable std::mutex                    mutex_;
    CacheKeysType                         keys_;
    std::vector<BlockIdxType>             blocks_;
    std::vector<std::string>              group_tags_;
    std::vector<size_t>                   key_handle_counts_;
    std::vector<void*>                    addresses_;
    bool                                  submitted_outside_tree_lock_{false};
};

std::shared_ptr<ControlledPerRankBlockTransferEngine> installStoreTransferEngine(
    StoreEnvironment& env, TransferCopyAction action, std::shared_ptr<CallbackBarrier> barrier = nullptr) {
    auto engine = std::make_shared<ControlledPerRankBlockTransferEngine>(env.groups, action, std::move(barrier));
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*env.cache, engine);
    return engine;
}

std::vector<std::vector<GroupSetResource>>
deviceSourceResources(const std::vector<std::vector<BlockIdxType>>& per_group_blocks) {
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(per_group_blocks.size()));
    for (size_t group_set_id = 0; group_set_id < per_group_blocks.size(); ++group_set_id) {
        resources[0][group_set_id].device_blocks = per_group_blocks[group_set_id];
    }
    return resources;
}

size_t candidateCountForTier(const BlockTreeCache& cache, Tier tier) {
    const CacheStats stats = cache.getStats();
    switch (tier) {
        case Tier::DEVICE:
            return stats.device_heap_total_size;
        case Tier::HOST:
            return stats.host_heap_total_size;
        default:
            return stats.disk_heap_total_size;
    }
}

std::shared_ptr<LoadAsyncContext> takeLoadContext(BlockTreeMatchResult& result) {
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    result.async_context.reset();
    return context;
}

class ResidentTieredCacheTest: public ::testing::TestWithParam<Tier> {
protected:
    void SetUp() override {
        if (!cudaAvailable()) {
            GTEST_SKIP() << "CUDA not available";
        }
        env_ = std::make_shared<StoreEnvironment>(makeStoreEnvironment("resident_" + std::string(tierName(GetParam())),
                                                                       true,
                                                                       GetParam() == Tier::HOST,
                                                                       GetParam() == Tier::DISK,
                                                                       {4},
                                                                       2,
                                                                       nullptr));
        const MultiNodeBlocks sources = allocateDeviceBlocksForTest(*env_->groups[0], keys_.size());
        for (const BlockIndicesType& blocks : sources) {
            request_holds_.push_back(blocks);
            resources_.push_back({GroupSetResource{}});
            resources_.back()[0].device_blocks = blocks;
        }
        ASSERT_EQ(sources.size(), keys_.size());
        engine_ = installStoreTransferEngine(*env_, TransferCopyAction::Succeed, nullptr);
        ASSERT_NE(engine_, nullptr);
        env_->cache->insert(keys_, resources_, GetParam(), /*is_resident=*/false);
        ASSERT_TRUE(waitForIdle());
        const std::vector<TreeNode*> path = env_->cache->tree()->findNode(keys_);
        ASSERT_EQ(path.size(), keys_.size());
        for (const TreeNode* node : path) {
            EXPECT_EQ(node->group_set_resources[0].getTopTier(), GetParam());
        }
    }

    void TearDown() override {
        if (!env_) {
            return;
        }
        BoundedThread<void> cleanup([env = env_, holds = request_holds_]() {
            BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env->cache);
            for (const BlockIndicesType& blocks : holds) {
                releaseDeviceBlocks(*env->cache, env->device_pools[0], blocks);
            }
        });
        ASSERT_EQ(cleanup.waitFor(std::chrono::seconds(5)), std::future_status::ready);
        cleanup.get();
        EXPECT_EQ(env_->storeRefCount(), 0u);
        EXPECT_EQ(env_->device_pools[0]->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    }

    bool waitForIdle() {
        BoundedThread<void> waiter([env = env_]() { BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env->cache); });
        if (waiter.waitFor(std::chrono::seconds(5)) != std::future_status::ready) {
            return false;
        }
        waiter.get();
        return true;
    }

    void expectResidentPrefix(const BlockIndicesType& expected_blocks) {
        const std::vector<TreeNode*> path = env_->cache->tree()->findNode(keys_);
        ASSERT_EQ(path.size(), keys_.size());
        for (const TreeNode* node : path) {
            EXPECT_TRUE(node->is_resident);
            EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(node->group_set_resources[0].getTopTier(), Tier::DEVICE);
            EXPECT_FALSE(node->group_set_resources[0].hasTier(GetParam()));
            EXPECT_EQ(node->group_set_resources[0].servingTierCount(), 1u);
        }
        for (Tier tier : {Tier::DEVICE, GetParam()}) {
            EXPECT_EQ(candidateCountForTier(*env_->cache, tier), 0u);
            EXPECT_FALSE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*env_->cache, 0, tier, true));
            EXPECT_FALSE(BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(*env_->cache, 0, tier, false));
        }
        BlockTreeMatchResult match = env_->cache->match(keys_);
        EXPECT_EQ(match.matched_device_blocks, keys_.size());
        EXPECT_EQ(env_->cache->matchedBlocksForGroup(0, match.matched_device_resources), expected_blocks);
        releaseRequestRefsForTest(*env_->cache, match.matched_device_resources);
        EXPECT_EQ(candidateCountForTier(*env_->cache, Tier::DEVICE), 0u);
    }

    void exerciseLoadCompletion(TransferCopyAction action) {
        auto                                   barrier = std::make_shared<CallbackBarrier>();
        block_tree_cache_detail::ScopeRollback release_barrier([barrier]() { barrier->release(); });
        ASSERT_TRUE(waitForIdle());
        const size_t submitted_before = engine_->submittedBatchCount();
        engine_->setCopyBehavior(action, barrier);
        BlockTreeMatchResult                    result  = env_->cache->match(keys_);
        const std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->loadDescs().size(), keys_.size());
        const size_t request_holds_before = request_holds_.size();
        BlockIndicesType target_blocks;
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const MultiNodeBlocks target = allocateDeviceBlocksForTest(*env_->groups[0], 1);
            ASSERT_EQ(target.size(), 1u);
            request_holds_.push_back(target.front());
            target_blocks.push_back(target.front().front());
            context->setTargetBlocks(i, target.front());
        }
        auto                   completion = std::make_shared<std::promise<ErrorInfo>>();
        std::future<ErrorInfo> completed  = completion->get_future();
        context->onDone([completion](ErrorInfo error) { completion->set_value(std::move(error)); });
        ASSERT_TRUE(context->commit());
        ASSERT_TRUE(barrier->waitUntilEnteredFor(1, std::chrono::seconds(5)));

        env_->cache->insert(keys_, resources_, Tier::DEVICE, /*is_resident=*/true);
        const std::vector<TreeNode*> busy_path = env_->cache->tree()->findNode(keys_);
        ASSERT_EQ(busy_path.size(), keys_.size());
        for (const TreeNode* node : busy_path) {
            EXPECT_FALSE(node->is_resident);
            EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
        }
        barrier->release();
        ASSERT_EQ(completed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        const bool succeeded = action == TransferCopyAction::Succeed;
        EXPECT_EQ(completed.get().ok(), succeeded);
        ASSERT_TRUE(waitForIdle());
        EXPECT_GT(engine_->submittedBatchCount(), submitted_before);
        EXPECT_EQ(env_->device_pools[0]->referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
        const std::vector<TreeNode*> settled_path = env_->cache->tree()->findNode(keys_);
        ASSERT_EQ(settled_path.size(), keys_.size());
        for (const TreeNode* node : settled_path) {
            EXPECT_FALSE(node->is_resident);
            EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
            EXPECT_EQ(node->group_set_resources[0].getTopTier(), succeeded ? Tier::DEVICE : GetParam());
        }

        env_->cache->insert(keys_, resources_, Tier::DEVICE, /*is_resident=*/true);
        if (succeeded) {
            expectResidentPrefix(target_blocks);
            EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
        } else {
            // A failed load retains the only source; resident insertion must not
            // publish a second serving tier or mark an unpromoted prefix resident.
            for (const TreeNode* node : settled_path) {
                EXPECT_FALSE(node->is_resident);
                EXPECT_EQ(node->group_set_resources[0].getTopTier(), GetParam());
                EXPECT_EQ(node->group_set_resources[0].servingTierCount(), 1u);
            }
            EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::CACHE), keys_.size());
            EXPECT_EQ(candidateCountForTier(*env_->cache, GetParam()), 1u);
            // The caller owns REQUEST refs even when loading fails. Release the
            // failed destination allocation before the same request retries.
            for (size_t i = request_holds_before; i < request_holds_.size(); ++i) {
                releaseDeviceBlocks(*env_->cache, env_->device_pools[0], request_holds_[i]);
            }
            request_holds_.resize(request_holds_before);
            EXPECT_EQ(env_->device_pools[0]->freeBlocksNum(), keys_.size());
        }
    }

    std::shared_ptr<StoreEnvironment>                     env_;
    std::shared_ptr<ControlledPerRankBlockTransferEngine> engine_;
    const CacheKeysType                                   keys_{100, 200};
    std::vector<std::vector<GroupSetResource>>            resources_;
    std::vector<BlockIndicesType>                         request_holds_;
};

TEST_P(ResidentTieredCacheTest, ResidentInsertDoesNotDuplicateExistingLowerTier) {
    EXPECT_EQ(candidateCountForTier(*env_->cache, GetParam()), 1u);
    env_->cache->insert(keys_, resources_, Tier::DEVICE, /*is_resident=*/true);
    const auto path = env_->cache->tree()->findNode(keys_);
    ASSERT_EQ(path.size(), keys_.size());
    for (const TreeNode* node : path) {
        EXPECT_FALSE(node->is_resident);
        EXPECT_EQ(node->group_set_resources[0].getTopTier(), GetParam());
        EXPECT_EQ(node->group_set_resources[0].servingTierCount(), 1u);
    }
    EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::CACHE), keys_.size());
    EXPECT_EQ(candidateCountForTier(*env_->cache, GetParam()), 1u);
    EXPECT_EQ(candidateCountForTier(*env_->cache, Tier::DEVICE), 0u);
}

TEST_P(ResidentTieredCacheTest, SuccessfulAsyncLoadCanBePromotedToResidentAfterSettlement) {
    exerciseLoadCompletion(TransferCopyAction::Succeed);
}

TEST_P(ResidentTieredCacheTest, FailedAsyncLoadPreservesSourceForResidentRetry) {
    exerciseLoadCompletion(TransferCopyAction::Fail);
    ASSERT_FALSE(HasFatalFailure());
    exerciseLoadCompletion(TransferCopyAction::Succeed);
}

TEST_P(ResidentTieredCacheTest, ResidentDeviceMatchDoesNotSubmitLowerTierReadOrReadmitNodes) {
    exerciseLoadCompletion(TransferCopyAction::Succeed);
    ASSERT_FALSE(HasFatalFailure());
    const size_t submitted_before = engine_->submittedBatchCount();
    engine_->setCopyBehavior(TransferCopyAction::Fail, nullptr);
    BlockTreeMatchResult result = env_->cache->match(keys_);
    EXPECT_EQ(result.matched_device_blocks, keys_.size());
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(engine_->submittedBatchCount(), submitted_before);
    releaseRequestRefsForTest(*env_->cache, result.matched_device_resources);
    for (const TreeNode* node : env_->cache->tree()->findNode(keys_)) {
        EXPECT_TRUE(node->is_resident);
        EXPECT_EQ(node->group_set_resources[0].servingTierCount(), 1u);
    }
    EXPECT_EQ(candidateCountForTier(*env_->cache, Tier::DEVICE), 0u);
    EXPECT_EQ(candidateCountForTier(*env_->cache, GetParam()), 0u);
}

TEST_P(ResidentTieredCacheTest, CancelPendingLoadThenRetryBeforeRegisteringResident) {
    BlockTreeMatchResult                    result  = env_->cache->match(keys_);
    const std::shared_ptr<LoadAsyncContext> context = takeLoadContext(result);
    ASSERT_NE(context, nullptr);
    env_->cache->insert(keys_, resources_, Tier::DEVICE, /*is_resident=*/true);
    const std::vector<TreeNode*> path = env_->cache->tree()->findNode(keys_);
    ASSERT_EQ(path.size(), keys_.size());
    for (const TreeNode* node : path) {
        EXPECT_FALSE(node->is_resident);
        EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::LOAD_PENDING);
    }
    ASSERT_TRUE(env_->cache->abortPendingLoad(context));
    EXPECT_EQ(env_->poolFor(GetParam()).referencedBlocksNum(BlockTreeRefType::LOAD), 0u);
    EXPECT_EQ(candidateCountForTier(*env_->cache, GetParam()), 1u);
    for (const TreeNode* node : path) {
        EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(node->group_set_resources[0].servingTierCount(), 1u);
    }
    exerciseLoadCompletion(TransferCopyAction::Succeed);
}

INSTANTIATE_TEST_SUITE_P(HostAndDisk, ResidentTieredCacheTest, ::testing::Values(Tier::HOST, Tier::DISK));

TEST(BlockTreeStorerTest, StorePublishesTargetTierOnlyWithoutDeviceResidency) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    for (const Tier target_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(target_tier));
        StoreEnvironment env         = makeStoreEnvironment("store_publish_" + std::string(tierName(target_tier)),
                                                    /*device_cache_on=*/false,
                                                    /*host_cache_on=*/target_tier == Tier::HOST,
                                                    /*disk_cache_on=*/target_tier == Tier::DISK);
        IBlockPool&      target_pool = env.poolFor(target_tier);
        const size_t     free_before = target_pool.freeBlocksNum();

        MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
        ASSERT_EQ(request_holder.size(), 1u);
        env.cache->insert({100},
                          deviceSourceResources({request_holder[0]}),
                          target_tier,
                          /*is_resident=*/false);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

        auto find = env.cache->tree()->findNode({100});
        ASSERT_EQ(find.size(), 1u);
        const GroupSetResource& resource = find.back()->group_set_resources[0];
        EXPECT_TRUE(resource.device_blocks.empty());
        ASSERT_TRUE(resource.hasTier(target_tier));
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);

        const BlockIdxType target_block = resource.getBlocks(target_tier).front();
        EXPECT_EQ(target_pool.treeRefCount(target_block), 1u) << "only the tree may hold the published block";
        EXPECT_EQ(target_pool.freeBlocksNum(), free_before - 1);
        EXPECT_EQ(candidateCountForTier(*env.cache, target_tier), 1u)
            << "the accepted target must reach its eviction heap";
        EXPECT_EQ(env.storeRefCount(), 0u);
        releaseDeviceBlocks(*env.cache, env.device_pools[0], request_holder.front());
    }
}

TEST(BlockTreeStorerTest, DeviceInsertSubmitsAllBlocksOutsideTreeLockAndPinsUntilCompletion) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto             backend = std::make_shared<PendingWriteBackend>();
    StoreEnvironment env     = makeStoreEnvironment("storage_write",
                                                /*device_cache_on=*/true,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{2},
                                                /*task_pool_size=*/4,
                                                backend);
    backend->setCache(env.cache.get());
    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env.groups[0], 2);
    ASSERT_EQ(holder.size(), 2u);
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = holder[0];
    resources[1][0].device_blocks = holder[1];

    CacheKeysType             expected_keys;
    size_t                    expected_submissions = 0;
    std::vector<BlockIdxType> expected_blocks;
    std::vector<size_t>       expected_counts;
    int64_t                   version = -1;
    for (const bool is_resident : {false, false, true, true}) {
        SCOPED_TRACE(is_resident);
        env.cache->insert({100, 101}, resources, Tier::DEVICE, is_resident);
        EXPECT_EQ(backend->submittedCount(), ++expected_submissions);
        const auto snapshot = env.cache->getKeySnapshot();
        if (version >= 0) {
            EXPECT_EQ(snapshot.version, version) << "duplicate inserts and resident promotion do not change tree data";
        }
        version = snapshot.version;
        EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 3u);
        EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 3u);
        backend->finishWrite();
        expected_keys.insert(expected_keys.end(), {100, 101});
        EXPECT_EQ(backend->keys(), expected_keys);
        expected_counts.insert(expected_counts.end(), {1, 1});
        expected_blocks.insert(expected_blocks.end(), {holder[0][0], holder[1][0]});
        EXPECT_TRUE(backend->submittedOutsideTreeLock());
        EXPECT_EQ(backend->keyHandleCounts(), expected_counts);
        EXPECT_EQ(backend->blocks(), expected_blocks);
        EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 2u);
        EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 2u);
        if (is_resident) {
            EXPECT_EQ(candidateCountForTier(*env.cache, Tier::DEVICE), 0u);
        }
    }
    env.cache->insert({}, {}, Tier::DEVICE, /*is_resident=*/false);
    EXPECT_EQ(backend->submittedCount(), expected_submissions);
    backend->finishWrite();
    EXPECT_EQ(backend->blocks(), expected_blocks);

    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[1]);
}

TEST(BlockTreeStorerTest, UnsupportedInsertTargetsRejectWithoutPublishingOrPinning) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto             backend = std::make_shared<PendingWriteBackend>();
    StoreEnvironment env     = makeStoreEnvironment("storage_remote_only",
                                                /*device_cache_on=*/false,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{2},
                                                /*task_pool_size=*/4,
                                                backend);
    backend->setCache(env.cache.get());
    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env.groups[0], 2);
    ASSERT_EQ(holder.size(), 2u);
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = holder[0];
    resources[1][0].device_blocks = holder[1];

    CoreDumpGuard guard;
    const auto    version = env.cache->getKeySnapshot().version;
    for (const Tier target : {Tier::REMOTE, Tier::NONE}) {
        SCOPED_TRACE(tierName(target));
        EXPECT_ANY_THROW(env.cache->insert({100, 101}, resources, target, /*is_resident=*/false));
        EXPECT_TRUE(env.cache->tree()->findNode({100, 101}).empty());
        EXPECT_EQ(env.cache->getKeySnapshot().version, version);
        EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 1u);
        EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 1u);
        backend->finishWrite();
        EXPECT_TRUE(backend->blocks().empty());
        EXPECT_TRUE(backend->keyHandleCounts().empty());
        EXPECT_EQ(backend->submittedCount(), 0u);
    }

    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[1]);
}

TEST(BlockTreeStorerTest, DeviceInsertDoesNotWaitForBackendWriteOrLocalTaskPool) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto backend = std::make_shared<PendingWriteBackend>();
    auto env     = std::make_shared<StoreEnvironment>(makeStoreEnvironment("storage_device_async",
                                                                       /*device_cache_on=*/true,
                                                                       /*host_cache_on=*/false,
                                                                       /*disk_cache_on=*/false,
                                                                       /*lower_tier_blocks=*/{2},
                                                                       /*task_pool_size=*/2,
                                                                       backend));
    backend->setCache(env->cache.get());
    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env->groups[0], 1);
    ASSERT_EQ(holder.size(), 1u);

    auto local_task_entered        = std::make_shared<std::promise<void>>();
    auto local_task_entered_future = local_task_entered->get_future();
    auto release_local_task        = std::make_shared<std::promise<void>>();
    auto release_local_task_future = release_local_task->get_future().share();
    ASSERT_TRUE(
        env->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [local_task_entered, release_local_task_future] {
            local_task_entered->set_value();
            release_local_task_future.wait();
        }));
    const auto local_task_status = local_task_entered_future.wait_for(std::chrono::seconds(5));
    if (local_task_status != std::future_status::ready) {
        release_local_task->set_value();
        FAIL() << "unrelated BlockTree task did not start";
    }

    const auto          resources = deviceSourceResources({holder[0]});
    BoundedThread<void> insert(
        [env, resources] { env->cache->insert({100}, resources, Tier::DEVICE, /*is_resident=*/false); });
    if (!backend->waitForPendingWrite()) {
        backend->finishWrite();
        release_local_task->set_value();
        if (insert.waitFor(std::chrono::seconds(5)) == std::future_status::ready) {
            insert.get();
        }
        FAIL() << "remote write was not submitted";
    }
    const auto before_backend_completion = insert.waitFor(std::chrono::seconds(5));
    backend->finishWrite();
    const auto before_local_release = insert.waitFor(std::chrono::seconds(5));
    release_local_task->set_value();
    const auto after_cleanup = before_local_release == std::future_status::ready ?
                                   before_local_release :
                                   insert.waitFor(std::chrono::seconds(5));
    if (after_cleanup != std::future_status::ready) {
        FAIL() << "DEVICE insert did not finish after releasing every controlled dependency";
    }
    insert.get();
    EXPECT_EQ(before_backend_completion, std::future_status::ready);
    EXPECT_EQ(before_local_release, std::future_status::ready)
        << "DEVICE insert must not wait for unrelated BlockTree tasks";
    BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env->cache);

    EXPECT_EQ(env->cache->tree()->findNode({100}).size(), 1u);
    EXPECT_EQ(backend->blocks(), (std::vector<BlockIdxType>{holder[0][0]}));
    EXPECT_EQ(env->device_pools[0]->refCount(holder[0][0]), 2u);
    releaseDeviceBlocks(*env->cache, env->device_pools[0], holder[0]);
}

TEST(BlockTreeStorerTest, HostAndDiskInsertReturnBeforeTransferSettlement) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    for (const Tier target_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(std::string(tierName(target_tier)));
        auto backend = std::make_shared<PendingWriteBackend>();
        auto env     = std::make_shared<StoreEnvironment>(
            makeStoreEnvironment("store_settlement_" + std::string(tierName(target_tier)) + "_async",
                                 /*device_cache_on=*/false,
                                 /*host_cache_on=*/target_tier == Tier::HOST,
                                 /*disk_cache_on=*/target_tier == Tier::DISK,
                                 /*lower_tier_blocks=*/{2},
                                 /*task_pool_size=*/2,
                                 backend));
        backend->setCache(env->cache.get());
        auto barrier = std::make_shared<CallbackBarrier>();
        installStoreTransferEngine(*env, TransferCopyAction::Succeed, barrier);
        MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env->groups[0], 1);
        ASSERT_EQ(holder.size(), 1u);

        const auto          resources = deviceSourceResources({holder[0]});
        BoundedThread<void> insert([env, resources, target_tier] {
            env->cache->insert({100}, resources, target_tier, /*is_resident=*/false);
        });
        if (!barrier->waitUntilEnteredFor(1, std::chrono::seconds(5))) {
            barrier->release();
            if (insert.waitFor(std::chrono::seconds(5)) == std::future_status::ready) {
                insert.get();
            }
            FAIL() << "HOST/DISK transfer did not enter the controlled barrier";
        }
        const auto before_release =
            insert.waitFor(std::chrono::seconds(5));
        barrier->release();
        const auto after_cleanup =
            before_release == std::future_status::ready ? before_release : insert.waitFor(std::chrono::seconds(5));
        if (after_cleanup != std::future_status::ready) {
            FAIL() << "HOST/DISK insert did not finish after releasing the transfer barrier";
        }
        insert.get();
        EXPECT_EQ(before_release, std::future_status::ready)
            << "async HOST/DISK insert must return while transfer is blocked";
        BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env->cache);

        const auto path = env->cache->tree()->findNode({100});
        ASSERT_EQ(path.size(), 1u);
        const GroupSetResource& resource = path.back()->group_set_resources[0];
        ASSERT_TRUE(resource.hasTier(target_tier));
        const BlockIdxType target_block = resource.getBlocks(target_tier).front();
        EXPECT_EQ(env->poolFor(target_tier).treeRefCount(target_block), 1u);
        EXPECT_EQ(env->storeRefCount(), 0u);
        backend->finishWrite();
        EXPECT_TRUE(backend->blocks().empty());
        EXPECT_TRUE(backend->keyHandleCounts().empty());
        EXPECT_EQ(backend->submittedCount(), 0u);
        EXPECT_EQ(env->device_pools[0]->refCount(holder[0][0]), 1u);
        releaseDeviceBlocks(*env->cache, env->device_pools[0], holder[0]);
    }
}

TEST(BlockTreeStorerTest, DeviceInsertWithoutBackendOnlyPublishesLocalCache) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    StoreEnvironment env    = makeStoreEnvironment("storage_local_only",
                                                /*device_cache_on=*/true,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{2},
                                                /*task_pool_size=*/4,
                                                nullptr);
    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(holder.size(), 1u);

    env.cache->insert({100}, deviceSourceResources({holder[0]}), Tier::DEVICE, /*is_resident=*/false);

    EXPECT_EQ(env.cache->storageBackend(), nullptr);
    EXPECT_EQ(env.cache->tree()->findNode({100}).size(), 1u);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 2u) << "only request and tree references remain";
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
}

TEST(BlockTreeStorerTest, StorageHandlesUseTopologyGroupsAndResolveGpuBuffers) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto make_group = [](std::string tag, int layer) {
        GroupBase group =
            block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {layer});
        auto spec  = group.spec->clone();
        spec->tag  = tag;
        group.tag  = std::move(tag);
        group.spec = std::move(spec);
        return group;
    };
    auto topology  = CacheTopology::create({make_group("z_group", 0), make_group("a_group", 1)},
                                           {{0, {"z_group"}}, {1, {"a_group"}}});
    auto pool_z    = makeDevicePool({{16, 0}}, kStoreDeviceBlocks, "storage_tag_z");
    auto pool_a    = makeDevicePool({{16, 0}}, kStoreDeviceBlocks, "storage_tag_a");
    auto group_set = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool_z, pool_a}, nullptr, nullptr);
    group_set->initialize(0, topology, {0, 1});

    auto                 backend = std::make_shared<PendingWriteBackend>();
    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_remote_cache = true;
    auto cache                 = makeBlockTreeCacheForTest({group_set}, std::move(config), backend);
    backend->setCache(cache.get());

    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*group_set, 1);
    ASSERT_EQ(holder.size(), 1u);
    cache->insert({100}, deviceSourceResources({holder.front()}), Tier::DEVICE, /*is_resident=*/false);
    backend->finishWrite();
    EXPECT_EQ(backend->keyHandleCounts(), (std::vector<size_t>{2}));
    EXPECT_EQ(backend->groupTags(), (std::vector<std::string>{"z_group", "a_group"}));
    EXPECT_EQ(backend->blocks(), (std::vector<BlockIdxType>{holder[0][0], holder[0][1]}));
    EXPECT_EQ(backend->addresses(),
              (std::vector<void*>{pool_z->convertIndexToBuffer(0, holder[0][0]).front().addr,
                                  pool_a->convertIndexToBuffer(0, holder[0][1]).front().addr}));
    releaseDeviceBlocks(*cache, pool_z, {holder[0][0]});
    releaseDeviceBlocks(*cache, pool_a, {holder[0][1]});
}

TEST(BlockTreeStorerTest, StoreToDiskStaysDiscoverableWhenDeviceCacheIsEnabled) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    StoreEnvironment env = makeStoreEnvironment("store_l1_l3_promotion",
                                                /*device_cache_on=*/true,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/true);

    MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(request_holder.size(), 1u);

    env.cache->insert({100}, deviceSourceResources({request_holder[0]}), Tier::DISK, /*is_resident=*/false);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

    auto find = env.cache->tree()->findNode({100});
    ASSERT_EQ(find.size(), 1u);
    const GroupSetResource& resource = find.back()->group_set_resources[0];
    ASSERT_TRUE(resource.hasTier(Tier::DISK));
    EXPECT_TRUE(resource.device_blocks.empty()) << "the request forbade L1, so nothing may be published there";
    const BlockIdxType disk_block = resource.disk_block;

    BlockTreeMatchResult              result       = env.cache->match({100});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_EQ(load_context->matchedBlocks(), 1u);
    ASSERT_EQ(load_context->loadDescs().size(), 1u);
    EXPECT_EQ(load_context->loadDescs()[0].source_tier, Tier::DISK);
    EXPECT_EQ(load_context->loadDescs()[0].target_tier, Tier::DEVICE);
    EXPECT_EQ(load_context->loadDescs()[0].source_blocks, (BlockIndicesType{disk_block}));

    load_context.reset();
    releaseDeviceBlocks(*env.cache, env.device_pools[0], request_holder.front());
}

TEST(BlockTreeStorerTest, StoreKeepsDeviceSourceAliveAfterRequestRelease) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    for (const bool device_cache_on : {false, true}) {
        SCOPED_TRACE(device_cache_on ? "device_cache_on" : "device_cache_off");
        StoreEnvironment env =
            makeStoreEnvironment(std::string("store_source_lifetime_") + (device_cache_on ? "l1" : "no_l1"),
                                 device_cache_on,
                                 /*host_cache_on=*/true,
                                 /*disk_cache_on=*/false);
        auto barrier = std::make_shared<CallbackBarrier>();
        installStoreTransferEngine(env, TransferCopyAction::Succeed, barrier);

        const size_t device_free_before = env.device_pools[0]->freeBlocksNum();
        const size_t host_free_before   = env.host_pools[0]->freeBlocksNum();

        MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
        ASSERT_EQ(request_holder.size(), 1u);
        const BlockIdxType device_block = request_holder[0][0];

        if (device_cache_on) {
            env.cache->insert({100},
                              deviceSourceResources({request_holder[0]}),
                              Tier::DEVICE,
                              /*is_resident=*/false);
        }
        env.cache->insert({100},
                          deviceSourceResources({request_holder[0]}),
                          Tier::HOST,
                          /*is_resident=*/false);
        barrier->waitUntilEntered();
        EXPECT_EQ(env.device_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), 1u);

        releaseDeviceBlocks(*env.cache, env.device_pools[0], request_holder.front());
        EXPECT_TRUE(env.device_pools[0]->isAllocated(device_block)) << "the pending store still owns the source";
        EXPECT_EQ(env.device_pools[0]->refCount(device_block), 1u);
        EXPECT_EQ(env.device_pools[0]->treeRefCount(device_block), device_cache_on ? 2u : 1u);
        EXPECT_EQ(candidateCountForTier(*env.cache, Tier::DEVICE), device_cache_on ? 1u : 0u)
            << "a store hold must not make a cached source ineligible for eviction";
        if (device_cache_on) {
            EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*env.cache, 1, Tier::DEVICE), 1);
        }

        barrier->release();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

        EXPECT_EQ(env.storeRefCount(), 0u);
        if (device_cache_on) {
            EXPECT_FALSE(env.device_pools[0]->isAllocated(device_block));
            EXPECT_EQ(env.device_pools[0]->freeBlocksNum(), device_free_before);
            const std::vector<TreeNode*> path = env.cache->tree()->findNode({100});
            if (path.empty()) {
                EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), host_free_before)
                    << "eviction settled before STORE publication";
            } else {
                const GroupSetResource& resource = path.back()->group_set_resources[0];
                EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
                EXPECT_TRUE(resource.hasTier(Tier::HOST));
                EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), host_free_before - 1)
                    << "STORE publication won the settlement race";
            }
        } else {
            ASSERT_EQ(env.cache->tree()->findNode({100}).size(), 1u);
            EXPECT_FALSE(env.device_pools[0]->isAllocated(device_block));
            EXPECT_EQ(env.device_pools[0]->freeBlocksNum(), device_free_before);
            EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), host_free_before - 1);
            EXPECT_EQ(candidateCountForTier(*env.cache, Tier::HOST), 1u);
        }
    }
}

TEST(BlockTreeStorerTest, StoreCopyFailureLeavesTreeAndPoolsUntouched) {
    for (const Tier target_tier : {Tier::HOST, Tier::DISK}) {
        for (const TransferCopyAction action : {TransferCopyAction::Fail, TransferCopyAction::Throw}) {
            SCOPED_TRACE(std::string(tierName(target_tier)) + "/" + transferCopyActionName(action));
            StoreEnvironment env = makeStoreEnvironment("store_copy_failure_" + std::string(tierName(target_tier)) + "_"
                                                            + transferCopyActionName(action),
                                                        /*device_cache_on=*/false,
                                                        /*host_cache_on=*/target_tier == Tier::HOST,
                                                        /*disk_cache_on=*/target_tier == Tier::DISK);
            auto             engine = installStoreTransferEngine(env, action);

            const size_t target_free_before = env.poolFor(target_tier).freeBlocksNum();
            const size_t device_free_before = env.device_pools[0]->freeBlocksNum();

            MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
            ASSERT_EQ(request_holder.size(), 1u);

            env.cache->insert({100},
                              deviceSourceResources({request_holder[0]}),
                              target_tier,
                              /*is_resident=*/false);
            block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

            EXPECT_EQ(engine->submittedBatchCount(), 1u);
            EXPECT_TRUE(env.cache->tree()->findNode({100}).empty());
            EXPECT_EQ(env.poolFor(target_tier).freeBlocksNum(), target_free_before);
            EXPECT_EQ(env.storeRefCount(), 0u);

            releaseDeviceBlocks(*env.cache, env.device_pools[0], request_holder.front());
            EXPECT_EQ(env.device_pools[0]->freeBlocksNum(), device_free_before);
        }
    }
}

enum class StoreRejection {
    TargetExhausted,
    PartialPrepare
};

const char* storeRejectionName(StoreRejection rejection) {
    switch (rejection) {
        case StoreRejection::TargetExhausted:
            return "target_exhausted";
        case StoreRejection::PartialPrepare:
            return "partial_prepare";
    }
    return "unknown";
}

TEST(BlockTreeStorerTest, StoreRejectionRollsBackEveryTemporaryHolderExactlyOnce) {
    for (const StoreRejection rejection : {StoreRejection::TargetExhausted, StoreRejection::PartialPrepare}) {
        SCOPED_TRACE(storeRejectionName(rejection));
        const std::vector<size_t> lower_tier_blocks =
            rejection == StoreRejection::PartialPrepare ? std::vector<size_t>{1, 1} : std::vector<size_t>{1};
        StoreEnvironment env    = makeStoreEnvironment(std::string("store_reject_") + storeRejectionName(rejection),
                                                    /*device_cache_on=*/false,
                                                    /*host_cache_on=*/true,
                                                    /*disk_cache_on=*/false,
                                                    lower_tier_blocks);
        auto             engine = installStoreTransferEngine(env, TransferCopyAction::Succeed);

        std::vector<BlockIdxType> squatters(env.groups.size(), NULL_BLOCK_IDX);
        const size_t              failing_group_set_id = env.groups.size() - 1;
        const auto                squatter             = env.host_pools[failing_group_set_id]->malloc();
        ASSERT_TRUE(squatter.has_value());
        env.host_pools[failing_group_set_id]->incTreeRef(*squatter, BlockTreeRefType::LOAD);
        squatters[failing_group_set_id] = *squatter;

        std::vector<MultiNodeBlocks>           holders;
        std::vector<std::vector<BlockIdxType>> sources;
        std::vector<size_t>                    device_free_before;
        for (const GroupSetPtr& group : env.groups) {
            holders.push_back(allocateDeviceBlocksForTest(*group, 1));
            ASSERT_EQ(holders.back().size(), 1u);
            sources.push_back(holders.back()[0]);
            device_free_before.push_back(kStoreDeviceBlocks);
        }

        env.cache->insert({100}, deviceSourceResources(sources), Tier::HOST, /*is_resident=*/false);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

        EXPECT_EQ(engine->submittedBatchCount(), 0u) << "a rejected store must never copy";
        EXPECT_TRUE(env.cache->tree()->findNode({100}).empty());
        EXPECT_EQ(env.storeRefCount(), 0u);
        for (size_t group_set_id = 0; group_set_id < env.groups.size(); ++group_set_id) {
            const size_t squatted = isNullBlockIdx(squatters[group_set_id]) ? 0u : 1u;
            EXPECT_EQ(env.host_pools[group_set_id]->freeBlocksNum(), lower_tier_blocks[group_set_id] - squatted);
            if (squatted > 0) {
                env.host_pools[group_set_id]->decTreeRef(squatters[group_set_id], BlockTreeRefType::LOAD);
            }
            releaseDeviceBlocks(*env.cache, env.device_pools[group_set_id], holders[group_set_id].front());
            EXPECT_EQ(env.device_pools[group_set_id]->freeBlocksNum(), device_free_before[group_set_id]);
        }
    }
}

TEST(BlockTreeStorerTest, DuplicateStoreForSameKeyReleasesLoserBlock) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    StoreEnvironment env              = makeStoreEnvironment("store_duplicate_key",
                                                /*device_cache_on=*/false,
                                                /*host_cache_on=*/true,
                                                /*disk_cache_on=*/false);
    const size_t     host_free_before = env.host_pools[0]->freeBlocksNum();

    MultiNodeBlocks first_holder  = allocateDeviceBlocksForTest(*env.groups[0], 1);
    MultiNodeBlocks second_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(first_holder.size(), 1u);
    ASSERT_EQ(second_holder.size(), 1u);

    auto barrier = std::make_shared<CallbackBarrier>();
    auto engine  = installStoreTransferEngine(env, TransferCopyAction::Succeed, barrier);

    env.cache->insert({100}, deviceSourceResources({first_holder[0]}), Tier::HOST, /*is_resident=*/false);
    env.cache->insert({100}, deviceSourceResources({second_holder[0]}), Tier::HOST, /*is_resident=*/false);
    barrier->waitUntilEntered(2);
    EXPECT_EQ(engine->submittedBatchCount(), 2u);
    barrier->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

    auto find = env.cache->tree()->findNode({100});
    ASSERT_EQ(find.size(), 1u);
    const GroupSetResource& resource = find.back()->group_set_resources[0];
    ASSERT_TRUE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(env.host_pools[0]->treeRefCount(resource.host_block), 1u);
    EXPECT_EQ(candidateCountForTier(*env.cache, Tier::HOST), 1u);
    EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), host_free_before - 1) << "the losing copy must be returned";
    EXPECT_EQ(env.storeRefCount(), 0u);

    releaseDeviceBlocks(*env.cache, env.device_pools[0], first_holder.front());
    releaseDeviceBlocks(*env.cache, env.device_pools[0], second_holder.front());
}

}  // namespace
}  // namespace rtp_llm
