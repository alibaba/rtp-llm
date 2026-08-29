#include <gtest/gtest.h>

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
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
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
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

private:
    std::mutex       mutex_;
    std::deque<Task> tasks_;
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
        env.cache->insert({100}, deviceSourceResources({request_holder[0]}), target_tier);
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

    env.cache->insert({100, 101}, resources, Tier::DEVICE);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 3u);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 3u);
    backend->finishWrite();
    EXPECT_TRUE(backend->submittedOutsideTreeLock());
    EXPECT_EQ(backend->keyHandleCounts(), (std::vector<size_t>{1, 1}));
    EXPECT_EQ(backend->blocks(), (std::vector<BlockIdxType>{holder[0][0], holder[1][0]}));
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 2u);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 2u);

    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[1]);
}

TEST(BlockTreeStorerTest, RemoteOnlyInsertWritesWithoutPublishingDeviceResidency) {
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

    env.cache->insert({100, 101}, resources, Tier::REMOTE, /*write_remote=*/true);
    EXPECT_TRUE(env.cache->tree()->findNode({100, 101}).empty());
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 2u);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 2u);

    backend->finishWrite();
    EXPECT_TRUE(backend->submittedOutsideTreeLock());
    EXPECT_EQ(backend->keyHandleCounts(), (std::vector<size_t>{1, 1}));
    EXPECT_EQ(backend->blocks(), (std::vector<BlockIdxType>{holder[0][0], holder[1][0]}));
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 1u);
    EXPECT_EQ(env.device_pools[0]->refCount(holder[1][0]), 1u);

    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[1]);
}

TEST(BlockTreeStorerTest, DeviceInsertSkipsRemoteWriteWhenRequestDisablesIt) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto             backend = std::make_shared<PendingWriteBackend>();
    StoreEnvironment env     = makeStoreEnvironment("storage_remote_disabled",
                                                /*device_cache_on=*/true,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{2},
                                                /*task_pool_size=*/4,
                                                backend);
    backend->setCache(env.cache.get());
    MultiNodeBlocks holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(holder.size(), 1u);

    env.cache->insert({100}, deviceSourceResources({holder[0]}), Tier::DEVICE, /*write_remote=*/false);

    EXPECT_TRUE(backend->blocks().empty());
    EXPECT_FALSE(backend->submittedOutsideTreeLock());
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), 2u) << "only request and tree references remain";
    releaseDeviceBlocks(*env.cache, env.device_pools[0], holder[0]);
}

TEST(BlockTreeStorerTest, RemoteOnlyInsertRejectsDisabledRemoteWriteWithoutPinning) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto             backend = std::make_shared<PendingWriteBackend>();
    StoreEnvironment env     = makeStoreEnvironment("storage_invalid_remote_only",
                                                /*device_cache_on=*/false,
                                                /*host_cache_on=*/false,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{2},
                                                /*task_pool_size=*/4,
                                                backend);
    MultiNodeBlocks  holder  = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(holder.size(), 1u);
    const size_t ref_count_before = env.device_pools[0]->refCount(holder[0][0]);

    CoreDumpGuard guard;
    EXPECT_ANY_THROW(
        env.cache->insert({100}, deviceSourceResources({holder[0]}), Tier::REMOTE, /*write_remote=*/false));
    EXPECT_TRUE(env.cache->tree()->findNode({100}).empty());
    EXPECT_EQ(env.device_pools[0]->refCount(holder[0][0]), ref_count_before);
    EXPECT_TRUE(backend->blocks().empty());

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
    cache->insert({100}, deviceSourceResources({holder.front()}), Tier::DEVICE);
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

    env.cache->insert({100}, deviceSourceResources({request_holder[0]}), Tier::DISK);
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
            env.cache->insert({100}, deviceSourceResources({request_holder[0]}), Tier::DEVICE);
        }
        env.cache->insert({100}, deviceSourceResources({request_holder[0]}), Tier::HOST);
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

            env.cache->insert({100}, deviceSourceResources({request_holder[0]}), target_tier);
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

        env.cache->insert({100}, deviceSourceResources(sources), Tier::HOST);
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

    env.cache->insert({100}, deviceSourceResources({first_holder[0]}), Tier::HOST);
    env.cache->insert({100}, deviceSourceResources({second_holder[0]}), Tier::HOST);
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

TEST(BlockTreeStorerTest, StoreShutdownCutoffSettlesQueuedAndInFlightTasksWithoutPublishing) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kStoreTasks = 3;
    StoreEnvironment env         = makeStoreEnvironment("store_shutdown_cutoff",
                                                /*device_cache_on=*/false,
                                                /*host_cache_on=*/true,
                                                /*disk_cache_on=*/false,
                                                /*lower_tier_blocks=*/{kStoreTasks},
                                                /*task_pool_size=*/2);
    auto             barrier     = std::make_shared<CallbackBarrier>();
    installStoreTransferEngine(env, TransferCopyAction::Succeed, barrier);

    const size_t device_free_before = env.device_pools[0]->freeBlocksNum();
    const size_t host_free_before   = env.host_pools[0]->freeBlocksNum();

    std::vector<MultiNodeBlocks> holders;
    for (size_t index = 0; index < kStoreTasks; ++index) {
        holders.push_back(allocateDeviceBlocksForTest(*env.groups[0], 1));
        ASSERT_EQ(holders.back().size(), 1u);
        env.cache->insert(
            {static_cast<CacheKeyType>(100 + index)}, deviceSourceResources({holders.back()[0]}), Tier::HOST);
    }
    barrier->waitUntilEntered(2);
    EXPECT_EQ(env.host_pools[0]->referencedBlocksNum(BlockTreeRefType::STORE), kStoreTasks);

    BlockTreeCacheTestPeer::beginStoreShutdownForTest(*env.cache);
    barrier->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

    for (size_t index = 0; index < kStoreTasks; ++index) {
        EXPECT_TRUE(env.cache->tree()->findNode({static_cast<CacheKeyType>(100 + index)}).empty())
            << "no store may publish after the shutdown cutoff";
    }
    EXPECT_EQ(env.storeRefCount(), 0u);
    EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), host_free_before);

    env.cache.reset();
    for (const MultiNodeBlocks& holder : holders) {
        env.device_pools[0]->decRef(holder.front());
    }
    EXPECT_EQ(env.device_pools[0]->freeBlocksNum(), device_free_before);
}

TEST(BlockTreeStorerTest, StoreDerivedEvictionIsDrainedByWaitForPendingTasks) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    StoreEnvironment env = makeStoreEnvironment("store_derived_eviction",
                                                /*device_cache_on=*/false,
                                                /*host_cache_on=*/true,
                                                /*disk_cache_on=*/true);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*env.cache, Tier::HOST, 0.01);

    MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*env.groups[0], 1);
    ASSERT_EQ(request_holder.size(), 1u);

    env.cache->insert({100}, deviceSourceResources({request_holder[0]}), Tier::HOST);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*env.cache);

    auto find = env.cache->tree()->findNode({100});
    ASSERT_EQ(find.size(), 1u);
    const GroupSetResource& resource = find.back()->group_set_resources[0];
    EXPECT_TRUE(resource.hasTier(Tier::DISK)) << "store-derived shared-pool work must be waited for too";
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(env.host_pools[0]->freeBlocksNum(), 2u);
    EXPECT_EQ(env.storeRefCount(), 0u);

    releaseDeviceBlocks(*env.cache, env.device_pools[0], request_holder.front());
}

}  // namespace
}  // namespace rtp_llm
