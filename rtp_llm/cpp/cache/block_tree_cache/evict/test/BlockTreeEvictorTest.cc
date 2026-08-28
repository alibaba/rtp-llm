#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::allocateDeviceBlocksForTest;
using block_tree_cache_test::MultiNodeBlocks;
using block_tree_cache_test::releaseLowerTierSeedRefs;
using ScriptedTransferEngine = block_tree_cache_test::ScriptedPerRankBlockTransferEngine;
using block_tree_cache_test::unreferenceDeviceBlocksForTest;

static_assert(!noexcept(std::declval<BlockTreeEvictor&>().settleEvictionLocked(
    std::declval<const EvictionTask&>(), std::declval<const EvictionTaskResult&>())));
static_assert(!noexcept(std::declval<BlockTreeEvictor&>().abortEvictionLocked(std::declval<const EvictionTask&>())));
static_assert(noexcept(std::declval<BlockTreeEvictor&>().runEvictionTask(
    std::declval<std::shared_ptr<const EvictionTask>>())));
static_assert(noexcept(std::declval<BlockTreeEvictor&>().finalizeEvictionLocked(
    std::declval<const EvictionTask&>(), std::declval<const EvictionTaskResult&>())));

class DisableCoreDumpGuard {
public:
    DisableCoreDumpGuard(): old_value_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    ~DisableCoreDumpGuard() {
        StaticConfig::user_ft_core_dump_on_exception = old_value_;
    }

private:
    bool old_value_;
};

std::shared_ptr<FullGroupSet> makeFullGroup(const DeviceBlockPoolPtr& device_pool) {
    return std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, nullptr, nullptr);
}

TreeNode* insertedNode(const BlockTreeInsertResult& result) {
    return result.inserted_nodes.back();
}

std::shared_ptr<HostBlockPool> makePageableHostPool(size_t usable_blocks) {
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = "block_tree_evictor_test_host";
    config->physical_block_count = usable_blocks + 1;
    config->payload_bytes        = 64;
    config->stride_bytes         = 4096;
    config->enable_pinned        = false;
    config->alignment            = 4096;

    auto pool = std::make_shared<HostBlockPool>(config);
    if (!pool->init()) {
        return nullptr;
    }
    return pool;
}

class NoopDiskBlockIO final: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(uint64_t, void*, size_t) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(uint64_t, const void*, size_t) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus read(const std::vector<DiskRead>&) override {
        return DiskBlockIOStatus::OK;
    }

    DiskBlockIOStatus write(const std::vector<DiskWrite>&) override {
        return DiskBlockIOStatus::OK;
    }

    void close() override {}

    std::string debugString() const override {
        return "NoopDiskBlockIO";
    }
};

std::shared_ptr<BlockTreeDiskBlockPool> makeTestDiskPool(size_t usable_blocks, const std::string& name) {
    auto config                  = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type            = BlockPoolType::DISK;
    config->pool_name            = name;
    config->work_dir             = "/tmp";
    config->payload_bytes        = 64;
    config->stride_bytes         = 64;
    config->disk_size_bytes      = (usable_blocks + 1) * config->stride_bytes;
    config->physical_block_count = usable_blocks + 1;
    config->buffered_io          = true;

    auto pool = std::make_shared<BlockTreeDiskBlockPool>(config, std::make_unique<NoopDiskBlockIO>());
    if (!pool->init()) {
        return nullptr;
    }
    return pool;
}

DeviceBlockPoolPtr makeTestDevicePool(size_t usable_blocks, const std::string& name) {
    const size_t physical_blocks = usable_blocks + 1;
    const size_t block_bytes     = 16;

    MemoryLayoutConfig layout;
    layout.layer_num                  = 1;
    layout.block_num                  = static_cast<uint32_t>(physical_blocks);
    layout.dtype                      = TYPE_INT8;
    layout.kv_cache_offset_bytes      = 0;
    layout.kv_block_stride_bytes      = block_bytes;
    layout.kv_block_pool_size_bytes   = physical_blocks * block_bytes;
    layout.block_stride_bytes         = block_bytes;
    layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
    layout.local_head_num_kv          = 1;
    layout.seq_size_per_block         = 1;
    layout.kernel_blocks_per_kv_block = 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = name;
    config->physical_block_count    = physical_blocks;
    config->total_size_bytes        = layout.total_size_bytes;
    config->memory_layouts          = {layout};
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    if (!pool->init()) {
        return nullptr;
    }
    return pool;
}

void initializeGroups(const std::vector<GroupSetPtr>&        groups,
                      const std::vector<DeviceBlockPoolPtr>& device_pools,
                      std::vector<GroupBase>                 group_bases) {
    RTP_LLM_CHECK(groups.size() == device_pools.size());
    RTP_LLM_CHECK(groups.size() == group_bases.size());
    auto topology = block_transfer_engine_test::makeTestTopology(std::move(group_bases));
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        groups[group_set_id]->initialize(group_set_id, topology, {group_set_id});
    }
}

void initializeFullGroup(const GroupSetPtr& group, const DeviceBlockPoolPtr& device_pool) {
    initializeGroups(
        {group},
        {device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16)});
}

std::vector<GroupSetPtr> makeCascadeGroups() {
    auto full_device_pool   = makeTestDevicePool(2, "cascade_policy_full");
    auto swa_device_pool    = makeTestDevicePool(2, "cascade_policy_swa");
    auto linear_device_pool = makeTestDevicePool(2, "cascade_policy_linear");
    auto full               = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{full_device_pool}, makePageableHostPool(4), nullptr);
    auto swa = std::make_shared<SWAGroupSet>(
        2, 1, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, makePageableHostPool(4), nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{linear_device_pool}, makePageableHostPool(4), nullptr);
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    full_policy.enable_prefix_reuse   = true;
    swa_policy.enable_prefix_reuse    = true;
    linear_policy.enable_prefix_reuse = true;
    swa_policy.sliding_window_size    = 2;
    std::vector<GroupSetPtr> groups   = {full, swa, linear};
    initializeGroups(groups,
                     {full_device_pool, swa_device_pool, linear_device_pool},
                     {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16),
                      block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16),
                      block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16)});
    return groups;
}

GroupSetResource makeResource(Tier tier, BlockIdxType block) {
    GroupSetResource resource;
    switch (tier) {
        case Tier::DEVICE:
            resource.device_blocks = {block};
            break;
        case Tier::HOST:
            resource.host_block = block;
            break;
        case Tier::DISK:
            resource.disk_block = block;
            break;
        default:
            break;
    }
    return resource;
}

class TestEvictorRuntime {
public:
    std::unique_ptr<BlockTreeEvictor> make(
        BlockTree*                        tree,
        EvictionPolicy                    device_policy   = EvictionPolicy::LRU,
        EvictionPolicy                    host_policy     = EvictionPolicy::LRU,
        EvictionPolicy                    disk_policy     = EvictionPolicy::FIFO,
        BlockTreeEvictor::IsTierEnabledFn is_tier_enabled = [](Tier) { return true; }) {
        transfer_engine_     = std::make_shared<ScriptedTransferEngine>(tree->groupSets(), false);
        transfer_dispatcher_ = std::make_unique<BlockTransferDispatcher>(transfer_engine_);
        return std::make_unique<BlockTreeEvictor>(tree,
                                                  device_policy,
                                                  host_policy,
                                                  disk_policy,
                                                  transfer_dispatcher_.get(),
                                                  nullptr,
                                                  metrics_reporter_,
                                                  mutex_,
                                                  0,
                                                  0,
                                                  std::move(is_tier_enabled),
                                                  [](bool, bool) {});
    }

    std::shared_ptr<ScriptedTransferEngine> transferEngine() const {
        return transfer_engine_;
    }

    size_t transferCount() const {
        return transfer_engine_->submittedBatchCount();
    }

private:
    BlockTreeCacheMetricsReporter            metrics_reporter_;
    std::shared_ptr<ScriptedTransferEngine>  transfer_engine_;
    std::unique_ptr<BlockTransferDispatcher> transfer_dispatcher_;
    std::mutex                               mutex_;
};

class DeferredEvictionTransferEngine final: public PerRankBlockTransferEngine {
public:
    explicit DeferredEvictionTransferEngine(const std::vector<GroupSetPtr>& groups):
        PerRankBlockTransferEngine(groups) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        auto context = std::make_shared<TransferBatchAsyncContext>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            descriptors_.push_back(descriptors);
            contexts_.push_back(context);
        }
        cv_.notify_all();
        return context;
    }

    bool waitForBatchCount(size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return contexts_.size() >= expected; });
    }

    void complete(size_t index, bool success) {
        std::shared_ptr<TransferBatchAsyncContext> context;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context = contexts_.at(index);
        }
        context->complete(success ? ErrorInfo::OkStatus() :
                                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
    }

private:
    std::mutex                                             mutex_;
    std::condition_variable                                cv_;
    std::vector<std::vector<TransferDescriptor>>           descriptors_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;
};

class BlockTreeEvictorTestPeer {
public:
    static void reserveSource(BlockTreeEvictor& evictor, const TransferDescriptor& eviction_desc) {
        evictor.reserveSource(eviction_desc);
    }

    static EvictionTask selectCascades(BlockTreeEvictor& evictor, TransferDescriptor primary_desc) {
        EvictionTask task;
        task.primary_desc = std::move(primary_desc);
        std::vector<std::pair<TreeNode*, size_t>> detached_resources;
        evictor.selectCascades(task, detached_resources);
        return task;
    }

    static void rollbackDesc(BlockTreeEvictor& evictor, const TransferDescriptor& eviction_desc) {
        EvictionTask task;
        task.primary_desc = eviction_desc;
        evictor.abortEvictionLocked(task);
    }
};

TEST(BlockTreeEvictorAsyncTest, PendingTransferDoesNotOccupyBusinessWorker) {
    auto device_pool = makeTestDevicePool(1, "async_eviction_device");
    auto host_pool   = makePageableHostPool(1);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);

    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/1, "eviction_async_test");
    ASSERT_TRUE(task_pool.start());
    auto deferred_engine = std::make_shared<DeferredEvictionTransferEngine>(groups);
    BlockTransferDispatcher dispatcher(deferred_engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    std::mutex cache_mutex;
    size_t     settled_count = 0;
    BlockTreeEvictor evictor(&tree,
                             EvictionPolicy::LRU,
                             EvictionPolicy::LRU,
                             EvictionPolicy::FIFO,
                             &dispatcher,
                             &task_pool,
                             metrics_reporter,
                             cache_mutex,
                             0,
                             0,
                             [](Tier) { return true; },
                             [&](bool, bool) { ++settled_count; });

    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    const BlockIdxType source = device_blocks.front().front();
    auto inserted = tree.insertNode({100}, {{makeResource(Tier::DEVICE, source)}}, /*collect_path=*/false);
    unreferenceDeviceBlocksForTest(*group, device_blocks, BlockTreeRefType::CACHE);
    evictor.onInserted(inserted);

    ASSERT_TRUE(evictor.evictLocked(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/false));
    ASSERT_TRUE(deferred_engine->waitForBatchCount(1, std::chrono::seconds(2)));
    EXPECT_FALSE(task_pool.acquireBusinessCredit());

    std::mutex              marker_mutex;
    std::condition_variable marker_cv;
    bool                    marker_ran = false;
    ASSERT_TRUE(task_pool.submit([&] {
        {
            std::lock_guard<std::mutex> lock(marker_mutex);
            marker_ran = true;
        }
        marker_cv.notify_one();
    }));
    {
        std::unique_lock<std::mutex> lock(marker_mutex);
        EXPECT_TRUE(marker_cv.wait_for(lock, std::chrono::milliseconds(300), [&] { return marker_ran; }));
    }
    EXPECT_EQ(settled_count, 0u);

    deferred_engine->complete(0, true);
    task_pool.waitForIdle();
    EXPECT_EQ(settled_count, 1u);
    EXPECT_TRUE(task_pool.acquireBusinessCredit());
    task_pool.releaseBusinessCredit();
    task_pool.shutdown();
}

std::vector<size_t> cascadeGroupSetIds(const EvictionTask& task) {
    std::vector<size_t> result;
    result.reserve(task.cascade_descs.size());
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        result.push_back(cascade_desc.group_set_id);
    }
    return result;
}

TransferDescriptor makeSelectionDesc(TreeNode* node, size_t group_set_id, Tier source_tier, Tier target_tier) {
    return TransferDescriptor(node,
                              group_set_id,
                              /*path_index=*/0,
                              source_tier,
                              target_tier,
                              node->group_set_resources[group_set_id].getBlocks(source_tier));
}

std::vector<BlockIdxType> exhaustPool(IBlockPool& pool) {
    std::vector<BlockIdxType> blocks;
    while (true) {
        auto block = pool.malloc();
        if (!block.has_value()) {
            break;
        }
        pool.incTreeRef(*block, BlockTreeRefType::STORE);
        blocks.push_back(*block);
    }
    return blocks;
}

void releaseBlocks(IBlockPool& pool, const std::vector<BlockIdxType>& blocks) {
    for (BlockIdxType block : blocks) {
        pool.decTreeRef(block, BlockTreeRefType::STORE);
    }
}

class CascadeTestEnvironment {
public:
    bool init() {
        std::vector<DeviceBlockPoolPtr> device_pools = {
            makeTestDevicePool(2, "cascade_environment_full"),
            makeTestDevicePool(2, "cascade_environment_swa"),
            makeTestDevicePool(2, "cascade_environment_linear"),
        };
        for (size_t group_set_id = 0; group_set_id < device_pools.size(); ++group_set_id) {
            auto host = makePageableHostPool(2);
            auto disk = makeTestDiskPool(2, "block_tree_evictor_cascade_" + std::to_string(group_set_id));
            if (host == nullptr || disk == nullptr) {
                return false;
            }
            host_pools_.push_back(std::move(host));
            disk_pools_.push_back(std::move(disk));
        }

        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pools[0]}, host_pools_[0], disk_pools_[0]);
        auto swa = std::make_shared<SWAGroupSet>(
            2, 1, std::vector<DeviceBlockPoolPtr>{device_pools[1]}, host_pools_[1], disk_pools_[1]);
        auto linear = std::make_shared<LinearGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pools[2]}, host_pools_[2], disk_pools_[2]);
        groups_                           = {full, swa, linear};
        auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
        auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
        auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        full_policy.enable_prefix_reuse   = true;
        swa_policy.enable_prefix_reuse    = true;
        linear_policy.enable_prefix_reuse = true;
        swa_policy.sliding_window_size    = 2;
        initializeGroups(groups_,
                         device_pools,
                         {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16),
                          block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16),
                          block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16)});

        tree_ = std::make_unique<BlockTree>(groups_);

        evictor_ = evictor_runtime_.make(tree_.get());

        std::vector<GroupSetResource> resources(groups_.size());
        host_blocks_.resize(groups_.size(), NULL_BLOCK_IDX);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            host_blocks_[group_set_id] =
                groups_[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            if (isNullBlockIdx(host_blocks_[group_set_id])) {
                return false;
            }
            resources[group_set_id].host_block = host_blocks_[group_set_id];
        }

        auto result = tree_->insertNode({100}, {resources}, /*collect_path=*/false);
        releaseLowerTierSeedRefs(groups_, {resources});
        evictor_->onInserted(result);
        node_ = insertedNode(result);
        return node_ != nullptr;
    }

    std::optional<EvictionTask> prepareTask(size_t primary_group_set_id) {
        auto victim = evictor_->chooseVictim(primary_group_set_id, Tier::HOST);
        if (!victim.has_value()) {
            return std::nullopt;
        }
        return evictor_->prepareEvictionLocked(*victim);
    }

    MultiNodeResource hostSet(size_t group_set_id) const {
        return MultiNodeResource{
            group_set_id, Tier::HOST, {{node_, {host_blocks_[static_cast<size_t>(group_set_id)]}}}};
    }

    void setTransferResults(std::initializer_list<bool> results) {
        auto transfer_engine = evictor_runtime_.transferEngine();
        transfer_engine->clear();
        for (bool success : results) {
            transfer_engine->enqueue(success);
        }
    }

    EvictionTaskResult runTransfer(const EvictionTask& task) {
        std::optional<EvictionTaskResult> result;
        evictor_->task_runner_->runTransfer(
            std::make_shared<EvictionTask>(task),
            *evictor_->metrics_reporter_,
            [&](EvictionTaskResult completed) { result = std::move(completed); });
        if (!result.has_value()) {
            ADD_FAILURE() << "scripted eviction transfer did not complete inline";
            return {};
        }
        return std::move(*result);
    }

    std::vector<size_t> transferGroupSetIds() const {
        std::vector<size_t> group_set_ids;
        for (const auto& descriptor : evictor_runtime_.transferEngine()->descriptors()) {
            group_set_ids.push_back(descriptor.group_set_id);
        }
        return group_set_ids;
    }

    void releaseResidentBlocks() {
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            auto& resource = node_->group_set_resources[group_set_id];
            if (resource.hasTier(Tier::HOST)) {
                const BlockIdxType block = resource.host_block;
                resource.host_block      = NULL_BLOCK_IDX;
                groups_[group_set_id]->releaseSingleBlock(Tier::HOST, block, BlockTreeRefType::CACHE);
            }
            if (resource.hasTier(Tier::DISK)) {
                const BlockIdxType block = resource.disk_block;
                resource.disk_block       = NULL_BLOCK_IDX;
                groups_[group_set_id]->releaseSingleBlock(Tier::DISK, block, BlockTreeRefType::CACHE);
            }
        }
    }

    void expectAllPoolsFree() const {
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            EXPECT_EQ(host_pools_[group_set_id]->freeBlocksNum(), 2u);
            EXPECT_EQ(disk_pools_[group_set_id]->freeBlocksNum(), 2u);
        }
    }

    std::vector<GroupSetPtr>                             groups_;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools_;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools_;
    std::vector<BlockIdxType>                            host_blocks_;
    std::unique_ptr<BlockTree>                           tree_;
    TestEvictorRuntime                                   evictor_runtime_;
    std::unique_ptr<BlockTreeEvictor>                    evictor_;
    TreeNode*                                            node_{nullptr};
};

class BlockTreeEvictorTest: public ::testing::Test {
protected:
    void SetUp() override {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(test_info, nullptr);
        device_pool_ = makeTestDevicePool(128, "block_tree_evictor_fixture_" + std::string(test_info->name()));
        ASSERT_NE(device_pool_, nullptr);
        resetGroup();
    }

    void resetGroup(std::shared_ptr<HostBlockPool> host_pool = nullptr, BlockTreeDiskBlockPoolPtr disk_pool = nullptr) {
        group_ = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{device_pool_}, std::move(host_pool), std::move(disk_pool));
        initializeFullGroup(group_, device_pool_);
        groups_  = {group_};
        tree_    = std::make_unique<BlockTree>(groups_);
        evictor_ = evictor_runtime_.make(tree_.get());
    }

    BlockTreeInsertResult insert(const CacheKeysType&                              keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources) {
        auto result = tree_->insertNode(keys, resources, /*collect_path=*/false);
        releaseLowerTierSeedRefs(groups_, resources);
        evictor_->onInserted(result);
        return result;
    }

    bool reserveAndBeginLoad(TreeNode* node, size_t group_set_id, Tier source) {
        GroupSetResource& resource = node->group_set_resources[group_set_id];
        if (resource.transfer_state != GroupSetTransferState::IDLE || resource.getTopTier() != source) {
            return false;
        }
        resource.transfer_state = GroupSetTransferState::LOAD_PENDING;
        evictor_->suspendCandidate(node, group_set_id, source);
        resource.transfer_state = GroupSetTransferState::LOADING;
        return true;
    }

    void settleLoad(TreeNode* node, size_t group_set_id, bool copy_ok) {
        node->group_set_resources[group_set_id].transfer_state = GroupSetTransferState::IDLE;
        if (copy_ok) {
            evictor_->onLoaded(node, group_set_id);
        } else {
            evictor_->admitCandidate(node, group_set_id, node->group_set_resources[group_set_id].getTopTier());
        }
    }

    std::shared_ptr<FullGroupSet>     group_;
    DeviceBlockPoolPtr                device_pool_;
    std::vector<GroupSetPtr>          groups_;
    std::unique_ptr<BlockTree>        tree_;
    TestEvictorRuntime                evictor_runtime_;
    std::unique_ptr<BlockTreeEvictor> evictor_;
};

TEST_F(BlockTreeEvictorTest, PendingReleasesFollowAsyncTaskSourcePools) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(2, "pending_release_task_disk");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    auto verify_tier = [this](Tier source_tier, Tier target_tier, IBlockPool* pool, BlockIdxType block) {
        EvictionTask task;
        task.primary_desc.group_set_id  = 0;
        task.primary_desc.source_tier   = source_tier;
        task.primary_desc.target_tier   = target_tier;
        task.primary_desc.source_blocks = {block};

        evictor_->updatePendingReleases(task, true);
        ASSERT_EQ(evictor_->pending_release_counts_.at(pool), 1u);
        evictor_->updatePendingReleases(task, false);
        EXPECT_EQ(evictor_->pending_release_counts_.at(pool), 0u);
    };

    verify_tier(Tier::DEVICE, Tier::HOST, device_pool_.get(), 7);
    verify_tier(Tier::HOST, Tier::DISK, host_pool.get(), 8);
}

TEST_F(BlockTreeEvictorTest, PendingReleasesCountEveryDeviceMemberBlock) {
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology =
        block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(policy, {0}, 16),
                                                      block_transfer_engine_test::makeTestGroupBase(policy, {1}, 16)});
    group_ =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool_, device_pool_}, nullptr, nullptr);
    group_->initialize(0, std::move(topology), {0, 1});
    groups_  = {group_};
    tree_    = std::make_unique<BlockTree>(groups_);
    evictor_ = evictor_runtime_.make(tree_.get());

    EvictionTask task;
    task.primary_desc.group_set_id  = 0;
    task.primary_desc.source_tier   = Tier::DEVICE;
    task.primary_desc.target_tier   = Tier::HOST;
    task.primary_desc.source_blocks = {7, 8};

    evictor_->updatePendingReleases(task, true);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 2u);
    evictor_->updatePendingReleases(task, false);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
}

TEST_F(BlockTreeEvictorTest, UpdatePendingReleasesReportsPoolAndBlockOnInvalidSettlement) {
    constexpr BlockIdxType block = 17;
    EvictionTask           task;
    task.primary_desc.group_set_id  = 0;
    task.primary_desc.source_tier   = Tier::DEVICE;
    task.primary_desc.target_tier   = Tier::HOST;
    task.primary_desc.source_blocks = {block};

    try {
        evictor_->updatePendingReleases(task, false);
        FAIL() << "settling an unreserved pending release should fail";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("pool=" + device_pool_->poolName()), std::string::npos);
        EXPECT_NE(message.find("block=17"), std::string::npos);
        EXPECT_NE(message.find("pending=0"), std::string::npos);
    }
}

TEST_F(BlockTreeEvictorTest, CompleteEvictRejectsNonDemotingResource) {
    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    TreeNode*          node         = insertedNode(result);
    ASSERT_NE(node, nullptr);

    const TransferDescriptor desc{node, /*group_set_id=*/0, /*path_index=*/0, Tier::DEVICE, Tier::NONE, {source_block}};
    DisableCoreDumpGuard     core_dump_guard;
    EXPECT_THROW(evictor_->completeEvict(desc), std::exception);

    evictor_->suspendCandidate(node, 0, Tier::DEVICE);
    node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, RunEvictionTaskTerminatesOnSettledCallbackException) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto task = evictor_->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    evictor_->updatePendingReleases(*task, true);
    evictor_->settled_ = [](bool, bool) { throw std::runtime_error("settled callback failure"); };
    evictor_runtime_.transferEngine()->enqueue(false);

    EXPECT_DEATH(evictor_->runEvictionTask(std::make_shared<EvictionTask>(*task)), "");

    evictor_->settled_ = [](bool, bool) {};
    evictor_->updatePendingReleases(*task, false);
    evictor_->abortEvictionLocked(*task);
}

TEST_F(BlockTreeEvictorTest, RunEvictionTaskReleasesPendingCapacityBeforeSettledCallback) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto task = evictor_->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    ASSERT_TRUE(task->needsCopy());
    evictor_->updatePendingReleases(*task, true);
    ASSERT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 1u);
    size_t settled_count = 0;
    evictor_->settled_   = [this, &settled_count](bool tree_data_mutated, bool check_watermark) {
        ++settled_count;
        EXPECT_FALSE(tree_data_mutated);
        EXPECT_FALSE(check_watermark);
        EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
    };
    evictor_runtime_.transferEngine()->enqueue(false);

    evictor_->runEvictionTask(std::make_shared<EvictionTask>(*task));

    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
    EXPECT_EQ(settled_count, 1u);
    GroupSetResource&       resource = insertedNode(result)->group_set_resources[0];
    const MultiNodeResource source{0, Tier::DEVICE, {{insertedNode(result), {source_block}}}};
    evictor_->suspendCandidate(insertedNode(result), 0, Tier::DEVICE);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, PoolWatermarkExcessRejectsPendingReleasesAboveUsedBlocks) {
    ASSERT_EQ(device_pool_->usedBlocksNum(), 0u);
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_[device_pool_.get()] = 1;
    }

    std::string error_message;
    try {
        (void)evictor_->poolWatermarkExcess(device_pool_.get(), 0.5);
    } catch (const std::runtime_error& error) {
        error_message = error.what();
    }
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_.clear();
    }

    ASSERT_FALSE(error_message.empty()) << "pending releases above used blocks should fail";
    EXPECT_NE(error_message.find("pool=" + device_pool_->poolName()), std::string::npos);
    EXPECT_NE(error_message.find("pending=1"), std::string::npos);
    EXPECT_NE(error_message.find("used=0"), std::string::npos);
}

TEST_F(BlockTreeEvictorTest, ComputeGroupSetExcessRejectsNonPositiveRatio) {
    for (double ratio : {0.0, -0.1}) {
        try {
            (void)evictor_->computeGroupSetExcess(*group_, Tier::DEVICE, ratio);
            FAIL() << "non-positive watermark ratio should fail: " << ratio;
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("group_set=0"), std::string::npos);
            EXPECT_NE(message.find("tier=DEVICE"), std::string::npos);
            EXPECT_NE(message.find("ratio="), std::string::npos);
        }
    }
}

TEST(BlockTreeEvictorCascadeTest, NonLeafCascadeFollowsGroupPriority) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder = runtime.make(&tree);
    BlockTreeEvictor&  evictor        = *evictor_holder;

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(groups.size()));
    for (auto& node_resources : resources) {
        for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
            const BlockIdxType block = groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_FALSE(isNullBlockIdx(block));
            node_resources[group_set_id] = makeResource(Tier::HOST, block);
        }
    }
    auto path = tree.insertNode({100, 200}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    ASSERT_EQ(path.inserted_nodes.size(), 2u);
    TreeNode* non_leaf = path.inserted_nodes.front();
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/0, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{1, 2}));
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/1, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{2}));
    EXPECT_TRUE(
        cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                               evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/2, Tier::HOST, Tier::DISK)))
            .empty());
}

TEST(BlockTreeEvictorCascadeTest, ReverseCascadeSelectsAllOtherGroupsAtLeaf) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder = runtime.make(&tree);
    BlockTreeEvictor&  evictor        = *evictor_holder;

    std::vector<GroupSetResource> resources(groups.size());
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        const BlockIdxType block = groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
        ASSERT_FALSE(isNullBlockIdx(block));
        resources[group_set_id] = makeResource(Tier::HOST, block);
    }
    auto inserted = tree.insertNode({100}, {resources}, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, {resources});

    ASSERT_NE(insertedNode(inserted), nullptr);
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  evictor, makeSelectionDesc(insertedNode(inserted), /*group_set_id=*/0, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{1, 2}));
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  evictor, makeSelectionDesc(insertedNode(inserted), /*group_set_id=*/1, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{0, 2}));
}

TEST(BlockTreeEvictorCascadeTest, DemotionDoesNotCascadeToUnmatchableParent) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(2, "upward_cascade_no_demotion");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, host_pool, disk_pool);
    initializeFullGroup(group, nullptr);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor = runtime.make(&tree);

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> resources = {
        {GroupSetResource{}},
        {makeResource(Tier::HOST, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200, 300}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    ASSERT_EQ(inserted.inserted_nodes.size(), 3u);

    const EvictionTask task = BlockTreeEvictorTestPeer::selectCascades(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::DISK));

    EXPECT_TRUE(task.cascade_descs.empty());
}

TEST(BlockTreeEvictorCascadeTest, StopsAtLogicallyMatchableParent) {
    auto device_pool = makeTestDevicePool(1, "upward_cascade_matchable_device");
    auto host_pool   = makePageableHostPool(2);
    auto disk_pool   = makeTestDiskPool(2, "upward_cascade_matchable_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor = runtime.make(&tree);

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> resources = {
        {makeResource(Tier::HOST, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    evictor->onInserted(inserted);
    ASSERT_EQ(inserted.inserted_nodes.size(), 2u);

    const EvictionTask task = BlockTreeEvictorTestPeer::selectCascades(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::DISK));

    EXPECT_TRUE(task.cascade_descs.empty());
}

TEST(BlockTreeEvictorCascadeTest, StopsAtParentWithAnotherEmptyChild) {
    auto host_pool = makePageableHostPool(3);
    auto disk_pool = makeTestDiskPool(3, "upward_cascade_branch_disk");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, host_pool, disk_pool);
    initializeFullGroup(group, nullptr);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor = runtime.make(&tree);

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> leaf_resources = {
        {GroupSetResource{}},
        {makeResource(Tier::HOST, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200, 300}, leaf_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, leaf_resources);
    ASSERT_EQ(inserted.inserted_nodes.size(), 3u);

    group->referenceBlocks(MultiNodeResource{0, Tier::HOST, {{nullptr, {parent_block}}}}, BlockTreeRefType::CACHE);
    const std::vector<std::vector<GroupSetResource>> sibling_resources = {
        {GroupSetResource{}},
        {makeResource(Tier::HOST, parent_block)},
        {GroupSetResource{}},
    };
    auto sibling = tree.insertNode({100, 200, 400}, sibling_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, sibling_resources);
    ASSERT_EQ(sibling.inserted_nodes.size(), 1u);

    const EvictionTask task = BlockTreeEvictorTestPeer::selectCascades(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::DISK));

    EXPECT_TRUE(task.cascade_descs.empty());
}

TEST(BlockTreeEvictorCascadeTest, DemotionPrepareLeavesAncestorUnchanged) {
    auto device_pool = makeTestDevicePool(1, "upward_cascade_exhausted_device");
    auto host_pool   = makePageableHostPool(2);
    auto disk_pool   = makeTestDiskPool(1, "upward_cascade_exhausted_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor = runtime.make(&tree);

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> resources = {
        {GroupSetResource{}},
        {makeResource(Tier::HOST, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200, 300}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    evictor->onInserted(inserted);

    auto victim = evictor->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    auto task = evictor->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    EXPECT_TRUE(task->cascade_descs.empty());
    EXPECT_EQ(inserted.inserted_nodes[1]->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(inserted.inserted_nodes[1]->group_set_resources[0].host_block, parent_block);

    evictor->abortEvictionLocked(*task);
}

TEST(BlockTreeEvictorCascadeTest, ForceDropRemovesUnmatchableParentChain) {
    auto device_pool = makeTestDevicePool(1, "upward_cascade_drop_device");
    auto host_pool   = makePageableHostPool(2);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor =
        runtime.make(&tree, EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO, [](Tier) { return false; });

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> resources = {
        {GroupSetResource{}},
        {makeResource(Tier::HOST, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200, 300}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    evictor->onInserted(inserted);

    EXPECT_TRUE(evictor->evictLocked(/*group_set_id=*/0, Tier::HOST, /*force_drop=*/true));
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
}

TEST(BlockTreeEvictorCascadeTest, PruneCascadesAncestorResourcesFromTheirActualTiers) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "upward_cascade_mixed_tier_prune");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, host_pool, disk_pool);
    initializeFullGroup(group, nullptr);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor = runtime.make(&tree);

    const BlockIdxType parent_block = group->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    const BlockIdxType leaf_block   = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_block));
    ASSERT_FALSE(isNullBlockIdx(leaf_block));
    const std::vector<std::vector<GroupSetResource>> resources = {
        {GroupSetResource{}},
        {makeResource(Tier::DISK, parent_block)},
        {makeResource(Tier::HOST, leaf_block)},
    };
    auto inserted = tree.insertNode({100, 200, 300}, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, resources);
    ASSERT_EQ(inserted.inserted_nodes.size(), 3u);

    const EvictionTask task = BlockTreeEvictorTestPeer::selectCascades(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::NONE));

    ASSERT_EQ(task.cascade_descs.size(), 1u);
    EXPECT_EQ(task.cascade_descs[0].node, inserted.inserted_nodes[1]);
    EXPECT_EQ(task.cascade_descs[0].source_tier, Tier::DISK);
    EXPECT_EQ(task.cascade_descs[0].target_tier, Tier::NONE);

    evictor->onInserted(inserted);
    EXPECT_TRUE(evictor->evictLocked(/*group_set_id=*/0, Tier::HOST, /*force_drop=*/true));
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, TierEntryRefreshesLastAccessTime) {
    const std::optional<BlockIdList> allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    BlockTreeInsertResult result = insert({100}, {{makeResource(Tier::DEVICE, allocated->front())}});
    TreeNode*             node   = insertedNode(result);
    ASSERT_NE(node, nullptr);

    CandidateMeta& candidate_meta      = node->group_set_resources[0].candidate_meta;
    candidate_meta.tier_enter_time_us  = 0;
    candidate_meta.last_access_time_us = 0;
    evictor_->onLoaded(node, 0);

    EXPECT_GT(candidate_meta.tier_enter_time_us, 0);
    EXPECT_EQ(candidate_meta.last_access_time_us, candidate_meta.tier_enter_time_us);
}

TEST_F(BlockTreeEvictorTest, SuspendAndAdmitCandidateTrackTransferState) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_is_evictable");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    TreeNode*         node     = insertedNode(result);
    GroupSetResource& resource = node->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    resource.transfer_state = GroupSetTransferState::DEMOTING;
    evictor_->suspendCandidate(node, /*group_set_id=*/0, Tier::HOST);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);

    resource.transfer_state = GroupSetTransferState::IDLE;
    evictor_->admitCandidate(node, /*group_set_id=*/0, Tier::HOST);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    evictor_->suspendCandidate(node, /*group_set_id=*/0, Tier::HOST);
    resource.host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimRejectsUnsupportedTier) {
    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::REMOTE).has_value());
    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::NONE).has_value());
}

TEST_F(BlockTreeEvictorTest, ChooseVictimSkipsInvalidHeapEntry) {
    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 2, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 2u);

    auto      first       = insert({100}, {{makeResource(Tier::DEVICE, device_blocks[0][0])}});
    auto      second      = insert({200}, {{makeResource(Tier::DEVICE, device_blocks[1][0])}});
    TreeNode* first_node  = insertedNode(first);
    TreeNode* second_node = insertedNode(second);
    ASSERT_NE(first_node, nullptr);
    ASSERT_NE(second_node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);

    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    const auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, second_node);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);

    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
    first_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    second_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, EvictForceDropsSelectedVictim) {
    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    ASSERT_EQ(device_blocks.front().size(), 1u);
    const BlockIdxType source = device_blocks.front().front();

    auto      result = insert({100}, {{makeResource(Tier::DEVICE, source)}});
    TreeNode* node   = insertedNode(result);
    ASSERT_NE(node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_pool_->refCount(source), 1u);
    size_t settled_count = 0;
    evictor_->settled_   = [&settled_count](bool tree_data_mutated, bool check_watermark) {
        ++settled_count;
        EXPECT_TRUE(tree_data_mutated);
        EXPECT_FALSE(check_watermark);
    };

    EXPECT_TRUE(evictor_->evictLocked(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/true));
    EXPECT_EQ(settled_count, 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(tree_->size(), 0u);
    EXPECT_FALSE(device_pool_->isAllocated(source));
    EXPECT_EQ(device_pool_->freeBlocksNum(), 128u);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimUsesNearestEnabledTargetTier) {
    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    const BlockIdxType source = device_blocks.front().front();
    auto               result = insert({100}, {{makeResource(Tier::DEVICE, source)}});
    TreeNode*          node   = insertedNode(result);
    ASSERT_NE(node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);

    evictor_ = evictor_runtime_.make(
        tree_.get(), EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO, [](Tier tier) {
            return tier == Tier::HOST || tier == Tier::DISK;
        });
    evictor_->onInserted(result);
    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->target_tier, Tier::HOST);

    evictor_ = evictor_runtime_.make(
        tree_.get(), EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO, [](Tier tier) {
            return tier == Tier::DISK;
        });
    evictor_->onInserted(result);
    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->target_tier, Tier::DISK);

    evictor_ = evictor_runtime_.make(
        tree_.get(), EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO, [](Tier) { return false; });
    evictor_->onInserted(result);
    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->target_tier, Tier::NONE);

    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/true);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->target_tier, Tier::NONE);

    node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{node, {source}}}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimSkipsDeviceParentOfLoadingChildUnlessForceDropping) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto device_blocks = device_pool_->malloc(2);
    ASSERT_TRUE(device_blocks.has_value());
    ASSERT_EQ(device_blocks->size(), 2u);
    const BlockIdxType host_block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));

    auto path =
        insert({100, 200}, {{makeResource(Tier::DEVICE, (*device_blocks)[0])}, {makeResource(Tier::HOST, host_block)}});
    auto rival = insert({300}, {{makeResource(Tier::DEVICE, (*device_blocks)[1])}});
    ASSERT_EQ(path.inserted_nodes.size(), 2u);
    TreeNode* parent = path.inserted_nodes[0];
    TreeNode* child  = path.inserted_nodes[1];
    ASSERT_NE(insertedNode(rival), nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 2u);

    for (GroupSetTransferState state : {GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING}) {
        child->group_set_resources[0].transfer_state = state;

        auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/false);
        ASSERT_TRUE(victim.has_value());
        EXPECT_EQ(victim->node, insertedNode(rival));
        EXPECT_EQ(evictor_->candidateStats().device_candidates, 2u);

        victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/true);
        ASSERT_TRUE(victim.has_value());
        EXPECT_EQ(victim->node, parent);
        EXPECT_EQ(evictor_->candidateStats().device_candidates, 2u);
    }

    child->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
    const auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/false);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, parent);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimReturnsEmptyWhenAllDeviceCandidatesAreSkipped) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto device_blocks = device_pool_->malloc(1);
    ASSERT_TRUE(device_blocks.has_value());
    ASSERT_EQ(device_blocks->size(), 1u);
    const BlockIdxType host_block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));

    auto path = insert({100, 200},
                       {{makeResource(Tier::DEVICE, device_blocks->front())}, {makeResource(Tier::HOST, host_block)}});
    ASSERT_EQ(path.inserted_nodes.size(), 2u);
    TreeNode* parent = path.inserted_nodes[0];
    TreeNode* child  = path.inserted_nodes[1];
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    child->group_set_resources[0].transfer_state = GroupSetTransferState::LOAD_PENDING;
    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/false).has_value());
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    const auto forced = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE, /*force_drop=*/true);
    ASSERT_TRUE(forced.has_value());
    EXPECT_EQ(forced->node, parent);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
}

TEST_F(BlockTreeEvictorTest, MatchUpdatesIntermediateHistoryWithoutAdmittingIt) {
    const auto allocated = device_pool_->malloc(3);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 3u);
    const BlockIdxType                         parent_block = (*allocated)[0];
    const BlockIdxType                         leaf_block   = (*allocated)[1];
    const BlockIdxType                         rival_block  = (*allocated)[2];
    std::vector<std::vector<GroupSetResource>> resources    = {{makeResource(Tier::DEVICE, parent_block)},
                                                               {makeResource(Tier::DEVICE, leaf_block)}};
    auto                                       result       = insert({100, 200}, resources);
    ASSERT_EQ(result.inserted_nodes.size(), 2u);
    auto rival = insert({300}, {{makeResource(Tier::DEVICE, rival_block)}});
    ASSERT_NE(insertedNode(rival), nullptr);

    TreeNode* parent = result.inserted_nodes[0];
    TreeNode* leaf   = result.inserted_nodes[1];
    ASSERT_NE(parent, nullptr);
    ASSERT_NE(leaf, nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 2u);

    const int64_t parent_insert_time_us = parent->group_set_resources[0].candidate_meta.insert_time_us;
    evictor_->onMatched({parent, leaf});

    const auto parent_meta = parent->group_set_resources[0].candidate_meta;
    const auto leaf_meta   = leaf->group_set_resources[0].candidate_meta;
    EXPECT_EQ(parent_meta.last_access_seq, leaf_meta.last_access_seq);
    EXPECT_EQ(parent_meta.insert_time_us, parent_insert_time_us);
    EXPECT_EQ(parent_meta.last_access_time_us, leaf_meta.last_access_time_us);
    EXPECT_GE(parent_meta.last_access_time_us, parent_insert_time_us);
    EXPECT_EQ(parent_meta.hit_count, 1u);
    EXPECT_EQ(leaf_meta.hit_count, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 2u);

    const MultiNodeResource leaf_resource{0, Tier::DEVICE, {{leaf, {leaf_block}}}};
    evictor_->suspendCandidate(leaf, 0, Tier::DEVICE);
    leaf->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(leaf_resource, BlockTreeRefType::CACHE);
    tree_->removeNodeAndEmptyAncestors(leaf);
    evictor_->onTopologyChanged(parent);

    ASSERT_EQ(evictor_->candidateStats().device_candidates, 2u);
    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(rival));
    EXPECT_EQ(parent->group_set_resources[0].candidate_meta.last_access_seq, parent_meta.last_access_seq);
    EXPECT_EQ(parent->group_set_resources[0].candidate_meta.hit_count, parent_meta.hit_count);

    const MultiNodeResource parent_resource{0, Tier::DEVICE, {{parent, {parent_block}}}};
    evictor_->suspendCandidate(parent, 0, Tier::DEVICE);
    parent->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(parent_resource, BlockTreeRefType::CACHE);
    const MultiNodeResource rival_resource{0, Tier::DEVICE, {{insertedNode(rival), {rival_block}}}};
    evictor_->suspendCandidate(insertedNode(rival), 0, Tier::DEVICE);
    insertedNode(rival)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    group_->unreferenceBlocks(rival_resource, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ExistingGroupFillAdmitsChildAndRemovesFullParentCandidate) {
    const auto allocated = device_pool_->malloc(2);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    const BlockIdxType parent_block = (*allocated)[0];
    const BlockIdxType child_block  = (*allocated)[1];

    BlockTreeInsertResult parent_result = insert({100}, {{makeResource(Tier::DEVICE, parent_block)}});
    ASSERT_NE(insertedNode(parent_result), nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    GroupSetResource empty_resource;
    empty_resource.device_blocks      = {NULL_BLOCK_IDX};
    BlockTreeInsertResult empty_child = tree_->insertNode(
        {100, 200}, {{makeResource(Tier::DEVICE, parent_block)}, {empty_resource}}, /*collect_path=*/false);
    evictor_->onInserted(empty_child);
    ASSERT_NE(insertedNode(empty_child), nullptr);
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    const BlockTreeInsertResult fill_result =
        insert({100, 200}, {{makeResource(Tier::DEVICE, parent_block)}, {makeResource(Tier::DEVICE, child_block)}});
    ASSERT_TRUE(fill_result.inserted_nodes.empty());
    ASSERT_EQ(fill_result.adopted_nodes.size(), 1u);
    EXPECT_EQ(fill_result.adopted_nodes.front().first, insertedNode(empty_child));

    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    const std::optional<TransferDescriptor> victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(empty_child));
    EXPECT_EQ(victim->source_blocks, (std::vector<BlockIdxType>{child_block}));
}

TEST_F(BlockTreeEvictorTest, ExtraTreeReferencesKeepCandidateEligible) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const BlockIdxType block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(block));
    ASSERT_EQ(host_pool->treeRefCount(block), 1u);

    auto result = insert({100}, {{makeResource(Tier::HOST, block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    MultiNodeResource match_set{0, Tier::HOST, {{insertedNode(result), {block}}}};
    group_->referenceBlocks(match_set, BlockTreeRefType::LOAD);
    group_->referenceBlocks(match_set, BlockTreeRefType::LOAD);
    ASSERT_EQ(host_pool->treeRefCount(block), 3u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));
    EXPECT_EQ(victim->target_tier, Tier::DISK);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    group_->unreferenceBlocks(match_set, BlockTreeRefType::LOAD);
    EXPECT_EQ(host_pool->treeRefCount(block), 2u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    group_->unreferenceBlocks(match_set, BlockTreeRefType::LOAD);
    EXPECT_EQ(host_pool->treeRefCount(block), 1u);
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));
    EXPECT_EQ(victim->target_tier, Tier::DISK);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, block, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimAllowsNewLoadReferenceWithoutSideEffects) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_pin");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    MultiNodeResource pin{0, Tier::HOST, {{insertedNode(result), {source}}}};
    group_->referenceBlocks(pin, BlockTreeRefType::LOAD);
    ASSERT_EQ(host_pool->treeRefCount(source), 2u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));
    EXPECT_EQ(victim->source_blocks, (BlockIndicesType{source}));
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(insertedNode(result)->group_set_resources[0].transfer_detached);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_EQ(host_pool->treeRefCount(source), 2u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);

    group_->unreferenceBlocks(pin, BlockTreeRefType::LOAD);
    EXPECT_EQ(host_pool->treeRefCount(source), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockTreeRefType::CACHE);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimPreservesLoadOwner) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_load");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    ASSERT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST).has_value());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_EQ(host_pool->treeRefCount(source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);

    settleLoad(insertedNode(result), 0, false);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimPreservesExistingDemotionOwnerAndTarget) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_demotion");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);

    TransferDescriptor owner(insertedNode(result),
                             /*group_set_id=*/0,
                             /*path_index=*/0,
                             Tier::HOST,
                             Tier::DISK,
                             insertedNode(result)->group_set_resources[0].getBlocks(Tier::HOST));
    owner.target_blocks = {group_->allocateSingleBlock(Tier::DISK, BlockTreeRefType::EVICTION)};
    ASSERT_FALSE(isNullBlockIdx(owner.target_blocks.front()));
    BlockTreeEvictorTestPeer::reserveSource(*evictor_, owner);
    const BlockIdxType owner_target = owner.target_blocks[0];
    ASSERT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);

    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST).has_value());
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].host_block, source);
    EXPECT_TRUE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->treeRefCount(owner_target), 1u);
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);

    BlockTreeEvictorTestPeer::rollbackDesc(*evictor_, owner);
    EXPECT_EQ(insertedNode(result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(host_pool->treeRefCount(source), 1u);
    EXPECT_FALSE(disk_pool->isAllocated(owner_target));
    EXPECT_EQ(disk_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);

    insertedNode(result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, source, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimRejectsSourceTierChangedByLoad) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_tier_change");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    auto&           resource   = insertedNode(result)->group_set_resources[0];
    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_EQ(device_set.front().size(), 1u);
    const BlockIdxType device_block = device_set.front().front();
    resource.setBlocks(Tier::DEVICE, device_set.front());
    group_->unreferenceBlocks(MultiNodeResource{0, Tier::HOST, {{insertedNode(result), {source}}}},
                              BlockTreeRefType::CACHE);
    resource.evictFromTier(Tier::HOST);
    settleLoad(insertedNode(result), 0, true);
    ASSERT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    ASSERT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    ASSERT_FALSE(host_pool->isAllocated(source));

    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST).has_value());
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{device_block}));
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);

    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, device_set, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimSkipsFullNodeThatBecameNonLeaf) {
    auto host_pool = makePageableHostPool(2);
    auto disk_pool = makeTestDiskPool(1, "block_tree_evictor_topology");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType parent_source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType child_source  = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(parent_source));
    ASSERT_FALSE(isNullBlockIdx(child_source));
    auto parent_result = insert({100}, {{makeResource(Tier::HOST, parent_source)}});
    ASSERT_NE(insertedNode(parent_result), nullptr);
    const std::vector<std::vector<GroupSetResource>> child_resources = {{makeResource(Tier::HOST, parent_source)},
                                                                        {makeResource(Tier::HOST, child_source)}};
    // The duplicate entry reuses a tree-owned block, so it needs its own seed hold
    // for the release below to stay balanced.
    group_->referenceBlocks(MultiNodeResource{0, Tier::HOST, {{nullptr, {parent_source}}}}, BlockTreeRefType::CACHE);
    auto child_result = tree_->insertNode({100, 101}, child_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups_, child_resources);
    evictor_->onInserted(child_result);
    ASSERT_NE(insertedNode(child_result), nullptr);
    ASSERT_FALSE(tree_->isLeafAtTier(insertedNode(parent_result), 0, Tier::HOST));

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(child_result));
    EXPECT_EQ(insertedNode(parent_result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(parent_result)->group_set_resources[0].host_block, parent_source);
    EXPECT_EQ(host_pool->treeRefCount(parent_source), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);

    insertedNode(parent_result)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    insertedNode(child_result)->group_set_resources[0].host_block  = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, parent_source, BlockTreeRefType::CACHE);
    group_->releaseSingleBlock(Tier::HOST, child_source, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, LoadingStateExcludesAndIdleStateReadmitsSource) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(source));
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateStats().host_candidates, 1u);

    EXPECT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);

    settleLoad(insertedNode(result), 0, false);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
}

TEST_F(BlockTreeEvictorTest, LoadSuccessAdmitsOnlyStableDeviceResource) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(source, NULL_BLOCK_IDX);
    auto result = insert({100}, {{makeResource(Tier::HOST, source)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];

    ASSERT_TRUE(reserveAndBeginLoad(insertedNode(result), 0, Tier::HOST));
    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_EQ(device_set.front().size(), 1u);
    group_->unreferenceBlocks(MultiNodeResource{0, Tier::HOST, {{insertedNode(result), {source}}}},
                              BlockTreeRefType::CACHE);
    resource.evictFromTier(Tier::HOST);
    resource.setBlocks(Tier::DEVICE, device_set.front());
    settleLoad(insertedNode(result), 0, true);

    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(result));

    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, device_set, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, DemotionExcludesSourceAndRollbackOrSuccessRestoresOneTier) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    const CandidateMeta candidate_meta = resource.candidate_meta;
    auto                victim         = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto task = evictor_->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    ASSERT_EQ(task->primary_desc.target_blocks.size(), 1u);
    EXPECT_EQ(task->primary_timing.tier_enter_time_us, candidate_meta.tier_enter_time_us);
    EXPECT_EQ(task->primary_timing.insert_time_us, candidate_meta.insert_time_us);
    EXPECT_EQ(task->primary_timing.last_access_time_us, candidate_meta.last_access_time_us);
    EXPECT_GE(task->primary_timing.selected_time_us, candidate_meta.last_access_time_us);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);

    evictor_->abortEvictionLocked(*task);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source_block}));
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);

    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    task = evictor_->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    const BlockIdxType target_block = task->primary_desc.target_blocks[0];
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    evictor_->settleEvictionLocked(*task, EvictionTaskResult{true, {}});

    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
    EXPECT_EQ(resource.host_block, target_block);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(host_pool->treeRefCount(target_block), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);

    resource.host_block = NULL_BLOCK_IDX;
    group_->releaseSingleBlock(Tier::HOST, target_block, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ChooseVictimKeepsCandidateUntilTaskActivation) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(evictor_->candidateNodes(/*group_set_id=*/0, Tier::DEVICE),
              (std::vector<TreeNode*>{insertedNode(result)}));

    auto task = evictor_->prepareEvictionLocked(*victim);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);

    evictor_->abortEvictionLocked(*task);
    const MultiNodeResource source{0, Tier::DEVICE, {{insertedNode(result), {source_block}}}};
    evictor_->suspendCandidate(insertedNode(result), 0, Tier::DEVICE);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, PrimaryTargetExhaustionLeavesSourceAndCandidateUnchanged) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const auto exhausted_target_result = host_pool->malloc();
    ASSERT_TRUE(exhausted_target_result.has_value());
    host_pool->incTreeRef(*exhausted_target_result, BlockTreeRefType::STORE);
    const BlockIdxType exhausted_target = *exhausted_target_result;
    ASSERT_EQ(host_pool->freeBlocksNum(), 0u);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);
    auto& resource = insertedNode(result)->group_set_resources[0];

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(evictor_->candidateStats().device_candidates, 1u);

    EXPECT_FALSE(evictor_->prepareEvictionLocked(*victim).has_value());
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source_block}));
    EXPECT_EQ(device_pool_->refCount(source_block), 1u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);

    auto retry = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(retry->node, insertedNode(result));

    const MultiNodeResource source{0, Tier::DEVICE, {{insertedNode(result), {source_block}}}};
    evictor_->suspendCandidate(insertedNode(result), 0, Tier::DEVICE);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
    host_pool->decTreeRef(exhausted_target, BlockTreeRefType::STORE);
}

TEST(BlockTreeEvictorCascadeTest, PrepareTaskIncludesReferencedSibling) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  *environment.evictor_, makeSelectionDesc(environment.node_, 0, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{1, 2}));

    MultiNodeResource pin = environment.hostSet(1);
    environment.groups_[1]->referenceBlocks(pin, BlockTreeRefType::LOAD);
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  *environment.evictor_, makeSelectionDesc(environment.node_, 0, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{1, 2}));
    ASSERT_EQ(environment.host_pools_[1]->treeRefCount(environment.host_blocks_[1]), 2u);

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{1, 2}));
    EXPECT_EQ(task->primary_desc.group_set_id, 0);
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transferGroupSetIds().empty());

    environment.evictor_->abortEvictionLocked(*task);
    environment.groups_[1]->unreferenceBlocks(pin, BlockTreeRefType::LOAD);
    EXPECT_EQ(environment.host_pools_[1]->treeRefCount(environment.host_blocks_[1]), 1u);

    auto retry = environment.prepareTask(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->abortEvictionLocked(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrepareTaskSkipsLoadingSiblingAndReadmitsAfterFinish) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.node_->group_set_resources[1].transfer_state = GroupSetTransferState::LOADING;
    environment.evictor_->suspendCandidate(environment.node_, 1, Tier::HOST);

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::LOADING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->treeRefCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_TRUE(environment.transferGroupSetIds().empty());

    environment.evictor_->abortEvictionLocked(*task);
    environment.node_->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    environment.evictor_->admitCandidate(environment.node_, 1, Tier::HOST);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    auto retry = environment.prepareTask(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->abortEvictionLocked(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrepareTaskSkipsDemotingSiblingWithoutAdoptingItsTarget) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    TransferDescriptor sibling_owner(environment.node_,
                                     /*group_set_id=*/1,
                                     /*path_index=*/0,
                                     Tier::HOST,
                                     Tier::DISK,
                                     environment.node_->group_set_resources[1].getBlocks(Tier::HOST));
    sibling_owner.target_blocks = {environment.groups_[1]->allocateSingleBlock(Tier::DISK, BlockTreeRefType::EVICTION)};
    ASSERT_FALSE(isNullBlockIdx(sibling_owner.target_blocks.front()));
    BlockTreeEvictorTestPeer::reserveSource(*environment.evictor_, sibling_owner);
    const BlockIdxType owner_target = sibling_owner.target_blocks[0];

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    EXPECT_EQ(environment.disk_pools_[1]->treeRefCount(owner_target), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transferGroupSetIds().empty());

    environment.evictor_->abortEvictionLocked(*task);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_TRUE(environment.disk_pools_[1]->isAllocated(owner_target));
    BlockTreeEvictorTestPeer::rollbackDesc(*environment.evictor_, sibling_owner);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(owner_target));

    auto retry = environment.prepareTask(0);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{1, 2}));
    environment.evictor_->abortEvictionLocked(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, LeafPrepareTaskIncludesReferencedFullSibling) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    ASSERT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::selectCascades(
                  *environment.evictor_, makeSelectionDesc(environment.node_, 2, Tier::HOST, Tier::DISK))),
              (std::vector<size_t>{0, 1}));

    MultiNodeResource pin = environment.hostSet(0);
    environment.groups_[0]->referenceBlocks(pin, BlockTreeRefType::LOAD);
    auto task = environment.prepareTask(2);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{0, 1}));
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transferGroupSetIds().empty());

    environment.evictor_->abortEvictionLocked(*task);
    environment.groups_[0]->unreferenceBlocks(pin, BlockTreeRefType::LOAD);
    auto retry = environment.prepareTask(2);
    ASSERT_TRUE(retry.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*retry), (std::vector<size_t>{0, 1}));
    environment.evictor_->abortEvictionLocked(*retry);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, CascadeTargetExhaustionRestoresOnlyFailedSibling) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    const std::vector<BlockIdxType> exhausted = exhaustPool(*environment.disk_pools_[1]);
    ASSERT_EQ(exhausted.size(), 2u);
    ASSERT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 0u);
    const size_t exhausted_capacity = environment.disk_pools_[1]->freeBlocksNum();

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{2}));
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].host_block, environment.host_blocks_[1]);
    EXPECT_EQ(environment.host_pools_[1]->treeRefCount(environment.host_blocks_[1]), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), exhausted_capacity);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 1u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 1u);
    EXPECT_TRUE(environment.transferGroupSetIds().empty());

    environment.evictor_->abortEvictionLocked(*task);
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(environment.host_pools_[group_set_id]->treeRefCount(environment.host_blocks_[group_set_id]), 1u);
    }
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 0u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 2u);

    releaseBlocks(*environment.disk_pools_[1], exhausted);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrimaryFailureAbortsResourcesEvenWithCascadeSuccessFlags) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    ASSERT_EQ(task->cascade_descs.size(), 2u);

    environment.evictor_->settleEvictionLocked(*task, EvictionTaskResult{false, {true, true}});

    for (const GroupSetResource& resource : environment.node_->group_set_resources) {
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrimaryFailureSkipsCascadesAndAbortsFullTask) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferResults({false, true, true});

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    ASSERT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{1, 2}));
    environment.evictor_->updatePendingReleases(*task, true);
    auto task_result = environment.runTransfer(*task);
    environment.evictor_->updatePendingReleases(*task, false);
    EXPECT_FALSE(task_result.primary_success);
    EXPECT_EQ(task_result.cascade_success, (std::vector<bool>{false, false}));
    EXPECT_EQ(environment.transferGroupSetIds(), (std::vector<size_t>{0}));

    environment.evictor_->settleEvictionLocked(*task, task_result);
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(environment.node_->group_set_resources[group_set_id].host_block,
                  environment.host_blocks_[group_set_id]);
        EXPECT_EQ(environment.host_pools_[group_set_id]->treeRefCount(environment.host_blocks_[group_set_id]), 1u);
        EXPECT_EQ(environment.disk_pools_[group_set_id]->freeBlocksNum(), 2u);
    }
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 3u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, OrderedTransferSuccessPublishesFullTask) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferResults({true, true, true});

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    ASSERT_EQ(cascadeGroupSetIds(*task), (std::vector<size_t>{1, 2}));
    environment.evictor_->updatePendingReleases(*task, true);
    const BlockIdxType primary_target = task->primary_desc.target_blocks[0];
    const BlockIdxType first_target   = task->cascade_descs[0].target_blocks[0];
    const BlockIdxType success_target = task->cascade_descs[1].target_blocks[0];

    auto task_result = environment.runTransfer(*task);
    environment.evictor_->updatePendingReleases(*task, false);
    ASSERT_TRUE(task_result.primary_success);
    EXPECT_EQ(task_result.cascade_success, (std::vector<bool>{true, true}));
    EXPECT_EQ(environment.transferGroupSetIds(), (std::vector<size_t>{0, 1, 2}));
    environment.evictor_->settleEvictionLocked(*task, task_result);

    const auto& primary_resource = environment.node_->group_set_resources[0];
    const auto& first_resource   = environment.node_->group_set_resources[1];
    const auto& success_resource = environment.node_->group_set_resources[2];
    EXPECT_EQ(primary_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(primary_resource.hasTier(Tier::HOST));
    EXPECT_EQ(primary_resource.disk_block, primary_target);
    EXPECT_EQ(environment.disk_pools_[0]->treeRefCount(primary_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));

    EXPECT_EQ(first_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(first_resource.hasTier(Tier::HOST));
    EXPECT_EQ(first_resource.disk_block, first_target);
    EXPECT_EQ(environment.disk_pools_[1]->treeRefCount(first_target), 1u);
    EXPECT_FALSE(environment.host_pools_[1]->isAllocated(environment.host_blocks_[1]));
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 0u);

    EXPECT_EQ(success_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(success_resource.hasTier(Tier::HOST));
    EXPECT_EQ(success_resource.disk_block, success_target);
    EXPECT_EQ(environment.disk_pools_[2]->treeRefCount(success_target), 1u);
    EXPECT_FALSE(environment.host_pools_[2]->isAllocated(environment.host_blocks_[2]));

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, RejectsResourcesReservedByAnotherTask) {
    auto full_device_pool   = makeTestDevicePool(2, "reserved_task_full_device");
    auto swa_device_pool    = makeTestDevicePool(2, "reserved_task_swa_device");
    auto linear_device_pool = makeTestDevicePool(2, "reserved_task_linear_device");
    ASSERT_NE(full_device_pool, nullptr);
    ASSERT_NE(swa_device_pool, nullptr);
    ASSERT_NE(linear_device_pool, nullptr);
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto swa_policy                   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.sliding_window_size    = 128;
    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    full_policy.enable_prefix_reuse   = true;
    swa_policy.enable_prefix_reuse    = true;
    linear_policy.enable_prefix_reuse = true;

    auto full_host_pool   = makePageableHostPool(2);
    auto swa_host_pool    = makePageableHostPool(2);
    auto linear_host_pool = makePageableHostPool(2);
    ASSERT_NE(full_host_pool, nullptr);
    ASSERT_NE(swa_host_pool, nullptr);
    ASSERT_NE(linear_host_pool, nullptr);
    auto full =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{full_device_pool}, full_host_pool, nullptr);
    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{swa_device_pool}, swa_host_pool, nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{linear_device_pool}, linear_host_pool, nullptr);
    initializeGroups({full, swa, linear},
                     {full_device_pool, swa_device_pool, linear_device_pool},
                     {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16, 0, 128, 64),
                      block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, 16, 0, 128, 64),
                      block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16, 0, 128, 64)});

    std::vector<GroupSetPtr> groups = {full, swa, linear};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree);
    BlockTreeEvictor&        evictor        = *evictor_holder;

    MultiNodeBlocks full_blocks   = allocateDeviceBlocksForTest(*full, 1, BlockTreeRefType::CACHE);
    MultiNodeBlocks swa_blocks    = allocateDeviceBlocksForTest(*swa, 1, BlockTreeRefType::CACHE);
    MultiNodeBlocks linear_blocks = allocateDeviceBlocksForTest(*linear, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(full_blocks.size(), 1u);
    ASSERT_EQ(swa_blocks.size(), 1u);
    ASSERT_EQ(linear_blocks.size(), 1u);

    auto insert_result = tree.insertNode({100},
                                         {{makeResource(Tier::DEVICE, full_blocks[0][0]),
                                           makeResource(Tier::DEVICE, swa_blocks[0][0]),
                                           makeResource(Tier::DEVICE, linear_blocks[0][0])}},
                                         /*collect_path=*/false);
    ASSERT_NE(insertedNode(insert_result), nullptr);
    unreferenceDeviceBlocksForTest(*full, full_blocks, BlockTreeRefType::CACHE);
    unreferenceDeviceBlocksForTest(*swa, swa_blocks, BlockTreeRefType::CACHE);
    unreferenceDeviceBlocksForTest(*linear, linear_blocks, BlockTreeRefType::CACHE);
    evictor.onInserted(insert_result);

    auto swa_victim = evictor.chooseVictim(1, Tier::DEVICE);
    ASSERT_TRUE(swa_victim.has_value());
    auto first_task = evictor.prepareEvictionLocked(*swa_victim);
    ASSERT_TRUE(first_task.has_value());
    ASSERT_EQ(first_task->cascade_descs.size(), 2u);
    EXPECT_EQ(first_task->cascade_timings.size(), first_task->cascade_descs.size());
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[2].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 1u);

    EXPECT_FALSE(evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE).has_value());
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 1u);

    evictor.abortEvictionLocked(*first_task);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(insertedNode(insert_result)->group_set_resources[2].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(full_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(swa_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(linear_host_pool->freeBlocksNum(), 2u);
}

TEST(BlockTreeEvictorStatsTest, AggregatesCandidatesAcrossGroupsAndTiers) {
    auto device_pool = makeTestDevicePool(1, "block_tree_evictor_stats_device");
    auto host_pool   = makePageableHostPool(1);
    auto disk_pool   = makeTestDiskPool(1, "block_tree_evictor_stats_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    auto group1_device_pool = makeTestDevicePool(1, "block_tree_evictor_stats_unused_device");
    ASSERT_NE(group1_device_pool, nullptr);
    auto group0 = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    auto group1 =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{group1_device_pool}, nullptr, disk_pool);
    initializeGroups(
        {group0, group1},
        {device_pool, group1_device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16),
         block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::FULL), {0}, 16)});
    std::vector<GroupSetPtr> groups = {group0, group1};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto                     evictor_holder = runtime.make(&tree);
    BlockTreeEvictor&        evictor        = *evictor_holder;

    MultiNodeBlocks    device_set = allocateDeviceBlocksForTest(*group0, 1, BlockTreeRefType::CACHE);
    const BlockIdxType host_block = group0->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType disk_block = group1->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 1u);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    const std::vector<std::vector<GroupSetResource>> first_resources = {
        {makeResource(Tier::DEVICE, device_set.front().front()), makeResource(Tier::DISK, disk_block)}};
    auto first = tree.insertNode({100}, first_resources, /*collect_path=*/false);
    unreferenceDeviceBlocksForTest(*group0, device_set, BlockTreeRefType::CACHE);
    releaseLowerTierSeedRefs(groups, first_resources);
    evictor.onInserted(first);
    const std::vector<std::vector<GroupSetResource>> second_resources = {
        {makeResource(Tier::HOST, host_block), GroupSetResource{}}};
    auto second = tree.insertNode({200}, second_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, second_resources);
    evictor.onInserted(second);

    const CandidateStats stats = evictor.candidateStats();
    EXPECT_EQ(stats.device_candidates, 1u);
    EXPECT_EQ(stats.host_candidates, 1u);
    EXPECT_EQ(stats.disk_candidates, 1u);

    EXPECT_FALSE(evictor.chooseVictim(0, Tier::DISK).has_value());
    auto disk_victim = evictor.chooseVictim(1, Tier::DISK);
    ASSERT_TRUE(disk_victim.has_value());
    EXPECT_EQ(disk_victim->node, insertedNode(first));
    EXPECT_EQ(disk_victim->group_set_id, 1);

    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    insertedNode(first)->group_set_resources[1].disk_block   = NULL_BLOCK_IDX;
    insertedNode(second)->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    unreferenceDeviceBlocksForTest(*group0, device_set, BlockTreeRefType::CACHE);
    group0->releaseSingleBlock(Tier::HOST, host_block, BlockTreeRefType::CACHE);
    group1->releaseSingleBlock(Tier::DISK, disk_block, BlockTreeRefType::CACHE);
}

TEST(BlockTreeEvictorPolicyTest, MatchDoesNotChangeFifoAdmissionOrder) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_fifo_policy");
    ASSERT_NE(device_pool, nullptr);
    auto group = makeFullGroup(device_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto evictor_holder       = runtime.make(&tree, EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO);
    BlockTreeEvictor& evictor = *evictor_holder;

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto first = tree.insertNode({100}, {{makeResource(Tier::DEVICE, device_set[0][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}},
                             BlockTreeRefType::CACHE);
    evictor.onInserted(first);
    auto second = tree.insertNode({200}, {{makeResource(Tier::DEVICE, device_set[1][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}},
                             BlockTreeRefType::CACHE);
    evictor.onInserted(second);
    const uint64_t first_admission = insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq;

    evictor.onMatched({insertedNode(first)});

    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq, first_admission);
    auto first_victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(first_victim.has_value());
    EXPECT_EQ(first_victim->node, insertedNode(first));

    // No Host pool is configured, so target allocation fails before the source
    // is reserved. FIFO admission and relative victim order stay unchanged.
    EXPECT_FALSE(evictor.prepareEvictionLocked(*first_victim).has_value());
    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.admission_seq, first_admission);
    auto retried_victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(retried_victim.has_value());
    EXPECT_EQ(retried_victim->node, insertedNode(first));

    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    insertedNode(second)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockTreeRefType::CACHE);
}

TEST(BlockTreeEvictorPolicyTest, ExistingGroupFillPrecedesNewSuffixAdmission) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_existing_fill_fifo");
    ASSERT_NE(device_pool, nullptr);
    auto group = std::make_shared<LinearGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, nullptr, nullptr);
    initializeGroups(
        {group},
        {device_pool},
        {block_transfer_engine_test::makeTestGroupBase(defaultCacheGroupPolicy(CacheGroupType::LINEAR), {0}, 16)});

    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto evictor_holder       = runtime.make(&tree, EvictionPolicy::FIFO, EvictionPolicy::LRU, EvictionPolicy::FIFO);
    BlockTreeEvictor& evictor = *evictor_holder;

    GroupSetResource empty_resource;
    empty_resource.device_blocks = {NULL_BLOCK_IDX};
    auto existing                = tree.insertNode({100}, {{empty_resource}}, /*collect_path=*/false);
    evictor.onInserted(existing);

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto mixed = tree.insertNode(
        {100, 200},
        {{makeResource(Tier::DEVICE, device_set[0][0])}, {makeResource(Tier::DEVICE, device_set[1][0])}},
        /*collect_path=*/false);
    ASSERT_EQ(mixed.adopted_nodes.size(), 1u);
    ASSERT_EQ(mixed.inserted_nodes.size(), 1u);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockTreeRefType::CACHE);
    evictor.onInserted(mixed);

    TreeNode* filled_node = mixed.adopted_nodes.front().first;
    TreeNode* new_node    = mixed.inserted_nodes.front();
    ASSERT_NE(filled_node, nullptr);
    ASSERT_NE(new_node, nullptr);
    EXPECT_LT(filled_node->group_set_resources[0].candidate_meta.admission_seq,
              new_node->group_set_resources[0].candidate_meta.admission_seq);

    auto victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, filled_node);

    filled_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    new_node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockTreeRefType::CACHE);
}

TEST(BlockTreeEvictorPolicyTest, MatchUpdatesLfuHitCountAndOrder) {
    auto device_pool = makeTestDevicePool(2, "block_tree_evictor_lfu_policy");
    ASSERT_NE(device_pool, nullptr);
    auto group = makeFullGroup(device_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);
    TestEvictorRuntime       runtime;
    auto evictor_holder       = runtime.make(&tree, EvictionPolicy::LFU, EvictionPolicy::LRU, EvictionPolicy::FIFO);
    BlockTreeEvictor& evictor = *evictor_holder;

    MultiNodeBlocks device_set = allocateDeviceBlocksForTest(*group, 2, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_set.size(), 2u);
    auto first = tree.insertNode({100}, {{makeResource(Tier::DEVICE, device_set[0][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(first), device_set[0]}}},
                             BlockTreeRefType::CACHE);
    evictor.onInserted(first);
    auto second = tree.insertNode({200}, {{makeResource(Tier::DEVICE, device_set[1][0])}}, /*collect_path=*/false);
    group->unreferenceBlocks(MultiNodeResource{0, Tier::DEVICE, {{insertedNode(second), device_set[1]}}},
                             BlockTreeRefType::CACHE);
    evictor.onInserted(second);

    evictor.onMatched({insertedNode(first)});

    EXPECT_EQ(insertedNode(first)->group_set_resources[0].candidate_meta.hit_count, 1u);
    EXPECT_EQ(insertedNode(second)->group_set_resources[0].candidate_meta.hit_count, 0u);
    auto victim = evictor.chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->node, insertedNode(second));

    insertedNode(first)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    insertedNode(second)->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group, device_set, BlockTreeRefType::CACHE);
}

}  // namespace
}  // namespace rtp_llm
