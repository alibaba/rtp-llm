#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
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
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::allocateDeviceBlocksForTest;
using block_tree_cache_test::MultiNodeBlocks;
using block_tree_cache_test::releaseLowerTierSeedRefs;
using ScriptedTransferEngine = block_tree_cache_test::ScriptedPerRankBlockTransferEngine;
using block_tree_cache_test::unreferenceDeviceBlocksForTest;

double snapshotQps(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags& tags) {
    if (metric == nullptr) {
        ADD_FAILURE() << "metric is null";
        return -1;
    }
    kmonitor::Metric* qps_metric = metric->DeclareMetric(&tags);
    if (qps_metric == nullptr) {
        ADD_FAILURE() << "metric series is missing for tags=" << tags.ToString();
        return -1;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    qps_metric->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(qps_metric));
    if (record.Values().size() != 1) {
        ADD_FAILURE() << "unexpected metric value count=" << record.Values().size();
        return -1;
    }
    return std::stod(record.Values().front()->Value());
}

static_assert(!noexcept(std::declval<BlockTreeEvictor&>().runDropTask(std::declval<TransferDescriptor>())));
static_assert(!noexcept(
    std::declval<BlockTreeEvictor&>().rollbackTransferLocked(std::declval<const std::vector<TransferDescriptor>&>())));
static_assert(noexcept(
    std::declval<BlockTreeEvictor&>().runEvictionTask(std::declval<std::shared_ptr<const EvictionTransferTask>>())));

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

TreeNode* insertCascadeLeafWithMultiTierLinear(BlockTree& tree, const std::vector<GroupSetPtr>& groups) {
    std::vector<GroupSetResource> resources(groups.size());
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        const BlockIdxType host_block = groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
        RTP_LLM_CHECK(!isNullBlockIdx(host_block));
        resources[group_set_id].host_block = host_block;
    }
    const MultiNodeBlocks linear_device_blocks = allocateDeviceBlocksForTest(*groups[2], 1, BlockTreeRefType::CACHE);
    RTP_LLM_CHECK(linear_device_blocks.size() == 1);
    resources[2].device_blocks = linear_device_blocks.front();

    auto inserted = tree.insertNode({100}, {resources}, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, {resources});
    unreferenceDeviceBlocksForTest(*groups[2], linear_device_blocks, BlockTreeRefType::CACHE);
    RTP_LLM_CHECK(inserted.inserted_nodes.size() == 1);
    return inserted.inserted_nodes.front();
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
        EvictionPolicy                    device_policy             = EvictionPolicy::LRU,
        EvictionPolicy                    host_policy               = EvictionPolicy::LRU,
        EvictionPolicy                    disk_policy               = EvictionPolicy::FIFO,
        BlockTreeEvictor::IsTierEnabledFn is_tier_enabled           = [](Tier) { return true; },
        BlockTreeTaskPool*                task_pool                 = nullptr,
        size_t                            max_device_host_batch     = 8,
        size_t                            max_non_device_host_batch = 16) {
        transfer_engine_     = std::make_shared<ScriptedTransferEngine>(tree->groupSets(), false);
        transfer_dispatcher_ = std::make_unique<BlockTransferDispatcher>(
            transfer_engine_, nullptr, max_device_host_batch, max_non_device_host_batch);
        return std::make_unique<BlockTreeEvictor>(tree,
                                                  device_policy,
                                                  host_policy,
                                                  disk_policy,
                                                  transfer_dispatcher_.get(),
                                                  task_pool,
                                                  metrics_reporter_,
                                                  mutex_,
                                                  0,
                                                  0,
                                                  max_device_host_batch,
                                                  max_non_device_host_batch,
                                                  std::move(is_tier_enabled),
                                                  [](bool, bool) {});
    }

    std::unique_ptr<BlockTreeEvictor> make(BlockTree* tree, BlockTreeTaskPool* task_pool) {
        return make(
            tree, EvictionPolicy::LRU, EvictionPolicy::LRU, EvictionPolicy::FIFO, [](Tier) { return true; }, task_pool);
    }

    std::shared_ptr<ScriptedTransferEngine> transferEngine() const {
        return transfer_engine_;
    }

    size_t transferCount() const {
        return transfer_engine_->submittedBatchCount();
    }

    void setMetricsReporter(const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter) {
        metrics_reporter_.setMetricsReporter(metrics_reporter);
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

    bool completeGroupSet(size_t group_set_id, bool success) {
        std::shared_ptr<TransferBatchAsyncContext> context;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t index = 0; index < descriptors_.size(); ++index) {
                if (descriptors_[index].size() == 1 && descriptors_[index].front().group_set_id == group_set_id) {
                    context = contexts_[index];
                    break;
                }
            }
        }
        if (context == nullptr) {
            return false;
        }
        context->complete(success ? ErrorInfo::OkStatus() :
                                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
        return true;
    }

    bool completeBatch(size_t batch_index, bool success) {
        std::shared_ptr<TransferBatchAsyncContext> context;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (batch_index >= contexts_.size()) {
                return false;
            }
            context = contexts_[batch_index];
        }
        context->complete(success ? ErrorInfo::OkStatus() :
                                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected failure"));
        return true;
    }

    std::vector<TransferDescriptor> batchDescriptors(size_t batch_index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch_index >= descriptors_.size()) {
            return {};
        }
        return descriptors_[batch_index];
    }

    std::optional<TransferDescriptor> descriptorForGroupSet(size_t group_set_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& batch : descriptors_) {
            if (batch.size() == 1 && batch.front().group_set_id == group_set_id) {
                return batch.front();
            }
        }
        return std::nullopt;
    }

private:
    mutable std::mutex                                      mutex_;
    std::condition_variable                                 cv_;
    std::vector<std::vector<TransferDescriptor>>            descriptors_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;
};

class BlockTreeEvictorTestPeer {
public:
    static void reserveSource(BlockTreeEvictor& evictor, const TransferDescriptor& eviction_desc) {
        evictor.reserveSource({eviction_desc});
    }

    static EvictionDropTask createDropTask(BlockTreeEvictor& evictor, TransferDescriptor primary_desc) {
        EvictionDropTask                task = evictor.createDropTask(std::move(primary_desc));
        std::vector<TransferDescriptor> descs{task.primary_desc};
        descs.insert(descs.end(), task.cascade_descs.begin(), task.cascade_descs.end());
        descs.insert(descs.end(), task.dependent_prune_descs.begin(), task.dependent_prune_descs.end());
        evictor.restoreSource(descs);
        return task;
    }

    static void runDropTask(BlockTreeEvictor& evictor, TransferDescriptor primary_desc) {
        evictor.runDropTask(std::move(primary_desc), /*notify_settled=*/false);
    }

    static void rollbackDesc(BlockTreeEvictor& evictor, const TransferDescriptor& eviction_desc) {
        evictor.rollbackTransferLocked({eviction_desc});
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
    auto                          deferred_engine = std::make_shared<DeferredEvictionTransferEngine>(groups);
    BlockTransferDispatcher       dispatcher(deferred_engine);
    BlockTreeCacheMetricsReporter metrics_reporter;
    std::mutex                    cache_mutex;
    size_t                        settled_count = 0;
    BlockTreeEvictor              evictor(
        &tree,
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        &dispatcher,
        &task_pool,
        metrics_reporter,
        cache_mutex,
        0,
        0,
        8,
        16,
        [](Tier) { return true; },
        [&](bool, bool) { ++settled_count; });

    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    const BlockIdxType source = device_blocks.front().front();
    auto inserted             = tree.insertNode({100}, {{makeResource(Tier::DEVICE, source)}}, /*collect_path=*/false);
    unreferenceDeviceBlocksForTest(*group, device_blocks, BlockTreeRefType::CACHE);
    evictor.onInserted(inserted);

    ASSERT_TRUE(evictor.batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_TRUE(deferred_engine->waitForBatchCount(1, std::chrono::seconds(2)));

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

    ASSERT_TRUE(deferred_engine->completeGroupSet(0, true));
    task_pool.waitForIdle();
    EXPECT_EQ(settled_count, 1u);
    task_pool.shutdown();
}

class MultiGroupAsyncEvictionEnvironment {
public:
    ~MultiGroupAsyncEvictionEnvironment() {
        if (task_pool_ != nullptr) {
            task_pool_->shutdown();
        }
    }

    bool init() {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (test_info == nullptr) {
            return false;
        }
        const std::string test_name = test_info->name();
        device_pools_               = {makeTestDevicePool(4, test_name + "_device_0"),
                                       makeTestDevicePool(4, test_name + "_device_1")};
        host_pools_                 = {makePageableHostPool(4), makePageableHostPool(4)};
        disk_pools_ = {makeTestDiskPool(4, test_name + "_disk_0"), makeTestDiskPool(4, test_name + "_disk_1")};
        if (device_pools_[0] == nullptr || device_pools_[1] == nullptr || host_pools_[0] == nullptr
            || host_pools_[1] == nullptr || disk_pools_[0] == nullptr || disk_pools_[1] == nullptr) {
            return false;
        }

        groups_                           = {std::make_shared<FullGroupSet>(
                       std::vector<DeviceBlockPoolPtr>{device_pools_[0]}, host_pools_[0], disk_pools_[0]),
                                             std::make_shared<LinearGroupSet>(
                       std::vector<DeviceBlockPoolPtr>{device_pools_[1]}, host_pools_[1], disk_pools_[1])};
        auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
        auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        full_policy.enable_prefix_reuse   = true;
        linear_policy.enable_prefix_reuse = true;
        initializeGroups(groups_,
                         device_pools_,
                         {block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, 16),
                          block_transfer_engine_test::makeTestGroupBase(linear_policy, {0}, 16)});

        tree_      = std::make_unique<BlockTree>(groups_);
        task_pool_ = std::make_unique<BlockTreeTaskPool>(/*thread_count=*/2, /*queue_size=*/2, test_name);
        if (!task_pool_->start()) {
            return false;
        }
        transfer_engine_     = std::make_shared<DeferredEvictionTransferEngine>(groups_);
        transfer_dispatcher_ = std::make_unique<BlockTransferDispatcher>(transfer_engine_);
        evictor_             = std::make_unique<BlockTreeEvictor>(
            tree_.get(),
            EvictionPolicy::LRU,
            EvictionPolicy::LRU,
            EvictionPolicy::FIFO,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            metrics_reporter_,
            cache_mutex_,
            0,
            0,
            /*max_device_host_batch=*/8,
            /*max_non_device_host_batch=*/16,
            [](Tier) { return true; },
            [this](bool tree_data_mutated, bool check_watermark) {
                {
                    std::lock_guard<std::mutex> lock(settled_mutex_);
                    settled_events_.emplace_back(tree_data_mutated, check_watermark);
                }
                settled_cv_.notify_all();
            });
        return true;
    }

    TreeNode* insertDeviceNode() {
        std::vector<GroupSetResource> resources(groups_.size());
        std::vector<MultiNodeBlocks>  seed_blocks(groups_.size());
        device_sources_.resize(groups_.size(), NULL_BLOCK_IDX);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            seed_blocks[group_set_id] = allocateDeviceBlocksForTest(*groups_[group_set_id], 1, BlockTreeRefType::CACHE);
            if (seed_blocks[group_set_id].size() != 1 || seed_blocks[group_set_id].front().size() != 1) {
                return nullptr;
            }
            device_sources_[group_set_id]         = seed_blocks[group_set_id].front().front();
            resources[group_set_id].device_blocks = seed_blocks[group_set_id].front();
        }
        auto result = tree_->insertNode({100}, {resources}, /*collect_path=*/false);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            unreferenceDeviceBlocksForTest(*groups_[group_set_id], seed_blocks[group_set_id], BlockTreeRefType::CACHE);
        }
        evictor_->onInserted(result);
        return insertedNode(result);
    }

    std::vector<TreeNode*> insertParentDeviceChildHost() {
        std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(groups_.size()));
        std::vector<MultiNodeBlocks>               device_seed_blocks(groups_.size());
        device_sources_.resize(groups_.size(), NULL_BLOCK_IDX);
        host_sources_.resize(groups_.size(), NULL_BLOCK_IDX);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            device_seed_blocks[group_set_id] =
                allocateDeviceBlocksForTest(*groups_[group_set_id], 1, BlockTreeRefType::CACHE);
            if (device_seed_blocks[group_set_id].size() != 1 || device_seed_blocks[group_set_id].front().size() != 1) {
                return {};
            }
            device_sources_[group_set_id]            = device_seed_blocks[group_set_id].front().front();
            resources[0][group_set_id].device_blocks = device_seed_blocks[group_set_id].front();
            host_sources_[group_set_id] =
                groups_[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            if (isNullBlockIdx(host_sources_[group_set_id])) {
                return {};
            }
            resources[1][group_set_id].host_block = host_sources_[group_set_id];
        }
        auto result = tree_->insertNode({100, 200}, resources, /*collect_path=*/false);
        for (size_t group_set_id = 0; group_set_id < groups_.size(); ++group_set_id) {
            unreferenceDeviceBlocksForTest(
                *groups_[group_set_id], device_seed_blocks[group_set_id], BlockTreeRefType::CACHE);
        }
        releaseLowerTierSeedRefs(groups_, resources);
        evictor_->onInserted(result);
        return tree_->findNode({100, 200});
    }

    bool waitForSettledCount(size_t expected) {
        std::unique_lock<std::mutex> lock(settled_mutex_);
        return settled_cv_.wait_for(lock, std::chrono::seconds(2), [&] { return settled_events_.size() >= expected; });
    }

    std::vector<std::pair<bool, bool>> settledEvents() const {
        std::lock_guard<std::mutex> lock(settled_mutex_);
        return settled_events_;
    }

    size_t pendingReleaseCount() const {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        size_t                      count = 0;
        for (const auto& [_, pending] : evictor_->pending_release_counts_) {
            count += pending;
        }
        return count;
    }

    std::vector<DeviceBlockPoolPtr>                      device_pools_;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools_;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools_;
    std::vector<GroupSetPtr>                             groups_;
    std::vector<BlockIdxType>                            device_sources_;
    std::vector<BlockIdxType>                            host_sources_;
    std::unique_ptr<BlockTree>                           tree_;
    std::unique_ptr<BlockTreeTaskPool>                   task_pool_;
    std::shared_ptr<DeferredEvictionTransferEngine>      transfer_engine_;
    std::unique_ptr<BlockTransferDispatcher>             transfer_dispatcher_;
    BlockTreeCacheMetricsReporter                        metrics_reporter_;
    std::mutex                                           cache_mutex_;
    std::unique_ptr<BlockTreeEvictor>                    evictor_;

private:
    mutable std::mutex                 settled_mutex_;
    std::condition_variable            settled_cv_;
    std::vector<std::pair<bool, bool>> settled_events_;
};

TEST(BlockTreeEvictorAsyncTest, SameNodeGroupSetsSettleInReverseCompletionOrder) {
    MultiGroupAsyncEvictionEnvironment environment;
    ASSERT_TRUE(environment.init());
    TreeNode* node = environment.insertDeviceNode();
    ASSERT_NE(node, nullptr);

    ASSERT_TRUE(environment.evictor_->batchEvictLocked(0, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.evictor_->batchEvictLocked(1, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.transfer_engine_->waitForBatchCount(2, std::chrono::seconds(2)));
    ASSERT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    ASSERT_EQ(node->group_set_resources[1].transfer_state, GroupSetTransferState::DEMOTING);

    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(1, true));
    ASSERT_TRUE(environment.waitForSettledCount(1));
    EXPECT_EQ(node->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_TRUE(node->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_EQ(node->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_TRUE(node->group_set_resources[1].hasTier(Tier::HOST));
    EXPECT_EQ(environment.pendingReleaseCount(), 1u);

    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(0, true));
    environment.task_pool_->waitForIdle();
    EXPECT_TRUE(node->group_set_resources[0].hasTier(Tier::HOST));
    EXPECT_TRUE(node->group_set_resources[1].hasTier(Tier::HOST));
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 2u);
    EXPECT_EQ(environment.pendingReleaseCount(), 0u);
    EXPECT_EQ(environment.settledEvents(), (std::vector<std::pair<bool, bool>>{{true, true}, {true, true}}));
}

TEST(BlockTreeEvictorAsyncTest, SameNodeGroupSetsSettleSuccessAndFailureIndependently) {
    MultiGroupAsyncEvictionEnvironment environment;
    ASSERT_TRUE(environment.init());
    TreeNode* node = environment.insertDeviceNode();
    ASSERT_NE(node, nullptr);

    ASSERT_TRUE(environment.evictor_->batchEvictLocked(0, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.evictor_->batchEvictLocked(1, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.transfer_engine_->waitForBatchCount(2, std::chrono::seconds(2)));
    const auto group_0_desc = environment.transfer_engine_->descriptorForGroupSet(0);
    const auto group_1_desc = environment.transfer_engine_->descriptorForGroupSet(1);
    ASSERT_TRUE(group_0_desc.has_value());
    ASSERT_TRUE(group_1_desc.has_value());

    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(1, false));
    ASSERT_TRUE(environment.waitForSettledCount(1));
    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(0, true));
    environment.task_pool_->waitForIdle();

    EXPECT_TRUE(node->group_set_resources[0].hasTier(Tier::HOST));
    EXPECT_FALSE(node->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_TRUE(node->group_set_resources[1].hasTier(Tier::DEVICE));
    EXPECT_FALSE(node->group_set_resources[1].hasTier(Tier::HOST));
    EXPECT_FALSE(environment.device_pools_[0]->isAllocated(environment.device_sources_[0]));
    EXPECT_TRUE(environment.device_pools_[1]->isAllocated(environment.device_sources_[1]));
    EXPECT_TRUE(environment.host_pools_[0]->isAllocated(group_0_desc->target_blocks.front()));
    EXPECT_FALSE(environment.host_pools_[1]->isAllocated(group_1_desc->target_blocks.front()));
    EXPECT_EQ(environment.evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 1u);
    EXPECT_EQ(environment.pendingReleaseCount(), 0u);
    EXPECT_EQ(environment.settledEvents(), (std::vector<std::pair<bool, bool>>{{false, false}, {true, true}}));
}

TEST(BlockTreeEvictorAsyncTest, ForceDropDetachesTwoGroupSetsBeforeLateCompletions) {
    MultiGroupAsyncEvictionEnvironment environment;
    ASSERT_TRUE(environment.init());
    kmonitor::MetricsTags                      base_tags;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter =
        std::make_shared<kmonitor::MetricsReporter>("", "", base_tags);
    environment.metrics_reporter_.setMetricsReporter(metrics_reporter);
    const std::vector<TreeNode*> path = environment.insertParentDeviceChildHost();
    ASSERT_EQ(path.size(), 2u);
    TreeNode* const parent = path[0];
    TreeNode* const child  = path[1];

    ASSERT_TRUE(environment.evictor_->batchEvictLocked(0, Tier::HOST, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.evictor_->batchEvictLocked(1, Tier::HOST, /*max_victim_count=*/1));
    ASSERT_TRUE(environment.transfer_engine_->waitForBatchCount(2, std::chrono::seconds(2)));
    const auto group_0_desc = environment.transfer_engine_->descriptorForGroupSet(0);
    const auto group_1_desc = environment.transfer_engine_->descriptorForGroupSet(1);
    ASSERT_TRUE(group_0_desc.has_value());
    ASSERT_TRUE(group_1_desc.has_value());

    ASSERT_TRUE(environment.evictor_->dropLocked(0, Tier::DEVICE, /*notify_settled=*/true));
    EXPECT_TRUE(parent->group_set_resources[0].is_empty());
    EXPECT_TRUE(parent->group_set_resources[1].is_empty());
    EXPECT_TRUE(child->group_set_resources[0].transfer_detached);
    EXPECT_TRUE(child->group_set_resources[1].transfer_detached);
    EXPECT_EQ(environment.settledEvents(), (std::vector<std::pair<bool, bool>>{{true, false}}));

    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(1, true));
    ASSERT_TRUE(environment.waitForSettledCount(2));
    EXPECT_TRUE(child->group_set_resources[1].is_empty());
    EXPECT_FALSE(child->group_set_resources[1].transfer_detached);
    EXPECT_TRUE(child->group_set_resources[0].transfer_detached);
    EXPECT_FALSE(environment.host_pools_[1]->isAllocated(environment.host_sources_[1]));
    EXPECT_FALSE(environment.disk_pools_[1]->isAllocated(group_1_desc->target_blocks.front()));

    ASSERT_TRUE(environment.transfer_engine_->completeGroupSet(0, false));
    environment.task_pool_->waitForIdle();
    EXPECT_TRUE(environment.tree_->findNode({100, 200}).empty());
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        EXPECT_FALSE(environment.host_pools_[group_set_id]->isAllocated(environment.host_sources_[group_set_id]));
    }
    EXPECT_FALSE(environment.disk_pools_[0]->isAllocated(group_0_desc->target_blocks.front()));
    EXPECT_EQ(environment.pendingReleaseCount(), 0u);
    EXPECT_EQ(environment.settledEvents(),
              (std::vector<std::pair<bool, bool>>{{true, false}, {true, false}, {true, false}}));

    RtpLLMCacheEvictionMetrics* eviction_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheEvictionMetrics>();
    ASSERT_NE(eviction_metrics, nullptr);
    for (CacheGroupType group_type : {CacheGroupType::FULL, CacheGroupType::LINEAR}) {
        kmonitor::MetricsTags transfer_tags("source_tier", tierName(Tier::HOST));
        transfer_tags.AddTag("target_tier", tierName(Tier::DISK));
        transfer_tags.AddTag("group_type", metricCacheGroupTypeName(group_type));
        EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_qps_metric, transfer_tags), 0);

        kmonitor::MetricsTags drop_tags("source_tier", tierName(Tier::HOST));
        drop_tags.AddTag("target_tier", tierName(Tier::NONE));
        drop_tags.AddTag("group_type", metricCacheGroupTypeName(group_type));
        EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_qps_metric, drop_tags), 1);
    }
}

void verifyMixedDetachedBatchSettlement(bool transfer_success) {
    const std::string suffix      = transfer_success ? "success" : "failure";
    auto              device_pool = makeTestDevicePool(3, "mixed_batch_settlement_" + suffix + "_device");
    auto              host_pool   = makePageableHostPool(3);
    auto              disk_pool   = makeTestDiskPool(3, "mixed_batch_settlement_" + suffix + "_disk");
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);

    auto group = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    initializeFullGroup(group, device_pool);
    std::vector<GroupSetPtr> groups = {group};
    BlockTree                tree(groups);

    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/2, "mixed_batch_settlement_" + suffix);
    ASSERT_TRUE(task_pool.start());
    auto                               transfer_engine = std::make_shared<DeferredEvictionTransferEngine>(groups);
    BlockTransferDispatcher            dispatcher(transfer_engine);
    BlockTreeCacheMetricsReporter      metrics_reporter;
    std::mutex                         cache_mutex;
    std::mutex                         settled_mutex;
    std::vector<std::pair<bool, bool>> settled_events;
    BlockTreeEvictor                   evictor(
        &tree,
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        &dispatcher,
        &task_pool,
        metrics_reporter,
        cache_mutex,
        0,
        0,
        /*max_device_host_batch=*/8,
        /*max_non_device_host_batch=*/16,
        [](Tier) { return true; },
        [&](bool tree_data_mutated, bool check_watermark) {
            std::lock_guard<std::mutex> lock(settled_mutex);
            settled_events.emplace_back(tree_data_mutated, check_watermark);
        });

    const MultiNodeBlocks parent_device_blocks = allocateDeviceBlocksForTest(*group, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(parent_device_blocks.size(), 1u);
    ASSERT_EQ(parent_device_blocks.front().size(), 1u);
    const BlockIdxType child_host_source = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(child_host_source));
    std::vector<std::vector<GroupSetResource>> path_resources(2, std::vector<GroupSetResource>(1));
    path_resources[0][0].device_blocks = parent_device_blocks.front();
    path_resources[1][0].host_block    = child_host_source;
    auto path_insert                   = tree.insertNode({100, 200}, path_resources, /*collect_path=*/false);
    unreferenceDeviceBlocksForTest(*group, parent_device_blocks, BlockTreeRefType::CACHE);
    releaseLowerTierSeedRefs(groups, path_resources);
    evictor.onInserted(path_insert);
    const std::vector<TreeNode*> path = tree.findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    TreeNode* const child = path.back();

    const BlockIdxType sibling_host_source = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(sibling_host_source));
    std::vector<std::vector<GroupSetResource>> sibling_resources(1, std::vector<GroupSetResource>(1));
    sibling_resources[0][0].host_block = sibling_host_source;
    auto sibling_insert                = tree.insertNode({300}, sibling_resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(groups, sibling_resources);
    evictor.onInserted(sibling_insert);
    TreeNode* const sibling = insertedNode(sibling_insert);
    ASSERT_NE(sibling, nullptr);

    ASSERT_TRUE(evictor.batchEvictLocked(/*group_set_id=*/0, Tier::HOST, /*max_victim_count=*/2));
    ASSERT_TRUE(transfer_engine->waitForBatchCount(1, std::chrono::seconds(2)));
    const std::vector<TransferDescriptor> descriptors = transfer_engine->batchDescriptors(0);
    ASSERT_EQ(descriptors.size(), 2u);
    const auto detached_it =
        std::find_if(descriptors.begin(), descriptors.end(), [child](const auto& desc) { return desc.node == child; });
    const auto normal_it = std::find_if(
        descriptors.begin(), descriptors.end(), [sibling](const auto& desc) { return desc.node == sibling; });
    ASSERT_NE(detached_it, descriptors.end());
    ASSERT_NE(normal_it, descriptors.end());
    const BlockIdxType detached_target = detached_it->target_blocks.front();
    const BlockIdxType normal_target   = normal_it->target_blocks.front();
    {
        std::lock_guard<std::mutex> lock(evictor.pending_release_mutex_);
        ASSERT_EQ(evictor.pending_release_counts_.at(host_pool.get()), 2u);
    }

    ASSERT_TRUE(evictor.dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/true));
    ASSERT_TRUE(child->group_set_resources[0].transfer_detached);
    {
        std::lock_guard<std::mutex> lock(settled_mutex);
        ASSERT_EQ(settled_events, (std::vector<std::pair<bool, bool>>{{true, false}}));
    }

    ASSERT_TRUE(transfer_engine->completeBatch(0, transfer_success));
    task_pool.waitForIdle();

    EXPECT_TRUE(tree.findNode({100, 200}).empty());
    EXPECT_FALSE(host_pool->isAllocated(child_host_source));
    EXPECT_FALSE(disk_pool->isAllocated(detached_target));
    {
        std::lock_guard<std::mutex> lock(evictor.pending_release_mutex_);
        EXPECT_EQ(evictor.pending_release_counts_.at(host_pool.get()), 0u);
    }
    {
        std::lock_guard<std::mutex> lock(settled_mutex);
        ASSERT_EQ(settled_events.size(), 2u);
        EXPECT_EQ(settled_events.back(), std::make_pair(true, transfer_success));
    }

    const std::vector<TreeNode*> sibling_path = tree.findNode({300});
    ASSERT_EQ(sibling_path.size(), 1u);
    if (transfer_success) {
        EXPECT_FALSE(host_pool->isAllocated(sibling_host_source));
        EXPECT_TRUE(disk_pool->isAllocated(normal_target));
        EXPECT_TRUE(sibling->group_set_resources[0].hasTier(Tier::DISK));
        EXPECT_FALSE(sibling->group_set_resources[0].hasTier(Tier::HOST));
        EXPECT_TRUE(evictor.dropLocked(/*group_set_id=*/0, Tier::DISK, /*notify_settled=*/true));
    } else {
        EXPECT_TRUE(host_pool->isAllocated(sibling_host_source));
        EXPECT_FALSE(disk_pool->isAllocated(normal_target));
        EXPECT_TRUE(sibling->group_set_resources[0].hasTier(Tier::HOST));
        EXPECT_FALSE(sibling->group_set_resources[0].hasTier(Tier::DISK));
        EXPECT_TRUE(evictor.dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
    }
    task_pool.shutdown();
}

TEST(BlockTreeEvictorAsyncTest, MixedDetachedBatchSuccessSettlesOnceAndPublishesOnlyNormalTarget) {
    verifyMixedDetachedBatchSettlement(true);
}

TEST(BlockTreeEvictorAsyncTest, MixedDetachedBatchFailureSettlesOnceAndRollsBackNormalDescriptor) {
    verifyMixedDetachedBatchSettlement(false);
}

std::vector<size_t> cascadeGroupSetIds(const EvictionDropTask& task) {
    std::vector<size_t> result;
    result.reserve(task.cascade_descs.size());
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        result.push_back(cascade_desc.group_set_id);
    }
    return result;
}

std::vector<size_t> rootDependentPruneGroupSetIds(const EvictionDropTask& task, const TreeNode* root) {
    std::vector<size_t> result;
    for (const TransferDescriptor& dependent_desc : task.dependent_prune_descs) {
        if (dependent_desc.node == root) {
            EXPECT_EQ(dependent_desc.source_tier, Tier::NONE);
            result.push_back(dependent_desc.group_set_id);
        }
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

std::optional<EvictionTransferTask> activateTransferForTest(BlockTreeEvictor&  evictor,
                                                            TransferDescriptor eviction_desc) {
    EvictionTransferTask task;
    task.timings.emplace_back(eviction_desc.node->group_set_resources[eviction_desc.group_set_id].candidate_meta);
    BlockIdxType target = evictor.tree_->groupSets()[eviction_desc.group_set_id]->allocateSingleBlock(
        eviction_desc.target_tier, BlockTreeRefType::EVICTION);
    if (isNullBlockIdx(target)) {
        return std::nullopt;
    }
    eviction_desc.target_blocks = {target};
    evictor.reserveSource({eviction_desc});
    task.descs.push_back(std::move(eviction_desc));
    return task;
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

    std::optional<EvictionTransferTask> prepareTask(size_t primary_group_set_id) {
        auto victim = evictor_->chooseVictim(primary_group_set_id, Tier::HOST);
        if (!victim.has_value()) {
            return std::nullopt;
        }
        return activateTransferForTest(*evictor_, *victim);
    }

    void setTransferResults(std::initializer_list<bool> results) {
        auto transfer_engine = evictor_runtime_.transferEngine();
        transfer_engine->clear();
        for (bool success : results) {
            transfer_engine->enqueue(success);
        }
    }

    bool runTransfer(const EvictionTransferTask& task) {
        std::optional<bool> result;
        evictor_->task_runner_->runTransfer(std::make_shared<EvictionTransferTask>(task),
                                            *evictor_->metrics_reporter_,
                                            [&](bool success) { result = success; });
        if (!result.has_value()) {
            ADD_FAILURE() << "scripted eviction transfer did not complete inline";
            return false;
        }
        return *result;
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
                resource.disk_block      = NULL_BLOCK_IDX;
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
        TransferDescriptor desc;
        desc.group_set_id  = 0;
        desc.source_tier   = source_tier;
        desc.target_tier   = target_tier;
        desc.source_blocks = {block};

        evictor_->updatePendingRelease({desc}, true);
        ASSERT_EQ(evictor_->pending_release_counts_.at(pool), 1u);
        evictor_->updatePendingRelease({desc}, false);
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

    TransferDescriptor desc;
    desc.group_set_id  = 0;
    desc.source_tier   = Tier::DEVICE;
    desc.target_tier   = Tier::HOST;
    desc.source_blocks = {7, 8};

    evictor_->updatePendingRelease({desc}, true);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 2u);
    evictor_->updatePendingRelease({desc}, false);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
}

TEST_F(BlockTreeEvictorTest, PendingReleasesCountEveryDescriptorInBatch) {
    TransferDescriptor first;
    first.group_set_id        = 0;
    first.source_tier         = Tier::DEVICE;
    first.target_tier         = Tier::HOST;
    first.source_blocks       = {7};
    TransferDescriptor second = first;
    second.source_blocks      = {8};
    const std::vector<TransferDescriptor> descs{first, second};

    evictor_->updatePendingRelease(descs, true);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 2u);
    evictor_->updatePendingRelease(descs, false);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
}

TEST_F(BlockTreeEvictorTest, UpdatePendingReleasesReportsPoolAndRequiredCountOnInvalidSettlement) {
    TransferDescriptor desc;
    desc.group_set_id  = 0;
    desc.source_tier   = Tier::DEVICE;
    desc.target_tier   = Tier::HOST;
    desc.source_blocks = {17};

    try {
        evictor_->updatePendingRelease({desc}, false);
        FAIL() << "settling an unreserved pending release should fail";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("pool=" + device_pool_->poolName()), std::string::npos);
        EXPECT_NE(message.find("required=1"), std::string::npos);
        EXPECT_NE(message.find("pending=0"), std::string::npos);
    }
}

TEST_F(BlockTreeEvictorTest, PendingReleaseSettlementIsTransactionalAcrossDevicePools) {
    auto second_device_pool = makeTestDevicePool(2, "pending_release_second_device");
    ASSERT_NE(second_device_pool, nullptr);
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology =
        block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(policy, {0}, 16),
                                                      block_transfer_engine_test::makeTestGroupBase(policy, {1}, 16)});
    group_ = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{device_pool_, second_device_pool}, nullptr, nullptr);
    group_->initialize(0, std::move(topology), {0, 1});
    groups_  = {group_};
    tree_    = std::make_unique<BlockTree>(groups_);
    evictor_ = evictor_runtime_.make(tree_.get());

    TransferDescriptor desc;
    desc.group_set_id                                     = 0;
    desc.source_tier                                      = Tier::DEVICE;
    desc.target_tier                                      = Tier::HOST;
    desc.source_blocks                                    = {7, 8};
    evictor_->pending_release_counts_[device_pool_.get()] = 1;

    EXPECT_THROW(evictor_->updatePendingRelease({desc}, false), std::exception);
    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 1u);
    EXPECT_EQ(evictor_->pending_release_counts_.count(second_device_pool.get()), 0u);

    evictor_->pending_release_counts_.clear();
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
    EXPECT_THROW(evictor_->completeEvict({desc}), std::exception);

    evictor_->suspendCandidate(node, 0, Tier::DEVICE);
    node->group_set_resources[0].evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, RunEvictionTaskReleasesPendingSourceBeforeSettledCallback) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/1, "eviction_settlement_order");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(tree_.get(), &task_pool);

    const auto allocated = device_pool_->malloc(1);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    const BlockIdxType source_block = allocated->front();
    auto               result       = insert({100}, {{makeResource(Tier::DEVICE, source_block)}});
    ASSERT_NE(insertedNode(result), nullptr);

    auto victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    auto task = activateTransferForTest(*evictor_, *victim);
    ASSERT_TRUE(task.has_value());
    ASSERT_NE(task->descs.front().target_tier, Tier::NONE);
    evictor_->updatePendingRelease(task->descs, true);
    ASSERT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 1u);
    size_t settled_count = 0;
    evictor_->settled_   = [this, &settled_count](bool tree_data_mutated, bool check_watermark) {
        ++settled_count;
        EXPECT_FALSE(tree_data_mutated);
        EXPECT_FALSE(check_watermark);
        EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
    };
    evictor_runtime_.transferEngine()->enqueue(false);

    // This test bypasses evictLocked(), so mirror its workflow admission
    // precondition before invoking the asynchronous runner directly.
    ASSERT_TRUE(task_pool.acquireWorkflowCredit());
    evictor_->runEvictionTask(std::make_shared<EvictionTransferTask>(*task));
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_->pending_release_counts_.at(device_pool_.get()), 0u);
    EXPECT_EQ(settled_count, 1u);
    GroupSetResource&       resource = insertedNode(result)->group_set_resources[0];
    const MultiNodeResource source{0, Tier::DEVICE, {{insertedNode(result), {source_block}}}};
    evictor_->suspendCandidate(insertedNode(result), 0, Tier::DEVICE);
    resource.evictFromTier(Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*group_, MultiNodeBlocks{{source_block}}, BlockTreeRefType::CACHE);
}

TEST_F(BlockTreeEvictorTest, ComputeWatermarkEvictCountRejectsPendingReleasesAboveUsedBlocks) {
    ASSERT_EQ(device_pool_->usedBlocksNum(), 0u);
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_[device_pool_.get()] = 1;
    }

    std::string error_message;
    try {
        (void)evictor_->computeWatermarkEvictCount(
            *group_, Tier::DEVICE, TierWatermark{/*low_ratio=*/0.4, /*high_ratio=*/0.5});
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

TEST_F(BlockTreeEvictorTest, WatermarkRequiresHighAndRequestsDownToLowWithIntegerBoundaries) {
    const auto first = device_pool_->malloc(115);
    ASSERT_TRUE(first.has_value());
    device_pool_->incRef(*first);
    ASSERT_EQ(device_pool_->totalBlocksNum(), 128u);

    const TierWatermark watermark{/*low_ratio=*/0.82, /*high_ratio=*/0.90};
    EXPECT_EQ(evictor_->computeWatermarkEvictCount(*group_, Tier::DEVICE, watermark), 0u);

    const auto second = device_pool_->malloc(1);
    ASSERT_TRUE(second.has_value());
    device_pool_->incRef(*second);
    EXPECT_EQ(evictor_->computeWatermarkEvictCount(*group_, Tier::DEVICE, watermark), 12u);

    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_[device_pool_.get()] = 1;
    }
    EXPECT_EQ(evictor_->computeWatermarkEvictCount(*group_, Tier::DEVICE, watermark), 11u);
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        evictor_->pending_release_counts_.clear();
    }

    device_pool_->decRef(*first);
    device_pool_->decRef(*second);
}

TEST_F(BlockTreeEvictorTest, WatermarkRequiredGaugeClearsWhenNextCheckHasNoDeficit) {
    kmonitor::MetricsTags base_tags;
    auto                  metrics_reporter = std::make_shared<kmonitor::MetricsReporter>("", "", base_tags);
    evictor_runtime_.setMetricsReporter(metrics_reporter);

    const auto blocks = device_pool_->malloc(116);
    ASSERT_TRUE(blocks.has_value());
    device_pool_->incRef(*blocks);

    const TierWatermark watermark{/*low_ratio=*/0.82, /*high_ratio=*/0.90};
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);

    RtpLLMCacheEvictionMetrics* eviction_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheEvictionMetrics>();
    ASSERT_NE(eviction_metrics, nullptr);
    kmonitor::MetricsTags watermark_tags("tier", tierName(Tier::DEVICE));
    watermark_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->watermark_required_blocks_metric, watermark_tags), 12);

    device_pool_->decRef(*blocks);
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->watermark_required_blocks_metric, watermark_tags), 0);
}

TEST_F(BlockTreeEvictorTest, DeviceHostWatermarkCapsBatchByRemainingRequiredCount) {
    device_pool_ = makeTestDevicePool(10, "configured_device_host_watermark_batch");
    ASSERT_NE(device_pool_, nullptr);
    auto host_pool = makePageableHostPool(4);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "configured_device_host_watermark_batch");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(
        tree_.get(),
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        [](Tier) { return true; },
        &task_pool,
        /*max_device_host_batch=*/4,
        /*max_non_device_host_batch=*/16);

    for (int64_t key = 100; key < 900; key += 100) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }
    ASSERT_EQ(device_pool_->usedBlocksNum(), 8u);

    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, TierWatermark{/*low_ratio=*/0.70, /*high_ratio=*/0.80});
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 1u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 1u);
    EXPECT_EQ(device_pool_->usedBlocksNum(), 7u);
    EXPECT_EQ(host_pool->usedBlocksNum(), 1u);

    evictor_->settled_ = [](bool, bool) {};
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/false)) {}
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/false)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DeviceHostWatermarkSubmitsOneLogicalBatchCappedByTransferLimit) {
    auto host_pool = makePageableHostPool(4);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    BlockTreeTaskPool     task_pool(/*thread_count=*/1, /*queue_size=*/4, "device_host_watermark_batch");
    kmonitor::MetricsTags base_tags;
    auto                  metrics_reporter = std::make_shared<kmonitor::MetricsReporter>("", "", base_tags);
    evictor_runtime_.setMetricsReporter(metrics_reporter);

    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(
        tree_.get(),
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        [](Tier) { return true; },
        &task_pool,
        /*max_device_host_batch=*/2,
        /*max_non_device_host_batch=*/16);

    for (int64_t key : {100, 200, 300}) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }

    evictor_runtime_.transferEngine()->enqueue(false);
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, TierWatermark{/*low_ratio=*/0.001, /*high_ratio=*/0.02});
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 1u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 2u);
    RtpLLMCacheEvictionMetrics* eviction_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheEvictionMetrics>();
    ASSERT_NE(eviction_metrics, nullptr);
    kmonitor::MetricsTags watermark_tags("tier", tierName(Tier::DEVICE));
    watermark_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->watermark_required_blocks_metric, watermark_tags), 3);

    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/true)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DirectDeviceDropsConvergePastTransferBatchLimit) {
    device_pool_ = makeTestDevicePool(20, "direct_device_watermark_convergence");
    ASSERT_NE(device_pool_, nullptr);
    auto host_pool = makePageableHostPool(18);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    BlockTreeTaskPool     task_pool(/*thread_count=*/1, /*queue_size=*/4, "direct_device_watermark_convergence");
    kmonitor::MetricsTags base_tags;
    auto                  metrics_reporter = std::make_shared<kmonitor::MetricsReporter>("", "", base_tags);
    evictor_runtime_.setMetricsReporter(metrics_reporter);
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(
        tree_.get(),
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        [](Tier) { return true; },
        &task_pool,
        /*max_device_host_batch=*/8,
        /*max_non_device_host_batch=*/16);

    for (int64_t key = 100; key < 1900; key += 100) {
        const MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(device_blocks.size(), 1u);
        ASSERT_EQ(device_blocks.front().size(), 1u);
        const BlockIdxType host_block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
        ASSERT_FALSE(isNullBlockIdx(host_block));

        GroupSetResource resource = makeResource(Tier::DEVICE, device_blocks.front().front());
        resource.host_block       = host_block;
        auto result               = insert({key}, {{resource}});
        ASSERT_NE(insertedNode(result), nullptr);
        unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
    }
    ASSERT_EQ(device_pool_->usedBlocksNum(), 18u);
    ASSERT_EQ(host_pool->usedBlocksNum(), 18u);

    std::vector<std::pair<bool, bool>> settled_events;
    evictor_->settled_ = [&settled_events](bool tree_data_mutated, bool check_watermark) {
        settled_events.emplace_back(tree_data_mutated, check_watermark);
    };
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, TierWatermark{/*low_ratio=*/0.40, /*high_ratio=*/0.80});

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 0u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 0u);
    EXPECT_EQ(device_pool_->usedBlocksNum(), 8u);
    EXPECT_EQ(host_pool->usedBlocksNum(), 18u);
    EXPECT_EQ(device_pool_->referencedBlocksNum(BlockTreeRefType::CACHE), 8u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), 18u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 8u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 10u);
    EXPECT_EQ(tree_->size(), 18u);
    EXPECT_EQ(settled_events, (std::vector<std::pair<bool, bool>>{{true, false}, {true, false}}));

    RtpLLMCacheEvictionMetrics* eviction_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheEvictionMetrics>();
    ASSERT_NE(eviction_metrics, nullptr);
    kmonitor::MetricsTags watermark_tags("tier", tierName(Tier::DEVICE));
    watermark_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->watermark_required_blocks_metric, watermark_tags), 0);
    kmonitor::MetricsTags trigger_tags("trigger_type", "watermark");
    trigger_tags.AddTag("source_tier", tierName(Tier::DEVICE));
    trigger_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_trigger_qps_metric, trigger_tags), 1);

    evictor_->settled_ = [](bool, bool) {};
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/false)) {}
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/false)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DeviceWatermarkStaysTriggeredAcrossBatchesUntilLow) {
    device_pool_ = makeTestDevicePool(10, "persistent_device_watermark");
    ASSERT_NE(device_pool_, nullptr);
    auto host_pool = makePageableHostPool(4);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "persistent_device_watermark");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(
        tree_.get(),
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        [](Tier) { return true; },
        &task_pool,
        /*max_device_host_batch=*/2,
        /*max_non_device_host_batch=*/16);

    for (int64_t key = 100; key < 900; key += 100) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }
    ASSERT_EQ(device_pool_->usedBlocksNum(), 8u);

    const TierWatermark                watermark{/*low_ratio=*/0.50, /*high_ratio=*/0.80};
    std::vector<std::pair<bool, bool>> settled_events;
    evictor_->settled_ = [this, &watermark, &settled_events](bool tree_data_mutated, bool check_watermark) {
        settled_events.emplace_back(tree_data_mutated, check_watermark);
        if (check_watermark) {
            evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
        }
    };
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_runtime_.transferEngine()->enqueue(true);

    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 2u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 3u);
    EXPECT_EQ(device_pool_->usedBlocksNum(), 5u);
    EXPECT_EQ(host_pool->usedBlocksNum(), 3u);
    EXPECT_EQ(settled_events, (std::vector<std::pair<bool, bool>>{{true, true}, {true, true}}));

    const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(blocks.size(), 1u);
    auto result = insert({900}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
    ASSERT_NE(insertedNode(result), nullptr);
    unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_pool_->usedBlocksNum(), 6u);

    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
    task_pool.waitForIdle();
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 2u);

    evictor_->settled_ = [](bool, bool) {};
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/false)) {}
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/false)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, HostDiskWatermarkConvergesAcrossBoundedBatches) {
    auto host_pool = makePageableHostPool(100);
    auto disk_pool = makeTestDiskPool(32, "host_disk_watermark_batch");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);
    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "host_disk_watermark_batch");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(tree_.get(), &task_pool);

    for (int64_t key = 100; key < 195; ++key) {
        const BlockIdxType source = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
        ASSERT_FALSE(isNullBlockIdx(source));
        auto result = insert({key}, {{makeResource(Tier::HOST, source)}});
        ASSERT_NE(insertedNode(result), nullptr);
    }

    const TierWatermark watermark{/*low_ratio=*/0.78, /*high_ratio=*/0.95};
    evictor_->settled_ = [this, &watermark](bool, bool check_watermark) {
        if (check_watermark) {
            evictor_->scheduleWatermarkEvictionsLocked(Tier::HOST, watermark);
        }
    };
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_->scheduleWatermarkEvictionsLocked(Tier::HOST, watermark);
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 2u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 17u);
    EXPECT_EQ(host_pool->usedBlocksNum(), 78u);
    EXPECT_EQ(disk_pool->usedBlocksNum(), 17u);
    evictor_->settled_ = [](bool, bool) {};
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true)) {}
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DISK, /*notify_settled=*/false)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DeviceDiskEvictionRespectsPerRankSingleDescriptorContract) {
    device_pool_ = makeTestDevicePool(4, "device_disk_single_descriptor");
    ASSERT_NE(device_pool_, nullptr);
    auto disk_pool = makeTestDiskPool(3, "device_disk_single_descriptor");
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(nullptr, disk_pool);

    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "device_disk_single_descriptor");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(
        tree_.get(),
        EvictionPolicy::LRU,
        EvictionPolicy::LRU,
        EvictionPolicy::FIFO,
        [](Tier tier) { return tier != Tier::HOST; },
        &task_pool);

    for (int64_t key : {100, 200, 300}) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }

    const TierWatermark watermark{/*low_ratio=*/0.01, /*high_ratio=*/0.02};
    evictor_->settled_ = [this, &watermark](bool, bool check_watermark) {
        if (check_watermark) {
            evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
        }
    };
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_runtime_.transferEngine()->enqueue(true);
    evictor_->scheduleWatermarkEvictionsLocked(Tier::DEVICE, watermark);
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 3u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 3u);
    EXPECT_EQ(device_pool_->usedBlocksNum(), 0u);
    EXPECT_EQ(disk_pool->usedBlocksNum(), 3u);

    evictor_->settled_ = [](bool, bool) {};
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/false)) {}
    while (evictor_->dropLocked(/*group_set_id=*/0, Tier::DISK, /*notify_settled=*/false)) {}
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DirectDropWatermarkBatchNotifiesSettlementOnce) {
    auto disk_pool = makeTestDiskPool(3, "direct_drop_watermark_batch");
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(nullptr, disk_pool);

    for (int64_t key : {100, 200, 300}) {
        const BlockIdxType source = group_->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
        ASSERT_FALSE(isNullBlockIdx(source));
        auto result = insert({key}, {{makeResource(Tier::DISK, source)}});
        ASSERT_NE(insertedNode(result), nullptr);
    }

    size_t settled_count = 0;
    evictor_->settled_   = [&](bool tree_data_mutated, bool check_watermark) {
        ++settled_count;
        EXPECT_TRUE(tree_data_mutated);
        EXPECT_FALSE(check_watermark);
    };

    EXPECT_TRUE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DISK, /*max_victim_count=*/3));

    EXPECT_EQ(settled_count, 1u);
    EXPECT_EQ(tree_->size(), 0u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 3u);
}

TEST_F(BlockTreeEvictorTest, BatchDropLockedHonorsVictimLimit) {
    auto disk_pool = makeTestDiskPool(2, "batch_drop_victim_limit");
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(nullptr, disk_pool);

    for (int64_t key : {100, 200}) {
        const BlockIdxType source = group_->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
        ASSERT_FALSE(isNullBlockIdx(source));
        auto result = insert({key}, {{makeResource(Tier::DISK, source)}});
        ASSERT_NE(insertedNode(result), nullptr);
    }

    EXPECT_TRUE(evictor_->batchDropLocked(/*group_set_id=*/0, Tier::DISK, /*max_victim_count=*/1));

    EXPECT_EQ(tree_->size(), 1u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::DISK, /*notify_settled=*/true));
}

TEST_F(BlockTreeEvictorTest, BatchAdmissionRejectionRollsBackEveryPlannedDescriptor) {
    auto host_pool = makePageableHostPool(2);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "batch_admission_rejection");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(tree_.get(), &task_pool);

    std::vector<std::pair<TreeNode*, BlockIdxType>> sources;
    for (int64_t key : {100, 200}) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        ASSERT_EQ(blocks.front().size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        sources.emplace_back(insertedNode(result), blocks.front().front());
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }
    size_t settled_count = 0;
    evictor_->settled_   = [&](bool, bool) { ++settled_count; };
    task_pool.stopAdmission();

    EXPECT_FALSE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/2));

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 0u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 2u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_EQ(settled_count, 0u);
    for (const auto& [node, source] : sources) {
        const GroupSetResource& resource = node->group_set_resources[0];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source}));
        EXPECT_TRUE(device_pool_->isAllocated(source));
    }
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        for (const auto& [_, pending] : evictor_->pending_release_counts_) {
            EXPECT_EQ(pending, 0u);
        }
    }
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, BatchQueueTimeoutRollsBackEveryPlannedDescriptorOnce) {
    auto host_pool = makePageableHostPool(2);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "batch_queue_timeout");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(tree_.get(), &task_pool);

    std::vector<std::pair<TreeNode*, BlockIdxType>> sources;
    for (int64_t key : {100, 200}) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        ASSERT_EQ(blocks.front().size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        sources.emplace_back(insertedNode(result), blocks.front().front());
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }

    std::promise<void> worker_ready;
    std::promise<void> release_worker;
    auto               ready_future   = worker_ready.get_future();
    auto               release_future = release_worker.get_future();
    ASSERT_TRUE(task_pool.submit([&] {
        worker_ready.set_value();
        release_future.wait();
    }));
    if (ready_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release_worker.set_value();
        task_pool.shutdown();
        FAIL() << "blocking task did not occupy the business worker";
    }

    std::vector<std::pair<bool, bool>> settled_events;
    evictor_->settled_ = [&](bool tree_data_mutated, bool check_watermark) {
        settled_events.emplace_back(tree_data_mutated, check_watermark);
    };
    const bool submitted        = evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/2);
    bool       deadline_rewound = false;
    {
        std::lock_guard<std::mutex> lock(task_pool.lifecycle_mutex_);
        if (task_pool.normal_queue_.size() == 1) {
            task_pool.normal_queue_.front().deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
            deadline_rewound                         = true;
        }
    }
    release_worker.set_value();
    task_pool.waitForIdle();

    ASSERT_TRUE(submitted);
    ASSERT_TRUE(deadline_rewound);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 0u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 2u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_EQ(settled_events, (std::vector<std::pair<bool, bool>>{{false, false}}));
    for (const auto& [node, source] : sources) {
        const GroupSetResource& resource = node->group_set_resources[0];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source}));
        EXPECT_TRUE(device_pool_->isAllocated(source));
    }
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        for (const auto& [_, pending] : evictor_->pending_release_counts_) {
            EXPECT_EQ(pending, 0u);
        }
    }
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, BatchTargetExhaustionLeavesEntirePlannedBatchUnchanged) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);
    BlockTreeTaskPool task_pool(/*thread_count=*/1, /*queue_size=*/4, "batch_target_exhaustion");
    ASSERT_TRUE(task_pool.start());
    evictor_ = evictor_runtime_.make(tree_.get(), &task_pool);

    std::vector<std::pair<TreeNode*, BlockIdxType>> sources;
    for (int64_t key : {100, 200}) {
        const MultiNodeBlocks blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
        ASSERT_EQ(blocks.size(), 1u);
        ASSERT_EQ(blocks.front().size(), 1u);
        auto result = insert({key}, {{makeResource(Tier::DEVICE, blocks.front().front())}});
        ASSERT_NE(insertedNode(result), nullptr);
        sources.emplace_back(insertedNode(result), blocks.front().front());
        unreferenceDeviceBlocksForTest(*group_, blocks, BlockTreeRefType::CACHE);
    }

    std::vector<std::pair<bool, bool>> settled_events;
    evictor_->settled_ = [&](bool tree_data_mutated, bool check_watermark) {
        settled_events.emplace_back(tree_data_mutated, check_watermark);
    };
    evictor_runtime_.transferEngine()->enqueue(false);
    EXPECT_FALSE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/2));
    task_pool.waitForIdle();

    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedBatchCount(), 0u);
    EXPECT_EQ(evictor_runtime_.transferEngine()->submittedDescriptorCount(), 0u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 2u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);
    EXPECT_TRUE(settled_events.empty());
    for (const auto& [node, source] : sources) {
        const GroupSetResource& resource = node->group_set_resources[0];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source}));
        EXPECT_TRUE(device_pool_->isAllocated(source));
    }
    {
        std::lock_guard<std::mutex> lock(evictor_->pending_release_mutex_);
        for (const auto& [_, pending] : evictor_->pending_release_counts_) {
            EXPECT_EQ(pending, 0u);
        }
    }
    task_pool.shutdown();
}

TEST_F(BlockTreeEvictorTest, DeviceWatermarkUsesMaximumDeficitAcrossMemberPools) {
    auto narrow_pool = makeTestDevicePool(10, "watermark_member_narrow");
    auto wide_pool   = makeTestDevicePool(20, "watermark_member_wide");
    ASSERT_NE(narrow_pool, nullptr);
    ASSERT_NE(wide_pool, nullptr);

    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto topology =
        block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(policy, {0}, 16),
                                                      block_transfer_engine_test::makeTestGroupBase(policy, {1}, 16)});
    group_ = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{narrow_pool, wide_pool}, nullptr, nullptr);
    group_->initialize(0, std::move(topology), {0, 1});
    groups_  = {group_};
    tree_    = std::make_unique<BlockTree>(groups_);
    evictor_ = evictor_runtime_.make(tree_.get());

    const auto narrow_blocks = narrow_pool->malloc(8);
    const auto wide_blocks   = wide_pool->malloc(15);
    ASSERT_TRUE(narrow_blocks.has_value());
    ASSERT_TRUE(wide_blocks.has_value());
    narrow_pool->incRef(*narrow_blocks);
    wide_pool->incRef(*wide_blocks);

    const TierWatermark watermark{/*low_ratio=*/0.50, /*high_ratio=*/0.80};
    EXPECT_EQ(evictor_->computeWatermarkEvictCount(*group_, Tier::DEVICE, watermark), 5u);

    narrow_pool->decRef(*narrow_blocks);
    wide_pool->decRef(*wide_blocks);
}

TEST(BlockTreeEvictorCascadeTest, NonLeafDropCascadeFollowsGroupPriority) {
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
    TreeNode*  non_leaf  = path.inserted_nodes.front();
    const auto full_drop = BlockTreeEvictorTestPeer::createDropTask(
        evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/0, Tier::HOST, Tier::NONE));
    EXPECT_TRUE(cascadeGroupSetIds(full_drop).empty());
    EXPECT_EQ(rootDependentPruneGroupSetIds(full_drop, non_leaf), (std::vector<size_t>{1, 2}));
    EXPECT_EQ(cascadeGroupSetIds(BlockTreeEvictorTestPeer::createDropTask(
                  evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/1, Tier::HOST, Tier::NONE))),
              (std::vector<size_t>{2}));
    EXPECT_TRUE(
        cascadeGroupSetIds(BlockTreeEvictorTestPeer::createDropTask(
                               evictor, makeSelectionDesc(non_leaf, /*group_set_id=*/2, Tier::HOST, Tier::NONE)))
            .empty());
}

TEST(BlockTreeEvictorCascadeTest, LeafDropCascadeSelectsAllOtherGroups) {
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
    TreeNode* const leaf      = insertedNode(inserted);
    const auto      full_drop = BlockTreeEvictorTestPeer::createDropTask(
        evictor, makeSelectionDesc(leaf, /*group_set_id=*/0, Tier::HOST, Tier::NONE));
    EXPECT_TRUE(cascadeGroupSetIds(full_drop).empty());
    EXPECT_EQ(rootDependentPruneGroupSetIds(full_drop, leaf), (std::vector<size_t>{1, 2}));

    const auto cascade_to_full = BlockTreeEvictorTestPeer::createDropTask(
        evictor, makeSelectionDesc(leaf, /*group_set_id=*/1, Tier::HOST, Tier::NONE));
    EXPECT_EQ(cascade_to_full.primary_desc.group_set_id, 0u);
    EXPECT_TRUE(cascadeGroupSetIds(cascade_to_full).empty());
    EXPECT_EQ(rootDependentPruneGroupSetIds(cascade_to_full, leaf), (std::vector<size_t>{1, 2}));
}

TEST(BlockTreeEvictorCascadeTest, DirectFullPruneRemovesEveryTierFromClosureRoot) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder      = runtime.make(&tree);
    BlockTreeEvictor&  evictor             = *evictor_holder;
    TreeNode* const    leaf                = insertCascadeLeafWithMultiTierLinear(tree, groups);
    const BlockIdxType linear_device_block = leaf->group_set_resources[2].device_blocks.front();
    const BlockIdxType linear_host_block   = leaf->group_set_resources[2].host_block;

    const auto task = BlockTreeEvictorTestPeer::createDropTask(
        evictor, makeSelectionDesc(leaf, /*group_set_id=*/0, Tier::HOST, Tier::NONE));

    EXPECT_TRUE(task.hasFullPrune());
    EXPECT_EQ(task.primary_desc.group_set_id, 0u);
    EXPECT_EQ(rootDependentPruneGroupSetIds(task, leaf), (std::vector<size_t>{1, 2}));
    const auto linear_desc =
        std::find_if(task.dependent_prune_descs.begin(),
                     task.dependent_prune_descs.end(),
                     [leaf](const TransferDescriptor& desc) { return desc.node == leaf && desc.group_set_id == 2; });
    ASSERT_NE(linear_desc, task.dependent_prune_descs.end());
    EXPECT_EQ(linear_desc->source_tier, Tier::NONE);

    BlockTreeEvictorTestPeer::runDropTask(evictor, makeSelectionDesc(leaf, /*group_set_id=*/0, Tier::HOST, Tier::NONE));
    EXPECT_TRUE(tree.findNode({100}).empty());
    EXPECT_FALSE(groups[2]->devicePools().front()->isAllocated(linear_device_block));
    EXPECT_FALSE(groups[2]->hostPool()->isAllocated(linear_host_block));
}

TEST(BlockTreeEvictorCascadeTest, CascadedFullPruneRemovesEveryTierFromClosureRoot) {
    auto               groups = makeCascadeGroups();
    BlockTree          tree(groups);
    TestEvictorRuntime runtime;
    auto               evictor_holder      = runtime.make(&tree);
    BlockTreeEvictor&  evictor             = *evictor_holder;
    TreeNode* const    leaf                = insertCascadeLeafWithMultiTierLinear(tree, groups);
    const BlockIdxType linear_device_block = leaf->group_set_resources[2].device_blocks.front();
    const BlockIdxType linear_host_block   = leaf->group_set_resources[2].host_block;

    const auto task = BlockTreeEvictorTestPeer::createDropTask(
        evictor, makeSelectionDesc(leaf, /*group_set_id=*/1, Tier::HOST, Tier::NONE));

    EXPECT_TRUE(task.hasFullPrune());
    EXPECT_EQ(task.primary_desc.group_set_id, 0u);
    EXPECT_EQ(rootDependentPruneGroupSetIds(task, leaf), (std::vector<size_t>{1, 2}));
    const auto linear_desc =
        std::find_if(task.dependent_prune_descs.begin(),
                     task.dependent_prune_descs.end(),
                     [leaf](const TransferDescriptor& desc) { return desc.node == leaf && desc.group_set_id == 2; });
    ASSERT_NE(linear_desc, task.dependent_prune_descs.end());
    EXPECT_EQ(linear_desc->source_tier, Tier::NONE);

    BlockTreeEvictorTestPeer::runDropTask(evictor, makeSelectionDesc(leaf, /*group_set_id=*/1, Tier::HOST, Tier::NONE));
    EXPECT_TRUE(tree.findNode({100}).empty());
    EXPECT_FALSE(groups[2]->devicePools().front()->isAllocated(linear_device_block));
    EXPECT_FALSE(groups[2]->hostPool()->isAllocated(linear_host_block));
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

    const EvictionDropTask task = BlockTreeEvictorTestPeer::createDropTask(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::NONE));

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

    const EvictionDropTask task = BlockTreeEvictorTestPeer::createDropTask(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::NONE));

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
    auto task = activateTransferForTest(*evictor, *victim);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(inserted.inserted_nodes[1]->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(inserted.inserted_nodes[1]->group_set_resources[0].host_block, parent_block);

    evictor->rollbackTransferLocked(task->descs);
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

    EXPECT_TRUE(evictor->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
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

    const EvictionDropTask task = BlockTreeEvictorTestPeer::createDropTask(
        *evictor, makeSelectionDesc(inserted.inserted_nodes.back(), /*group_set_id=*/0, Tier::HOST, Tier::NONE));

    ASSERT_EQ(task.cascade_descs.size(), 1u);
    EXPECT_EQ(task.cascade_descs[0].node, inserted.inserted_nodes[1]);
    EXPECT_EQ(task.cascade_descs[0].source_tier, Tier::DISK);
    EXPECT_EQ(task.cascade_descs[0].target_tier, Tier::NONE);

    evictor->onInserted(inserted);
    EXPECT_TRUE(evictor->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
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

TEST_F(BlockTreeEvictorTest, DropLockedDropsSelectedVictim) {
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

    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/true));
    EXPECT_EQ(settled_count, 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(tree_->size(), 0u);
    EXPECT_FALSE(device_pool_->isAllocated(source));
    EXPECT_EQ(device_pool_->freeBlocksNum(), 128u);
}

TEST_F(BlockTreeEvictorTest, DeviceDropUsesExistingHostWhenDiskPoolIsFull) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "device_drop_existing_host_disk");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType disk_blocker = group_->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(disk_blocker));

    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    ASSERT_EQ(device_blocks.front().size(), 1u);
    const BlockIdxType device_block = device_blocks.front().front();
    const BlockIdxType host_block   = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));

    GroupSetResource resource = makeResource(Tier::DEVICE, device_block);
    resource.host_block       = host_block;
    auto      result          = insert({100}, {{resource}});
    TreeNode* node            = insertedNode(result);
    ASSERT_NE(node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
    ASSERT_EQ(node->group_set_resources[0].servingTierCount(), 2u);
    ASSERT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    ASSERT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);

    EXPECT_TRUE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_EQ(tree_->size(), 1u);
    const GroupSetResource& retained = node->group_set_resources[0];
    EXPECT_FALSE(retained.hasTier(Tier::DEVICE));
    EXPECT_TRUE(retained.hasTier(Tier::HOST));
    EXPECT_TRUE(retained.isValidSteadyState());
    EXPECT_FALSE(device_pool_->isAllocated(device_block));
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 0u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);
    EXPECT_TRUE(disk_pool->isAllocated(disk_blocker));

    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
    EXPECT_EQ(tree_->size(), 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    group_->releaseSingleBlock(Tier::DISK, disk_blocker, BlockTreeRefType::CACHE);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, DeviceDropUsesExistingDiskWhenHostPoolIsFull) {
    auto host_pool = makePageableHostPool(1);
    auto disk_pool = makeTestDiskPool(1, "device_drop_existing_disk_host");
    ASSERT_NE(host_pool, nullptr);
    ASSERT_NE(disk_pool, nullptr);
    resetGroup(host_pool, disk_pool);

    const BlockIdxType host_blocker = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_blocker));
    const BlockIdxType disk_block = group_->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(disk_block));
    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    const BlockIdxType device_block = device_blocks.front().front();

    GroupSetResource resource = makeResource(Tier::DEVICE, device_block);
    resource.disk_block       = disk_block;
    auto      result          = insert({100}, {{resource}});
    TreeNode* node            = insertedNode(result);
    ASSERT_NE(node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);

    EXPECT_TRUE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/1));
    ASSERT_EQ(tree_->size(), 1u);
    const GroupSetResource& retained = node->group_set_resources[0];
    EXPECT_FALSE(retained.hasTier(Tier::DEVICE));
    EXPECT_TRUE(retained.hasTier(Tier::DISK));
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DISK), 1u);
    EXPECT_EQ(evictor_runtime_.transferCount(), 0u);
    EXPECT_TRUE(host_pool->isAllocated(host_blocker));

    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::DISK, /*notify_settled=*/true));
    EXPECT_EQ(tree_->size(), 0u);
    group_->releaseSingleBlock(Tier::HOST, host_blocker, BlockTreeRefType::CACHE);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeEvictorTest, LowerTierAdoptionPreservesTopCandidateMetadata) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    auto      first = insert({100}, {{makeResource(Tier::DEVICE, device_blocks.front().front())}});
    TreeNode* node  = insertedNode(first);
    ASSERT_NE(node, nullptr);
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
    evictor_->onMatched({node});
    const CandidateMeta before = node->group_set_resources[0].candidate_meta;

    const BlockIdxType host_block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));
    GroupSetResource host_resource;
    host_resource.host_block = host_block;
    const auto adopted       = insert({100}, {{host_resource}});
    ASSERT_EQ(adopted.adopted_nodes.size(), 1u);
    EXPECT_EQ(adopted.adopted_nodes.front().old_top_tiers, (std::vector<Tier>{Tier::DEVICE}));
    EXPECT_EQ(adopted.adopted_nodes.front().new_top_tiers, (std::vector<Tier>{Tier::DEVICE}));

    const CandidateMeta& after = node->group_set_resources[0].candidate_meta;
    EXPECT_EQ(after.last_access_seq, before.last_access_seq);
    EXPECT_EQ(after.admission_seq, before.admission_seq);
    EXPECT_EQ(after.hit_count, before.hit_count);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);

    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/true));
    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
    EXPECT_EQ(tree_->size(), 0u);
}

TEST_F(BlockTreeEvictorTest, HigherTierAdoptionMovesTheSoleCandidate) {
    auto host_pool = makePageableHostPool(1);
    ASSERT_NE(host_pool, nullptr);
    resetGroup(host_pool);

    const BlockIdxType host_block = group_->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_FALSE(isNullBlockIdx(host_block));
    GroupSetResource host_resource;
    host_resource.host_block = host_block;
    auto      first          = insert({100}, {{host_resource}});
    TreeNode* node           = insertedNode(first);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    MultiNodeBlocks device_blocks = allocateDeviceBlocksForTest(*group_, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(device_blocks.size(), 1u);
    const auto adopted = insert({100}, {{makeResource(Tier::DEVICE, device_blocks.front().front())}});
    unreferenceDeviceBlocksForTest(*group_, device_blocks, BlockTreeRefType::CACHE);
    ASSERT_EQ(adopted.adopted_nodes.size(), 1u);
    EXPECT_EQ(adopted.adopted_nodes.front().old_top_tiers, (std::vector<Tier>{Tier::HOST}));
    EXPECT_EQ(adopted.adopted_nodes.front().new_top_tiers, (std::vector<Tier>{Tier::DEVICE}));
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);
    EXPECT_EQ(evictor_->candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    EXPECT_FALSE(evictor_->chooseVictim(/*group_set_id=*/0, Tier::HOST).has_value());

    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::DEVICE, /*notify_settled=*/true));
    EXPECT_TRUE(evictor_->dropLocked(/*group_set_id=*/0, Tier::HOST, /*notify_settled=*/true));
    EXPECT_EQ(tree_->size(), 0u);
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
    evictor_->updateFullCandidate(parent);

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
    EXPECT_EQ(fill_result.adopted_nodes.front().node, insertedNode(empty_child));

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

TEST_F(BlockTreeEvictorTest, BatchSettlementRemovesDescriptorsSharingAnAncestor) {
    const std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    auto                                             result = insert({100, 200}, resources);
    ASSERT_EQ(result.inserted_nodes.size(), 2u);
    TreeNode* const parent = result.inserted_nodes[0];
    TreeNode* const child  = result.inserted_nodes[1];

    const TransferDescriptor child_desc(child, 0, 1, Tier::HOST, Tier::DISK, {});
    const TransferDescriptor parent_desc(parent, 0, 0, Tier::HOST, Tier::DISK, {});
    evictor_->settleEviction({child_desc, parent_desc});

    EXPECT_TRUE(tree_->findNode({100}).empty());
    EXPECT_TRUE(tree_->findNode({100, 200}).empty());
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
    auto task = activateTransferForTest(*evictor_, *victim);
    ASSERT_TRUE(task.has_value());
    ASSERT_EQ(task->descs.front().target_blocks.size(), 1u);
    EXPECT_EQ(task->timings.front().tier_enter_time_us, candidate_meta.tier_enter_time_us);
    EXPECT_EQ(task->timings.front().insert_time_us, candidate_meta.insert_time_us);
    EXPECT_EQ(task->timings.front().last_access_time_us, candidate_meta.last_access_time_us);
    EXPECT_GE(task->timings.front().selected_time_us, candidate_meta.last_access_time_us);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);
    EXPECT_EQ(evictor_->candidateStats().host_candidates, 0u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);

    evictor_->rollbackTransferLocked(task->descs);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(resource.device_blocks, (std::vector<BlockIdxType>{source_block}));
    EXPECT_FALSE(resource.hasTier(Tier::HOST));
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u);

    victim = evictor_->chooseVictim(/*group_set_id=*/0, Tier::DEVICE);
    ASSERT_TRUE(victim.has_value());
    task = activateTransferForTest(*evictor_, *victim);
    ASSERT_TRUE(task.has_value());
    const BlockIdxType target_block = task->descs.front().target_blocks[0];
    EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 1u);
    evictor_->completeEvict(task->descs);
    evictor_->settleEviction(task->descs);

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

    auto task = activateTransferForTest(*evictor_, *victim);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(evictor_->candidateStats().device_candidates, 0u);

    evictor_->rollbackTransferLocked(task->descs);
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

    EXPECT_FALSE(evictor_->batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/1));
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

TEST(BlockTreeEvictorCascadeTest, PrepareDemotionReservesOnlyPrimary) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    EXPECT_EQ(task->descs.front().group_set_id, 0);
    EXPECT_EQ(environment.node_->group_set_resources[0].transfer_state, GroupSetTransferState::DEMOTING);
    EXPECT_EQ(environment.node_->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.node_->group_set_resources[2].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(environment.disk_pools_[0]->freeBlocksNum(), 1u);
    EXPECT_EQ(environment.disk_pools_[1]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.disk_pools_[2]->freeBlocksNum(), 2u);
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 2u);

    environment.evictor_->rollbackTransferLocked(task->descs);
    for (const GroupSetResource& resource : environment.node_->group_set_resources) {
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrimaryFailureRestoresPrimaryAndLeavesSiblingsResident) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferResults({false});

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    environment.evictor_->updatePendingRelease(task->descs, true);
    const bool success = environment.runTransfer(*task);
    environment.evictor_->updatePendingRelease(task->descs, false);

    EXPECT_FALSE(success);
    EXPECT_EQ(environment.transferGroupSetIds(), (std::vector<size_t>{0}));

    environment.evictor_->rollbackTransferLocked(task->descs);
    for (size_t group_set_id = 0; group_set_id < environment.groups_.size(); ++group_set_id) {
        const GroupSetResource& resource = environment.node_->group_set_resources[group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.host_block, environment.host_blocks_[group_set_id]);
        EXPECT_EQ(environment.host_pools_[group_set_id]->treeRefCount(environment.host_blocks_[group_set_id]), 1u);
        EXPECT_EQ(environment.disk_pools_[group_set_id]->freeBlocksNum(), 2u);
    }
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 3u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
}

TEST(BlockTreeEvictorCascadeTest, PrimarySuccessPublishesOnlyPrimary) {
    CascadeTestEnvironment environment;
    ASSERT_TRUE(environment.init());
    environment.setTransferResults({true});

    auto task = environment.prepareTask(0);
    ASSERT_TRUE(task.has_value());
    environment.evictor_->updatePendingRelease(task->descs, true);
    const BlockIdxType primary_target = task->descs.front().target_blocks[0];

    const bool success = environment.runTransfer(*task);
    environment.evictor_->updatePendingRelease(task->descs, false);
    ASSERT_TRUE(success);
    EXPECT_EQ(environment.transferGroupSetIds(), (std::vector<size_t>{0}));
    environment.evictor_->completeEvict(task->descs);
    environment.evictor_->settleEviction(task->descs);

    const GroupSetResource& primary_resource = environment.node_->group_set_resources[0];
    EXPECT_EQ(primary_resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_FALSE(primary_resource.hasTier(Tier::HOST));
    EXPECT_EQ(primary_resource.disk_block, primary_target);
    EXPECT_EQ(environment.disk_pools_[0]->treeRefCount(primary_target), 1u);
    EXPECT_FALSE(environment.host_pools_[0]->isAllocated(environment.host_blocks_[0]));

    for (size_t group_set_id : {1u, 2u}) {
        const GroupSetResource& resource = environment.node_->group_set_resources[group_set_id];
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
        EXPECT_EQ(resource.host_block, environment.host_blocks_[group_set_id]);
        EXPECT_FALSE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(environment.disk_pools_[group_set_id]->freeBlocksNum(), 2u);
    }
    EXPECT_EQ(environment.evictor_->candidateStats().host_candidates, 2u);

    environment.releaseResidentBlocks();
    environment.expectAllPoolsFree();
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
    insertedNode(first)->group_set_resources[1].disk_block  = NULL_BLOCK_IDX;
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
    EXPECT_FALSE(evictor.batchEvictLocked(/*group_set_id=*/0, Tier::DEVICE, /*max_victim_count=*/1));
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

    TreeNode* filled_node = mixed.adopted_nodes.front().node;
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
