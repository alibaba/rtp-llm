#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

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

size_t metricSeriesCount(kmonitor::MutableMetric* metric) {
    if (metric == nullptr) {
        ADD_FAILURE() << "metric is null";
        return 0;
    }
    return static_cast<size_t>(metric->metric_data_->Size());
}

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
                            size_t                                 logical_layer_bytes = 1,
                            size_t                                 group_set_id        = 0) {
    RTP_LLM_CHECK(group_set != nullptr && !device_pools.empty());
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

    std::vector<GroupBase> groups;
    std::vector<size_t>    group_ids;
    groups.reserve(device_pools.size());
    group_ids.reserve(device_pools.size());
    for (size_t group_id = 0; group_id < device_pools.size(); ++group_id) {
        groups.push_back(block_transfer_engine_test::makeTestGroupBase(
            policy, {0}, logical_layer_bytes, 0, 128, seq_size_per_block));
        group_ids.push_back(group_id);
    }
    group_set->initialize(
        group_set_id, block_transfer_engine_test::makeTestTopology(std::move(groups)), std::move(group_ids));
}

void initializeSingleMemberGroupSets(const std::vector<GroupSetPtr>&        group_sets,
                                     const std::vector<DeviceBlockPoolPtr>& device_pools,
                                     size_t                                 logical_layer_bytes = 1) {
    RTP_LLM_CHECK(!group_sets.empty() && group_sets.size() == device_pools.size());
    std::vector<GroupBase> groups;
    groups.reserve(group_sets.size());
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
        groups.push_back(block_transfer_engine_test::makeTestGroupBase(
            policy, {static_cast<int>(group_set_id)}, logical_layer_bytes, 0, 128, seq_size_per_block));
    }
    auto topology = block_transfer_engine_test::makeTestTopology(std::move(groups));
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        group_sets[group_set_id]->initialize(group_set_id, topology, {group_set_id});
    }
}

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

class BlockTreeCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto full_group = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr> groups = {full_group};

        cache_ = makeBlockTreeCacheForTest(std::move(groups));
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

TEST_F(BlockTreeCacheTest, MatchEmptyThenFullAndPartialPath) {
    BlockTreeMatchResult empty_result = cache_->match({100, 200, 300});
    EXPECT_EQ(empty_result.matched_device_blocks, 0u);
    EXPECT_TRUE(empty_result.matched_device_resources.empty());

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    resources[1][0].device_blocks = {43};
    resources[2][0].device_blocks = {44};
    cache_->insert({100, 200, 300}, resources, Tier::DEVICE);

    BlockTreeMatchResult full_result = cache_->match({100, 200, 300});
    EXPECT_EQ(full_result.matched_device_blocks, 3u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, full_result.matched_device_resources), (BlockIndicesType{42, 43, 44}));
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, full_result.matched_device_resources);

    BlockTreeMatchResult partial_result = cache_->match({100, 200, 999});
    EXPECT_EQ(partial_result.matched_device_blocks, 2u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, partial_result.matched_device_resources), (BlockIndicesType{42, 43}));
    block_tree_cache_test::releaseRequestRefsForTest(*cache_, partial_result.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, CollectReuseTimeMetricsAggregatesPerTierAndGroupType) {
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    resources[1][0].device_blocks = {43};
    cache_->insert({100, 200}, resources, Tier::DEVICE);

    const std::vector<TreeNode*> path = cache_->tree()->findNode({100, 200});
    ASSERT_EQ(path.size(), 2u);
    CandidateMeta& first_meta = path[0]->group_set_resources[0].candidate_meta;
    EXPECT_GT(first_meta.insert_time_us, 0);
    EXPECT_EQ(first_meta.last_access_time_us, first_meta.insert_time_us);
    EXPECT_EQ(first_meta.tier_enter_time_us, first_meta.insert_time_us);

    BlockTreeCacheMetricsReporter                             reporter;
    const std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> snapshots =
        reporter.collectCacheReuseTimeMetrics({{Tier::DEVICE, CacheGroupType::FULL, 1000, 3000, 11000},
                                               {Tier::DEVICE, CacheGroupType::FULL, 2000, 5000, 11000}});
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_EQ(snapshots[0].tier, Tier::DEVICE);
    EXPECT_EQ(snapshots[0].group_type, CacheGroupType::FULL);
    EXPECT_EQ(snapshots[0].reuse_interval_avg_ms, 7);
    EXPECT_EQ(snapshots[0].reuse_interval_max_ms, 8);
    EXPECT_EQ(snapshots[0].hit_entry_age_avg_ms, 9);
    EXPECT_EQ(snapshots[0].hit_entry_age_max_ms, 10);
}

TEST_F(BlockTreeCacheTest, AccumulateTransferBytesAggregatesDescriptors) {
    const std::vector<GroupSetPtr>&       group_sets  = cache_->groupSets();
    const DeviceBlockPoolPtr&             device_pool = group_sets[0]->devicePools()[0];
    const std::vector<TransferDescriptor> descs       = {
        TransferDescriptor::deviceToHost(0, {1}, 10),
        TransferDescriptor::deviceToHost(0, {2}, 11),
    };
    BlockTreeTransferBytes transfer_bytes;

    BlockTreeCacheMetricsReporter reporter;
    reporter.accumulateTransferBytes(descs, group_sets, transfer_bytes);

    ASSERT_EQ(transfer_bytes.size(), 1u);
    const auto bytes_it = transfer_bytes.find({device_pool->poolName(), CacheGroupType::FULL});
    ASSERT_NE(bytes_it, transfer_bytes.end());
    EXPECT_EQ(bytes_it->second, 2 * device_pool->blockSizeBytes());
}

TEST_F(BlockTreeCacheTest, ReportTransferFinishedAcceptsSuccessfulDescriptors) {
    const std::vector<GroupSetPtr>&       group_sets = cache_->groupSets();
    const std::vector<TransferDescriptor> descs      = {
        TransferDescriptor::deviceToHost(0, {1}, 10),
        TransferDescriptor::deviceToHost(0, {2}, 11),
    };
    kmonitor::MetricsTags         tags;
    BlockTreeCacheMetricsReporter reporter;
    reporter.setMetricsReporter(std::make_shared<kmonitor::MetricsReporter>("", "", tags));

    const int64_t begin_time_us =
        reporter.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, Tier::HOST);
    reporter.reportTransferFinished(
        CacheTransferOperation::STORE, Tier::DEVICE, Tier::HOST, descs.size(), begin_time_us, true, descs, group_sets);

    const size_t operation_index = static_cast<size_t>(CacheTransferOperation::STORE);
    const size_t direction_index =
        static_cast<size_t>(BlockTreeCacheMetricsReporter::transferDirectionIndex(Tier::DEVICE, Tier::HOST));
    EXPECT_EQ(reporter.transfer_in_flight_[operation_index][direction_index].load(), 0);
}

TEST(BlockTreeCacheMetricsTest, FailedQpsMetricsPublishZeroForSuccessfulOperations) {
    kmonitor::MetricsTags                      tags;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter =
        std::make_shared<kmonitor::MetricsReporter>("", "", tags);

    RtpLLMCacheOperationMetricsCollector malloc_collector;
    malloc_collector.operation_type = RtpLLMCacheOperationMetricsCollector::OpType::MALLOC;
    malloc_collector.success        = true;
    ASSERT_TRUE((metrics_reporter->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(
        nullptr, &malloc_collector)));
    RtpLLMCacheOperationMetrics* operation_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheOperationMetrics>();
    ASSERT_NE(operation_metrics, nullptr);
    EXPECT_EQ(metricSeriesCount(operation_metrics->malloc_failed_qps_metric), 1u);
    EXPECT_DOUBLE_EQ(snapshotQps(operation_metrics->malloc_failed_qps_metric, tags), 0);
    malloc_collector.success = false;
    ASSERT_TRUE((metrics_reporter->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(
        nullptr, &malloc_collector)));
    EXPECT_DOUBLE_EQ(snapshotQps(operation_metrics->malloc_failed_qps_metric, tags), 1);

    RtpLLMCacheTransferMetricsCollector transfer_collector;
    transfer_collector.operation   = "load";
    transfer_collector.source_tier = "host";
    transfer_collector.target_tier = "device";
    transfer_collector.success     = true;
    ASSERT_TRUE((metrics_reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(
        nullptr, &transfer_collector)));
    RtpLLMCacheTransferMetrics* transfer_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheTransferMetrics>();
    ASSERT_NE(transfer_metrics, nullptr);
    kmonitor::MetricsTags transfer_tags("operation", "load");
    transfer_tags.AddTag("source_tier", "host");
    transfer_tags.AddTag("target_tier", "device");
    EXPECT_EQ(metricSeriesCount(transfer_metrics->transfer_failed_qps_metric), 1u);
    EXPECT_DOUBLE_EQ(snapshotQps(transfer_metrics->transfer_failed_qps_metric, transfer_tags), 0);
    transfer_collector.success = false;
    ASSERT_TRUE((metrics_reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(
        nullptr, &transfer_collector)));
    EXPECT_DOUBLE_EQ(snapshotQps(transfer_metrics->transfer_failed_qps_metric, transfer_tags), 1);

    RtpLLMCacheReuseMetricsCollector no_load_collector;
    ASSERT_TRUE((metrics_reporter->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(
        nullptr, &no_load_collector)));
    RtpLLMCacheReuseMetrics* reuse_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheReuseMetrics>();
    ASSERT_NE(reuse_metrics, nullptr);
    EXPECT_EQ(metricSeriesCount(reuse_metrics->load_qps_metric), 0u);
    EXPECT_EQ(metricSeriesCount(reuse_metrics->load_fail_qps_metric), 0u);

    RtpLLMCacheReuseMetricsCollector load_collector;
    load_collector.report_load_metrics = true;
    load_collector.load_success        = true;
    ASSERT_TRUE((
        metrics_reporter->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &load_collector)));
    EXPECT_EQ(metricSeriesCount(reuse_metrics->load_qps_metric), 1u);
    EXPECT_EQ(metricSeriesCount(reuse_metrics->load_fail_qps_metric), 1u);
    EXPECT_DOUBLE_EQ(snapshotQps(reuse_metrics->load_qps_metric, tags), 1);
    EXPECT_DOUBLE_EQ(snapshotQps(reuse_metrics->load_fail_qps_metric, tags), 0);
    load_collector.load_success = false;
    ASSERT_TRUE((
        metrics_reporter->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &load_collector)));
    EXPECT_DOUBLE_EQ(snapshotQps(reuse_metrics->load_qps_metric, tags), 1);
    EXPECT_DOUBLE_EQ(snapshotQps(reuse_metrics->load_fail_qps_metric, tags), 1);
}

TEST_F(BlockTreeCacheTest, EvictionTriggerQpsPublishesOnlyExistingGroupTypes) {
    kmonitor::MetricsTags                      tags;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter =
        std::make_shared<kmonitor::MetricsReporter>("", "", tags);
    BlockTreeCacheMetricsReporter reporter;
    reporter.setMetricsReporter(metrics_reporter);

    const std::vector<BlockTreeEvictableMetricsSnapshot> snapshots =
        reporter.collectEvictableMetricsSnapshots(cache_->groupSets(), cache_->evictor_);
    ASSERT_EQ(snapshots.size(), 3u);
    for (const BlockTreeEvictableMetricsSnapshot& snapshot : snapshots) {
        EXPECT_EQ(snapshot.group_type, CacheGroupType::FULL);
    }
    reporter.reportEvictableCandidateCount(snapshots);
    RtpLLMCacheEvictionMetrics* eviction_metrics = metrics_reporter->getMetricsGroup<RtpLLMCacheEvictionMetrics>();
    ASSERT_NE(eviction_metrics, nullptr);
    EXPECT_EQ(metricSeriesCount(eviction_metrics->evictable_candidate_count_metric), 3u);
    EXPECT_EQ(metricSeriesCount(eviction_metrics->eviction_trigger_qps_metric), 6u);

    kmonitor::MetricsTags watermark_tags("trigger_type", "watermark");
    watermark_tags.AddTag("source_tier", tierName(Tier::DEVICE));
    watermark_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_trigger_qps_metric, watermark_tags), 0);

    kmonitor::MetricsTags force_drop_tags("trigger_type", "force_drop");
    force_drop_tags.AddTag("source_tier", tierName(Tier::DEVICE));
    force_drop_tags.AddTag("group_type", metricCacheGroupTypeName(CacheGroupType::FULL));
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_trigger_qps_metric, force_drop_tags), 0);

    reporter.reportEvictionTriggered(Tier::DEVICE, CacheGroupType::FULL, false);
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_trigger_qps_metric, watermark_tags), 1);
    reporter.reportEvictionTriggered(Tier::DEVICE, CacheGroupType::FULL, true);
    EXPECT_DOUBLE_EQ(snapshotQps(eviction_metrics->eviction_trigger_qps_metric, force_drop_tags), 1);
}

TEST_F(BlockTreeCacheTest, KeySnapshotTracksMutationVersionAndLimit) {
    const auto empty = cache_->getKeySnapshot(/*limit=*/10);
    EXPECT_EQ(empty.version, 0u);
    EXPECT_TRUE(empty.keys.empty());

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    resources[1][0].device_blocks = {43};
    resources[2][0].device_blocks = {44};
    cache_->insert({100, 200, 300}, resources, Tier::DEVICE);

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
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[2][0].device_blocks = {12};

    ASSERT_TRUE(insertGroupSetResources(*cache_, {100, 200, 300}, resources));

    BlockTreeMatchResult result = cache_->match({100, 200, 300});
    EXPECT_EQ(result.matched_device_blocks, 1u);
    EXPECT_EQ(cache_->matchedBlocksForGroup(0, result.matched_device_resources), (BlockIndicesType{10}));

    block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, MatchFailsFastAtIdleResourceWithMultipleServingTiers) {
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    cache_->insert({100, 200}, resources, Tier::DEVICE);

    TreeNode* first_node                          = cache_->tree()->root()->children.at(100);
    first_node->group_set_resources[0].host_block = 7;

    EXPECT_THROW(cache_->match({100, 200}), std::runtime_error);

    first_node->group_set_resources[0].host_block = NULL_BLOCK_IDX;
}

TEST_F(BlockTreeCacheTest, MatchDoesNotReuseBusyFullResource) {
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    cache_->insert({100, 200}, resources, Tier::DEVICE);

    TreeNode* first_node                              = cache_->tree()->root()->children.at(100);
    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    const BlockTreeMatchResult result = cache_->match({100, 200});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());

    first_node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
}

TEST_F(BlockTreeCacheTest, MatchSkipsBusySwaResourceWithoutTruncatingFullPrefix) {
    // FULL + SWA(window=2 blocks): a busy SWA resource on a middle node is outside
    // the trailing window and must not truncate the FULL prefix match.
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto full_group = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        auto swa_group = std::make_shared<SWAGroupSet>(
            2,
            1,
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            nullptr,
            nullptr);
        std::vector<GroupSetPtr>        groups      = {full_group, swa_group};
        std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(groups));
        ASSERT_NE(multi_cache, nullptr);

        std::vector<std::vector<GroupSetResource>> resources(4, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < 4; ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            resources[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        multi_cache->insert({100, 200, 300, 400}, resources, Tier::DEVICE);

        TreeNode* busy_node = multi_cache->tree()->root()->children.at(100)->children.at(200);
        busy_node->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result = multi_cache->match({100, 200, 300, 400});
        EXPECT_EQ(result.matched_device_blocks, 4u);
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_device_resources),
                  (BlockIndicesType{10, 11, 12, 13}));
        // SWA locks only the trailing window; the busy middle resource stays untouched.
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(1, result.matched_device_resources), (BlockIndicesType{22, 23}));
        const auto& swa_pool = multi_cache->groupSets()[1]->devicePools()[0];
        EXPECT_EQ(swa_pool->refCount(21), 1u);  // cache hold only, no match reference
        EXPECT_EQ(swa_pool->refCount(22), 2u);

        block_tree_cache_test::releaseRequestRefsForTest(*multi_cache, result.matched_device_resources);
        busy_node->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    }
}

TEST_F(BlockTreeCacheTest, MatchStillTruncatesAtBusyFullResource) {
    // The FULL prefix latch must keep truncating when the FULL resource itself is busy.
    auto full_group = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    auto swa_group = std::make_shared<SWAGroupSet>(
        2, 1, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr>        groups      = {full_group, swa_group};
    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(groups));
    ASSERT_NE(multi_cache, nullptr);

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(2));
    for (size_t i = 0; i < 3; ++i) {
        resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        resources[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
    }
    multi_cache->insert({100, 200, 300}, resources, Tier::DEVICE);

    TreeNode* busy_node                              = multi_cache->tree()->root()->children.at(100)->children.at(200);
    busy_node->group_set_resources[0].transfer_state = GroupSetTransferState::DEMOTING;

    BlockTreeMatchResult result = multi_cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_device_blocks, 1u);
    EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_device_resources), (BlockIndicesType{10}));

    block_tree_cache_test::releaseRequestRefsForTest(*multi_cache, result.matched_device_resources);
    busy_node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
}

TEST_F(BlockTreeCacheTest, MatchSkipsBusyLinearResourceAndReusesTailState) {
    // LINEAR only consumes the deepest node's state; a busy middle resource must
    // not truncate the match nor be referenced.
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto full_group = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        auto linear_group = std::make_shared<LinearGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        std::vector<GroupSetPtr>        groups      = {full_group, linear_group};
        std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(groups));
        ASSERT_NE(multi_cache, nullptr);

        std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < 3; ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            resources[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        }
        multi_cache->insert({100, 200, 300}, resources, Tier::DEVICE);

        TreeNode* busy_node = multi_cache->tree()->root()->children.at(100)->children.at(200);
        busy_node->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result = multi_cache->match({100, 200, 300});
        EXPECT_EQ(result.matched_device_blocks, 3u);
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(0, result.matched_device_resources),
                  (BlockIndicesType{10, 11, 12}));
        EXPECT_EQ(multi_cache->matchedBlocksForGroup(1, result.matched_device_resources), (BlockIndicesType{22}));
        const auto& linear_pool = multi_cache->groupSets()[1]->devicePools()[0];
        EXPECT_EQ(linear_pool->refCount(21), 1u);  // busy middle resource not referenced
        EXPECT_EQ(linear_pool->refCount(22), 2u);

        block_tree_cache_test::releaseRequestRefsForTest(*multi_cache, result.matched_device_resources);
        busy_node->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
    }
}

TEST_F(BlockTreeCacheTest, InsertFailsFastForNonIdleOrMultiTierResource) {
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks  = {10};
    resources[0][0].transfer_state = GroupSetTransferState::DEMOTING;
    EXPECT_THROW(cache_->insert({100}, resources, Tier::DEVICE), std::runtime_error);
    EXPECT_EQ(cache_->tree()->size(), 0u);

    resources[0][0].transfer_state = GroupSetTransferState::IDLE;
    resources[0][0].host_block     = 7;
    EXPECT_THROW(cache_->insert({100}, resources, Tier::DEVICE), std::runtime_error);
    EXPECT_EQ(cache_->tree()->size(), 0u);
}

TEST_F(BlockTreeCacheTest, DuplicateInsertDoesNotCreateNodes) {
    CacheStats stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);

    std::vector<std::vector<GroupSetResource>> original_resources(2, std::vector<GroupSetResource>(1));
    original_resources[0][0].device_blocks = {10};
    original_resources[1][0].device_blocks = {11};
    cache_->insert({100, 200}, original_resources, Tier::DEVICE);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    std::vector<std::vector<GroupSetResource>> duplicate_resources(2, std::vector<GroupSetResource>(1));
    duplicate_resources[0][0].device_blocks = {20};
    duplicate_resources[1][0].device_blocks = {21};
    cache_->insert({100, 200}, duplicate_resources, Tier::DEVICE);

    stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 2u);
    EXPECT_EQ(stats.device_heap_total_size, 1u);

    auto find_result = cache_->tree()->findNode({100, 200});
    ASSERT_EQ(find_result.size(), 2u);
    EXPECT_EQ(find_result[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
    EXPECT_EQ(find_result[1]->group_set_resources[0].device_blocks, (BlockIndicesType{11}));
}

TEST_F(BlockTreeCacheTest, ReclaimCascadesToLowerPriorityGroup) {
    // Build a cache with Full + SWA groups

    auto full_group = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    auto swa_group = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::vector<GroupSetPtr> groups = {full_group, swa_group};

    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(groups));

    // Insert a node with both Full and SWA data
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};  // Full
    resources[0][1].device_blocks = {20};  // SWA

    multi_cache->insert({100}, resources, Tier::DEVICE);

    // Reclaim Full group at DEVICE → should cascade to SWA.
    int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*multi_cache, 1, Tier::DEVICE);
    EXPECT_EQ(reclaimed, 1);

    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*multi_cache);
}

TEST_F(BlockTreeCacheTest, MultiGroupConstruction) {

    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    auto swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::vector<GroupSetPtr> groups = {full, swa, linear};

    std::unique_ptr<BlockTreeCache> multi_cache = makeBlockTreeCacheForTest(std::move(groups));

    EXPECT_EQ(multi_cache->groupSets().size(), 3u);
    EXPECT_EQ(multi_cache->tree()->groupSets().size(), 3);
}

TEST_F(BlockTreeCacheTest, EmptyKeysAreNoOps) {
    const CacheStats stats_before = cache_->getStats();
    cache_->insert({}, {}, Tier::DEVICE);
    const CacheStats stats_after = cache_->getStats();
    EXPECT_EQ(stats_after.tree_node_count, stats_before.tree_node_count);
    EXPECT_EQ(stats_after.device_heap_total_size, stats_before.device_heap_total_size);

    BlockTreeMatchResult result = cache_->match({});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    EXPECT_EQ(result.async_context, nullptr);
}

TEST_F(BlockTreeCacheTest, ThreadSafety) {
    // Basic thread safety test: concurrent inserts
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, i]() {
            std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
            resources[0][0].device_blocks = {static_cast<BlockIdxType>(i * 100 + 1)};
            CacheKeysType keys            = {static_cast<CacheKeyType>(i * 1000 + 1)};
            cache_->insert(keys, resources, Tier::DEVICE);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto stats = cache_->getStats();
    EXPECT_EQ(stats.tree_node_count, 4u);
}

TEST_F(BlockTreeCacheTest, ConcurrentDoubleMatch_EvictsBeforeLastRelease) {
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    cache_->insert({100}, resources, Tier::DEVICE);
    ASSERT_EQ(cache_->getStats().device_heap_total_size, 1u);

    std::mutex               mutex;
    std::condition_variable  cv;
    bool                     start{false};
    size_t                   matched_count{0};
    size_t                   released_count{0};
    std::array<bool, 2>      release_match{false, false};
    std::array<size_t, 2>    matched_blocks{0, 0};
    std::array<bool, 2>      node_independent{false, false};
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
                matched_blocks[thread_id]   = result.matched_device_blocks;
                node_independent[thread_id] = std::all_of(
                    result.matched_device_resources.begin(),
                    result.matched_device_resources.end(),
                    [](const MultiNodeResource& resource) {
                        return std::all_of(resource.node_blocks.begin(),
                                           resource.node_blocks.end(),
                                           [](const auto& node_blocks) { return node_blocks.first == nullptr; });
                    });
                ++matched_count;
                cv.notify_all();
                cv.wait(lock, [&] { return release_match[thread_id]; });
            }
            block_tree_cache_test::releaseRequestRefsForTest(*cache_, result.matched_device_resources);
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
    EXPECT_EQ(node_independent, (std::array<bool, 2>{true, true}));

    const DeviceBlockPoolPtr& pool = cache_->groupSets()[0]->devicePools()[0];
    EXPECT_EQ(pool->refCount(42), 3u);
    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
    EXPECT_TRUE(pool->isAllocated(42));
    EXPECT_EQ(pool->refCount(42), 2u);
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
    EXPECT_TRUE(pool->isAllocated(42));
    EXPECT_EQ(pool->refCount(42), 1u);

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_match[1] = true;
        cv.notify_all();
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(released_count, 2u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
    EXPECT_FALSE(pool->isAllocated(42));
}

TEST_F(BlockTreeCacheTest, ExtraReferencesDoNotChangeCandidateMembership) {
    const GroupSetPtr&        group_set = cache_->groupSets()[0];
    const DeviceBlockPoolPtr& pool      = group_set->devicePools()[0];
    pool->incRef(42);
    pool->incRef(43);

    std::vector<std::vector<GroupSetResource>> first_resources(1, std::vector<GroupSetResource>(1));
    first_resources[0][0].device_blocks = {42};
    cache_->insert({100}, first_resources, Tier::DEVICE);
    std::vector<std::vector<GroupSetResource>> second_resources(1, std::vector<GroupSetResource>(1));
    second_resources[0][0].device_blocks = {43};
    cache_->insert({200}, second_resources, Tier::DEVICE);
    ASSERT_EQ(cache_->getStats().device_heap_total_size, 2u);

    releaseDeviceBlocks(*cache_, pool, {42});
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 2u);
    EXPECT_EQ(pool->refCount(43), 2u);

    releaseDeviceBlocks(*cache_, pool, {43});
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 2u);
}

const BlockTreePoolMetricsSnapshot* findTreePoolSnapshot(const std::vector<BlockTreePoolMetricsSnapshot>& snapshots,
                                                         const std::string&                               pool_name,
                                                         const std::string&                               stage) {
    const BlockTreePoolMetricsSnapshot* found = nullptr;
    for (const BlockTreePoolMetricsSnapshot& snapshot : snapshots) {
        if (snapshot.pool_name != pool_name) {
            continue;
        }
        if (found != nullptr) {
            ADD_FAILURE() << "duplicate pool snapshot: " << pool_name << " stage=" << stage;
            return nullptr;
        }
        found = &snapshot;
    }
    if (found == nullptr) {
        ADD_FAILURE() << "missing pool snapshot: " << pool_name << " stage=" << stage;
    }
    return found;
}

// pool_name and block_size_bytes are intentionally excluded: member pools keep separate
// identities, and a GroupSet does not require its members to own the same layer count.
void expectSameJointLifecycleMetrics(const BlockTreePoolMetricsSnapshot& expected,
                                     const BlockTreePoolMetricsSnapshot& actual,
                                     const std::string&                  stage) {
    const std::string context = "pool=" + actual.pool_name + " stage=" + stage;
    EXPECT_EQ(actual.tier, expected.tier) << context;
    EXPECT_EQ(actual.total_blocks, expected.total_blocks) << context;
    EXPECT_EQ(actual.free_blocks, expected.free_blocks) << context;
    EXPECT_EQ(actual.used_blocks, expected.used_blocks) << context;
    EXPECT_EQ(actual.available_blocks, expected.available_blocks) << context;
    EXPECT_EQ(actual.active_tree_cached_blocks, expected.active_tree_cached_blocks) << context;
    EXPECT_EQ(actual.request_ref_blocks, expected.request_ref_blocks) << context;
    EXPECT_EQ(actual.block_cache_ref_blocks, expected.block_cache_ref_blocks) << context;
    EXPECT_EQ(actual.load_ref_blocks, expected.load_ref_blocks) << context;
    EXPECT_EQ(actual.eviction_ref_blocks, expected.eviction_ref_blocks) << context;
    EXPECT_EQ(actual.store_ref_blocks, expected.store_ref_blocks) << context;
}

TEST_F(BlockTreeCacheTest, MultiMemberPoolMetricsStayAlignedThroughJointEviction) {
    const std::vector<DeviceBlockPoolPtr> pools = makeStructuralDevicePools(3, "multi_member_metrics_pool");
    ASSERT_EQ(pools.size(), 3u);
    auto full = std::make_shared<FullGroupSet>(pools, nullptr, nullptr);
    initializeTestGroupSet(full, pools);
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));
    ASSERT_NE(cache, nullptr);

    for (size_t left = 0; left < pools.size(); ++left) {
        for (size_t right = left + 1; right < pools.size(); ++right) {
            EXPECT_NE(pools[left]->poolName(), pools[right]->poolName());
        }
    }

    const std::vector<BlockTreePoolMetricsSnapshot> initial_snapshots = cache->poolMetricsSnapshots();
    ASSERT_EQ(initial_snapshots.size(), pools.size());
    const size_t total_blocks = initial_snapshots.front().total_blocks;
    ASSERT_GT(total_blocks, 0u);
    for (const BlockTreePoolMetricsSnapshot& snapshot : initial_snapshots) {
        ASSERT_EQ(snapshot.tier, Tier::DEVICE);
        ASSERT_EQ(snapshot.total_blocks, total_blocks);
        ASSERT_EQ(snapshot.free_blocks, total_blocks);
    }

    struct JointStage {
        size_t used_blocks;
        size_t available_blocks;
        size_t request_ref_blocks;
        size_t block_cache_ref_blocks;
        size_t active_tree_cached_blocks;
    };
    auto expect_joint_stage = [&](const std::string& stage, const JointStage& expected) {
        const std::vector<BlockTreePoolMetricsSnapshot> snapshots = cache->poolMetricsSnapshots();
        ASSERT_EQ(snapshots.size(), pools.size()) << stage;
        const BlockTreePoolMetricsSnapshot* first = nullptr;
        for (const DeviceBlockPoolPtr& pool : pools) {
            const BlockTreePoolMetricsSnapshot* snapshot = findTreePoolSnapshot(snapshots, pool->poolName(), stage);
            ASSERT_NE(snapshot, nullptr) << pool->poolName() << " stage=" << stage;
            const std::string context = "pool=" + pool->poolName() + " stage=" + stage;
            EXPECT_EQ(snapshot->tier, Tier::DEVICE) << context;
            EXPECT_EQ(snapshot->total_blocks, total_blocks) << context;
            EXPECT_EQ(snapshot->used_blocks, expected.used_blocks) << context;
            EXPECT_EQ(snapshot->free_blocks, total_blocks - expected.used_blocks) << context;
            EXPECT_EQ(snapshot->available_blocks, expected.available_blocks) << context;
            EXPECT_EQ(snapshot->request_ref_blocks, expected.request_ref_blocks) << context;
            EXPECT_EQ(snapshot->block_cache_ref_blocks, expected.block_cache_ref_blocks) << context;
            EXPECT_EQ(snapshot->active_tree_cached_blocks, expected.active_tree_cached_blocks) << context;
            EXPECT_EQ(snapshot->load_ref_blocks, 0u) << context;
            EXPECT_EQ(snapshot->eviction_ref_blocks, 0u) << context;
            EXPECT_EQ(snapshot->store_ref_blocks, 0u) << context;
            if (first == nullptr) {
                first = snapshot;
            } else {
                expectSameJointLifecycleMetrics(*first, *snapshot, stage);
            }
        }
    };

    std::vector<BlockIdxType> device_blocks;
    for (const DeviceBlockPoolPtr& pool : pools) {
        const BlockIdxType block = pool->malloc().value();
        pool->incRef(block);
        device_blocks.push_back(block);
    }

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = device_blocks;
    cache->insert({100}, resources, Tier::DEVICE);

    // Stage A: insertion admits one GroupSet-level candidate even while the request still holds every member block.
    ASSERT_EQ(cache->getStats().tree_node_count, 1u);
    ASSERT_EQ(cache->getStats().device_heap_total_size, 1u);
    expect_joint_stage("tree_and_request_held",
                       JointStage{/*used_blocks=*/1,
                                  /*available_blocks=*/total_blocks,
                                  /*request_ref_blocks=*/1,
                                  /*block_cache_ref_blocks=*/1,
                                  /*active_tree_cached_blocks=*/1});

    for (size_t member_group_id = 0; member_group_id < pools.size(); ++member_group_id) {
        pools[member_group_id]->decRef(device_blocks[member_group_id]);
    }

    // Stage B: releasing external references does not change candidate membership.
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    expect_joint_stage("single_joint_candidate",
                       JointStage{/*used_blocks=*/1,
                                  /*available_blocks=*/total_blocks,
                                  /*request_ref_blocks=*/0,
                                  /*block_cache_ref_blocks=*/1,
                                  /*active_tree_cached_blocks=*/0});

    // Stage C: host/disk tiers are disabled and evictForGroup force-drops, so the joint eviction
    // completes synchronously without any transfer task.
    EXPECT_EQ(cache->evictForGroup(full->groupIds().front(), 1), 1);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 0u);
    expect_joint_stage("after_joint_eviction",
                       JointStage{/*used_blocks=*/0,
                                  /*available_blocks=*/total_blocks,
                                  /*request_ref_blocks=*/0,
                                  /*block_cache_ref_blocks=*/0,
                                  /*active_tree_cached_blocks=*/0});
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
                std::vector<std::vector<GroupSetResource>> same_resources(2, std::vector<GroupSetResource>(1));
                same_resources[0][0].device_blocks = {10};
                same_resources[1][0].device_blocks = {11};
                cache_->insert({100, 200}, same_resources, Tier::DEVICE);

                std::vector<std::vector<GroupSetResource>> fork_resources(2, std::vector<GroupSetResource>(1));
                fork_resources[0][0].device_blocks = {10};
                fork_resources[1][0].device_blocks = {fork_block};
                cache_->insert({100, fork_key}, fork_resources, Tier::DEVICE);

                for (const CacheKeysType& keys : {CacheKeysType{100, 200}, CacheKeysType{100, fork_key}}) {
                    BlockTreeMatchResult match  = cache_->match(keys);
                    const auto           blocks = cache_->matchedBlocksForGroup(0, match.matched_device_resources);
                    if (match.matched_device_blocks != 2 || blocks.size() != 2 || blocks[0] != 10) {
                        consistent.store(false);
                    }
                    block_tree_cache_test::releaseRequestRefsForTest(*cache_, match.matched_device_resources);
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
        ASSERT_EQ(found.size(), 2u);
        EXPECT_EQ(found[0]->group_set_resources[0].device_blocks, (BlockIndicesType{10}));
        EXPECT_EQ(found[1]->group_set_resources[0].device_blocks, (BlockIndicesType{fork_block}));
        EXPECT_EQ(pool->refCount(fork_block), 1u);
    }

    // Final reclaim: drain leaves first, then the promoted shared parent, until
    // the tree is empty and every cache hold is released back to the pool.
    for (size_t attempt = 0; attempt < (kThreadCount + 2) * 2; ++attempt) {
        if (BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache_, 1, Tier::DEVICE) == 0) {
            break;
        }
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache_);
    }
    EXPECT_EQ(cache_->getStats().tree_node_count, 0u);
    EXPECT_EQ(cache_->getStats().device_heap_total_size, 0u);
    EXPECT_FALSE(pool->isAllocated(10));
    EXPECT_FALSE(pool->isAllocated(11));
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        EXPECT_FALSE(pool->isAllocated(static_cast<BlockIdxType>(20 + thread_id)));
    }
}

TEST(BlockTreeCacheFinalizationTest, CopyExceptionSettlesPendingReleasesBeforeTaskCompletion) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length          = 1;
    options.usable_device_blocks = 4;
    options.usable_host_blocks   = 4;
    options.enable_disk          = false;
    auto environment             = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    ASSERT_NE(environment->cache, nullptr);

    auto barrier = std::make_shared<CallbackBarrier>();
    auto per_rank_transfer_engine =
        std::make_shared<ControlledPerRankBlockTransferEngine>(environment->groups, TransferCopyAction::Throw, barrier);
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache, per_rank_transfer_engine);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));

    std::vector<BlockIdxType> source_blocks;
    std::vector<size_t>       source_free_before;
    std::vector<size_t>       source_refs_before;
    source_blocks.reserve(environment->device_pools.size());
    source_free_before.reserve(environment->device_pools.size());
    source_refs_before.reserve(environment->device_pools.size());
    for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
        const auto blocks = environment->blocksForDevicePool(pool_id);
        ASSERT_EQ(blocks.size(), 1u);
        source_blocks.push_back(blocks.front());
        source_free_before.push_back(environment->device_pools[pool_id]->freeBlocksNum());
        source_refs_before.push_back(environment->device_pools[pool_id]->refCount(blocks.front()));
        ASSERT_EQ(source_refs_before.back(), 1u);
    }

    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment->cache, Tier::DEVICE, 0.01);
    BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);
    ASSERT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    barrier->waitUntilEntered();

    EXPECT_GT(BlockTreeCacheTestPeer::pendingEvictionReleasesForTest(*environment->cache), 0u);
    const int    pending_tasks = BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache);
    const size_t submit_count  = per_rank_transfer_engine->submittedBatchCount();
    BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), pending_tasks);
    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), submit_count);
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment->cache, Tier::DEVICE, 0.0);
    barrier->release();
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*environment->cache);

    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*environment->cache), 0);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingEvictionReleasesForTest(*environment->cache), 0u);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    for (size_t pool_id = 0; pool_id < environment->device_pools.size(); ++pool_id) {
        EXPECT_EQ(environment->device_pools[pool_id]->freeBlocksNum(), source_free_before[pool_id]);
        EXPECT_EQ(environment->device_pools[pool_id]->refCount(source_blocks[pool_id]), source_refs_before[pool_id]);
    }

    EXPECT_NO_THROW(environment->cache.reset());
}

TEST(BlockTreeCacheFinalizationTest, EvictionQueueRejectionSettlesPendingReleasesAndRestoresSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length          = 1;
    options.usable_device_blocks = 4;
    options.usable_host_blocks   = 4;
    auto environment             = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    BlockTreeCacheTestPeer::setTierWatermarkForTest(*environment->cache, Tier::DEVICE, 0.01);

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*environment->cache);
    ASSERT_TRUE(rejection_guard.armed());
    BlockTreeCacheTestPeer::runMaintenanceForTest(*environment->cache);

    EXPECT_EQ(BlockTreeCacheTestPeer::pendingEvictionReleasesForTest(*environment->cache), 0u);
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    ASSERT_TRUE(rejection_guard.restore());
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

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool0, pool1}, nullptr, nullptr);
    initializeTestGroupSet(full, {pool0, pool1});
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));

    MultiNodeBlocks request_blocks = allocateDeviceBlocksForTest(*full, 2);
    ASSERT_EQ(request_blocks.size(), 2u);
    ASSERT_EQ(request_blocks[0].size(), 2u);
    ASSERT_EQ(request_blocks[1].size(), 2u);

    const BlockIdxType a_pool0 = request_blocks[0][0];
    const BlockIdxType a_pool1 = request_blocks[0][1];
    const BlockIdxType b_pool0 = request_blocks[1][0];
    const BlockIdxType b_pool1 = request_blocks[1][1];
    EXPECT_NE(a_pool0, a_pool1);
    EXPECT_NE(b_pool0, b_pool1);

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {a_pool0, a_pool1};
    resources[1][0].device_blocks = {b_pool0, b_pool1};
    cache->insert({100, 200}, resources, Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*full, request_blocks);
    EXPECT_TRUE(pool0->isAllocated(a_pool0));
    EXPECT_TRUE(pool0->isAllocated(b_pool0));
    EXPECT_TRUE(pool1->isAllocated(a_pool1));
    EXPECT_TRUE(pool1->isAllocated(b_pool1));

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_device_blocks, 2u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_device_resources), (BlockIndicesType{a_pool0, b_pool0}));
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_device_resources), (BlockIndicesType{a_pool1, b_pool1}));
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 2, Tier::DEVICE), 2);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_FALSE(pool0->isAllocated(a_pool0));
    EXPECT_FALSE(pool0->isAllocated(b_pool0));
    EXPECT_FALSE(pool1->isAllocated(a_pool1));
    EXPECT_FALSE(pool1->isAllocated(b_pool1));

    pool0->incRef(*pool0_prefix);
    pool0->decRef(*pool0_prefix);
    pool1->incRef(*pool1_prefix);
    pool1->decRef(*pool1_prefix);
    EXPECT_EQ(pool0->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(pool1->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, DuplicateInsert_KeepsExistingResourceAndCallerOwnsLoser) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "duplicate_insert_pool");

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool}, nullptr, nullptr);
    initializeTestGroupSet(full, {pool});
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));

    MultiNodeBlocks existing = allocateDeviceBlocksForTest(*full, 1);
    MultiNodeBlocks loser    = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(existing.size(), 1u);
    ASSERT_EQ(loser.size(), 1u);
    ASSERT_EQ(existing[0].size(), 1u);
    ASSERT_EQ(loser[0].size(), 1u);
    const BlockIdxType existing_block = existing[0][0];
    const BlockIdxType loser_block    = loser[0][0];
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    std::vector<std::vector<GroupSetResource>> first_resources(1, std::vector<GroupSetResource>(1));
    first_resources[0][0].device_blocks = existing[0];
    cache->insert({100}, first_resources, Tier::DEVICE);
    EXPECT_EQ(pool->refCount(existing_block), 2u);
    auto initial_find = cache->tree()->findNode({100});
    ASSERT_FALSE(initial_find.empty());
    block_tree_cache_test::releaseRequestRefsForTest(
        *cache, {makeMultiNodeResourceForTest(full->groupSetId(), Tier::DEVICE, {initial_find.back()}, existing)});
    EXPECT_EQ(pool->refCount(existing_block), 1u);

    std::vector<std::vector<GroupSetResource>> duplicate_resources(1, std::vector<GroupSetResource>(1));
    duplicate_resources[0][0].device_blocks = loser[0];
    cache->insert({100}, duplicate_resources, Tier::DEVICE);

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(find.back()->group_set_resources[0].device_blocks, (std::vector<BlockIdxType>{existing_block}));
    EXPECT_EQ(pool->refCount(existing_block), 1u);
    EXPECT_EQ(pool->refCount(loser_block), 1u);

    unreferenceDeviceBlocksForTest(*full, loser);
    EXPECT_FALSE(pool->isAllocated(loser_block));
    EXPECT_TRUE(pool->isAllocated(existing_block));

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_FALSE(pool->isAllocated(existing_block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
}

TEST_F(BlockTreeCacheTest, DuplicateInsert_FillsExistingEmptyGroupAndAddsOneCacheHold) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "existing_group_fill_pool");

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool}, nullptr, nullptr);
    initializeTestGroupSet(full, {pool});
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));

    std::vector<std::vector<GroupSetResource>> empty_resources(1, std::vector<GroupSetResource>(1));
    empty_resources[0][0].device_blocks = {NULL_BLOCK_IDX};
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, empty_resources));
    TreeNode* existing_node = cache->tree()->root()->children.at(100);
    ASSERT_NE(existing_node, nullptr);

    MultiNodeBlocks request_blocks = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(request_blocks.size(), 1u);
    ASSERT_EQ(request_blocks[0].size(), 1u);
    const BlockIdxType block = request_blocks[0][0];
    ASSERT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = request_blocks[0];
    cache->insert({100}, resources, Tier::DEVICE);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);
    EXPECT_EQ(existing_node->group_set_resources[0].device_blocks, request_blocks[0]);
    EXPECT_EQ(pool->refCount(block), 2u);

    block_tree_cache_test::releaseRequestRefsForTest(
        *cache, {makeMultiNodeResourceForTest(full->groupSetId(), Tier::DEVICE, {existing_node}, request_blocks)});
    EXPECT_EQ(pool->refCount(block), 1u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
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

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool0, pool1}, nullptr, nullptr);
    initializeTestGroupSet(full, {pool0, pool1});
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));

    MultiNodeBlocks request_blocks = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(request_blocks.size(), 1u);
    ASSERT_EQ(request_blocks[0].size(), 2u);
    const BlockIdxType block0 = request_blocks[0][0];
    const BlockIdxType block1 = request_blocks[0][1];

    std::vector<std::vector<GroupSetResource>> partial_resources(1, std::vector<GroupSetResource>(1));
    partial_resources[0][0].device_blocks = {block0, NULL_BLOCK_IDX};
    EXPECT_THROW(cache->insert({100}, partial_resources, Tier::DEVICE), std::runtime_error);
    EXPECT_EQ(cache->tree()->size(), 0u);
    EXPECT_EQ(pool0->refCount(block0), 1u);
    EXPECT_EQ(pool1->refCount(block1), 1u);

    unreferenceDeviceBlocksForTest(*full, request_blocks);
    EXPECT_FALSE(pool0->isAllocated(block0));
    EXPECT_FALSE(pool1->isAllocated(block1));
}

TEST_F(BlockTreeCacheTest, InsertMatchReclaimRelease_RefcountLifecycle) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr size_t kUsableBlocks = 4;
    auto             pool          = makeDevicePool({{64, 0}}, kUsableBlocks, "refcount_lifecycle_pool");

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{pool}, nullptr, nullptr);
    initializeTestGroupSet(full, {pool});
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));

    MultiNodeBlocks request_blocks = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(request_blocks.size(), 1u);
    ASSERT_EQ(request_blocks[0].size(), 1u);
    const BlockIdxType block = request_blocks[0][0];
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks - 1);
    EXPECT_EQ(pool->refCount(block), 1u);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = request_blocks[0];
    cache->insert({100}, resources, Tier::DEVICE);
    EXPECT_EQ(pool->refCount(block), 2u);

    unreferenceDeviceBlocksForTest(*full, request_blocks);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);

    BlockTreeMatchResult result = cache->match({100});
    EXPECT_EQ(result.matched_device_blocks, 1u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_device_resources), (BlockIndicesType{block}));
    ASSERT_EQ(result.matched_device_resources.size(), 1u);
    EXPECT_EQ(result.matched_device_resources[0].group_set_id, 0);
    EXPECT_EQ(result.matched_device_resources[0].tier, Tier::DEVICE);
    ASSERT_EQ(result.matched_device_resources[0].node_blocks.size(), 1u);
    EXPECT_EQ(result.matched_device_resources[0].node_blocks[0].second, (std::vector<BlockIdxType>{block}));
    EXPECT_EQ(pool->refCount(block), 2u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_TRUE(pool->isAllocated(block));
    EXPECT_EQ(pool->refCount(block), 1u);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);

    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    result.matched_device_resources.clear();
    EXPECT_FALSE(pool->isAllocated(block));
    EXPECT_EQ(pool->freeBlocksNum(), kUsableBlocks);
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, SequentialReclaimDrainsChainWithoutHostBlocks) {
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr> groups = {full};

    // No Host pool, Host disabled → direct release on reclaim.
    BlockTreeCacheConfig seq_cfg;
    seq_cfg.task_pool_size      = 2;
    seq_cfg.enable_device_cache = true;
    seq_cfg.enable_host_cache   = false;

    std::unique_ptr<BlockTreeCache> ce_cache = makeBlockTreeCacheForTest(std::move(groups), std::move(seq_cfg));

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    resources[1][0].device_blocks = {43};
    resources[2][0].device_blocks = {44};
    ce_cache->insert({100, 200, 300}, resources, Tier::DEVICE);

    // Reclaim all 3 nodes sequentially (synchronous direct release)
    for (int i = 0; i < 3; ++i) {
        int reclaimed = BlockTreeCacheTestPeer::reclaimBlocksForTest(*ce_cache, 1, Tier::DEVICE);
        EXPECT_EQ(reclaimed, 1) << "Reclaim " << i << " should succeed";
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*ce_cache);
    }

    EXPECT_EQ(ce_cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, HostDisabledDirectRelease) {
    auto host_pool = makeHostPool(256, 4);

    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr> groups = {full};

    // Host disabled (default): Device reclaim → direct release.
    std::unique_ptr<BlockTreeCache> cache =
        makeBlockTreeCacheForTest(std::move(groups), BlockTreeCacheConfig{.task_pool_size = 2});

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    cache->insert({100}, resources, Tier::DEVICE);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    // No host block allocated (Host disabled → direct release)
    EXPECT_EQ(host_pool->freeBlocksNum(), 4u);
    // Node deleted (direct release, no host data to keep it alive)
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, TierEnableQueries) {
    auto host_pool = makeHostPool(1, 2);
    auto disk_pool = makeDiskPool(1, 2, std::make_unique<MemoryDiskBlockIO>());

    auto device_pools = makeStructuralDevicePools(1, "tier_enable_queries");
    auto full         = std::make_shared<FullGroupSet>(device_pools, host_pool, disk_pool);
    initializeTestGroupSet(full, device_pools);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = true;
    cfg.enable_host_cache   = true;
    cfg.enable_disk_cache   = true;
    cfg.enable_remote_cache = true;

    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(groups), std::move(cfg));

    EXPECT_TRUE(cache->isDeviceCacheEnabled());
    EXPECT_TRUE(cache->isHostCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_TRUE(cache->isRemoteCacheEnabled());
}

TEST_F(BlockTreeCacheTest, NodeDeletedWhenAllGroupsEmpty) {

    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::vector<GroupSetPtr>        groups = {full};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups));

    // Insert
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    cache->insert({100}, resources, Tier::DEVICE);

    EXPECT_EQ(cache->getStats().tree_node_count, 1u);

    // Reclaim device data.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    // Node should be deleted: group empty
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
}

TEST_F(BlockTreeCacheTest, MatchCollectsBlocksSelectedByGroupPolicy) {

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::shared_ptr<LinearGroupSet> linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::shared_ptr<SWAGroupSet> swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::vector<GroupSetPtr>        group_sets = {full, linear, swa};
    std::unique_ptr<BlockTreeCache> cache      = makeBlockTreeCacheForTest(std::move(group_sets));

    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(3));
    for (size_t i = 0; i < resources.size(); ++i) {
        resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        resources[i][1].device_blocks = {static_cast<BlockIdxType>(20 + i)};
        resources[i][2].device_blocks = {static_cast<BlockIdxType>(30 + i)};
    }
    cache->insert({100, 200, 300}, resources, Tier::DEVICE);

    BlockTreeMatchResult result = cache->match({100, 200, 300});
    EXPECT_EQ(result.matched_device_blocks, 3u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_device_resources), (BlockIndicesType{10, 11, 12}));
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_device_resources), (BlockIndicesType{22}));
    EXPECT_EQ(cache->matchedBlocksForGroup(2, result.matched_device_resources), (BlockIndicesType{31, 32}));
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, MatchKeepsAggregatedDevicePoolsSeparate) {
    std::vector<DeviceBlockPoolPtr> device_pools = makeStructuralDevicePools(2, "aggregated_device_pool");
    std::shared_ptr<FullGroupSet>   full         = std::make_shared<FullGroupSet>(device_pools, nullptr, nullptr);
    auto                            pool0_prefix = device_pools[0]->malloc(1);
    auto                            pool1_prefix = device_pools[1]->malloc(3);
    ASSERT_TRUE(pool0_prefix.has_value());
    ASSERT_TRUE(pool1_prefix.has_value());
    initializeTestGroupSet(full, device_pools);

    std::vector<GroupSetPtr> group_sets = {full};
    auto                     cache      = makeBlockTreeCacheForTest(std::move(group_sets));
    ASSERT_NE(cache, nullptr);

    MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*full, 2);
    ASSERT_EQ(request_holder.size(), 2u);
    ASSERT_EQ(request_holder[0].size(), 2u);
    ASSERT_EQ(request_holder[1].size(), 2u);
    const BlockIndicesType group0_blocks = {request_holder[0][0], request_holder[1][0]};
    const BlockIndicesType group1_blocks = {request_holder[0][1], request_holder[1][1]};
    EXPECT_NE(group0_blocks, group1_blocks);

    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = request_holder[0];
    resources[1][0].device_blocks = request_holder[1];
    cache->insert({100, 200}, resources, Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*full, request_holder);
    device_pools[0]->incRef(*pool0_prefix);
    device_pools[0]->decRef(*pool0_prefix);
    device_pools[1]->incRef(*pool1_prefix);
    device_pools[1]->decRef(*pool1_prefix);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_device_blocks, 2u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_device_resources), group0_blocks);
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_device_resources), group1_blocks);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, ReorderedMembershipMapsBlocksByGroupId) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse = true;
    auto topology =
        block_transfer_engine_test::makeTestTopology({block_transfer_engine_test::makeTestGroupBase(policy, {0}, 1),
                                                      block_transfer_engine_test::makeTestGroupBase(policy, {0}, 1)});
    auto device_pools = makeStructuralDevicePools(2, "reordered_membership");
    auto full         = std::make_shared<FullGroupSet>(device_pools, nullptr, nullptr);
    full->initialize(0, topology, {1, 0});

    std::vector<GroupSetPtr> group_sets = {full};
    auto                     cache      = makeBlockTreeCacheForTest(std::move(group_sets));
    ASSERT_NE(cache, nullptr);

    MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*full, 2);
    ASSERT_EQ(request_holder.size(), 2u);
    const BlockIndicesType                     group1_blocks = {request_holder[0][0], request_holder[1][0]};
    const BlockIndicesType                     group0_blocks = {request_holder[0][1], request_holder[1][1]};
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = request_holder[0];
    resources[1][0].device_blocks = request_holder[1];
    cache->insert({100, 200}, resources, Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*full, request_holder);

    BlockTreeMatchResult result = cache->match({100, 200});
    EXPECT_EQ(result.matched_device_blocks, 2u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, result.matched_device_resources), group0_blocks);
    EXPECT_EQ(cache->matchedBlocksForGroup(1, result.matched_device_resources), group1_blocks);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, MatchRequiresSWAWindowAfterGap) {

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::shared_ptr<SWAGroupSet> swa = std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::vector<GroupSetPtr>        groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups));

    std::vector<std::vector<GroupSetResource>> resources(4, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks = {10};
    resources[1][0].device_blocks = {11};
    resources[2][0].device_blocks = {12};
    resources[3][0].device_blocks = {13};
    resources[0][1].device_blocks = {20};
    resources[2][1].device_blocks = {22};
    resources[3][1].device_blocks = {23};

    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200, 300, 400}, resources));

    BlockTreeMatchResult partial = cache->match({100, 200, 300});
    EXPECT_EQ(partial.matched_device_blocks, 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, partial.matched_device_resources);

    BlockTreeMatchResult restored = cache->match({100, 200, 300, 400});
    EXPECT_EQ(restored.matched_device_blocks, 4u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, restored.matched_device_resources);
}

TEST_F(BlockTreeCacheTest, ParentBecomesDeviceLeafAfterChildReclaim) {
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
    std::vector<GroupSetPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(groups));

    // Insert: root -> A -> B -> C
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    resources[1][0].device_blocks = {43};
    resources[2][0].device_blocks = {44};
    cache->insert({100, 200, 300}, resources, Tier::DEVICE);

    // Initially only C (leaf) is in heap
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim C -> B becomes DeviceLeaf -> enters heap.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);

    // Reclaim B -> A becomes DeviceLeaf -> enters heap.
    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheTest, LoadOnlyReloadsSWAWindow) {
    auto host_pool = makeHostPool(1, 4);
    ASSERT_NE(host_pool, nullptr);
    const auto host_blocks = host_pool->malloc(4);
    ASSERT_TRUE(host_blocks.has_value());
    host_pool->incTreeRef(*host_blocks, BlockTreeRefType::CACHE);

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);

    std::shared_ptr<SWAGroupSet> swa = std::make_shared<SWAGroupSet>(
        128,
        64,
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
        host_pool,
        nullptr);

    std::vector<GroupSetPtr>        groups = {full, swa};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups));

    std::vector<std::vector<GroupSetResource>> resources(4, std::vector<GroupSetResource>(2));
    for (size_t i = 0; i < resources.size(); ++i) {
        resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
        resources[i][1].host_block    = (*host_blocks)[i];
    }

    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200, 300, 400}, resources));

    BlockTreeMatchResult result = cache->match({100, 200, 300, 400});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 4u);
    const std::vector<TransferDescriptor>& load_descs = load_context->loadDescs();
    ASSERT_EQ(load_descs.size(), 6u);
    const std::function<size_t(size_t, Tier, size_t, BlockIdxType)> count_exact_desc =
        [&load_descs](size_t group_set_id, Tier source_tier, size_t path_index, BlockIdxType source_block) {
            size_t count = 0;
            for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
                count += load_descs[desc_index].group_set_id == group_set_id
                         && load_descs[desc_index].source_tier == source_tier
                         && load_descs[desc_index].path_index == path_index
                         && load_descs[desc_index].source_blocks == std::vector<BlockIdxType>{source_block};
            }
            return count;
        };
    for (size_t path_index = 0; path_index < 4; ++path_index) {
        EXPECT_EQ(
            count_exact_desc(/*group_id=*/0, Tier::DEVICE, path_index, static_cast<BlockIdxType>(10 + path_index)), 1);
    }
    for (size_t path_index = 2; path_index < 4; ++path_index) {
        EXPECT_EQ(count_exact_desc(/*group_id=*/1, Tier::HOST, path_index, (*host_blocks)[path_index]), 1);
    }
}

TEST_F(BlockTreeCacheTest, LoadPlanningIgnoresBusySwaResourceOutsideTrailingWindow) {
    for (GroupSetTransferState state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOAD_PENDING}) {
        auto host_pool = makeHostPool(1, 4);
        ASSERT_NE(host_pool, nullptr);
        const auto host_blocks = host_pool->malloc(4);
        ASSERT_TRUE(host_blocks.has_value());
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, nullptr, nullptr);
        auto swa = std::make_shared<SWAGroupSet>(
            2,
            1,
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)},
            host_pool,
            nullptr);
        std::vector<GroupSetPtr>        groups = {full, swa};
        std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups));
        ASSERT_NE(cache, nullptr);

        std::vector<std::vector<GroupSetResource>> resources(4, std::vector<GroupSetResource>(2));
        for (size_t i = 0; i < resources.size(); ++i) {
            resources[i][0].device_blocks = {static_cast<BlockIdxType>(10 + i)};
            resources[i][1].host_block    = (*host_blocks)[i];
        }
        if (state == GroupSetTransferState::DEMOTING) {
            resources[1][1].host_block    = NULL_BLOCK_IDX;
            resources[1][1].device_blocks = {21};
        }
        BlockIdList cached_host_blocks = *host_blocks;
        if (state == GroupSetTransferState::DEMOTING) {
            cached_host_blocks.erase(cached_host_blocks.begin() + 1);
        }
        host_pool->incTreeRef(cached_host_blocks, BlockTreeRefType::CACHE);
        ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200, 300, 400}, resources));

        const std::vector<TreeNode*> path = cache->tree()->findNode({100, 200, 300, 400});
        ASSERT_EQ(path.size(), 4u);
        path[1]->group_set_resources[1].transfer_state = state;

        BlockTreeMatchResult result                    = cache->match({100, 200, 300, 400});
        path[1]->group_set_resources[1].transfer_state = GroupSetTransferState::IDLE;
        EXPECT_EQ(result.matched_device_blocks, 0u);
        EXPECT_TRUE(result.matched_device_resources.empty());
        std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
        ASSERT_NE(load_context, nullptr);
        EXPECT_EQ(load_context->matchedBlocks(), 4u);
        const std::vector<TransferDescriptor>& load_descs = load_context->loadDescs();
        ASSERT_EQ(load_descs.size(), 6u);

        size_t swa_desc_count = 0;
        for (size_t desc_index = 0; desc_index < load_descs.size(); ++desc_index) {
            if (load_descs[desc_index].group_set_id != 1) {
                continue;
            }
            ++swa_desc_count;
            EXPECT_GE(load_descs[desc_index].path_index, 2u);
            EXPECT_EQ(load_descs[desc_index].source_tier, Tier::HOST);
        }
        EXPECT_EQ(swa_desc_count, 2u);

        load_context.reset();
    }
}

TEST_F(BlockTreeCacheTest, LoadDetectsHostData) {
    auto host_pool = makeHostPool(1, 1);
    ASSERT_NE(host_pool, nullptr);
    auto full = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, host_pool, nullptr);
    std::vector<GroupSetPtr> groups = {full};

    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(groups));

    // Insert a node and manually set host data (simulating prior demotion).
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = {42};
    cache->insert({100}, resources, Tier::DEVICE);

    BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    std::vector<std::vector<GroupSetResource>> resources2(1, std::vector<GroupSetResource>(1));
    resources2[0][0].device_blocks = {55};
    cache->insert({200}, resources2, Tier::DEVICE);

    // Manually set host_block and clear device_blocks to simulate a demoted state.
    auto find = cache->tree()->findNode({200});
    ASSERT_FALSE(find.empty());
    GroupSetResource& resource = find.back()->group_set_resources[0];
    resource.host_block        = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(resource.host_block, NULL_BLOCK_IDX);
    const auto device_blocks = resource.getBlocks(Tier::DEVICE);
    ASSERT_EQ(device_blocks, (BlockIndicesType{55}));
    const MultiNodeResource device_resource{full->groupSetId(), Tier::DEVICE, {{find.back(), device_blocks}}};
    resource.evictFromTier(Tier::DEVICE);
    full->unreferenceBlocks(device_resource, BlockTreeRefType::CACHE);

    const BlockIdxType host_block = resource.host_block;
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);

    BlockTreeMatchResult result = cache->match({200});
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 1u);
    ASSERT_EQ(load_context->loadDescs().size(), 1u);
    EXPECT_EQ(load_context->loadDescs()[0].group_set_id, 0u);
    EXPECT_EQ(load_context->loadDescs()[0].path_index, 0u);
    EXPECT_EQ(load_context->loadDescs()[0].source_tier, Tier::HOST);
    EXPECT_EQ(load_context->loadDescs()[0].source_blocks, (BlockIndicesType{host_block}));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    EXPECT_EQ(host_pool->treeRefCount(host_block), 2u);

    load_context.reset();
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);
}

static std::unique_ptr<BlockTreeCache> makeHostOnlyLoadCache(std::vector<DeviceBlockPoolPtr> device_pools = {}) {
    if (device_pools.empty()) {
        device_pools.push_back(makeDevicePool({{1, 0}}, 1, "load_context_abort"));
    }
    for (const DeviceBlockPoolPtr& device_pool : device_pools) {
        RTP_LLM_CHECK(device_pool != nullptr);
    }
    std::shared_ptr<HostBlockPool> host_pool = makeHostPool(/*payload_bytes=*/device_pools.size(), /*usable_count=*/1);
    RTP_LLM_CHECK(host_pool != nullptr);

    std::shared_ptr<FullGroupSet> full = std::make_shared<FullGroupSet>(device_pools, host_pool, nullptr);
    initializeTestGroupSet(full, device_pools);
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_host_cache              = true;
    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    RTP_LLM_CHECK(cache != nullptr);

    MultiNodeBlocks request_holder = allocateDeviceBlocksForTest(*full, 1);
    RTP_LLM_CHECK(request_holder.size() == 1);
    RTP_LLM_CHECK(request_holder[0].size() == device_pools.size());
    const std::vector<BlockIdxType> device_blocks = request_holder[0];

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = device_blocks;
    cache->insert({200}, resources, Tier::DEVICE);
    unreferenceDeviceBlocksForTest(*full, request_holder);

    auto find = cache->tree()->findNode({200});
    RTP_LLM_CHECK(!find.empty());
    GroupSetResource& resource = find.back()->group_set_resources[0];
    resource.host_block        = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    RTP_LLM_CHECK(resource.host_block != NULL_BLOCK_IDX);
    RTP_LLM_CHECK(resource.getBlocks(Tier::DEVICE) == device_blocks);
    const MultiNodeResource device_resource{full->groupSetId(), Tier::DEVICE, {{find.back(), device_blocks}}};
    resource.evictFromTier(Tier::DEVICE);
    full->unreferenceBlocks(device_resource, BlockTreeRefType::CACHE);
    return cache;
}

TEST_F(BlockTreeCacheTest, PendingLoadContextHardStopsSecondMatchUntilAbort) {
    std::unique_ptr<BlockTreeCache> cache = makeHostOnlyLoadCache();
    ASSERT_NE(cache, nullptr);

    const GroupSetPtr& group       = cache->groupSets().front();
    const auto         host_pool   = group->hostPool();
    const auto         source_path = cache->tree()->findNode({200});
    ASSERT_FALSE(source_path.empty());
    TreeNode*          source_node  = source_path.back();
    const BlockIdxType source_block = source_node->group_set_resources[0].host_block;
    ASSERT_NE(source_block, NULL_BLOCK_IDX);

    BlockTreeMatchResult              first_match   = cache->match({200});
    std::shared_ptr<LoadAsyncContext> first_context = takeLoadContext(first_match);
    ASSERT_NE(first_context, nullptr);
    EXPECT_EQ(source_node->group_set_resources[0].transfer_state, GroupSetTransferState::LOAD_PENDING);
    EXPECT_EQ(host_pool->treeRefCount(source_block), 2u);

    BlockTreeMatchResult second_match = cache->match({200});
    EXPECT_EQ(second_match.matched_device_blocks, 0u);
    EXPECT_EQ(second_match.async_context, nullptr);
    EXPECT_EQ(host_pool->treeRefCount(source_block), 2u);

    first_context.reset();
    EXPECT_EQ(source_node->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(host_pool->treeRefCount(source_block), 1u);
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

    auto first_group =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{first_device_pool}, first_host_pool, nullptr);
    auto second_group =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{second_device_pool}, second_host_pool, nullptr);
    initializeSingleMemberGroupSets({first_group, second_group}, {first_device_pool, second_device_pool});

    BlockTreeCacheConfig config;
    config.enable_host_cache                   = true;
    std::vector<GroupSetPtr>        group_sets = {first_group, second_group};
    std::unique_ptr<BlockTreeCache> cache      = makeBlockTreeCacheForTest(std::move(group_sets), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType first_source  = first_group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType second_source = second_group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(first_source, NULL_BLOCK_IDX);
    ASSERT_NE(second_source, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(2));
    resources[0][0].host_block                = first_source;
    resources[0][1].host_block                = second_source;
    const BlockTreeInsertResult insert_result = cache->tree()->insertNode({100}, resources, /*collect_path=*/false);
    ASSERT_EQ(insert_result.inserted_nodes.size(), 1u);
    releaseLowerTierSeedRefs(cache->groupSets(), resources);

    BlockTreeMatchResult              result       = cache->match({100});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    const std::vector<TransferDescriptor>& load_descs = load_context->loadDescs();
    ASSERT_EQ(load_descs.size(), 2u);
    ASSERT_EQ(load_descs[0].group_set_id, 0);
    ASSERT_EQ(load_descs[1].group_set_id, 1);
    EXPECT_EQ(first_host_pool->treeRefCount(first_source), 2u);
    EXPECT_EQ(second_host_pool->treeRefCount(second_source), 2u);

    // Duplicate the first descriptor immediately after itself. The complete batch passes
    // preflight while both resources are IDLE. Preparation then transitions the first descriptor
    // to LOADING, registers it, and takes its target holder; the duplicate observes the same
    // resource already LOADING and fails with one prepared descriptor and one
    // untouched trailing descriptor. Add the matching source planning hold explicitly
    // so every descriptor in the synthetic context owns exactly one source hold.
    TransferDescriptor                        duplicate_first_desc = load_descs.front();
    std::vector<TransferDescriptor>::iterator inserted_desc =
        load_context->load_descs_.insert(load_context->load_descs_.begin() + 1, std::move(duplicate_first_desc));
    ASSERT_EQ(inserted_desc, load_context->load_descs_.begin() + 1);
    std::vector<bool>::iterator inserted_joined =
        load_context->joined_load_.insert(load_context->joined_load_.begin() + 1, false);
    ASSERT_EQ(inserted_joined, load_context->joined_load_.begin() + 1);
    first_group->referenceBlocks(MultiNodeResource{0, Tier::HOST, {{load_descs.front().node, {first_source}}}},
                                 BlockTreeRefType::LOAD);
    ASSERT_EQ(load_descs.size(), 3u);
    EXPECT_EQ(first_host_pool->treeRefCount(first_source), 3u);
    EXPECT_EQ(second_host_pool->treeRefCount(second_source), 2u);

    const BlockIdList first_request_targets  = first_device_pool->malloc(1).value();
    const BlockIdList second_request_targets = second_device_pool->malloc(1).value();
    ASSERT_EQ(first_request_targets.size(), 1u);
    ASSERT_EQ(second_request_targets.size(), 1u);
    first_device_pool->incRef(first_request_targets);
    second_device_pool->incRef(second_request_targets);
    const BlockIdxType first_target  = first_request_targets.front();
    const BlockIdxType second_target = second_request_targets.front();
    load_context->setTargetBlocks(0, {first_target});
    load_context->setTargetBlocks(1, {first_target});
    load_context->setTargetBlocks(2, {second_target});

    const size_t first_refs_before  = first_device_pool->refCount(first_target);
    const size_t second_refs_before = second_device_pool->refCount(second_target);
    ASSERT_EQ(first_refs_before, 1u);
    ASSERT_EQ(second_refs_before, 1u);
    ASSERT_TRUE(first_device_pool->isAllocated(first_target));
    ASSERT_TRUE(second_device_pool->isAllocated(second_target));

    EXPECT_FALSE(load_context->commit());
    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), 0u);

    // The first descriptor's acquired target holder and both of its source planning
    // holds are gone; the unprepared trailing descriptor's source hold is also gone.
    // Request ownership remains untouched for both target blocks.
    EXPECT_EQ(first_host_pool->treeRefCount(first_source), 1u);
    EXPECT_EQ(second_host_pool->treeRefCount(second_source), 1u);
    EXPECT_TRUE(first_device_pool->isAllocated(first_target));
    EXPECT_TRUE(second_device_pool->isAllocated(second_target));
    EXPECT_EQ(first_device_pool->refCount(first_target), first_refs_before);
    EXPECT_EQ(second_device_pool->refCount(second_target), second_refs_before);

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    ASSERT_EQ(find.back()->group_set_resources.size(), 2u);
    EXPECT_EQ(find.back()->group_set_resources[0].host_block, first_source);
    EXPECT_EQ(find.back()->group_set_resources[1].host_block, second_source);
    EXPECT_TRUE(find.back()->group_set_resources[0].device_blocks.empty());
    EXPECT_TRUE(find.back()->group_set_resources[1].device_blocks.empty());
    EXPECT_EQ(find.back()->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(find.back()->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    load_context.reset();
    EXPECT_EQ(first_host_pool->treeRefCount(first_source), 1u) << "committed context must not release source twice";
    EXPECT_EQ(second_host_pool->treeRefCount(second_source), 1u) << "committed context must not release source twice";
    first_device_pool->decRef(first_request_targets);
    second_device_pool->decRef(second_request_targets);
}

TEST_F(BlockTreeCacheTest, LoadQueueRejectionRollsBackCoreHoldersAndRetainsRequestTarget) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             device_pool = makeDevicePool({{1, 0}}, 2, "load_queue_rejection");
    std::shared_ptr<HostBlockPool> host_pool   = makeHostPool(1, 2);
    ASSERT_NE(device_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, nullptr);
    initializeTestGroupSet(full, {device_pool});
    std::vector<GroupSetPtr> groups = {full};

    BlockTreeCacheConfig config;
    config.enable_host_cache              = true;
    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    const BlockIdxType source_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(source_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].host_block                = source_block;
    const BlockTreeInsertResult insert_result = cache->tree()->insertNode({100}, resources, /*collect_path=*/false);
    ASSERT_EQ(insert_result.inserted_nodes.size(), 1u);
    releaseLowerTierSeedRefs(cache->groupSets(), resources);
    const size_t source_tree_ref_before = host_pool->treeRefCount(source_block);

    BlockTreeMatchResult              result       = cache->match({100});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 1u);
    EXPECT_EQ(load_context->loadDescs()[0].group_set_id, 0);
    EXPECT_EQ(host_pool->treeRefCount(source_block), source_tree_ref_before + 1);

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    load_context->setTargetBlocks(0, {request_target});
    ASSERT_EQ(device_pool->refCount(request_target), 1u);

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    ASSERT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

    EXPECT_FALSE(load_context->commit());
    EXPECT_TRUE(load_context->done());
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), 0u);
    EXPECT_EQ(host_pool->treeRefCount(source_block), source_tree_ref_before);
    EXPECT_EQ(device_pool->refCount(request_target), 1u);

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    ASSERT_EQ(find.back()->group_set_resources.size(), 1u);
    EXPECT_EQ(find.back()->group_set_resources[0].host_block, source_block);
    EXPECT_TRUE(find.back()->group_set_resources[0].device_blocks.empty());
    EXPECT_EQ(find.back()->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    load_context.reset();
    EXPECT_EQ(host_pool->treeRefCount(source_block), source_tree_ref_before);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    device_pool->decRef(request_targets);
}

TEST_F(BlockTreeCacheTest, LoadQueueRejectionRollsBackMixedDeviceAndHostDescriptors) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    DeviceBlockPoolPtr             cache_device_pool  = makeDevicePool({{1, 0}}, 1, "load_mixed_cache");
    DeviceBlockPoolPtr             target_device_pool = makeDevicePool({{1, 0}}, 2, "load_mixed_target");
    std::shared_ptr<HostBlockPool> cache_host_pool    = makeHostPool(1, 1);
    std::shared_ptr<HostBlockPool> host_pool          = makeHostPool(1, 2);
    ASSERT_NE(cache_device_pool, nullptr);
    ASSERT_NE(target_device_pool, nullptr);
    ASSERT_NE(cache_host_pool, nullptr);
    ASSERT_NE(host_pool, nullptr);

    auto cache_group =
        std::make_shared<LinearGroupSet>(std::vector<DeviceBlockPoolPtr>{cache_device_pool}, cache_host_pool, nullptr);
    auto loading_group =
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{target_device_pool}, host_pool, nullptr);
    initializeSingleMemberGroupSets({cache_group, loading_group}, {cache_device_pool, target_device_pool});

    BlockTreeCacheConfig config;
    config.enable_host_cache               = true;
    std::vector<GroupSetPtr>        groups = {cache_group, loading_group};
    std::unique_ptr<BlockTreeCache> cache  = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);

    auto per_rank_transfer_engine = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    MultiNodeBlocks cache_holder = allocateDeviceBlocksForTest(*cache_group, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(cache_holder.size(), 1u);
    ASSERT_EQ(cache_holder.front().size(), 1u);
    const BlockIdxType cache_block = cache_holder.front().front();
    const BlockIdxType host_block  = loading_group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(2));
    resources[0][0].device_blocks             = {cache_block};
    resources[0][1].host_block                = host_block;
    const BlockTreeInsertResult insert_result = cache->tree()->insertNode({100}, resources, /*collect_path=*/false);
    ASSERT_EQ(insert_result.inserted_nodes.size(), 1u);
    releaseLowerTierSeedRefs(cache->groupSets(), resources);
    unreferenceDeviceBlocksForTest(*cache_group, cache_holder, BlockTreeRefType::CACHE);
    ASSERT_EQ(cache_device_pool->refCount(cache_block), 1u);
    ASSERT_EQ(host_pool->treeRefCount(host_block), 1u);
    cache->evictor_.admitCandidate(insert_result.inserted_nodes.front(), /*group_set_id=*/0, Tier::DEVICE);

    BlockTreeMatchResult              result       = cache->match({100});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 2u);
    EXPECT_EQ(load_context->loadDescs()[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(load_context->loadDescs()[0].node, nullptr);
    EXPECT_EQ(load_context->loadDescs()[1].source_tier, Tier::HOST);
    EXPECT_EQ(cache_device_pool->refCount(cache_block), 2u);
    EXPECT_EQ(cache_device_pool->treeRefCount(cache_block), 1u);
    EXPECT_EQ(host_pool->treeRefCount(host_block), 2u);

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_TRUE(cache_device_pool->isAllocated(cache_block));
    EXPECT_EQ(cache_device_pool->refCount(cache_block), 1u);
    EXPECT_EQ(cache_device_pool->treeRefCount(cache_block), 0u);

    const BlockIdxType request_target = poolMalloc(*target_device_pool);
    ASSERT_NE(request_target, NULL_BLOCK_IDX);
    target_device_pool->incRef(request_target);
    ASSERT_EQ(target_device_pool->refCount(request_target), 1u);
    load_context->setTargetBlocks(0, {cache_block});
    load_context->setTargetBlocks(1, {request_target});

    BlockTreeCacheTestPeer::ScopedQueueRejectionGuard rejection_guard(*cache);
    ASSERT_TRUE(rejection_guard.armed());
    EXPECT_FALSE(load_context->commit());
    EXPECT_TRUE(load_context->done());
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), 0u);
    EXPECT_EQ(cache_device_pool->refCount(cache_block), 1u);
    EXPECT_EQ(cache_device_pool->treeRefCount(cache_block), 0u);
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);
    EXPECT_EQ(target_device_pool->refCount(request_target), 1u);

    auto find = cache->tree()->findNode({100});
    ASSERT_FALSE(find.empty());
    ASSERT_EQ(find.back()->group_set_resources.size(), 2u);
    EXPECT_EQ(find.back()->group_set_resources[0].transfer_state, GroupSetTransferState::IDLE);
    EXPECT_EQ(find.back()->group_set_resources[1].transfer_state, GroupSetTransferState::IDLE);

    EXPECT_TRUE(rejection_guard.restore());
    load_context.reset();
    EXPECT_EQ(cache_device_pool->refCount(cache_block), 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    result.matched_device_resources.clear();
    EXPECT_FALSE(cache_device_pool->isAllocated(cache_block));
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);
    target_device_pool->decRef(request_target);
}

// Deferred load: match() plans and references source blocks without executing.
// The allocator binds request-owned targets before committing the context.

// Dropping an uncommitted context performs RAII abort without submitting a copy.
TEST_F(BlockTreeCacheTest, LoadContextAbortSkipsLoad) {
    auto cache = makeHostOnlyLoadCache();

    BlockTreeMatchResult              result       = cache->match({200});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_FALSE(load_context->empty());
    EXPECT_EQ(load_context->matchedBlocks(), 1u);
    // Counters reflect the planned load; match() submits nothing asynchronously.
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());
    load_context.reset();
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
}

// Committing the context uses the allocator-owned target and submits the copy.
TEST_F(BlockTreeCacheTest, LoadContextCommitTriggersLoad) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr device_pool = makeDevicePool({{1, 0}}, 1, "load_context_commit");
    ASSERT_NE(device_pool, nullptr);
    std::unique_ptr<BlockTreeCache> cache = makeHostOnlyLoadCache({device_pool});

    BlockTreeMatchResult              result       = cache->match({200});
    std::shared_ptr<LoadAsyncContext> load_context = takeLoadContext(result);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 1u);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    EXPECT_TRUE(result.matched_device_resources.empty());

    const BlockIdList request_targets = device_pool->malloc(1).value();
    ASSERT_EQ(request_targets.size(), 1u);
    device_pool->incRef(request_targets);
    const BlockIdxType request_target = request_targets.front();
    EXPECT_EQ(device_pool->refCount(request_target), 1u);
    ASSERT_EQ(load_context->loadDescs().size(), 1u);
    load_context->setTargetBlocks(0, {request_target});

    EXPECT_TRUE(load_context->commit());

    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    device_pool->decRef(request_targets);
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

    auto full = std::make_shared<FullGroupSet>(device_pools, host_pool, disk_pool);
    initializeTestGroupSet(full, device_pools, kBlockBytes);

    BlockTreeCacheConfig config;
    config.enable_device_cache      = true;
    config.enable_host_cache        = true;
    config.enable_disk_cache        = true;
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->tree()->groupSets().size(), 1u);
    EXPECT_EQ(cache->tree()->groupSets()[0], full);

    MultiNodeBlocks root_device_holds = allocateDeviceBlocksForTest(*full, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(root_device_holds.size(), 1u);
    ASSERT_EQ(root_device_holds[0].size(), 3u);
    const BlockIdxType device_block_0 = root_device_holds[0][0];
    const BlockIdxType device_block_1 = root_device_holds[0][1];
    const BlockIdxType device_block_2 = root_device_holds[0][2];
    ASSERT_NE(device_block_0, NULL_BLOCK_IDX);
    ASSERT_NE(device_block_1, NULL_BLOCK_IDX);
    ASSERT_NE(device_block_2, NULL_BLOCK_IDX);

    cache->tree()->root()->group_set_resources[0].setBlocks(Tier::DEVICE, root_device_holds[0]);
    const BlockIdxType host_block = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType disk_block = full->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);
    std::vector<std::vector<GroupSetResource>> lower_tier_resources(2, std::vector<GroupSetResource>(1));
    lower_tier_resources[0][0].host_block = host_block;
    lower_tier_resources[1][0].disk_slot  = disk_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100, 200}, lower_tier_resources));

    EXPECT_EQ(device_pools[0]->freeBlocksNum(), device_free_before[0] - 1);
    EXPECT_EQ(device_pools[1]->freeBlocksNum(), device_free_before[1] - 1);
    EXPECT_EQ(device_pools[2]->freeBlocksNum(), device_free_before[2] - 1);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);
    EXPECT_EQ(device_pools[0]->refCount(device_block_0), 1u);
    EXPECT_EQ(device_pools[1]->refCount(device_block_1), 1u);
    EXPECT_EQ(device_pools[2]->refCount(device_block_2), 1u);
    EXPECT_EQ(host_pool->treeRefCount(host_block), 1u);
    EXPECT_EQ(disk_pool->treeRefCount(disk_block), 1u);

    cache.reset();

    EXPECT_EQ(device_pools[0]->freeBlocksNum(), device_free_before[0]);
    EXPECT_EQ(device_pools[1]->freeBlocksNum(), device_free_before[1]);
    EXPECT_EQ(device_pools[2]->freeBlocksNum(), device_free_before[2]);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
    EXPECT_FALSE(device_pools[0]->isAllocated(device_block_0));
    EXPECT_FALSE(device_pools[1]->isAllocated(device_block_1));
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

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, nullptr, nullptr);
    initializeTestGroupSet(full, {device_pool}, kBlockBytes);
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups));
    ASSERT_NE(cache, nullptr);

    MultiNodeBlocks tree_holder = allocateDeviceBlocksForTest(*full, 1, BlockTreeRefType::CACHE);
    ASSERT_EQ(tree_holder.size(), 1u);
    ASSERT_EQ(tree_holder[0].size(), 1u);
    const BlockIdxType block = tree_holder[0][0];
    ASSERT_NE(block, NULL_BLOCK_IDX);
    MultiNodeBlocks external_holder = tree_holder;
    referenceDeviceBlocksForTest(*full, external_holder);
    EXPECT_EQ(device_pool->refCount(block), 2u);

    std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
    resources[0][0].device_blocks = tree_holder[0];
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));
    unreferenceDeviceBlocksForTest(*full, tree_holder, BlockTreeRefType::CACHE);

    cache.reset();

    EXPECT_TRUE(device_pool->isAllocated(block));
    EXPECT_EQ(device_pool->refCount(block), 1u);
    EXPECT_EQ(device_pool->freeBlocksNum(), free_before - 1);

    unreferenceDeviceBlocksForTest(*full, external_holder);
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

    auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{device_pool}, host_pool, disk_pool);
    initializeTestGroupSet(full, {device_pool}, kBlockBytes);
    BlockTreeCacheConfig config;
    config.enable_device_cache      = true;
    config.enable_host_cache        = true;
    config.enable_disk_cache        = true;
    std::vector<GroupSetPtr> groups = {full};
    auto                     cache  = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
    ASSERT_NE(cache, nullptr);
    auto per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(std::vector<GroupSetPtr>{full});
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, per_rank_transfer_engine);

    MultiNodeBlocks device_holder = allocateDeviceBlocksForTest(*full, 1);
    ASSERT_EQ(device_holder.size(), 1u);
    ASSERT_EQ(device_holder[0].size(), 1u);
    const BlockIdxType device_block = device_holder[0][0];
    const BlockIdxType host_block   = full->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
    const BlockIdxType disk_block   = full->allocateSingleBlock(Tier::DISK, BlockTreeRefType::CACHE);
    ASSERT_NE(device_block, NULL_BLOCK_IDX);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    ASSERT_NE(disk_block, NULL_BLOCK_IDX);

    std::vector<std::vector<GroupSetResource>> device_resources(1, std::vector<GroupSetResource>(1));
    device_resources[0][0].device_blocks = device_holder[0];
    std::vector<std::vector<GroupSetResource>> host_resources(1, std::vector<GroupSetResource>(1));
    host_resources[0][0].host_block = host_block;
    std::vector<std::vector<GroupSetResource>> disk_resources(1, std::vector<GroupSetResource>(1));
    disk_resources[0][0].disk_slot = disk_block;
    ASSERT_TRUE(insertGroupSetResources(*cache, {100}, device_resources));
    ASSERT_TRUE(insertGroupSetResources(*cache, {200}, host_resources));
    ASSERT_TRUE(insertGroupSetResources(*cache, {300}, disk_resources));
    releaseDeviceBlocks(*cache, device_pool, device_holder.front());

    EXPECT_EQ(BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, Tier::DEVICE), 1);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_FALSE(device_pool->isAllocated(device_block));
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before - 1);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before - 1);

    cache.reset();

    EXPECT_EQ(per_rank_transfer_engine->submittedBatchCount(), 0u);
    EXPECT_EQ(device_pool->freeBlocksNum(), device_free_before);
    EXPECT_EQ(host_pool->freeBlocksNum(), host_free_before);
    EXPECT_EQ(disk_pool->freeBlocksNum(), disk_free_before);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_FALSE(disk_pool->isAllocated(disk_block));
}

TEST_F(BlockTreeCacheTest, LoadContextOutlivesHostAndDiskCacheShutdown) {
    for (Tier source_tier : {Tier::HOST, Tier::DISK}) {
        SCOPED_TRACE(tierName(source_tier));

        auto host_pool = makeHostPool(1, 2);
        auto disk_pool = makeDiskPool(1, 2, std::make_unique<MemoryDiskBlockIO>());
        auto full      = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{block_tree_cache_test::makeStructuralDevicePool(0)}, host_pool, disk_pool);

        BlockTreeCacheConfig config;
        config.enable_host_cache        = true;
        config.enable_disk_cache        = true;
        std::vector<GroupSetPtr> groups = {full};
        auto                     cache  = makeBlockTreeCacheForTest(std::move(groups), std::move(config));
        ASSERT_NE(cache, nullptr);

        const BlockIdxType source_block = full->allocateSingleBlock(source_tier, BlockTreeRefType::CACHE);
        ASSERT_NE(source_block, NULL_BLOCK_IDX);
        IBlockPool& source_pool =
            source_tier == Tier::HOST ? static_cast<IBlockPool&>(*host_pool) : static_cast<IBlockPool&>(*disk_pool);
        EXPECT_EQ(source_pool.treeRefCount(source_block), 1u);

        std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
        if (source_tier == Tier::HOST) {
            resources[0][0].host_block = source_block;
        } else {
            resources[0][0].disk_slot = source_block;
        }
        ASSERT_TRUE(insertGroupSetResources(*cache, {100}, resources));

        BlockTreeMatchResult              result            = cache->match({100});
        std::shared_ptr<LoadAsyncContext> outliving_context = takeLoadContext(result);
        ASSERT_NE(outliving_context, nullptr);
        ASSERT_FALSE(outliving_context->empty());
        ASSERT_EQ(outliving_context->loadDescs().size(), 1u);
        EXPECT_EQ(outliving_context->loadDescs()[0].source_tier, source_tier);
        EXPECT_EQ(outliving_context->loadDescs()[0].source_blocks, (BlockIndicesType{source_block}));
        EXPECT_EQ(source_pool.treeRefCount(source_block), 2u);

        ThreadCompletion destruction;
        std::thread      destroy_thread([cache = std::move(cache), &destruction]() mutable {
            destruction.markEntered();
            cache.reset();
            destruction.markFinished();
        });
        destruction.waitUntilEntered();
        destroy_thread.join();

        EXPECT_TRUE(destruction.finished());
        EXPECT_FALSE(source_pool.isAllocated(source_block));
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);
        EXPECT_FALSE(outliving_context->commit());
        EXPECT_FALSE(outliving_context->commit());
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);

        outliving_context.reset();
        EXPECT_EQ(source_pool.freeBlocksNum(), 2u);
    }
}

// A no-match match() plans nothing and returns no async context.
TEST_F(BlockTreeCacheTest, EmptyMatchYieldsNoAsyncContext) {
    auto result = cache_->match({100, 200, 300});  // empty tree => no match
    EXPECT_EQ(result.async_context, nullptr);
}

}  // namespace
}  // namespace rtp_llm
