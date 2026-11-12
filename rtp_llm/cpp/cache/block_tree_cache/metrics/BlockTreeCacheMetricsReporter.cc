#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheStageMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

constexpr std::array<Tier, 3>           kMetricTiers      = {Tier::DEVICE, Tier::HOST, Tier::DISK};
constexpr std::array<CacheGroupType, 3> kMetricGroupTypes = {
    CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::LINEAR};

size_t transferOperationIndex(CacheTransferOperation operation) {
    return static_cast<size_t>(operation);
}

int metricGroupTypeIndex(CacheGroupType group_type) {
    if (group_type == CacheGroupType::FULL) {
        return 0;
    }
    if (group_type == CacheGroupType::SWA) {
        return 1;
    }
    if (group_type == CacheGroupType::LINEAR) {
        return 2;
    }
    return -1;
}

CachePoolMetrics makePoolMetrics(Tier tier, const IBlockPool& pool) {
    CachePoolMetrics metrics_data;
    metrics_data.tier                       = tier;
    metrics_data.pool_name                  = pool.poolName();
    metrics_data.block_size_bytes           = pool.blockSizeBytes();
    metrics_data.total_blocks               = pool.totalBlocksNum();
    metrics_data.free_blocks                = pool.freeBlocksNum();
    metrics_data.used_blocks                = metrics_data.total_blocks - metrics_data.free_blocks;
    metrics_data.active_blocks              = pool.activeBlocksNum();
    metrics_data.available_blocks           = pool.availableBlocksNum();
    metrics_data.block_cache_ref_blocks     = pool.referencedBlocksNum(BlockTreeRefType::CACHE);
    metrics_data.load_ref_blocks            = pool.referencedBlocksNum(BlockTreeRefType::LOAD);
    metrics_data.eviction_target_ref_blocks = pool.referencedBlocksNum(BlockTreeRefType::EVICTION);
    metrics_data.frozen_ref_blocks          = pool.referencedBlocksNum(BlockTreeRefType::FROZEN);
    return metrics_data;
}

CachePoolMetrics makeDevicePoolMetrics(const DeviceBlockPool& pool) {
    CachePoolMetrics metrics_data            = makePoolMetrics(Tier::DEVICE, pool);
    metrics_data.request_ref_blocks          = pool.referencedBlocksNum();
    metrics_data.remote_connector_ref_blocks = pool.referencedBlocksNum(DeviceBlockRefType::REMOTE_CONNECTOR);
    metrics_data.pd_connector_ref_blocks     = pool.referencedBlocksNum(DeviceBlockRefType::PD_CONNECTOR);
    return metrics_data;
}

}  // namespace

BlockTreeCacheMetricsReporter::BlockTreeCacheMetricsReporter(
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
    bool                                       enable_stage_metrics,
    std::chrono::milliseconds                  stage_interval):
    metrics_reporter_(std::move(metrics_reporter)) {
    if (metrics_reporter_ && enable_stage_metrics) {
        try {
            stage_metrics_ = std::make_unique<CacheStageMetrics>(metrics_reporter_, stage_interval);
        } catch (...) {}  // Optional observer initialization must never fail cache creation.
    }
}
BlockTreeCacheMetricsReporter::~BlockTreeCacheMetricsReporter() = default;

void BlockTreeCacheMetricsReporter::reportLockMetrics(const char* operation,
                                                      double      wait_latency_us,
                                                      double      hold_latency_us) const noexcept {
    if (!metrics_reporter_) {
        return;
    }
    try {
        RtpLLMCacheLockMetricsCollector collector;
        collector.wait_latency_us  = wait_latency_us;
        collector.hold_latency_us  = hold_latency_us;
        collector.total_latency_us = wait_latency_us + hold_latency_us;
        kmonitor::MetricsTags tags;
        tags.AddTag("operation", operation);
        metrics_reporter_->report<RtpLLMCacheLockMetrics, RtpLLMCacheLockMetricsCollector>(&tags, &collector);
    } catch (...) {}
}

void BlockTreeCacheMetricsReporter::flushStageMetrics() const {
    if (stage_metrics_)
        stage_metrics_->flush();
}

std::vector<CachePoolMetrics>
BlockTreeCacheMetricsReporter::collectPoolMetrics(const std::vector<GroupSetPtr>&       group_sets,
                                                  std::chrono::steady_clock::time_point now) {
    std::unordered_set<const IBlockPool*> reported_device_pools;
    std::vector<CachePoolMetrics>         metrics_list;
    for (const GroupSetPtr& group_set : group_sets) {
        const std::vector<DeviceBlockPoolPtr>& device_pools = group_set->devicePools();
        for (const DeviceBlockPoolPtr& pool : device_pools) {
            if (pool == nullptr) {
                continue;
            }
            const std::pair<std::unordered_set<const IBlockPool*>::iterator, bool> insert_result =
                reported_device_pools.insert(pool.get());
            if (!insert_result.second) {
                continue;
            }
            metrics_list.push_back(makeDevicePoolMetrics(*pool));
        }

        const std::shared_ptr<HostBlockPool> host_pool = group_set->hostPool();
        if (host_pool != nullptr) {
            metrics_list.push_back(makePoolMetrics(Tier::HOST, *host_pool));
        }

        const std::shared_ptr<DiskBlockPool> disk_pool = group_set->diskPool();
        if (disk_pool != nullptr) {
            auto               metrics_data = makePoolMetrics(Tier::DISK, *disk_pool);
            const PoolIOSample sample{disk_pool->readBytes(), disk_pool->writeBytes(), now};
            metrics_data.read_bytes  = static_cast<int64_t>(sample.read_bytes);
            metrics_data.write_bytes = static_cast<int64_t>(sample.write_bytes);
            auto [it, inserted]      = pool_io_samples_.emplace(disk_pool.get(), sample);
            if (inserted) {
                metrics_data.read_bandwidth_bytes_per_sec  = 0;
                metrics_data.write_bandwidth_bytes_per_sec = 0;
            } else {
                auto&        previous    = it->second;
                const double elapsed_sec = std::chrono::duration<double>(now - previous.time).count();
                if (elapsed_sec > 0) {
                    const size_t read_delta =
                        sample.read_bytes >= previous.read_bytes ? sample.read_bytes - previous.read_bytes : 0;
                    const size_t write_delta =
                        sample.write_bytes >= previous.write_bytes ? sample.write_bytes - previous.write_bytes : 0;
                    metrics_data.read_bandwidth_bytes_per_sec  = read_delta / elapsed_sec;
                    metrics_data.write_bandwidth_bytes_per_sec = write_delta / elapsed_sec;
                    previous                                   = sample;
                }
            }
            metrics_list.push_back(std::move(metrics_data));
        }
    }
    return metrics_list;
}

void BlockTreeCacheMetricsReporter::initEvictionMetricGroupTypes(const std::vector<GroupSetPtr>& group_sets) {
    std::array<bool, kMetricGroupTypes.size()> group_type_present{};
    for (const GroupSetPtr& group_set : group_sets) {
        if (group_set == nullptr) {
            continue;
        }
        const int group_type_index = metricGroupTypeIndex(group_set->groupType());
        if (group_type_index < 0) {
            continue;
        }
        group_type_present[static_cast<size_t>(group_type_index)] = true;
    }

    eviction_metric_group_types_.clear();
    eviction_metric_group_types_.reserve(kMetricGroupTypes.size());
    for (size_t i = 0; i < kMetricGroupTypes.size(); ++i) {
        if (group_type_present[i]) {
            eviction_metric_group_types_.push_back(kMetricGroupTypes[i]);
        }
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionMetrics(const EvictionVictimMetrics& metrics_data) const {
    if (!enabled()) {
        return;
    }
    for (const auto group_type : eviction_metric_group_types_) {
        for (const auto tier : kMetricTiers) {
            reportEvictionTrigger(tier, group_type, "watermark", 0);
            reportEvictionTrigger(tier, group_type, "force_drop", 0);
        }
    }
    for (const auto trigger : {EvictionVictimTrigger::WATERMARK, EvictionVictimTrigger::FORCE_DROP}) {
        for (const auto tier : kMetricTiers) {
            for (const auto group_type : kMetricGroupTypes) {
                for (const bool drop : {false, true}) {
                    const auto& counters = metrics_data.at(trigger, tier, group_type, drop);
                    if (!counters.observed) {
                        continue;
                    }
                    kmonitor::MetricsTags tags(
                        "trigger_type", trigger == EvictionVictimTrigger::WATERMARK ? "watermark" : "force_drop");
                    tags.AddTag("tier", tierName(tier));
                    tags.AddTag("group_type", metricCacheGroupTypeName(group_type));
                    tags.AddTag("mode", drop ? "drop" : "demote");
                    RtpLLMCacheEvictionVictimMetricsCollector collector;
                    collector.calls              = counters.calls;
                    collector.candidates_sum     = counters.candidates_sum;
                    collector.selected           = counters.selected;
                    collector.latency_sum_us     = static_cast<double>(counters.latency_sum_ns) / 1000.0;
                    collector.scanned_candidates = counters.scanned_candidates;
                    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionVictimMetricsCollector>(
                        &tags, &collector);
                    // Zero deltas keep previously active reason series current during idle periods.
                    for (size_t i = 0; i < counters.skipped.size(); ++i) {
                        collector.skip_reason = kEvictionVictimSkipReasonNames[i];
                        collector.skipped     = counters.skipped[i];
                        metrics_reporter_
                            ->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionVictimMetricsCollector>(&tags,
                                                                                                            &collector);
                    }
                }
            }
        }
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionTriggered(Tier           source_tier,
                                                            CacheGroupType group_type,
                                                            bool           force_drop) const {
    reportEvictionTrigger(source_tier, group_type, force_drop ? "force_drop" : "watermark", 1);
}

void BlockTreeCacheMetricsReporter::reportEvictionBlocks(Tier           source_tier,
                                                         CacheGroupType group_type,
                                                         bool           force_drop,
                                                         size_t         required_blocks,
                                                         size_t         scheduled_blocks) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheEvictionMetricsCollector collector;
    collector.source_tier               = tierName(source_tier);
    collector.group_type                = metricCacheGroupTypeName(group_type);
    collector.trigger_type              = force_drop ? "force_drop" : "watermark";
    collector.eviction_required_blocks  = static_cast<int64_t>(required_blocks);
    collector.eviction_scheduled_blocks = static_cast<int64_t>(scheduled_blocks);
    collector.report_eviction_blocks    = true;
    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportEvictionTrigger(Tier           source_tier,
                                                          CacheGroupType group_type,
                                                          const char*    trigger_type,
                                                          int64_t        count) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheEvictionMetricsCollector collector;
    collector.source_tier             = tierName(source_tier);
    collector.group_type              = metricCacheGroupTypeName(group_type);
    collector.trigger_type            = trigger_type;
    collector.eviction_trigger_count  = count;
    collector.report_eviction_trigger = true;
    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportCacheReuseTimeMetrics(
    const std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list) const {
    if (metrics_reporter_ == nullptr || metrics_list.empty()) {
        return;
    }
    struct Accumulator {
        int64_t first_hit_sum_ms{0};
        size_t  first_hit_count{0};
        int64_t hit_sum_ms{0};
        size_t  hit_count{0};
    };
    std::array<std::array<Accumulator, kMetricGroupTypes.size()>, kMetricTiers.size()> accumulators{};
    for (const BlockTreeCacheReuseTimeMetrics& metrics_data : metrics_list) {
        const size_t tier_index       = static_cast<size_t>(metrics_data.tier);
        const int    group_type_index = metricGroupTypeIndex(metrics_data.group_type);
        if (tier_index >= kMetricTiers.size() || group_type_index < 0) {
            continue;
        }
        auto& accumulator = accumulators[tier_index][group_type_index];
        if (metrics_data.reuse_block_first_hit_idletime_ms.has_value()) {
            accumulator.first_hit_sum_ms += *metrics_data.reuse_block_first_hit_idletime_ms;
            ++accumulator.first_hit_count;
        }
        accumulator.hit_sum_ms += metrics_data.reuse_block_hit_idletime_ms;
        ++accumulator.hit_count;
    }
    for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
        for (size_t group_type_index = 0; group_type_index < kMetricGroupTypes.size(); ++group_type_index) {
            const auto& accumulator = accumulators[tier_index][group_type_index];
            if (accumulator.hit_count == 0) {
                continue;
            }
            RtpLLMCacheReuseMetricsCollector collector;
            if (accumulator.first_hit_count != 0) {
                collector.reuse_block_first_hit_idletime_ms =
                    static_cast<double>(accumulator.first_hit_sum_ms) / accumulator.first_hit_count;
            }
            collector.reuse_block_hit_idletime_ms = static_cast<double>(accumulator.hit_sum_ms) / accumulator.hit_count;
            collector.report_reuse_time_metrics   = true;
            kmonitor::MetricsTags tags("tier", tierName(kMetricTiers[tier_index]));
            tags.AddTag("group_type", metricCacheGroupTypeName(kMetricGroupTypes[group_type_index]));
            metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(&tags, &collector);
        }
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionFinished(const EvictionTransferTask&     task,
                                                           const std::vector<GroupSetPtr>& group_sets,
                                                           bool                            success,
                                                           const std::vector<bool>&        detached) const {
    if (metrics_reporter_ == nullptr || task.descriptors().empty()) {
        return;
    }
    RtpLLMCacheEvictionMetricsCollector collector;
    collector.eviction_success = success;
    std::vector<std::pair<int64_t, int64_t>> block_times_ms;
    if (success) {
        block_times_ms.reserve(task.descriptors().size());
    }
    for (size_t i = 0; i < task.descriptors().size(); ++i) {
        if (!detached.empty() && detached[i]) {
            continue;
        }
        ++collector.eviction_blocks;
        if (success) {
            const auto& timing = task.timings[i];
            block_times_ms.emplace_back(
                std::max<int64_t>(0, (timing.selected_time_us - timing.tier_enter_time_us) / 1000),
                (timing.selected_time_us - timing.last_access_time_us) / 1000);
        }
    }
    collector.block_times_ms    = block_times_ms.data();
    collector.block_times_count = block_times_ms.size();
    reportEvictionMetrics(task.descriptors().front(), group_sets, "watermark", collector);
}

void BlockTreeCacheMetricsReporter::reportEvictionFinished(const EvictionDropTask&         task,
                                                           const std::vector<GroupSetPtr>& group_sets,
                                                           const char*                     trigger_type) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    reportEvictionDescriptor(task.primary_desc, task.primary_timing, group_sets, trigger_type, true);
    for (size_t desc_index = 0; desc_index < task.dependent_prune_descs.size(); ++desc_index) {
        reportEvictionDescriptor(task.dependent_prune_descs[desc_index],
                                 task.dependent_prune_timings[desc_index],
                                 group_sets,
                                 trigger_type,
                                 true);
    }
    for (size_t desc_index = 0; desc_index < task.cascade_descs.size(); ++desc_index) {
        reportEvictionDescriptor(
            task.cascade_descs[desc_index], task.cascade_timings[desc_index], group_sets, trigger_type, true);
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionDescriptor(const TransferDescriptor&       desc,
                                                             const EvictionTimingMetrics&    timing,
                                                             const std::vector<GroupSetPtr>& group_sets,
                                                             const char*                     trigger_type,
                                                             bool                            success) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheEvictionMetricsCollector collector;
    collector.eviction_blocks  = 1;
    collector.eviction_success = success;
    const std::pair<int64_t, int64_t> block_times_ms{
        std::max<int64_t>(0, (timing.selected_time_us - timing.tier_enter_time_us) / 1000),
        (timing.selected_time_us - timing.last_access_time_us) / 1000};
    collector.block_times_ms    = &block_times_ms;
    collector.block_times_count = success ? 1 : 0;
    reportEvictionMetrics(desc, group_sets, trigger_type, collector);
}

void BlockTreeCacheMetricsReporter::reportEvictionMetrics(const TransferDescriptor&            desc,
                                                          const std::vector<GroupSetPtr>&      group_sets,
                                                          const char*                          trigger_type,
                                                          RtpLLMCacheEvictionMetricsCollector& collector) const {
    if (collector.eviction_blocks == 0) {
        return;
    }
    const size_t group_set_id = desc.group_set_id;
    if (group_set_id >= group_sets.size()) {
        return;
    }
    const GroupSetPtr& group_set = group_sets[group_set_id];
    if (group_set == nullptr || group_set->groupSetId() != desc.group_set_id) {
        return;
    }

    collector.source_tier  = tierName(desc.source_tier);
    collector.group_type   = metricCacheGroupTypeName(group_set->groupType());
    collector.trigger_type = trigger_type;
    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportInsertHardStop(const BlockTreeInsertHardStopInfo& hard_stop) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheInsertMetricsCollector collector;
    collector.tier = tierName(hard_stop.existing_tier);
    if (hard_stop.transfer_state == GroupSetTransferState::DEMOTING) {
        collector.operation = cacheTransferOperationName(CacheTransferOperation::EVICT);
    } else if (hard_stop.transfer_state == GroupSetTransferState::LOAD_PENDING
               || hard_stop.transfer_state == GroupSetTransferState::LOADING) {
        collector.operation = cacheTransferOperationName(CacheTransferOperation::LOAD);
    } else {
        collector.operation = "idle";
    }
    metrics_reporter_->report<RtpLLMCacheInsertMetrics, RtpLLMCacheInsertMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportLoadJoin(size_t dependency_count) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheReuseMetricsCollector collector;
    collector.join_dependency_count = static_cast<int64_t>(dependency_count);
    collector.report_load_join      = true;
    metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportLoadJoinWait(int64_t join_wait_latency_us) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheReuseMetricsCollector collector;
    collector.join_wait_latency_us  = join_wait_latency_us;
    collector.report_load_join_wait = true;
    metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportDiskLoadStage(
    const char* stage, int64_t latency_us, bool success, int64_t rank, const char* scope) const noexcept {
    reportTransferStage(stage, latency_us, success, rank, scope, "load", "DISK", "DEVICE");
}

void BlockTreeCacheMetricsReporter::reportTransferStage(const char* stage,
                                                        int64_t     latency_us,
                                                        bool        success,
                                                        int64_t     rank,
                                                        const char* scope,
                                                        const char* operation,
                                                        const char* source_tier,
                                                        const char* target_tier) const noexcept {
    if (!stage_metrics_ || latency_us < 0) {
        return;
    }
    stage_metrics_->record(stage, latency_us, success, rank, scope, operation, source_tier, target_tier);
}

int BlockTreeCacheMetricsReporter::transferDirectionIndex(Tier source_tier, Tier target_tier) {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return 0;
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return 1;
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return 2;
    }
    if (source_tier == Tier::DISK && target_tier == Tier::DEVICE) {
        return 3;
    }
    return -1;
}

void BlockTreeCacheMetricsReporter::reportQueueWaitMetric(bool        callback,
                                                          const char* pool_type,
                                                          const char* operation,
                                                          Tier        source_tier,
                                                          Tier        target_tier,
                                                          int64_t     latency_us) const noexcept {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    try {
        RtpLLMCacheTransferMetricsCollector collector;
        collector.pool_type                 = pool_type;
        collector.operation                 = operation == nullptr ? "" : operation;
        collector.source_tier               = source_tier == Tier::NONE ? "" : tierName(source_tier);
        collector.target_tier               = target_tier == Tier::NONE ? "" : tierName(target_tier);
        collector.queue_waiting_tasks       = 1;
        collector.queue_wait_latency_us     = latency_us;
        collector.report_transfer           = false;
        collector.report_task_queue         = !callback;
        collector.report_callback_queue     = callback;
        collector.report_queue_wait_latency = latency_us >= 0;
        metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
    } catch (...) {}
}

void BlockTreeCacheMetricsReporter::reportQueueBacklog(const BlockTreeQueueSizes& queue_sizes,
                                                       const char*                pool_type) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    RtpLLMCacheTransferMetricsCollector collector;
    collector.load_queue_backlog       = queue_sizes.load;
    collector.background_queue_backlog = queue_sizes.background;
    collector.completion_queue_backlog = queue_sizes.completion;
    collector.pool_type                = pool_type;
    collector.report_transfer          = false;
    collector.report_queue_backlog     = true;
    metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
}

void BlockTreeCacheMetricsReporter::reportWorkLoopQueueWait(Tier    source,
                                                            Tier    target,
                                                            int64_t latency_us) const noexcept {
    if (stage_metrics_ && latency_us >= 0)
        stage_metrics_->recordQueueWait(tierName(source), tierName(target), latency_us);
}

void BlockTreeCacheMetricsReporter::maybeReportCopyError(Tier                  source_tier,
                                                         Tier                  target_tier,
                                                         const TransferResult& result) const {
    if (!metrics_reporter_ || result.ok()) {
        return;
    }
    RtpLLMCacheCopyErrorMetricsCollector collector;
    collector.error_type     = cacheCopyErrorName(result.copy_error);
    collector.copy_direction = source_tier == Tier::DEVICE ? "FROM_GPU" :
                               target_tier == Tier::DEVICE ? "TO_GPU" :
                               target_tier == Tier::DISK   ? "TO_DISK" :
                                                             "FROM_DISK";
    metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheCopyErrorMetricsCollector>(nullptr, &collector);
}

int64_t BlockTreeCacheMetricsReporter::reportTransferStarted(CacheTransferOperation operation,
                                                             Tier                   source_tier,
                                                             Tier                   target_tier) {
    if (metrics_reporter_ == nullptr) {
        return 0;
    }
    const int direction_index = transferDirectionIndex(source_tier, target_tier);
    if (direction_index < 0) {
        return 0;
    }
    const int64_t begin_time_us   = currentTimeUs();
    const size_t  operation_index = transferOperationIndex(operation);
    const size_t  direction       = static_cast<size_t>(direction_index);
    const int64_t in_flight       = transfer_in_flight_[operation_index][direction].fetch_add(1) + 1;

    RtpLLMCacheTransferMetricsCollector collector;
    collector.operation          = cacheTransferOperationName(operation);
    collector.source_tier        = tierName(source_tier);
    collector.target_tier        = tierName(target_tier);
    collector.in_flight          = in_flight;
    collector.transfer_completed = false;

    try {
        metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
    } catch (...) {
        transfer_in_flight_[operation_index][direction].fetch_sub(1);
        throw;
    }
    return begin_time_us;
}

void BlockTreeCacheMetricsReporter::reportTransferFinished(
    CacheTransferOperation                 operation,
    Tier                                   source_tier,
    Tier                                   target_tier,
    size_t                                 descriptors_per_transfer,
    int64_t                                begin_time_us,
    bool                                   success,
    const std::vector<TransferDescriptor>& successful_descriptors,
    const std::vector<GroupSetPtr>&        group_sets) {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    const int direction_index = transferDirectionIndex(source_tier, target_tier);
    if (direction_index < 0) {
        return;
    }
    const size_t  operation_index = transferOperationIndex(operation);
    const int64_t in_flight =
        transfer_in_flight_[operation_index][static_cast<size_t>(direction_index)].fetch_sub(1) - 1;
    BlockTreeTransferBytes transfer_bytes;
    accumulateTransferBytes(successful_descriptors, group_sets, transfer_bytes);

    RtpLLMCacheTransferMetricsCollector collector;
    collector.operation                = cacheTransferOperationName(operation);
    collector.source_tier              = tierName(source_tier);
    collector.target_tier              = tierName(target_tier);
    collector.descriptors_per_transfer = static_cast<int64_t>(descriptors_per_transfer);
    collector.latency_us               = currentTimeUs() - begin_time_us;
    collector.in_flight                = in_flight;
    collector.success                  = success;
    collector.transfer_bytes.reserve(transfer_bytes.size());
    for (const auto& transfer_bytes_entry : transfer_bytes) {
        RtpLLMCacheTransferMetricsCollector::TransferBytesEntry entry;
        entry.pool_name      = transfer_bytes_entry.first.pool_name;
        entry.group_type     = metricCacheGroupTypeName(transfer_bytes_entry.first.group_type);
        entry.transfer_bytes = static_cast<int64_t>(transfer_bytes_entry.second);
        collector.transfer_bytes.push_back(std::move(entry));
    }
    metrics_reporter_->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferMetricsCollector>(nullptr, &collector);
}

const char* cacheTransferOperationName(CacheTransferOperation operation) {
    switch (operation) {
        case CacheTransferOperation::LOAD:
            return "load";
        case CacheTransferOperation::EVICT:
            return "evict";
    }
    return "unknown";
}

void BlockTreeCacheMetricsReporter::accumulateTransferBytes(const TransferDescriptor& desc,
                                                            const GroupSetPtr&        group_set,
                                                            BlockTreeTransferBytes&   transfer_bytes) const {
    if (desc.source_tier == Tier::DEVICE) {
        for (const DeviceBlockPoolPtr& pool : group_set->devicePools()) {
            transfer_bytes[{pool->poolName(), group_set->groupType()}] += pool->blockSizeBytes();
        }
    } else if (desc.source_tier == Tier::HOST) {
        transfer_bytes[{group_set->hostPool()->poolName(), group_set->groupType()}] +=
            group_set->hostPool()->blockSizeBytes();
    } else {
        transfer_bytes[{group_set->diskPool()->poolName(), group_set->groupType()}] +=
            group_set->diskPool()->blockSizeBytes();
    }
}

void BlockTreeCacheMetricsReporter::accumulateTransferBytes(const std::vector<TransferDescriptor>& descs,
                                                            const std::vector<GroupSetPtr>&        group_sets,
                                                            BlockTreeTransferBytes& transfer_bytes) const {
    for (const TransferDescriptor& desc : descs) {
        accumulateTransferBytes(desc, group_sets[desc.group_set_id], transfer_bytes);
    }
}

}  // namespace rtp_llm
