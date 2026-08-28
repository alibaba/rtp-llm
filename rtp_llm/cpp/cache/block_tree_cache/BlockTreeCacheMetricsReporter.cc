#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <utility>

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

int metricTierIndex(Tier tier) {
    if (tier == Tier::DEVICE) {
        return 0;
    }
    if (tier == Tier::HOST) {
        return 1;
    }
    if (tier == Tier::DISK) {
        return 2;
    }
    return -1;
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

BlockTreePoolMetricsSnapshot makePoolMetricsSnapshot(Tier tier, const IBlockPool& pool) {
    BlockTreePoolMetricsSnapshot snapshot;
    snapshot.tier                   = tier;
    snapshot.pool_name              = pool.poolName();
    snapshot.block_size_bytes       = pool.blockSizeBytes();
    snapshot.total_blocks           = pool.totalBlocksNum();
    snapshot.free_blocks            = pool.freeBlocksNum();
    snapshot.used_blocks            = snapshot.total_blocks - snapshot.free_blocks;
    snapshot.active_blocks          = pool.activeBlocksNum();
    snapshot.available_blocks       = pool.availableBlocksNum();
    snapshot.block_cache_ref_blocks = pool.referencedBlocksNum(BlockTreeRefType::CACHE);
    snapshot.load_ref_blocks        = pool.referencedBlocksNum(BlockTreeRefType::LOAD);
    snapshot.eviction_ref_blocks    = pool.referencedBlocksNum(BlockTreeRefType::EVICTION);
    snapshot.store_ref_blocks       = pool.referencedBlocksNum(BlockTreeRefType::STORE);
    return snapshot;
}

BlockTreePoolMetricsSnapshot makeDevicePoolMetricsSnapshot(const DeviceBlockPool& pool) {
    BlockTreePoolMetricsSnapshot snapshot = makePoolMetricsSnapshot(Tier::DEVICE, pool);
    snapshot.request_ref_blocks           = pool.referencedBlocksNum();
    return snapshot;
}

}  // namespace

void BlockTreeCacheMetricsReporter::setMetricsReporter(
    const std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
}

bool BlockTreeCacheMetricsReporter::enabled() const {
    return metrics_reporter_ != nullptr;
}

std::vector<BlockTreePoolMetricsSnapshot>
BlockTreeCacheMetricsReporter::collectPoolMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets) const {
    std::unordered_set<const IBlockPool*>     reported_device_pools;
    std::vector<BlockTreePoolMetricsSnapshot> snapshots;
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
            snapshots.push_back(makeDevicePoolMetricsSnapshot(*pool));
        }

        const std::shared_ptr<HostBlockPool> host_pool = group_set->hostPool();
        if (host_pool != nullptr) {
            snapshots.push_back(makePoolMetricsSnapshot(Tier::HOST, *host_pool));
        }

        const std::shared_ptr<BlockTreeDiskBlockPool> disk_pool = group_set->diskPool();
        if (disk_pool != nullptr) {
            snapshots.push_back(makePoolMetricsSnapshot(Tier::DISK, *disk_pool));
        }
    }
    return snapshots;
}

std::vector<BlockTreeEvictableMetricsSnapshot>
BlockTreeCacheMetricsReporter::collectEvictableMetricsSnapshots(const std::vector<GroupSetPtr>& group_sets,
                                                                const BlockTreeEvictor&         evictor) const {
    std::array<std::array<size_t, kMetricGroupTypes.size()>, kMetricTiers.size()> candidate_counts{};
    std::array<bool, kMetricGroupTypes.size()>                                    group_type_present{};
    for (const GroupSetPtr& group_set : group_sets) {
        if (group_set == nullptr) {
            continue;
        }
        const int group_type_index = metricGroupTypeIndex(group_set->groupType());
        if (group_type_index < 0) {
            continue;
        }
        group_type_present[static_cast<size_t>(group_type_index)] = true;
        for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
            candidate_counts[tier_index][static_cast<size_t>(group_type_index)] +=
                evictor.candidateCount(group_set->groupSetId(), kMetricTiers[tier_index]);
        }
    }

    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    snapshots.reserve(kMetricTiers.size() * kMetricGroupTypes.size());
    for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
        for (size_t group_type_index = 0; group_type_index < kMetricGroupTypes.size(); ++group_type_index) {
            if (!group_type_present[group_type_index]) {
                continue;
            }
            BlockTreeEvictableMetricsSnapshot snapshot;
            snapshot.tier                      = kMetricTiers[tier_index];
            snapshot.group_type                = kMetricGroupTypes[group_type_index];
            snapshot.evictable_candidate_count = candidate_counts[tier_index][group_type_index];
            snapshots.push_back(snapshot);
        }
    }
    return snapshots;
}

void BlockTreeCacheMetricsReporter::reportEvictableCandidateCount(
    const std::vector<BlockTreeEvictableMetricsSnapshot>& snapshots) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    for (const BlockTreeEvictableMetricsSnapshot& snapshot : snapshots) {
        RtpLLMCacheEvictionMetricsCollector collector;
        collector.source_tier               = tierName(snapshot.tier);
        collector.group_type                = metricCacheGroupTypeName(snapshot.group_type);
        collector.evictable_candidate_count = static_cast<int64_t>(snapshot.evictable_candidate_count);
        collector.report_evictable          = true;
        metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
        reportEvictionTrigger(snapshot.tier, snapshot.group_type, "watermark", 0);
        reportEvictionTrigger(snapshot.tier, snapshot.group_type, "force_drop", 0);
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionTriggered(Tier           source_tier,
                                                            CacheGroupType group_type,
                                                            bool           force_drop) const {
    reportEvictionTrigger(source_tier, group_type, force_drop ? "force_drop" : "watermark", 1);
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

std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> BlockTreeCacheMetricsReporter::collectCacheReuseTimeMetrics(
    const std::vector<BlockTreeCacheReuseTimeSample>& samples) const {
    struct Accumulator {
        int64_t reuse_interval_sum_us{0};
        int64_t reuse_interval_max_us{0};
        int64_t entry_age_sum_us{0};
        int64_t entry_age_max_us{0};
        size_t  count{0};
    };

    std::array<std::array<Accumulator, kMetricGroupTypes.size()>, kMetricTiers.size()> accumulators{};
    for (const BlockTreeCacheReuseTimeSample& sample : samples) {
        const int tier_index       = metricTierIndex(sample.tier);
        const int group_type_index = metricGroupTypeIndex(sample.group_type);
        if (tier_index < 0) {
            continue;
        }
        assert(static_cast<size_t>(tier_index) < kMetricTiers.size());
        assert(group_type_index >= 0 && static_cast<size_t>(group_type_index) < kMetricGroupTypes.size());
        const int64_t reuse_interval_us = sample.access_time_us - sample.last_access_time_us;
        const int64_t entry_age_us      = sample.access_time_us - sample.insert_time_us;
        Accumulator& accumulator = accumulators[static_cast<size_t>(tier_index)][static_cast<size_t>(group_type_index)];
        accumulator.reuse_interval_sum_us += reuse_interval_us;
        accumulator.reuse_interval_max_us = std::max(accumulator.reuse_interval_max_us, reuse_interval_us);
        accumulator.entry_age_sum_us += entry_age_us;
        accumulator.entry_age_max_us = std::max(accumulator.entry_age_max_us, entry_age_us);
        ++accumulator.count;
    }

    std::vector<BlockTreeCacheReuseTimeMetricsSnapshot> snapshots;
    snapshots.reserve(kMetricTiers.size() * kMetricGroupTypes.size());
    for (size_t tier_index = 0; tier_index < kMetricTiers.size(); ++tier_index) {
        for (size_t group_type_index = 0; group_type_index < kMetricGroupTypes.size(); ++group_type_index) {
            const Accumulator& accumulator = accumulators[tier_index][group_type_index];
            if (accumulator.count == 0) {
                continue;
            }
            BlockTreeCacheReuseTimeMetricsSnapshot snapshot;
            snapshot.tier       = kMetricTiers[tier_index];
            snapshot.group_type = kMetricGroupTypes[group_type_index];
            snapshot.reuse_interval_avg_ms =
                accumulator.reuse_interval_sum_us / static_cast<int64_t>(accumulator.count) / 1000;
            snapshot.reuse_interval_max_ms = accumulator.reuse_interval_max_us / 1000;
            snapshot.hit_entry_age_avg_ms =
                accumulator.entry_age_sum_us / static_cast<int64_t>(accumulator.count) / 1000;
            snapshot.hit_entry_age_max_ms = accumulator.entry_age_max_us / 1000;
            snapshots.push_back(snapshot);
        }
    }
    return snapshots;
}

void BlockTreeCacheMetricsReporter::reportCacheReuseTimeMetrics(
    const std::vector<BlockTreeCacheReuseTimeMetricsSnapshot>& snapshots) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }
    for (const BlockTreeCacheReuseTimeMetricsSnapshot& snapshot : snapshots) {
        RtpLLMCacheReuseMetricsCollector collector;
        collector.reuse_interval_avg_ms     = snapshot.reuse_interval_avg_ms;
        collector.reuse_interval_max_ms     = snapshot.reuse_interval_max_ms;
        collector.hit_entry_age_avg_ms      = snapshot.hit_entry_age_avg_ms;
        collector.hit_entry_age_max_ms      = snapshot.hit_entry_age_max_ms;
        collector.report_reuse_time_metrics = true;
        kmonitor::MetricsTags tags("tier", tierName(snapshot.tier));
        tags.AddTag("group_type", metricCacheGroupTypeName(snapshot.group_type));
        metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(&tags, &collector);
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionFinished(const EvictionTask&             task,
                                                           const EvictionTaskResult&       task_result,
                                                           const std::vector<GroupSetPtr>& group_sets) const {
    if (metrics_reporter_ == nullptr) {
        return;
    }

    const int64_t finish_time_us = currentTimeUs();
    if (task_result.primary_success) {
        reportEvictionTransfer(task.primary_desc, task.primary_timing, group_sets, finish_time_us, true);
        for (size_t desc_index = 0; desc_index < task.dependent_prune_descs.size(); ++desc_index) {
            reportEvictionTransfer(task.dependent_prune_descs[desc_index],
                                   task.dependent_prune_timings[desc_index],
                                   group_sets,
                                   finish_time_us,
                                   false);
        }
    }
    for (size_t desc_index = 0; desc_index < task.cascade_descs.size(); ++desc_index) {
        if (desc_index < task_result.cascade_success.size() && task_result.cascade_success[desc_index]) {
            reportEvictionTransfer(
                task.cascade_descs[desc_index], task.cascade_timings[desc_index], group_sets, finish_time_us, false);
        }
    }
}

void BlockTreeCacheMetricsReporter::reportEvictionTransfer(const TransferDescriptor&       desc,
                                                           const EvictionTimingSnapshot&   timing,
                                                           const std::vector<GroupSetPtr>& group_sets,
                                                           int64_t                         finish_time_us,
                                                           bool report_candidate_times) const {
    const size_t group_set_id = desc.group_set_id;
    if (group_set_id >= group_sets.size()) {
        return;
    }
    const GroupSetPtr& group_set = group_sets[group_set_id];
    if (group_set == nullptr || group_set->groupSetId() != desc.group_set_id) {
        return;
    }

    RtpLLMCacheEvictionMetricsCollector collector;
    collector.source_tier                = tierName(desc.source_tier);
    collector.target_tier                = tierName(desc.target_tier);
    collector.group_type                 = metricCacheGroupTypeName(group_set->groupType());
    collector.report_eviction            = true;
    collector.tier_residence_time_ms     = (finish_time_us - timing.tier_enter_time_us) / 1000;
    collector.report_tier_residence_time = true;
    if (report_candidate_times) {
        collector.candidate_idle_time_ms     = (timing.selected_time_us - timing.last_access_time_us) / 1000;
        collector.candidate_age_ms           = (timing.selected_time_us - timing.insert_time_us) / 1000;
        collector.report_candidate_idle_time = true;
        collector.report_candidate_age       = true;
    }
    metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(nullptr, &collector);
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
    if (source_tier == Tier::DEVICE && target_tier == Tier::DISK) {
        return 4;
    }
    return -1;
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
    size_t                                 descriptor_count,
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
    collector.operation        = cacheTransferOperationName(operation);
    collector.source_tier      = tierName(source_tier);
    collector.target_tier      = tierName(target_tier);
    collector.descriptor_count = static_cast<int64_t>(descriptor_count);
    collector.latency_us       = currentTimeUs() - begin_time_us;
    collector.in_flight        = in_flight;
    collector.success          = success;
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
        case CacheTransferOperation::STORE:
            return "store";
    }
    return "unknown";
}

void BlockTreeCacheMetricsReporter::reportStorePublish(Tier   target_tier,
                                                       size_t accepted_blocks,
                                                       size_t duplicate_blocks) const {
    reportStoreBlocks(target_tier, "accepted", accepted_blocks);
    reportStoreBlocks(target_tier, "duplicate", duplicate_blocks);
}

void BlockTreeCacheMetricsReporter::reportStoreBlocks(Tier target_tier, const char* outcome, size_t block_count) const {
    if (metrics_reporter_ == nullptr || block_count == 0) {
        return;
    }
    RtpLLMTierStoreMetricsCollector collector;
    collector.target_tier = tierName(target_tier);
    collector.outcome     = outcome;
    collector.block_count = static_cast<int64_t>(block_count);
    metrics_reporter_->report<RtpLLMTierStoreMetrics, RtpLLMTierStoreMetricsCollector>(nullptr, &collector);
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
