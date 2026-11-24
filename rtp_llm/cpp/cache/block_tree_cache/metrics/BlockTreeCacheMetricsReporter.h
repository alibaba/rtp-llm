#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"
#include "rtp_llm/cpp/cache/metrics/CachePoolMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/ReuseMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/TransferMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

struct BlockTreeQueueSizes;
struct MockCacheHitDecision;
class CacheStageMetrics;
class RtpLLMCacheEvictionMetricsCollector;

using BlockTreePoolMetricsSnapshot = CachePoolMetrics;

// Owner-only logical admission/reuse. Physical records/bytes stay in their
// existing counters; neither TP endpoints nor FULL/SWA descriptors are summed.
struct MockCacheMetricsSnapshot {
    int64_t eligible_blocks{0};
    int64_t planned_memory_blocks{0};
    int64_t planned_disk_blocks{0};
    int64_t successful_memory_blocks{0};
    int64_t successful_disk_blocks{0};
    int64_t failed_blocks{0};
    int64_t planned_remote_blocks{0};
    int64_t successful_remote_blocks{0};
};

struct MockMemoryCacheMetricsSnapshot {
    int64_t eligible_blocks{0};
    int64_t planned_host_blocks{0};
    int64_t successful_h2d_blocks{0};
    int64_t failed_h2d_blocks{0};
    int64_t d2h_submitted_blocks{0};
    int64_t d2h_completed_blocks{0};
    int64_t d2h_failed_blocks{0};
};

struct LocalStoreSnapshot {
    uint64_t released_requests{0};
    uint64_t pending_requests{0};
    uint64_t abandoned_requests{0};
    uint64_t rejected_host_stores{0};
    uint64_t rejected_disk_stores{0};
};

// Physical destination records of accepted diagnostic disk stores. Completion
// means every copy in the task succeeded; it does not imply tree publication.
// Rejected admission is excluded, like the existing mock-memory D2H counters.
struct MockDiskWriteMetricsSnapshot {
    int64_t submitted_records{0};
    int64_t completed_records{0};
    int64_t failed_records{0};
};

class BlockTreeCacheMetricsReporter final {
public:
    explicit BlockTreeCacheMetricsReporter(std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
                                           bool                                       enable_stage_metrics = true,
                                           std::chrono::milliseconds stage_interval = std::chrono::seconds(1));
    ~BlockTreeCacheMetricsReporter();

    static CachePoolMetrics snapshotPool(Tier tier, const IBlockPool& pool);

    void                           startLocalStoreRequest();
    void                           finishLocalStoreRequest(bool released);
    void                           recordLocalStoreRejection(Tier tier);
    LocalStoreSnapshot             localStoreSnapshot() const;
    void                           updateMockMemorySource(const std::vector<GroupSetPtr>& group_sets);
    void                           recordMockMemoryD2HSubmitted(size_t host_records);
    void                           recordMockMemoryD2HFinished(size_t host_records, bool success);
    void                           reportMockMemoryMetrics() const;
    MockMemoryCacheMetricsSnapshot mockMemoryMetricsSnapshot() const;
    void                           recordMockDiskWriteSubmitted(size_t records);
    void                           recordMockDiskWriteFinished(size_t records, bool success);
    MockDiskWriteMetricsSnapshot   mockDiskWriteMetricsSnapshot() const;
    void                           updateMockCachePlan(const MockCacheHitDecision& decision);
    void recordMockCacheLoad(size_t memory_blocks, size_t disk_blocks, bool success, size_t remote_blocks = 0);
    MockCacheMetricsSnapshot mockCacheMetricsSnapshot() const;
    bool enabled() const noexcept {
        return metrics_reporter_ != nullptr;
    }
    bool stageMetricsEnabled() const noexcept {
        return stage_metrics_ != nullptr;
    }
    void flushStageMetrics() const;
    void reportUnfreezeMetrics(const FrozenTask::ExecutionInfo& execution, bool forced) const;
    void reportUnfreezeZeroMetrics(const std::vector<CachePoolMetrics>& pools) const;

    // Durations are per worker batch, except coordinator multi-rank stages.
    // Request/transfer IDs belong in the timeline, never in metric tags.
    void reportDiskLoadStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank  = -1,
                             const char* scope = "worker_batch") const noexcept;
    void reportTransferStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank,
                             const char* scope,
                             const char* operation,
                             const char* source_tier,
                             const char* target_tier) const noexcept;

    std::vector<CachePoolMetrics>
         collectPoolMetrics(const std::vector<GroupSetPtr>&       group_sets,
                            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
    void initEvictionMetricGroupTypes(const std::vector<GroupSetPtr>& group_sets);
    void reportEvictionMetrics(const EvictionVictimMetrics& metrics_data) const;
    void reportCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list) const;
    void reportEvictionFinished(const EvictionTransferTask&     task,
                                const std::vector<GroupSetPtr>& group_sets,
                                bool                            success,
                                const std::vector<bool>&        skip_eviction_metrics,
                                size_t                          cancelled_blocks) const;
    void reportEvictionFinished(const EvictionDropTask&         task,
                                const std::vector<GroupSetPtr>& group_sets,
                                const char*                     trigger_type) const;
    void reportEvictionDescriptor(const TransferDescriptor&       desc,
                                  const EvictionTimingMetrics&    timing,
                                  const std::vector<GroupSetPtr>& group_sets,
                                  const char*                     trigger_type,
                                  bool                            success) const;
    void reportEvictionTriggered(Tier source_tier, CacheGroupType group_type, bool force_drop) const;
    void reportEvictionBlocks(Tier           source_tier,
                              CacheGroupType group_type,
                              bool           force_drop,
                              size_t         required_blocks,
                              size_t         scheduled_blocks) const;

    void reportLoadJoin(size_t dependency_count) const;
    void reportLoadJoinWait(int64_t join_wait_latency_us) const;
    void reportInsertHardStop() const;
    void
    reportInvariantViolation(size_t group_set_id, const char* type, const char* reason, bool stable) const noexcept;

    int64_t reportTransferStarted(TransferOperation operation, Tier source_tier, Tier target_tier);
    // Report once after all batches/ranks of a logical copy have completed.
    void maybeReportCopyError(Tier source_tier, Tier target_tier, const TransferResult& result) const;
    void reportTransferFinished(TransferOperation               operation,
                                Tier                            source_tier,
                                Tier                            target_tier,
                                size_t                          descriptors_per_transfer,
                                int64_t                         begin_time_us,
                                bool                            success,
                                DescriptorView                  successful_descriptors,
                                const std::vector<GroupSetPtr>& group_sets);
    void reportQueueBacklog(const BlockTreeQueueSizes& queue_sizes, const char* pool_type) const;

private:
    std::vector<CacheGroupType> eviction_metric_group_types_;

    struct PoolIOSample {
        size_t                                read_bytes{0};
        size_t                                write_bytes{0};
        std::chrono::steady_clock::time_point time;
    };

    std::unordered_map<const IBlockPool*, PoolIOSample> pool_io_samples_;

    static constexpr size_t kOperationCount = 3;
    static constexpr size_t kDirectionCount = 5;

    static int transferDirectionIndex(Tier source_tier, Tier target_tier);
    void       reportEvictionMetrics(const TransferDescriptor&            desc,
                                     const std::vector<GroupSetPtr>&      group_sets,
                                     const char*                          trigger_type,
                                     RtpLLMCacheEvictionMetricsCollector& collector) const;
    void
    reportEvictionTrigger(Tier source_tier, CacheGroupType group_type, const char* trigger_type, int64_t count) const;
    void                                       accumulateTransferBytes(const TransferDescriptor& desc,
                                                                       const GroupSetPtr&        group_set,
                                                                       BlockTreeTransferBytes&   transfer_bytes) const;
    void                                       accumulateTransferBytes(DescriptorView                  descs,
                                                                       const std::vector<GroupSetPtr>& group_sets,
                                                                       BlockTreeTransferBytes&         transfer_bytes) const;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
    std::unique_ptr<CacheStageMetrics>         stage_metrics_;
    mutable std::mutex                         mock_memory_metrics_mutex_;
    RtpLLMMockMemoryCacheMetricsCollector      mock_memory_metrics_;
    LocalStoreSnapshot                         local_store_;
    MockDiskWriteMetricsSnapshot               mock_disk_write_metrics_;
    MockCacheMetricsSnapshot                   mock_cache_metrics_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
