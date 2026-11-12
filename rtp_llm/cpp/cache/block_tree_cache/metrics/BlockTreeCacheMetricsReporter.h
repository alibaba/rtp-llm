#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

struct BlockTreeInsertHardStopInfo;
struct BlockTreeQueueSizes;
class CacheStageMetrics;
class RtpLLMCacheEvictionMetricsCollector;

class BlockTreeCacheMetricsReporter final {
public:
    explicit BlockTreeCacheMetricsReporter(std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
                                           bool                                       enable_stage_metrics = true,
                                           std::chrono::milliseconds stage_interval = std::chrono::seconds(1));
    ~BlockTreeCacheMetricsReporter();
    bool enabled() const noexcept {
        return metrics_reporter_ != nullptr;
    }
    bool stageMetricsEnabled() const noexcept {
        return stage_metrics_ != nullptr;
    }
    void flushStageMetrics() const;
    void reportLockMetrics(const char* operation, double wait_latency_us, double hold_latency_us) const noexcept;

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
                                const std::vector<bool>&        detached = {}) const;
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
    void reportInsertHardStop(const BlockTreeInsertHardStopInfo& hard_stop) const;

    int64_t reportTransferStarted(CacheTransferOperation operation, Tier source_tier, Tier target_tier);
    // Report once after all batches/ranks of a logical copy have completed.
    void maybeReportCopyError(Tier source_tier, Tier target_tier, const TransferResult& result) const;
    void reportTransferFinished(CacheTransferOperation                 operation,
                                Tier                                   source_tier,
                                Tier                                   target_tier,
                                size_t                                 descriptors_per_transfer,
                                int64_t                                begin_time_us,
                                bool                                   success,
                                const std::vector<TransferDescriptor>& successful_descriptors,
                                const std::vector<GroupSetPtr>&        group_sets);
    void reportQueueWaitMetric(bool        callback,
                               const char* pool_type,
                               const char* operation,
                               Tier        source_tier,
                               Tier        target_tier,
                               int64_t     latency_us) const noexcept;
    void reportQueueBacklog(const BlockTreeQueueSizes& queue_sizes, const char* pool_type) const;
    // Buffered, bounded and nonblocking on the WorkLoop owner. Rejections before
    // admission are stage failures, not consumed queue tasks. Availability follows
    // the existing stage metrics switch; no buffer is created when it is off.
    void reportWorkLoopQueueWait(Tier source, Tier target, int64_t latency_us) const noexcept;

private:
    std::vector<CacheGroupType> eviction_metric_group_types_;

    struct PoolIOSample {
        size_t                                read_bytes{0};
        size_t                                write_bytes{0};
        std::chrono::steady_clock::time_point time;
    };

    std::unordered_map<const IBlockPool*, PoolIOSample> pool_io_samples_;

    static constexpr size_t kOperationCount = 2;
    static constexpr size_t kDirectionCount = 4;

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
    void                                       accumulateTransferBytes(const std::vector<TransferDescriptor>& descs,
                                                                       const std::vector<GroupSetPtr>&        group_sets,
                                                                       BlockTreeTransferBytes&                transfer_bytes) const;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
    std::unique_ptr<CacheStageMetrics>         stage_metrics_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
