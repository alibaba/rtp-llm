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

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"
#include "rtp_llm/cpp/cache/metrics/CachePoolMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/ReuseMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/TransferMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace kmonitor {
class MetricsReporter;
}

namespace rtp_llm {

struct BlockTreeQueueSizes;
class CacheStageMetrics;
class RtpLLMCacheEvictionMetricsCollector;

class BlockTreeCacheMetricsReporter final: public CacheObserver {
public:
    explicit BlockTreeCacheMetricsReporter(std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter,
                                           bool                                       enable_stage_metrics = true,
                                           std::chrono::milliseconds stage_interval = std::chrono::seconds(1));
    ~BlockTreeCacheMetricsReporter() override;
    bool enabled() const noexcept override {
        return metrics_reporter_ != nullptr;
    }
    bool stageMetricsEnabled() const noexcept override {
        return stage_metrics_ != nullptr;
    }
    void flushStageMetrics() const override;
    void reportUnfreezeMetrics(const FrozenTask::ExecutionInfo& execution, bool forced) const override;
    void reportUnfreezeZeroMetrics(const std::vector<CachePoolMetrics>& pools) const override;

    // Durations are per worker batch, except coordinator multi-rank stages.
    // Request/transfer IDs belong in the timeline, never in metric tags.
    void reportDiskLoadStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank  = -1,
                             const char* scope = "worker_batch") const noexcept override;
    void reportTransferStage(const char* stage,
                             int64_t     latency_us,
                             bool        success,
                             int64_t     rank,
                             const char* scope,
                             const char* operation,
                             const char* source_tier,
                             const char* target_tier) const noexcept override;

    std::vector<CachePoolMetrics>
         collectPoolMetrics(const std::vector<GroupSetPtr>&       group_sets,
                            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) override;
    void initEvictionMetricGroupTypes(const std::vector<GroupSetPtr>& group_sets) override;
    void reportEvictionMetrics(const EvictionVictimMetrics& metrics_data) const override;
    void reportCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeMetrics>& metrics_list) const override;
    void reportEvictionFinished(const EvictionTransferTask&     task,
                                const std::vector<GroupSetPtr>& group_sets,
                                bool                            success,
                                const std::vector<bool>&        skip_eviction_metrics,
                                size_t                          cancelled_blocks) const override;
    void reportEvictionFinished(const EvictionDropTask&         task,
                                const std::vector<GroupSetPtr>& group_sets,
                                const char*                     trigger_type) const override;
    void reportEvictionDescriptor(const TransferDescriptor&       desc,
                                  const EvictionTimingMetrics&    timing,
                                  const std::vector<GroupSetPtr>& group_sets,
                                  const char*                     trigger_type,
                                  bool                            success) const override;
    void reportEvictionTriggered(Tier source_tier, CacheGroupType group_type, bool force_drop) const override;
    void reportEvictionBlocks(Tier           source_tier,
                              CacheGroupType group_type,
                              bool           force_drop,
                              size_t         required_blocks,
                              size_t         scheduled_blocks) const override;

    void reportLoadJoin(size_t dependency_count) const override;
    void reportLoadJoinWait(int64_t join_wait_latency_us) const override;
    void reportInsertHardStop() const override;
    void
    reportInvariantViolation(size_t group_set_id, const char* type, const char* reason, bool stable) const noexcept override;

    int64_t reportTransferStarted(TransferOperation operation, Tier source_tier, Tier target_tier) override;
    // Report once after all batches/ranks of a logical copy have completed.
    void maybeReportCopyError(Tier source_tier, Tier target_tier, const TransferResult& result) const override;
    void reportTransferFinished(TransferOperation               operation,
                                Tier                            source_tier,
                                Tier                            target_tier,
                                size_t                          descriptors_per_transfer,
                                int64_t                         begin_time_us,
                                bool                            success,
                                DescriptorView                  successful_descriptors,
                                const std::vector<GroupSetPtr>& group_sets) override;
    void reportQueueBacklog(const BlockTreeQueueSizes& queue_sizes, const char* pool_type) const override;

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
    void                                       accumulateTransferBytes(DescriptorView                  descs,
                                                                       const std::vector<GroupSetPtr>& group_sets,
                                                                       BlockTreeTransferBytes&         transfer_bytes) const;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
    std::unique_ptr<CacheStageMetrics>         stage_metrics_;
    std::array<std::array<std::atomic<int64_t>, kDirectionCount>, kOperationCount> transfer_in_flight_{};
};

}  // namespace rtp_llm
