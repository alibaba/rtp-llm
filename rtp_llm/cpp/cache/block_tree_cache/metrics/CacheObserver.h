#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/metrics/CachePoolMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/ReuseMetrics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

namespace rtp_llm {

class GroupSet;
using GroupSetPtr = std::shared_ptr<GroupSet>;
struct BlockTreeQueueSizes;
struct EvictionTransferTask;
struct EvictionDropTask;

// Optional observer for the native workflows. The engine owns metric export;
// cache admission, settlement and physical holds do not depend on an observer.
class CacheObserver {
public:
    virtual ~CacheObserver() = default;
    virtual bool enabled() const noexcept {
        return false;
    }
    virtual bool stageMetricsEnabled() const noexcept {
        return false;
    }
    virtual void flushStageMetrics() const {}
    virtual void reportUnfreezeMetrics(const FrozenTask::ExecutionInfo&, bool) const {}
    virtual void reportUnfreezeZeroMetrics(const std::vector<CachePoolMetrics>&) const {}
    virtual void
    reportDiskLoadStage(const char*, int64_t, bool, int64_t = -1, const char* = "worker_batch") const noexcept {}
    virtual void reportTransferStage(
        const char*, int64_t, bool, int64_t, const char*, const char*, const char*, const char*) const noexcept {}
    virtual std::vector<CachePoolMetrics>
    collectPoolMetrics(const std::vector<GroupSetPtr>&,
                       std::chrono::steady_clock::time_point = std::chrono::steady_clock::now()) {
        return {};
    }
    virtual void initEvictionMetricGroupTypes(const std::vector<GroupSetPtr>&) {}
    virtual void reportEvictionMetrics(const EvictionVictimMetrics&) const {}
    virtual void reportCacheReuseTimeMetrics(const std::vector<BlockTreeCacheReuseTimeMetrics>&) const {}
    virtual void reportEvictionFinished(
        const EvictionTransferTask&, const std::vector<GroupSetPtr>&, bool, const std::vector<bool>&, size_t) const {}
    virtual void reportEvictionFinished(const EvictionDropTask&, const std::vector<GroupSetPtr>&, const char*) const {}
    virtual void reportEvictionDescriptor(const TransferDescriptor&,
                                          const EvictionTimingMetrics&,
                                          const std::vector<GroupSetPtr>&,
                                          const char*,
                                          bool) const {}
    virtual void reportEvictionTriggered(Tier, CacheGroupType, bool) const {}
    virtual void reportEvictionBlocks(Tier, CacheGroupType, bool, size_t, size_t) const {}
    virtual void reportLoadJoin(size_t) const {}
    virtual void reportLoadJoinWait(int64_t) const {}
    virtual void reportInsertHardStop() const {}
    virtual void reportInvariantViolation(size_t, const char*, const char*, bool) const noexcept {}
    virtual int64_t reportTransferStarted(TransferOperation, Tier, Tier) {
        return 0;
    }
    virtual void maybeReportCopyError(Tier, Tier, const TransferResult&) const {}
    virtual void reportTransferFinished(
        TransferOperation, Tier, Tier, size_t, int64_t, bool, DescriptorView, const std::vector<GroupSetPtr>&) {}
    virtual void reportQueueBacklog(const BlockTreeQueueSizes&, const char*) const {}
};

}  // namespace rtp_llm
