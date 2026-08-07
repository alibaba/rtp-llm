#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTreeTaskPool;

class EvictionTaskRunner {
public:
    using ExecuteTransferFn = BlockTreeEvictor::ExecuteTransferFn;
    using IsTierEnabledFn   = BlockTreeEvictor::IsTierEnabledFn;
    using SettledFn         = BlockTreeEvictor::SettledFn;
    using RemoteWriteFn     = BlockTreeEvictor::RemoteWriteFn;

    EvictionTaskRunner(ExecuteTransferFn              execute_transfer,
                       const std::vector<GroupSetPtr>& group_sets,
                       const BlockTransferDispatcher* transfer_dispatcher,
                       BlockTreeTaskPool*             task_pool,
                       BlockTreeCacheMetricsReporter& metrics_reporter,
                       std::mutex&                    mutex,
                       int                            memory_timeout_ms,
                       int                            disk_timeout_ms,
                       IsTierEnabledFn                is_tier_enabled,
                       SettledFn                      settled,
                       RemoteWriteFn                  remote_write);
    bool submitLocked(BlockTreeEvictor& evictor, TransferDescriptor& eviction_desc);

    BlockTreeEvictor::CopyResultSet performCopy(const BlockTreeEvictor::EvictionPlan& plan) const;
    BlockTreeEvictor::CopyResultSet runTransfer(const BlockTreeEvictor::EvictionPlan& plan) const;
    static bool                     buildTransferBatch(const BlockTreeEvictor::EvictionPlan& plan,
                                                       std::vector<TransferDescriptor>&      descriptors);

private:
    void runTask(BlockTreeEvictor& evictor, const BlockTreeEvictor::EvictionPlan& plan);
    bool executeTierCopy(const TransferDescriptor& eviction_desc) const;
    Tier normalizeTargetTier(Tier source_tier) const;
    static int
    transferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan, int memory_timeout_ms, int disk_timeout_ms);

    ExecuteTransferFn               execute_transfer_;
    const std::vector<GroupSetPtr>& group_sets_;
    const BlockTransferDispatcher* transfer_dispatcher_{nullptr};
    BlockTreeTaskPool*             task_pool_{nullptr};
    BlockTreeCacheMetricsReporter* metrics_reporter_{nullptr};
    std::mutex*                    mutex_{nullptr};
    int                            memory_timeout_ms_{0};
    int                            disk_timeout_ms_{0};
    IsTierEnabledFn                is_tier_enabled_;
    SettledFn                      settled_;
    RemoteWriteFn                  remote_write_;
};

}  // namespace rtp_llm
