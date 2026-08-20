#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        DeviceHostCopyOptions    device_host_options             = {},
                                        size_t                   device_disk_staging_block_count = 4,
                                        size_t                   max_descriptors_per_batch       = 64,
                                        size_t                   transfer_worker_count           = 1);
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine() = default;

    virtual std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors);

    void resetBenchmarkTimingStats();
    int64_t benchmarkQueueWaitNs() const;
    int64_t benchmarkExecutorNs() const;
    size_t  benchmarkExecutorCount() const;
    std::atomic<int64_t> benchmark_queue_wait_ns_{0};
    std::atomic<int64_t> benchmark_executor_ns_{0};
    std::atomic<size_t>  benchmark_executor_count_{0};

private:
    TransferStatus execute(const std::vector<HostBufferView>&       hosts,
                           const std::vector<TransferDescriptor>& descriptors,
                           const std::vector<const GroupSet*>&    group_sets) const;
    BlockTreeTaskPool* taskPoolForDirection(Tier source, Tier target) const;
    static HostBufferView resolveHostView(const GroupSet& group_set, BlockIdxType host_block);

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;  // nullable; present when a disk pool exists
    std::unique_ptr<BlockTreeTaskPool>          device_to_host_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>          host_to_device_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>          host_to_disk_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>          disk_to_host_task_pool_;
    size_t                                      max_descriptors_per_batch_{64};
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
