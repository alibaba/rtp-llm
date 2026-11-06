#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceDiskTransferExecutor.h"

namespace rtp_llm {

// Owns the synchronous queues, strategies and staging waiter lifecycle.
class LegacyTransferEngine {
public:
    LegacyTransferEngine(std::vector<GroupSetPtr>                       groups,
                         bool                                           enable_disk_cache,
                         DeviceHostCopyOptions                          device_host_options,
                         size_t                                         device_disk_staging_block_count,
                         size_t                                         max_descriptors_per_batch,
                         size_t                                         transfer_worker_count,
                         size_t                                         transfer_queue_max_size,
                         std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                         int64_t                                        world_rank);
    ~LegacyTransferEngine();
    std::shared_ptr<TransferAsyncContext> execute(TransferTask task, std::vector<const GroupSet*> groups);
    void                                  stopAdmission();
    void                                  drain();
    void                                  waitForIdle();
    void                                  cancelPendingStagingTransfers();
    BlockTreeQueueSizes                   queueSizes() const;

private:
    std::vector<GroupSetPtr>                    group_sets_;
    std::unique_ptr<BlockTreeTaskPool>          load_transfer_task_pool_, transfer_task_pool_;
    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;
};

}  // namespace rtp_llm
