#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class DeviceDiskTransferExecutor {
public:
    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count,
                               size_t                          transfer_worker_count = 1);
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    std::shared_ptr<AsyncContext> execute(const std::vector<TransferDescriptor>& descriptors,
                                          const std::vector<const GroupSet*>&    group_sets);

    TransferStatus execute(const TransferDescriptor& descriptor, const GroupSet& group_set);

    void resetBenchmarkTimingStats();
    std::atomic<int64_t> benchmark_queue_wait_ns_{0};
    std::atomic<int64_t> benchmark_executor_ns_{0};
    std::atomic<size_t>  benchmark_executor_count_{0};

private:
    HostStagingBlockPool* stagingPool(CacheGroupType group_type) const;
    BlockTreeTaskPool*    taskPool(CacheGroupType group_type) const;
    size_t                batchCapacity(CacheGroupType group_type) const;

    DeviceHostTransferExecutor&        device_host_executor_;
    HostDiskTransferExecutor&          host_disk_executor_;
    std::unique_ptr<HostStagingBlockPool> full_staging_pool_;
    std::unique_ptr<HostStagingBlockPool> swa_staging_pool_;
    std::unique_ptr<BlockTreeTaskPool> full_task_pool_;
    std::unique_ptr<BlockTreeTaskPool> swa_task_pool_;
    size_t                            full_batch_capacity_{0};
    size_t                            swa_batch_capacity_{0};
};

}  // namespace rtp_llm
