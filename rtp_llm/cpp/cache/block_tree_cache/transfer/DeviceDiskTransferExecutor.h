#pragma once

#include <cstddef>
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
                               BlockTreeTaskPool&              transfer_task_pool);
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    std::shared_ptr<AsyncContext> execute(const std::vector<TransferDescriptor>& descriptors,
                                          const std::vector<const GroupSet*>&    group_sets);

    std::shared_ptr<AsyncContext> executeDeviceToDisk(const TransferDescriptor& descriptor,
                                                      const GroupSet&           group_set);

    void cancelPendingTransfers();

private:
    HostStagingBlockPool* stagingPool(CacheGroupType group_type) const;
    size_t                batchCapacity(CacheGroupType group_type) const;

    DeviceHostTransferExecutor&          device_host_executor_;
    HostDiskTransferExecutor&            host_disk_executor_;
    BlockTreeTaskPool&                    transfer_task_pool_;
    std::unique_ptr<HostStagingBlockPool> full_staging_pool_;
    std::unique_ptr<HostStagingBlockPool> swa_staging_pool_;
    size_t                                full_batch_capacity_{0};
    size_t                                swa_batch_capacity_{0};
};

}  // namespace rtp_llm
