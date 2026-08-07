#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

// D2Disk: staging lease -> D2H -> H2Disk. Disk2D: staging lease -> Disk2H -> H2D.
class DeviceDiskTransferExecutor {
public:
    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count);
    ~DeviceDiskTransferExecutor() = default;

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    TransferStatus execute(const TransferDescriptor& desc, const GroupSet& group);

private:
    DeviceHostTransferExecutor&               device_host_executor_;
    HostDiskTransferExecutor&                 host_disk_executor_;
    std::unique_ptr<HostStagingBlockPool>     staging_pool_;
};

}  // namespace rtp_llm
