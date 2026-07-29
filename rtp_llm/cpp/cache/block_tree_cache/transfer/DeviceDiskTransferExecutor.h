#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

using StagingPinMemoryFn = std::function<torch::Tensor(const torch::Tensor&)>;

// D2Disk: staging lease -> D2H -> H2Disk. Disk2D: staging lease -> Disk2H -> H2D.
class DeviceDiskTransferExecutor {
public:
    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count,
                               StagingPinMemoryFn              pin_memory = {});
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    TransferStatus execute(const TransferDescriptor& desc, const GroupSet& group);

private:
    class TransientHostStagingPool;

    DeviceHostTransferExecutor&               device_host_executor_;
    HostDiskTransferExecutor&                 host_disk_executor_;
    std::unique_ptr<TransientHostStagingPool> staging_pool_;
};

}  // namespace rtp_llm
