#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor {
public:
    explicit DeviceHostTransferExecutor(DeviceHostCopyOptions options = {});
    ~DeviceHostTransferExecutor();

    TransferStatus deviceToHost(const TransferDescriptor& desc, const GroupSet& group_set, HostBufferView host);
    TransferStatus hostToDevice(HostBufferView host, const TransferDescriptor& desc, const GroupSet& group_set);

private:
    TransferStatus lowerAndExecute(const TransferDescriptor& desc,
                                   const GroupSet&           group_set,
                                   bool                      device_to_host,
                                   HostBufferView            host);

    DeviceHostCopyPlan lowerPlan(const TransferDescriptor& desc,
                                 const GroupSet&           group_set,
                                 bool                      device_to_host,
                                 HostBufferView            host,
                                 TransferStatus&           out_status) const;

    TransferStatus executeStrategies(const DeviceHostCopyPlan& plan);

    static std::vector<DeviceHostCopyPlan> splitByDevice(const DeviceHostCopyPlan& plan);

    DeviceHostCopyOptions                                options_;
    std::vector<std::unique_ptr<DeviceHostCopyStrategy>> strategies_;
};

}  // namespace rtp_llm
