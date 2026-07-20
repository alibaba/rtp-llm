#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngineLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor {
public:
    explicit DeviceHostTransferExecutor(DeviceHostCopyOptions options = {});
    ~DeviceHostTransferExecutor();

    CopyStatus execute(const TransferDescriptor& desc, const ResolvedGroupLayout& layout);

private:
    CopyStatus lowerAndExecute(const TransferDescriptor& desc, const ResolvedGroupLayout& layout, bool device_to_host);

    DeviceHostCopyPlan lowerPlan(const TransferDescriptor&  desc,
                                 const ResolvedGroupLayout& layout,
                                 bool                       device_to_host,
                                 CopyStatus&                out_status) const;

    CopyStatus executeStrategies(const DeviceHostCopyPlan& plan);

    static std::vector<DeviceHostCopyPlan> splitByDevice(const DeviceHostCopyPlan& plan);

    DeviceHostCopyOptions                                options_;
    std::vector<std::unique_ptr<DeviceHostCopyStrategy>> strategies_;
};

}  // namespace rtp_llm
