#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor {
public:
    explicit DeviceHostTransferExecutor(DeviceHostCopyOptions options = {});
    ~DeviceHostTransferExecutor() = default;

    TransferStatus execute(HostBufferView host, const TransferDescriptor& desc, const GroupSet& group_set);

private:
    std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
    generatePlan(const TransferDescriptor& desc, const GroupSet& group_set, HostBufferView host) const;

    DeviceHostCopyOptions                                options_;
    std::vector<std::unique_ptr<DeviceHostCopyStrategy>> strategies_;
};

}  // namespace rtp_llm
