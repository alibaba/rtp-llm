#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        DeviceHostCopyOptions    device_host_options = {});
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    // Currently executes synchronously on the calling thread and returns a non-null completed context.
    virtual std::shared_ptr<AsyncContext> submit(const TransferDescriptor& desc);

private:
    TransferStatus execute(const TransferDescriptor& desc);
    TransferStatus validateRequest(const TransferDescriptor& desc, const GroupSet*& group_set) const;

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
