#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class PerRankBlockTransferEngine {
public:
    PerRankBlockTransferEngine(std::vector<ComponentGroupPtr>                component_groups,
                               std::shared_ptr<const std::vector<Component>> components,
               DeviceHostCopyOptions                         device_host_options = {});
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    // Descriptor-based facade. Executes synchronously, returns completed handle.
    // TODO: change to async later
    virtual TransferHandle submit(const TransferDescriptor& desc);

private:
    TransferStatus execute(const TransferDescriptor& desc);
    TransferStatus validateRequest(const TransferDescriptor& desc, const ComponentGroup*& group) const;

    std::vector<ComponentGroupPtr>                component_groups_;
    std::shared_ptr<const std::vector<Component>> components_;
    std::atomic<uint64_t>                         next_request_id_{1};

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
