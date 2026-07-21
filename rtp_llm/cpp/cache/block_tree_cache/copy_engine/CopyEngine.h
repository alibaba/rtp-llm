#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class CopyEngine {
public:
    CopyEngine(std::vector<ComponentGroupPtr>                component_groups,
               std::shared_ptr<const std::vector<Component>> component_registry,
               DeviceHostCopyOptions                         device_host_options = {});
    CopyEngine() = delete;
    virtual ~CopyEngine();

    // Descriptor-based facade. Executes synchronously, returns completed handle.
    // TODO: change to async later
    virtual TransferHandle submit(const TransferDescriptor& desc);

private:
    CopyStatus execute(const TransferDescriptor& desc);
    CopyStatus validateRequest(const TransferDescriptor& desc, const ComponentGroup*& group) const;

    std::vector<ComponentGroupPtr>                component_groups_;
    std::shared_ptr<const std::vector<Component>> component_registry_;
    std::atomic<uint64_t>                         next_request_id_{1};

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
};

using CopyEnginePtr = std::shared_ptr<CopyEngine>;

}  // namespace rtp_llm
