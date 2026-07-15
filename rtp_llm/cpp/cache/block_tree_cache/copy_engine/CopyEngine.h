#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngineLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

// Internal executor implementation boundary; kept out of this public header.
class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class CopyEngine {
public:
    CopyEngine(const std::vector<ComponentGroupPtr>& component_groups,
               const std::vector<Component>&         components,
               DeviceHostCopyOptions                 device_host_options = {});
    CopyEngine() = delete;
    virtual ~CopyEngine();

    // Descriptor-based facade. Executes synchronously, returns completed handle.
    // TODO: change to async later
    virtual TransferHandle submit(const TransferDescriptor& desc);

    static size_t computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots);

private:
    CopyStatus execute(const TransferDescriptor& desc);

    void buildGroupLayouts(const std::vector<ComponentGroupPtr>& component_groups,
                           const std::vector<Component>&         components);

    static bool isDeviceHostTransfer(Tier source_tier, Tier target_tier);

    std::vector<ResolvedGroupLayout> group_layouts_;
    std::atomic<uint64_t>            next_request_id_{1};

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
};

using CopyEnginePtr = std::shared_ptr<CopyEngine>;

}  // namespace rtp_llm
