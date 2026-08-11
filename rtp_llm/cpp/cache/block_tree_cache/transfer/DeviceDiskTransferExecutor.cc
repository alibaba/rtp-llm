#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <chrono>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"

namespace rtp_llm {

namespace {

// Internal wait budget for a staging lease; not user-configurable.
constexpr std::chrono::milliseconds kStagingAcquireTimeout{1000};

}  // namespace

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count):
    device_host_executor_(device_host_executor), host_disk_executor_(host_disk_executor) {
    size_t max_payload_bytes = 0;
    for (const auto& group_set : group_sets) {
        max_payload_bytes = std::max(max_payload_bytes, group_set->payloadBytes());
    }
    const size_t alignment   = HostStagingBlockPool::kAlignment;
    const size_t stride_bytes = ((max_payload_bytes + alignment - 1) / alignment) * alignment;
    staging_pool_             = std::make_unique<HostStagingBlockPool>(staging_block_count, stride_bytes);
}

TransferStatus DeviceDiskTransferExecutor::execute(const TransferDescriptor& desc, const GroupSet& group) {
    const bool device_to_disk = desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::DISK;
    auto lease = staging_pool_->mallocWithBackoff(kStagingAcquireTimeout);
    if (!lease.has_value()) {
        return TransferStatus::RESOURCE_EXHAUSTED;
    }
    const HostBufferView host = lease->blockBuffer(group.payloadBytes());

    if (device_to_disk) {
        const TransferStatus stage_one = device_host_executor_.execute(host, desc, group);
        if (stage_one != TransferStatus::OK) {
            return stage_one;
        }
        return host_disk_executor_.execute(host, desc, group);
    }

    const TransferStatus stage_one = host_disk_executor_.execute(host, desc, group);
    if (stage_one != TransferStatus::OK) {
        return stage_one;
    }
    return device_host_executor_.execute(host, desc, group);
}

}  // namespace rtp_llm
