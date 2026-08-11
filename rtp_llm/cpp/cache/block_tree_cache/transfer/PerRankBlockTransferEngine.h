#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;
class BlockTreeTaskPool;
class TransferEndpointRegistry;

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        DeviceHostCopyOptions    device_host_options                = {},
                                        size_t                   device_disk_staging_block_count    = 4,
                                        size_t                   max_descriptors_per_transfer_batch = 64);
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    virtual std::shared_ptr<AsyncContext> submit(const TransferDescriptor& desc);
    virtual std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors);

private:
    TransferStatus     execute(const TransferDescriptor& desc, const GroupSet& group_set);
    TransferStatus     executeDirectBatch(const std::vector<TransferDescriptor>& descriptors,
                                          const std::vector<const GroupSet*>&    group_sets,
                                          const std::vector<HostBufferView>&     hosts);
    BlockTreeTaskPool* taskPoolForDirection(Tier source_tier, Tier target_tier) const;
    TransferStatus     validateRequest(const TransferDescriptor& desc, const GroupSet*& group_set) const;
    TransferStatus     validateDeviceBlocks(const TransferDescriptor& desc, const GroupSet& group_set) const;

    static HostBufferView resolveHostView(const GroupSet& group_set, BlockIdxType host_block);

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;

    std::unique_ptr<BlockTreeTaskPool> device_to_host_task_pool_;
    std::unique_ptr<BlockTreeTaskPool> host_to_device_task_pool_;
    std::unique_ptr<BlockTreeTaskPool> host_to_disk_task_pool_;
    std::unique_ptr<BlockTreeTaskPool> disk_to_host_task_pool_;

    std::shared_ptr<TransferEndpointRegistry> endpoint_registry_;

    size_t max_descriptors_per_transfer_batch_{64};
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
