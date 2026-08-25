#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        DeviceHostCopyOptions    device_host_options             = {},
                                        size_t                   device_disk_staging_block_count = 4,
                                        size_t                   max_descriptors_per_batch       = 64,
                                        size_t                   transfer_worker_count           = 1);
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    virtual std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors);
    void                                  cancelPendingStagingTransfers();
    void                                  stopAdmission();
    void                                  shutdown();

    size_t  transferWorkerCount() const {
        return transfer_worker_count_;
    }

private:
    TransferStatus execute(const std::vector<HostBufferView>&       hosts,
                           const std::vector<TransferDescriptor>& descriptors,
                           const std::vector<const GroupSet*>&    group_sets) const;
    static HostBufferView resolveHostView(const GroupSet& group_set, BlockIdxType host_block);

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<BlockTreeTaskPool>          transfer_task_pool_;
    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;  // nullable; present when a disk pool exists
    size_t                                      max_descriptors_per_batch_{64};
    size_t                                      transfer_worker_count_{1};
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
