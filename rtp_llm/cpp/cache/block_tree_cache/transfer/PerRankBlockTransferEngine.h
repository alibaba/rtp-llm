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

namespace block_tree_cache_test {
class BlockTreeCacheTestPeer;
}

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        bool                     enable_disk_cache                         = false,
                                        DeviceHostCopyOptions    device_host_options                       = {},
                                        size_t                   device_disk_staging_block_count           = 128,
                                        size_t                   max_descriptors_per_batch                  = 8,
                                        size_t                   transfer_worker_count                     = 4,
                                        size_t                   transfer_queue_max_size                   = 10000,
                                        std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter    = nullptr);
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    virtual std::shared_ptr<AsyncContext> execute(TransferTask task);
    void                                  cancelPendingStagingTransfers();
    BlockTreeQueueSizes                   queueSizes() const;

    size_t transferWorkerCount() const {
        return transfer_worker_count_;
    }

private:
    friend class block_tree_cache_test::BlockTreeCacheTestPeer;

    static HostBufferView resolveHostView(const GroupSet& group_set, BlockIdxType host_block);

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<BlockTreeTaskPool>          transfer_task_pool_;
    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;
    size_t                                      transfer_worker_count_{4};
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
