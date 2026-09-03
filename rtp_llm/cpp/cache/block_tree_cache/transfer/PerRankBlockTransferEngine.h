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

class BlockTreeCacheMetricsReporter;

namespace block_tree_cache_test {
class BlockTreeCacheTestPeer;
}

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        DeviceHostCopyOptions    device_host_options                       = {},
                                        size_t                   device_disk_staging_block_count           = 4,
                                        size_t                   max_device_host_descriptors_per_batch     = 8,
                                        size_t                   transfer_worker_count                     = 4,
                                        size_t                   max_non_device_host_descriptors_per_batch = 16,
                                        size_t                   transfer_queue_max_size                   = 10000,
                                        int                      host_queue_wait_timeout_ms                = 10000,
                                        int                      disk_queue_wait_timeout_ms                = 30000);
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    virtual std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors);
    void                                  cancelPendingStagingTransfers();
    void                                  stopAdmission();
    void                                  shutdown();
    void                                  setMetricsReporter(BlockTreeCacheMetricsReporter* metrics_reporter);

    size_t transferWorkerCount() const {
        return transfer_worker_count_;
    }

private:
    friend class block_tree_cache_test::BlockTreeCacheTestPeer;

    TransferStatus        execute(const std::vector<HostBufferView>&     hosts,
                                  const std::vector<TransferDescriptor>& descriptors,
                                  const std::vector<const GroupSet*>&    group_sets) const;
    static HostBufferView resolveHostView(const GroupSet& group_set, BlockIdxType host_block);

    std::vector<GroupSetPtr> group_sets_;

    std::unique_ptr<BlockTreeTaskPool>          transfer_task_pool_;
    std::unique_ptr<DeviceHostTransferExecutor> device_host_executor_;
    std::unique_ptr<HostDiskTransferExecutor>   host_disk_executor_;
    std::unique_ptr<DeviceDiskTransferExecutor> device_disk_executor_;  // nullable; present when a disk pool exists
    size_t                                      max_device_host_descriptors_per_batch_{8};
    size_t                                      max_non_device_host_descriptors_per_batch_{16};
    size_t                                      transfer_worker_count_{4};
    int                                         host_queue_wait_timeout_ms_{10000};
    int                                         disk_queue_wait_timeout_ms_{30000};
    BlockTreeCacheMetricsReporter*              metrics_reporter_{nullptr};
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
