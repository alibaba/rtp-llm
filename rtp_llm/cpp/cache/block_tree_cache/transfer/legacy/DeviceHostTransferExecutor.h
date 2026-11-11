#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/TransferExecutor.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class DeviceDiskTransferExecutor;
class CrcCopyExecutor;

class DeviceHostTransferExecutor: public TransferExecutor {
public:
    DeviceHostTransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                               size_t                                         max_descriptors_per_batch,
                               DeviceHostCopyOptions                          options            = {},
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter   = nullptr,
                               std::shared_ptr<CrcCopyExecutor>               load_crc_executor  = nullptr,
                               std::shared_ptr<CrcCopyExecutor>               evict_crc_executor = nullptr,
                               BlockTreeTaskPool*                             load_task_pool     = nullptr,
                               int64_t                                        world_rank         = -1);
    ~DeviceHostTransferExecutor() = default;

private:
    friend class DeviceDiskTransferExecutor;

    TransferResult executeBatch(const std::vector<HostBufferView>&     hosts,
                                const std::vector<TransferDescriptor>& descriptors,
                                const std::vector<const GroupSet*>&    group_sets) override;

    std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
    generatePlan(const std::vector<HostBufferView>&     hosts,
                 const std::vector<TransferDescriptor>& descriptors,
                 const std::vector<const GroupSet*>&    group_sets) const;

    std::shared_ptr<CrcCopyExecutor>    load_crc_executor_, evict_crc_executor_;
    DeviceHostCopyOptions               options_;
    // A rank uses one device. Bind its stream pair on the first copy; call_once
    // publishes it safely to concurrent transfer workers.
    std::once_flag                                       copy_streams_once_;
    std::shared_ptr<DeviceHostCopyStreams>               copy_streams_;
    std::vector<std::unique_ptr<DeviceHostCopyStrategy>> strategies_;
};

}  // namespace rtp_llm
