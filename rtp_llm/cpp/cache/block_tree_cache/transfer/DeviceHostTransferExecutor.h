#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferExecutor.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class DeviceDiskTransferExecutor;

class DeviceHostTransferExecutor: public TransferExecutor {
public:
    DeviceHostTransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                               size_t                                         max_descriptors_per_batch,
                               DeviceHostCopyOptions                          options          = {},
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr);
    ~DeviceHostTransferExecutor() = default;

private:
    friend class DeviceDiskTransferExecutor;

    TransferStatus executeBatch(const std::vector<HostBufferView>&     hosts,
                                const std::vector<TransferDescriptor>& descriptors,
                                const std::vector<const GroupSet*>&    group_sets) override;

    std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
    generatePlan(const std::vector<HostBufferView>&     hosts,
                 const std::vector<TransferDescriptor>& descriptors,
                 const std::vector<const GroupSet*>&    group_sets) const;

    DeviceHostCopyOptions                                options_;
    std::vector<std::unique_ptr<DeviceHostCopyStrategy>> strategies_;
};

}  // namespace rtp_llm
