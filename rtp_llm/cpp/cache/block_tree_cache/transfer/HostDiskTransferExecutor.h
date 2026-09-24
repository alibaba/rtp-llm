#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferExecutor.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class DeviceDiskTransferExecutor;
class CrcTransferService;
enum class BlockIOStatus;

// Consumes a validated HostBufferView; disk I/O always uses the padded stride.
class HostDiskTransferExecutor: public TransferExecutor {
public:
    HostDiskTransferExecutor(BlockTreeTaskPool&                             transfer_task_pool,
                             size_t                                         max_descriptors_per_batch,
                             std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr,
                             std::shared_ptr<CrcTransferService>            crc_service      = nullptr);

private:
    friend class DeviceDiskTransferExecutor;

    TransferStatus executeBatch(const std::vector<HostBufferView>&     hosts,
                                const std::vector<TransferDescriptor>& descriptors,
                                const std::vector<const GroupSet*>&    group_sets) override;

    std::shared_ptr<CrcTransferService> crc_service_;

    static TransferStatus blockIOStatusToTransferStatus(BlockIOStatus status);
    static const char*    blockIOStatusName(BlockIOStatus status);
};

}  // namespace rtp_llm
