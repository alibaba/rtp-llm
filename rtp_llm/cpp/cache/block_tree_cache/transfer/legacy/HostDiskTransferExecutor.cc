#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/HostDiskTransferExecutor.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DiskTransferUtils.h"

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HostDiskTransferExecutor::HostDiskTransferExecutor(BlockTreeTaskPool& transfer_task_pool,
                                                   size_t             max_descriptors_per_batch,
                                                   std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                   BlockTreeTaskPool*                             load_task_pool,
                                                   int64_t                                        world_rank):
    TransferExecutor(
        transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter), load_task_pool, world_rank) {}

TransferResult HostDiskTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                      const std::vector<TransferDescriptor>& descriptors,
                                                      const std::vector<const GroupSet*>&    group_sets) try {
    // Normal returns either reject before I/O or follow synchronous disk I/O,
    // including partial failures. Synchronous disk calls retain no access on unwind.
    const auto completed = [](TransferStatus status) {
        TransferResult result(status);
        result.buffer_safe = true;
        return result;
    };
    if (hosts.empty() || hosts.size() != descriptors.size() || hosts.size() != group_sets.size()) {
        return completed(TransferStatus::INVALID_ARGS);
    }
    const bool               write_to_disk = descriptors.front().target_tier == Tier::DISK;
    DiskBlockPool*  disk_pool     = group_sets.front()->diskPool().get();
    if (!disk_pool) {
        return completed(TransferStatus::INVALID_ARGS);
    }
    const size_t             disk_stride   = disk_pool->strideBytes();
    BlockIdList              disk_blocks;
    std::vector<void*>       read_buffers;
    std::vector<const void*> write_buffers;
    disk_blocks.reserve(descriptors.size());
    read_buffers.reserve(descriptors.size());
    write_buffers.reserve(descriptors.size());

    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto&  descriptor = descriptors[index];
        const auto&  host       = hosts[index];
        const auto*  group_set  = group_sets[index];
        const size_t payload    = group_set->payloadBytes();
        if (!isValidHostBufferView(host, payload, disk_stride)) {
            RTP_LLM_LOG_WARNING("invalid host-disk batch item index=%zu group=%zu", index, descriptor.group_set_id);
            return completed(TransferStatus::DISK_IO_ERROR);
        }
        disk_blocks.push_back(descriptor.singleBlockAt(Tier::DISK));
        read_buffers.push_back(host.base);
        write_buffers.push_back(host.base);
    }

    const BlockIOStatus status = write_to_disk ? disk_pool->write(disk_blocks, write_buffers, disk_stride) :
                                                 disk_pool->read(disk_blocks, read_buffers, disk_stride);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("batch %s failed, item_count=%zu, status=%s",
                            write_to_disk ? "write" : "read",
                            descriptors.size(),
                            blockIOStatusName(status));
        return completed(blockIOStatusToTransferStatus(status));
    }
    return completed(TransferStatus::OK);
} catch (const std::exception& error) {
    return TransferResult::executionException(true, error.what());
} catch (...) {
    return TransferResult::executionException(true);
}

}  // namespace rtp_llm
