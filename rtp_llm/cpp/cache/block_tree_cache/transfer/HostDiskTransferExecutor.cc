#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"

#include <cstring>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

const char* HostDiskTransferExecutor::blockIOStatusName(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return "OK";
        case BlockIOStatus::INVALID_BLOCK:
            return "INVALID_BLOCK";
        case BlockIOStatus::INVALID_SIZE:
            return "INVALID_SIZE";
        case BlockIOStatus::ALIGNMENT_ERROR:
            return "ALIGNMENT_ERROR";
        case BlockIOStatus::IO_ERROR:
            return "IO_ERROR";
        case BlockIOStatus::PARTIAL_FAILURE:
            return "PARTIAL_FAILURE";
    }
    return "UNKNOWN";
}

TransferStatus HostDiskTransferExecutor::blockIOStatusToTransferStatus(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return TransferStatus::OK;
        case BlockIOStatus::INVALID_BLOCK:
        case BlockIOStatus::INVALID_SIZE:
        case BlockIOStatus::ALIGNMENT_ERROR:
            return TransferStatus::INVALID_ARGS;
        case BlockIOStatus::IO_ERROR:
        case BlockIOStatus::PARTIAL_FAILURE:
            return TransferStatus::DISK_IO_ERROR;
    }
    return TransferStatus::DISK_IO_ERROR;
}

TransferStatus HostDiskTransferExecutor::execute(const std::vector<HostBufferView>&       hosts,
                                                 const std::vector<TransferDescriptor>& descriptors,
                                                 const std::vector<const GroupSet*>&    group_sets) const {
    const bool              write_to_disk = descriptors.front().target_tier == Tier::DISK;
    BlockTreeDiskBlockPool* disk_pool     = group_sets.front()->diskPool().get();
    const size_t            disk_stride   = disk_pool->strideBytes();
    BlockIdList             disk_blocks;
    std::vector<void*>      read_buffers;
    std::vector<const void*> write_buffers;
    disk_blocks.reserve(descriptors.size());
    read_buffers.reserve(descriptors.size());
    write_buffers.reserve(descriptors.size());

    for (size_t index = 0; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        const auto& host       = hosts[index];
        const auto* group_set  = group_sets[index];
        const size_t payload   = group_set->payloadBytes();
        if (!isValidHostBufferView(host, payload, disk_stride)) {
            RTP_LLM_LOG_WARNING("invalid host-disk batch item index=%zu group=%zu", index, descriptor.group_set_id);
            return TransferStatus::DISK_IO_ERROR;
        }
        if (write_to_disk && disk_stride > payload) {
            std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
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
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

}  // namespace rtp_llm
