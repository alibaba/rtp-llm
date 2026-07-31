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

TransferStatus
HostDiskTransferExecutor::hostToDisk(HostBufferView            host,
                                     const TransferDescriptor& desc,
                                     const GroupSet&           group_set) const {
    const BlockIdxType      disk_block  = desc.singleBlockAt(Tier::DISK);
    BlockTreeDiskBlockPool& disk_pool   = *group_set.diskPool();
    const size_t payload     = group_set.payloadBytes();
    const size_t disk_stride = disk_pool.strideBytes();
    if (!isValidHostBufferView(host, payload, disk_stride)) {
        RTP_LLM_LOG_WARNING("invalid host buffer for host->disk, disk=%d payload=%zu stride=%zu capacity=%zu",
                            disk_block,
                            payload,
                            disk_stride,
                            host.capacity_bytes);
        return TransferStatus::DISK_IO_ERROR;
    }
    // Write a full disk stride so O_DIRECT length stays block-aligned; zero the
    // [payload, stride) padding so no uninitialized host memory reaches disk.
    if (disk_stride > payload) {
        std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
    }
    const BlockIOStatus status = disk_pool.write(disk_block, host.base, disk_stride);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("write failed, disk=%d, status=%s", disk_block, blockIOStatusName(status));
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

TransferStatus
HostDiskTransferExecutor::diskToHost(const TransferDescriptor& desc,
                                     const GroupSet&           group_set,
                                     HostBufferView            host) const {
    const BlockIdxType      disk_block  = desc.singleBlockAt(Tier::DISK);
    BlockTreeDiskBlockPool& disk_pool   = *group_set.diskPool();
    const size_t payload     = group_set.payloadBytes();
    const size_t disk_stride = disk_pool.strideBytes();
    if (!isValidHostBufferView(host, payload, disk_stride)) {
        RTP_LLM_LOG_WARNING("invalid host buffer for disk->host, disk=%d payload=%zu stride=%zu capacity=%zu",
                            disk_block,
                            payload,
                            disk_stride,
                            host.capacity_bytes);
        return TransferStatus::DISK_IO_ERROR;
    }
    const BlockIOStatus status = disk_pool.read(disk_block, host.base, disk_stride);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("read failed, disk=%d, status=%s", disk_block, blockIOStatusName(status));
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

}  // namespace rtp_llm
