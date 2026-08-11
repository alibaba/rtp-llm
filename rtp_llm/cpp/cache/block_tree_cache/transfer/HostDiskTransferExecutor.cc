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

TransferStatus HostDiskTransferExecutor::execute(HostBufferView            host,
                                                 const TransferDescriptor& desc,
                                                 const GroupSet&           group_set) const {
    const bool              write_to_disk = desc.target_tier == Tier::DISK;
    const BlockIdxType      disk_block    = desc.singleBlockAt(Tier::DISK);
    BlockTreeDiskBlockPool& disk_pool     = *group_set.diskPool();
    const size_t            payload       = group_set.payloadBytes();
    const size_t            disk_stride   = disk_pool.strideBytes();
    if (!isValidHostBufferView(host, payload, disk_stride)) {
        RTP_LLM_LOG_WARNING("invalid host buffer for %s, disk=%d payload=%zu stride=%zu capacity=%zu",
                            write_to_disk ? "host->disk" : "disk->host",
                            disk_block,
                            payload,
                            disk_stride,
                            host.capacity_bytes);
        return TransferStatus::DISK_IO_ERROR;
    }
    if (write_to_disk && disk_stride > payload) {
        std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
    }
    const BlockIOStatus status =
        write_to_disk ? disk_pool.write(disk_block, host.base, disk_stride) :
                        disk_pool.read(disk_block, host.base, disk_stride);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "%s failed, disk=%d, status=%s", write_to_disk ? "write" : "read", disk_block, blockIOStatusName(status));
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

}  // namespace rtp_llm
