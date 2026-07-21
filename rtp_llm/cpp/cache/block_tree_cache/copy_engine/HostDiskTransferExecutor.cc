#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostDiskTransferExecutor.h"

#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
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

CopyStatus HostDiskTransferExecutor::blockIOStatusToCopyStatus(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return CopyStatus::OK;
        case BlockIOStatus::INVALID_BLOCK:
        case BlockIOStatus::INVALID_SIZE:
        case BlockIOStatus::ALIGNMENT_ERROR:
            return CopyStatus::INVALID_ARGS;
        case BlockIOStatus::IO_ERROR:
            return CopyStatus::DISK_IO_ERROR;
        case BlockIOStatus::PARTIAL_FAILURE:
            return CopyStatus::DISK_IO_ERROR;
    }
    return CopyStatus::DISK_IO_ERROR;
}

CopyStatus HostDiskTransferExecutor::hostToDisk(const TransferDescriptor& desc, const ComponentGroup& group) const {
    const auto  host_block = desc.host_block;
    const auto  disk_block = desc.disk_block;
    auto&       host_pool  = *group.hostPool();
    auto&       disk_pool  = *group.diskPool();
    const void* host_base  = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("null host buffer");
        return CopyStatus::DISK_IO_ERROR;
    }
    const size_t bytes  = group.layout().payloadBytes();
    const auto   status = disk_pool.write(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "write failed, host=%d, disk=%d, status=%s", host_block, disk_block, blockIOStatusName(status));
        return blockIOStatusToCopyStatus(status);
    }
    return CopyStatus::OK;
}

CopyStatus HostDiskTransferExecutor::diskToHost(const TransferDescriptor& desc, const ComponentGroup& group) const {
    const auto disk_block = desc.disk_block;
    const auto host_block = desc.host_block;
    auto&      host_pool  = *group.hostPool();
    auto&      disk_pool  = *group.diskPool();
    void*      host_base  = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("null host buffer");
        return CopyStatus::DISK_IO_ERROR;
    }
    const size_t bytes  = group.layout().payloadBytes();
    const auto   status = disk_pool.read(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "read failed, disk=%d, host=%d, status=%s", disk_block, host_block, blockIOStatusName(status));
        return blockIOStatusToCopyStatus(status);
    }
    return CopyStatus::OK;
}

}  // namespace rtp_llm
