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
    if (desc.target_tier == Tier::DISK) {
        return hostToDisk({host}, {desc}, {&group_set});
    }
    return diskToHost({desc}, {&group_set}, {host});
}

TransferStatus HostDiskTransferExecutor::hostToDisk(const std::vector<HostBufferView>&     hosts,
                                                    const std::vector<TransferDescriptor>& descriptors,
                                                    const std::vector<const GroupSet*>&    group_sets) const {
    BlockTreeDiskBlockPool*  disk_pool = group_sets.front()->diskPool().get();
    BlockIdList              disk_blocks;
    std::vector<const void*> sources;
    disk_blocks.reserve(descriptors.size());
    sources.reserve(descriptors.size());

    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const GroupSet* group_set = group_sets[descriptor_index];
        const BlockIdxType disk_block  = descriptors[descriptor_index].singleBlockAt(Tier::DISK);
        const size_t       payload     = group_set->payloadBytes();
        const size_t       disk_stride = disk_pool->strideBytes();
        const auto         host        = hosts[descriptor_index];
        if (disk_stride > payload) {
            std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
        }
        disk_blocks.push_back(disk_block);
        sources.push_back(host.base);
    }

    const BlockIOStatus status = disk_pool->write(disk_blocks, sources, disk_pool->strideBytes());
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "batch write failed, descriptors=%zu, status=%s", descriptors.size(), blockIOStatusName(status));
        for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
            RTP_LLM_LOG_WARNING("batch write failure candidate descriptor=%zu %s",
                                descriptor_index,
                                descriptors[descriptor_index].debugString().c_str());
        }
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

TransferStatus HostDiskTransferExecutor::diskToHost(const std::vector<TransferDescriptor>& descriptors,
                                                    const std::vector<const GroupSet*>&    group_sets,
                                                    const std::vector<HostBufferView>&     hosts) const {
    BlockTreeDiskBlockPool* disk_pool = group_sets.front()->diskPool().get();
    BlockIdList             disk_blocks;
    std::vector<void*>      destinations;
    disk_blocks.reserve(descriptors.size());
    destinations.reserve(descriptors.size());

    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const BlockIdxType disk_block  = descriptors[descriptor_index].singleBlockAt(Tier::DISK);
        const auto         host        = hosts[descriptor_index];
        disk_blocks.push_back(disk_block);
        destinations.push_back(host.base);
    }

    const BlockIOStatus status = disk_pool->read(disk_blocks, destinations, disk_pool->strideBytes());
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "batch read failed, descriptors=%zu, status=%s", descriptors.size(), blockIOStatusName(status));
        for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
            RTP_LLM_LOG_WARNING("batch read failure candidate descriptor=%zu %s",
                                descriptor_index,
                                descriptors[descriptor_index].debugString().c_str());
        }
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

}  // namespace rtp_llm
