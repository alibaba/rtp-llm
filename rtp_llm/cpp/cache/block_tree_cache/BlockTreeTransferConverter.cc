#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTransferConverter.h"

#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BlockTreeTransferConverter::validBlock(BlockIdxType block) {
    return block >= 0;
}

bool BlockTreeTransferConverter::validDeviceBlocks(const std::vector<BlockIdxType>& blocks) {
    if (blocks.empty()) {
        return false;
    }

    bool has_valid_block = false;
    for (BlockIdxType block : blocks) {
        if (block < NULL_BLOCK_IDX) {
            return false;
        }
        has_valid_block = has_valid_block || validBlock(block);
    }
    return has_valid_block;
}

bool BlockTreeTransferConverter::hasSourceMemory(const CopyItem& item) {
    return item.src_mem_block_presence_case() == CopyItem::kSrcMemBlock;
}

bool BlockTreeTransferConverter::hasSourceDisk(const CopyItem& item) {
    return item.src_disk_slot_presence_case() == CopyItem::kSrcDiskSlot;
}

bool BlockTreeTransferConverter::hasTargetDisk(const CopyItem& item) {
    return item.disk_slot_presence_case() == CopyItem::kDiskSlot;
}

bool BlockTreeTransferConverter::validateCommonItem(const CopyItem& item) {
    if (item.component_group_id_presence_case() != CopyItem::kComponentGroupId || item.component_group_id() < 0) {
        RTP_LLM_LOG_WARNING("block tree memory operation has invalid component_group_id");
        return false;
    }

    for (int index = 0; index < item.gpu_blocks_size(); ++index) {
        if (item.gpu_blocks(index) < NULL_BLOCK_IDX) {
            RTP_LLM_LOG_WARNING("block tree memory operation has invalid device block=%d", item.gpu_blocks(index));
            return false;
        }
    }
    return true;
}

bool BlockTreeTransferConverter::directionFor(const TransferDescriptor&                descriptor,
                                              MemoryOperationRequestPB::CopyDirection& request_direction) {
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::D2H;
        return validDeviceBlocks(descriptor.device_blocks) && validBlock(descriptor.host_block);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::H2D;
        return validBlock(descriptor.host_block) && validDeviceBlocks(descriptor.device_blocks);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::H2DISK;
        return descriptor.device_blocks.empty() && validBlock(descriptor.host_block)
               && validBlock(descriptor.disk_block);
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::DISK2H;
        return descriptor.device_blocks.empty() && validBlock(descriptor.disk_block)
               && validBlock(descriptor.host_block);
    }
    return false;
}

void BlockTreeTransferConverter::setDeviceBlocks(const std::vector<BlockIdxType>& blocks, CopyItem& item) {
    for (BlockIdxType block : blocks) {
        item.add_gpu_blocks(block);
    }
}

bool BlockTreeTransferConverter::decodeDeviceHostTransfer(const MemoryOperationRequestPB& request,
                                                          const CopyItem&                 item,
                                                          TransferDescriptor&             descriptor) {
    if (item.backing_type() != MemoryOperationRequestPB::MEMORY || !validBlock(item.mem_block()) || hasTargetDisk(item)
        || hasSourceMemory(item) || hasSourceDisk(item)) {
        return false;
    }

    std::vector<BlockIdxType> device_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());
    if (!validDeviceBlocks(device_blocks)) {
        return false;
    }

    if (request.copy_direction() == MemoryOperationRequestPB::D2H) {
        descriptor =
            TransferDescriptor::deviceToHost(item.component_group_id(), std::move(device_blocks), item.mem_block());
        return true;
    }
    if (request.copy_direction() == MemoryOperationRequestPB::H2D) {
        descriptor =
            TransferDescriptor::hostToDevice(item.component_group_id(), item.mem_block(), std::move(device_blocks));
        return true;
    }
    return false;
}

bool BlockTreeTransferConverter::decodeHostDiskTransfer(const MemoryOperationRequestPB& request,
                                                        const CopyItem&                 item,
                                                        TransferDescriptor&             descriptor) {
    if (item.gpu_blocks_size() != 0) {
        return false;
    }

    if (request.copy_direction() == MemoryOperationRequestPB::H2DISK
        && item.backing_type() == MemoryOperationRequestPB::DISK && hasTargetDisk(item) && validBlock(item.disk_slot())
        && isNullBlockIdx(item.mem_block()) && hasSourceMemory(item) && !hasSourceDisk(item)
        && item.src_backing_type() == MemoryOperationRequestPB::MEMORY && validBlock(item.src_mem_block())) {
        descriptor = TransferDescriptor::hostToDisk(item.component_group_id(), item.src_mem_block(), item.disk_slot());
        return true;
    }

    if (request.copy_direction() == MemoryOperationRequestPB::DISK2H
        && item.backing_type() == MemoryOperationRequestPB::MEMORY && validBlock(item.mem_block())
        && !hasTargetDisk(item) && !hasSourceMemory(item) && hasSourceDisk(item)
        && item.src_backing_type() == MemoryOperationRequestPB::DISK && validBlock(item.src_disk_slot())) {
        descriptor = TransferDescriptor::diskToHost(item.component_group_id(), item.src_disk_slot(), item.mem_block());
        return true;
    }
    return false;
}

bool BlockTreeTransferConverter::appendTransfer(const TransferDescriptor& descriptor,
                                                MemoryOperationRequestPB& request) {
    if (descriptor.component_group_id < 0) {
        RTP_LLM_LOG_WARNING("cannot encode transfer with invalid component_group_id=%d", descriptor.component_group_id);
        return false;
    }

    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(descriptor, request_direction)) {
        RTP_LLM_LOG_WARNING("cannot encode invalid block tree transfer, group=%d, source=%s, target=%s",
                            descriptor.component_group_id,
                            tierName(descriptor.source_tier),
                            tierName(descriptor.target_tier));
        return false;
    }
    if (request.copy_items_size() != 0 && request.copy_direction() != request_direction) {
        RTP_LLM_LOG_WARNING("cannot mix copy directions in one memory operation");
        return false;
    }

    // TODO: Stop populating legacy backing fields after removing the Memory Connector.
    CopyItem item;
    item.set_component_group_id(descriptor.component_group_id);
    item.set_is_complete(true);

    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.host_block);
        setDeviceBlocks(descriptor.device_blocks, item);
    } else if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.host_block);
        setDeviceBlocks(descriptor.device_blocks, item);
    } else if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        item.set_backing_type(MemoryOperationRequestPB::DISK);
        item.set_mem_block(NULL_BLOCK_IDX);
        item.set_disk_slot(descriptor.disk_block);
        item.set_src_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_src_mem_block(descriptor.host_block);
    } else if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.host_block);
        item.set_src_backing_type(MemoryOperationRequestPB::DISK);
        item.set_src_disk_slot(descriptor.disk_block);
    } else {
        RTP_LLM_LOG_WARNING("cannot encode unsupported block tree transfer, group=%d, source=%s, target=%s",
                            descriptor.component_group_id,
                            tierName(descriptor.source_tier),
                            tierName(descriptor.target_tier));
        return false;
    }

    if (request.copy_items_size() == 0) {
        request.set_copy_direction(request_direction);
    }
    request.add_copy_items()->CopyFrom(item);
    return true;
}

bool BlockTreeTransferConverter::decodeTransfer(const MemoryOperationRequestPB& request,
                                                int                             item_index,
                                                TransferDescriptor&             descriptor) {
    if (item_index < 0 || item_index >= request.copy_items_size()) {
        RTP_LLM_LOG_WARNING("cannot decode memory operation item, invalid index=%d", item_index);
        return false;
    }

    const CopyItem& item = request.copy_items(item_index);
    if (!validateCommonItem(item)) {
        return false;
    }

    bool success = false;
    switch (request.copy_direction()) {
        case MemoryOperationRequestPB::D2H:
        case MemoryOperationRequestPB::H2D:
            success = decodeDeviceHostTransfer(request, item, descriptor);
            break;
        case MemoryOperationRequestPB::H2DISK:
        case MemoryOperationRequestPB::DISK2H:
            success = decodeHostDiskTransfer(request, item, descriptor);
            break;
        default:
            success = false;
            break;
    }
    if (!success) {
        RTP_LLM_LOG_WARNING("cannot decode invalid block tree memory operation item, index=%d, group=%d",
                            item_index,
                            item.component_group_id());
    }
    return success;
}

}  // namespace rtp_llm
