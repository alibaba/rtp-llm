#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"

#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BlockTransferRequestConverter::hasSourceMemory(const CopyItem& item) {
    return item.src_mem_block_presence_case() == CopyItem::kSrcMemBlock;
}

bool BlockTransferRequestConverter::hasSourceDisk(const CopyItem& item) {
    return item.src_disk_slot_presence_case() == CopyItem::kSrcDiskSlot;
}

bool BlockTransferRequestConverter::hasTargetDisk(const CopyItem& item) {
    return item.disk_slot_presence_case() == CopyItem::kDiskSlot;
}

const GroupSet* BlockTransferRequestConverter::findGroupSet(const CopyItem&                 item,
                                                            const std::vector<GroupSetPtr>& group_sets) {
    const uint64_t group_set_id = item.group_set_id();
    if (group_set_id >= group_sets.size()) {
        return nullptr;
    }
    const auto& group_set = group_sets[static_cast<size_t>(group_set_id)];
    return group_set != nullptr && group_set->groupSetId() == group_set_id ? group_set.get() : nullptr;
}

bool BlockTransferRequestConverter::validDeviceBlocks(const std::vector<BlockIdxType>& blocks,
                                                      const GroupSet&                  group_set) {
    if (blocks.empty() || blocks.size() != group_set.groupIds().size()) {
        return false;
    }

    const std::vector<DeviceBlockPoolPtr>& device_pools = group_set.devicePools();
    if (blocks.size() != device_pools.size()) {
        return false;
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (device_pools[i] == nullptr || !device_pools[i]->validBlock(blocks[i])) {
            return false;
        }
    }
    return true;
}

bool BlockTransferRequestConverter::validHostBlock(BlockIdxType block, const GroupSet& group_set) {
    const auto host_pool = group_set.hostPool();
    return host_pool != nullptr && host_pool->validBlock(block);
}

bool BlockTransferRequestConverter::validDiskBlock(BlockIdxType block, const GroupSet& group_set) {
    const auto disk_pool = group_set.diskPool();
    return disk_pool != nullptr && disk_pool->validBlock(block);
}

bool BlockTransferRequestConverter::directionFor(const TransferDescriptor&                descriptor,
                                                 const GroupSet&                          group_set,
                                                 MemoryOperationRequestPB::CopyDirection& request_direction) {
    if (!descriptor.isExecutable()) {
        return false;
    }
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::D2H;
        return validDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), group_set)
               && validHostBlock(descriptor.singleBlockAt(Tier::HOST), group_set);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::H2D;
        return validHostBlock(descriptor.singleBlockAt(Tier::HOST), group_set)
               && validDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), group_set);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::H2DISK;
        return validHostBlock(descriptor.singleBlockAt(Tier::HOST), group_set)
               && validDiskBlock(descriptor.singleBlockAt(Tier::DISK), group_set);
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::DISK2H;
        return validDiskBlock(descriptor.singleBlockAt(Tier::DISK), group_set)
               && validHostBlock(descriptor.singleBlockAt(Tier::HOST), group_set);
    }
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::D2DISK;
        return validDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), group_set)
               && validDiskBlock(descriptor.singleBlockAt(Tier::DISK), group_set);
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::DISK2D;
        return validDiskBlock(descriptor.singleBlockAt(Tier::DISK), group_set)
               && validDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), group_set);
    }
    return false;
}

void BlockTransferRequestConverter::setDeviceBlocks(const std::vector<BlockIdxType>& blocks,
                                                    const GroupSet&                  group_set,
                                                    CopyItem&                        item) {
    const auto& group_ids = group_set.groupIds();
    RTP_LLM_CHECK(blocks.size() == group_ids.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        RTP_LLM_CHECK(group_ids[i] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));
        auto* group_block = item.add_group_blocks();
        group_block->set_group_id(static_cast<int32_t>(group_ids[i]));
        group_block->set_block_id(blocks[i]);
    }
}

bool BlockTransferRequestConverter::decodeDeviceBlocks(const CopyItem&            item,
                                                       const GroupSet&            group_set,
                                                       std::vector<BlockIdxType>& blocks) {
    const auto& group_ids = group_set.groupIds();
    if (item.group_blocks_size() != static_cast<int>(group_ids.size())) {
        return false;
    }
    std::unordered_map<size_t, BlockIdxType> blocks_by_group_id;
    for (const auto& group_block : item.group_blocks()) {
        if (group_block.group_id() < 0
            || !blocks_by_group_id.emplace(static_cast<size_t>(group_block.group_id()), group_block.block_id())
                    .second) {
            return false;
        }
    }
    blocks.clear();
    blocks.reserve(group_ids.size());
    for (const size_t group_id : group_ids) {
        const auto it = blocks_by_group_id.find(group_id);
        if (it == blocks_by_group_id.end()) {
            return false;
        }
        blocks.push_back(it->second);
    }
    return blocks_by_group_id.size() == group_ids.size() && validDeviceBlocks(blocks, group_set);
}

TransferDescriptor BlockTransferRequestConverter::decodeDeviceHostTransfer(const MemoryOperationRequestPB& request,
                                                                           const CopyItem&                 item,
                                                                           const GroupSet&                 group_set) {
    if (item.backing_type() != MemoryOperationRequestPB::MEMORY || !validHostBlock(item.mem_block(), group_set)
        || hasTargetDisk(item) || hasSourceMemory(item) || hasSourceDisk(item)) {
        return {};
    }
    std::vector<BlockIdxType> device_blocks;
    if (!decodeDeviceBlocks(item, group_set, device_blocks)) {
        return {};
    }
    if (request.copy_direction() == MemoryOperationRequestPB::D2H) {
        return TransferDescriptor::deviceToHost(group_set.groupSetId(), std::move(device_blocks), item.mem_block());
    }
    if (request.copy_direction() == MemoryOperationRequestPB::H2D) {
        return TransferDescriptor::hostToDevice(group_set.groupSetId(), item.mem_block(), std::move(device_blocks));
    }
    return {};
}

TransferDescriptor BlockTransferRequestConverter::decodeHostDiskTransfer(const MemoryOperationRequestPB& request,
                                                                         const CopyItem&                 item,
                                                                         const GroupSet&                 group_set) {
    if (item.group_blocks_size() != 0) {
        return {};
    }
    if (request.copy_direction() == MemoryOperationRequestPB::H2DISK
        && item.backing_type() == MemoryOperationRequestPB::DISK && hasTargetDisk(item)
        && validDiskBlock(item.disk_slot(), group_set) && isNullBlockIdx(item.mem_block()) && hasSourceMemory(item)
        && !hasSourceDisk(item) && item.src_backing_type() == MemoryOperationRequestPB::MEMORY
        && validHostBlock(item.src_mem_block(), group_set)) {
        return TransferDescriptor::hostToDisk(group_set.groupSetId(), item.src_mem_block(), item.disk_slot());
    }
    if (request.copy_direction() == MemoryOperationRequestPB::DISK2H
        && item.backing_type() == MemoryOperationRequestPB::MEMORY && validHostBlock(item.mem_block(), group_set)
        && !hasTargetDisk(item) && !hasSourceMemory(item) && hasSourceDisk(item)
        && item.src_backing_type() == MemoryOperationRequestPB::DISK
        && validDiskBlock(item.src_disk_slot(), group_set)) {
        return TransferDescriptor::diskToHost(group_set.groupSetId(), item.src_disk_slot(), item.mem_block());
    }
    return {};
}

TransferDescriptor BlockTransferRequestConverter::decodeDeviceDiskTransfer(const MemoryOperationRequestPB& request,
                                                                           const CopyItem&                 item,
                                                                           const GroupSet&                 group_set) {
    if (request.copy_direction() == MemoryOperationRequestPB::D2DISK
        && item.backing_type() == MemoryOperationRequestPB::DISK && hasTargetDisk(item)
        && validDiskBlock(item.disk_slot(), group_set) && isNullBlockIdx(item.mem_block()) && !hasSourceMemory(item)
        && !hasSourceDisk(item)) {
        std::vector<BlockIdxType> device_blocks;
        if (!decodeDeviceBlocks(item, group_set, device_blocks)) {
            return {};
        }
        return TransferDescriptor::deviceToDisk(group_set.groupSetId(), std::move(device_blocks), item.disk_slot());
    }
    if (request.copy_direction() == MemoryOperationRequestPB::DISK2D && !hasTargetDisk(item)
        && isNullBlockIdx(item.mem_block()) && !hasSourceMemory(item) && hasSourceDisk(item)
        && item.src_backing_type() == MemoryOperationRequestPB::DISK
        && validDiskBlock(item.src_disk_slot(), group_set)) {
        std::vector<BlockIdxType> device_blocks;
        if (!decodeDeviceBlocks(item, group_set, device_blocks)) {
            return {};
        }
        return TransferDescriptor::diskToDevice(group_set.groupSetId(), item.src_disk_slot(), std::move(device_blocks));
    }
    return {};
}

bool BlockTransferRequestConverter::appendTransfer(const TransferDescriptor&       descriptor,
                                                   const std::vector<GroupSetPtr>& group_sets,
                                                   MemoryOperationRequestPB&       request) {
    const GroupSet* group_set = nullptr;
    if (descriptor.group_set_id < group_sets.size()) {
        group_set = group_sets[descriptor.group_set_id].get();
    }
    if (group_set == nullptr || group_set->groupSetId() != descriptor.group_set_id || group_set->groupIds().empty()) {
        return false;
    }

    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(descriptor, *group_set, request_direction)
        || (request.copy_items_size() != 0 && request.copy_direction() != request_direction)) {
        return false;
    }

    CopyItem item;
    item.set_group_set_id(descriptor.group_set_id);
    item.set_is_complete(true);

    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        setDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), *group_set, item);
    } else if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        setDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), *group_set, item);
    } else if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        item.set_backing_type(MemoryOperationRequestPB::DISK);
        item.set_mem_block(NULL_BLOCK_IDX);
        item.set_disk_slot(descriptor.singleBlockAt(Tier::DISK));
        item.set_src_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_src_mem_block(descriptor.singleBlockAt(Tier::HOST));
    } else if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        item.set_src_backing_type(MemoryOperationRequestPB::DISK);
        item.set_src_disk_slot(descriptor.singleBlockAt(Tier::DISK));
    } else if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::DISK) {
        item.set_backing_type(MemoryOperationRequestPB::DISK);
        item.set_mem_block(NULL_BLOCK_IDX);
        item.set_disk_slot(descriptor.singleBlockAt(Tier::DISK));
        setDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), *group_set, item);
    } else if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
        // DISK2D has no primary non-device backing: source is src_disk_slot, target is group_blocks.
        // Leave primary backing_type at its proto3 default (not decoded).
        item.set_mem_block(NULL_BLOCK_IDX);
        item.set_src_backing_type(MemoryOperationRequestPB::DISK);
        item.set_src_disk_slot(descriptor.singleBlockAt(Tier::DISK));
        setDeviceBlocks(descriptor.blocksAt(Tier::DEVICE), *group_set, item);
    } else {
        return false;
    }

    if (request.copy_items_size() == 0) {
        request.set_copy_direction(request_direction);
    }
    request.add_copy_items()->CopyFrom(item);
    return true;
}

TransferDescriptor BlockTransferRequestConverter::decodeTransfer(const MemoryOperationRequestPB& request,
                                                                 int                             item_index,
                                                                 const std::vector<GroupSetPtr>& group_sets) {
    if (item_index < 0 || item_index >= request.copy_items_size()) {
        return {};
    }
    const CopyItem& item      = request.copy_items(item_index);
    const auto*     group_set = findGroupSet(item, group_sets);
    if (group_set == nullptr) {
        RTP_LLM_LOG_WARNING("cannot resolve BlockTree GroupSet id=%lu, item=%d", item.group_set_id(), item_index);
        return {};
    }

    switch (request.copy_direction()) {
        case MemoryOperationRequestPB::D2H:
        case MemoryOperationRequestPB::H2D:
            return decodeDeviceHostTransfer(request, item, *group_set);
        case MemoryOperationRequestPB::H2DISK:
        case MemoryOperationRequestPB::DISK2H:
            return decodeHostDiskTransfer(request, item, *group_set);
        case MemoryOperationRequestPB::D2DISK:
        case MemoryOperationRequestPB::DISK2D:
            return decodeDeviceDiskTransfer(request, item, *group_set);
        default:
            return {};
    }
}

}  // namespace rtp_llm
