#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
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

std::vector<std::string> BlockTransferRequestConverter::normalizedTags(const CopyItem& item) {
    std::vector<std::string> tags(item.group_set_tags().begin(), item.group_set_tags().end());
    if (tags.empty() || std::any_of(tags.begin(), tags.end(), [](const std::string& tag) { return tag.empty(); })) {
        return {};
    }
    std::sort(tags.begin(), tags.end());
    if (std::adjacent_find(tags.begin(), tags.end()) != tags.end()) {
        return {};
    }
    return tags;
}

const GroupSet*
BlockTransferRequestConverter::findGroupSet(const std::vector<std::string>&       normalized_tags,
                                                  const std::vector<GroupSetPtr>& group_sets) {
    if (normalized_tags.empty()) {
        return nullptr;
    }
    const GroupSet* match = nullptr;
    for (const auto& group_set : group_sets) {
        if (group_set == nullptr) {
            continue;
        }
        auto local_tags = group_set->groupTags();
        std::sort(local_tags.begin(), local_tags.end());
        if (local_tags != normalized_tags) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = group_set.get();
    }
    return match;
}

bool BlockTransferRequestConverter::validDeviceBlocks(const std::vector<BlockIdxType>& blocks,
                                                      const GroupSet&            group_set) {
    if (blocks.size() != group_set.groupIds().size() || blocks.empty()) {
        return false;
    }

    const std::vector<DeviceBlockPoolPtr>& device_pools = group_set.devicePools();
    if (blocks.size() != device_pools.size()) {
        return false;
    }
    bool has_valid_block = false;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const BlockIdxType block = blocks[i];
        if (isNullBlockIdx(block)) {
            continue;
        }
        if (device_pools[i] == nullptr || !device_pools[i]->validBlock(block)) {
            return false;
        }
        has_valid_block = true;
    }
    return has_valid_block;
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
                                                 const GroupSet&                    group_set,
                                                 MemoryOperationRequestPB::CopyDirection& request_direction) {
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::D2H;
        return validDeviceBlocks(descriptor.device_blocks, group_set)
               && validHostBlock(descriptor.host_block, group_set);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::H2D;
        return validHostBlock(descriptor.host_block, group_set)
               && validDeviceBlocks(descriptor.device_blocks, group_set);
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::H2DISK;
        return descriptor.device_blocks.empty() && validHostBlock(descriptor.host_block, group_set)
               && validDiskBlock(descriptor.disk_block, group_set);
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::DISK2H;
        return descriptor.device_blocks.empty() && validDiskBlock(descriptor.disk_block, group_set)
               && validHostBlock(descriptor.host_block, group_set);
    }
    return false;
}

void BlockTransferRequestConverter::setDeviceBlocks(const std::vector<BlockIdxType>& blocks,
                                                    const GroupSet&            group_set,
                                                    CopyItem&                        item) {
    const auto tags = group_set.groupTags();
    RTP_LLM_CHECK(blocks.size() == tags.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto* tagged_block = item.add_tagged_gpu_blocks();
        tagged_block->set_tag(tags[i]);
        tagged_block->set_block_id(blocks[i]);
        // BlockTreeCache transfers one block from each tag-owned physical pool.
        // -1 distinguishes this pool coordinate from target connector layer slots.
        tagged_block->set_layer_id(-1);
    }
}

bool BlockTransferRequestConverter::decodeDeviceBlocks(const CopyItem&            item,
                                                       const GroupSet&      group_set,
                                                       std::vector<BlockIdxType>& blocks) {
    const auto tags = group_set.groupTags();
    if (item.tagged_gpu_blocks_size() != static_cast<int>(tags.size())) {
        return false;
    }
    std::unordered_map<std::string, BlockIdxType> blocks_by_tag;
    for (const auto& tagged_block : item.tagged_gpu_blocks()) {
        if (tagged_block.layer_id() != -1 || tagged_block.tag().empty()
            || !blocks_by_tag.emplace(tagged_block.tag(), tagged_block.block_id()).second) {
            return false;
        }
    }
    blocks.clear();
    blocks.reserve(tags.size());
    for (const auto& tag : tags) {
        const auto it = blocks_by_tag.find(tag);
        if (it == blocks_by_tag.end()) {
            return false;
        }
        blocks.push_back(it->second);
    }
    return blocks_by_tag.size() == tags.size() && validDeviceBlocks(blocks, group_set);
}

bool BlockTransferRequestConverter::decodeDeviceHostTransfer(const MemoryOperationRequestPB& request,
                                                             const CopyItem&                 item,
                                                             const GroupSet&           group_set,
                                                             TransferDescriptor&             descriptor) {
    if (item.backing_type() != MemoryOperationRequestPB::MEMORY || !validHostBlock(item.mem_block(), group_set)
        || hasTargetDisk(item) || hasSourceMemory(item) || hasSourceDisk(item)) {
        return false;
    }
    std::vector<BlockIdxType> device_blocks;
    if (!decodeDeviceBlocks(item, group_set, device_blocks)) {
        return false;
    }
    if (request.copy_direction() == MemoryOperationRequestPB::D2H) {
        descriptor = TransferDescriptor::deviceToHost(
            group_set.groupSetId(), std::move(device_blocks), item.mem_block());
        return true;
    }
    if (request.copy_direction() == MemoryOperationRequestPB::H2D) {
        descriptor = TransferDescriptor::hostToDevice(
            group_set.groupSetId(), item.mem_block(), std::move(device_blocks));
        return true;
    }
    return false;
}

bool BlockTransferRequestConverter::decodeHostDiskTransfer(const MemoryOperationRequestPB& request,
                                                           const CopyItem&                 item,
                                                           const GroupSet&           group_set,
                                                           TransferDescriptor&             descriptor) {
    if (item.tagged_gpu_blocks_size() != 0) {
        return false;
    }
    if (request.copy_direction() == MemoryOperationRequestPB::H2DISK
        && item.backing_type() == MemoryOperationRequestPB::DISK && hasTargetDisk(item)
        && validDiskBlock(item.disk_slot(), group_set) && isNullBlockIdx(item.mem_block())
        && hasSourceMemory(item) && !hasSourceDisk(item) && item.src_backing_type() == MemoryOperationRequestPB::MEMORY
        && validHostBlock(item.src_mem_block(), group_set)) {
        descriptor =
            TransferDescriptor::hostToDisk(group_set.groupSetId(), item.src_mem_block(), item.disk_slot());
        return true;
    }
    if (request.copy_direction() == MemoryOperationRequestPB::DISK2H
        && item.backing_type() == MemoryOperationRequestPB::MEMORY && validHostBlock(item.mem_block(), group_set)
        && !hasTargetDisk(item) && !hasSourceMemory(item) && hasSourceDisk(item)
        && item.src_backing_type() == MemoryOperationRequestPB::DISK
        && validDiskBlock(item.src_disk_slot(), group_set)) {
        descriptor =
            TransferDescriptor::diskToHost(group_set.groupSetId(), item.src_disk_slot(), item.mem_block());
        return true;
    }
    return false;
}

bool BlockTransferRequestConverter::appendTransfer(const TransferDescriptor&             descriptor,
                                                   const std::vector<GroupSetPtr>& group_sets,
    MemoryOperationRequestPB&             request) {
    const GroupSet* group_set = nullptr;
    if (descriptor.group_set_id < group_sets.size()) {
        group_set = group_sets[descriptor.group_set_id].get();
    }
    if (group_set == nullptr || group_set->groupSetId() != descriptor.group_set_id
        || group_set->groupIds().empty()) {
        return false;
    }

    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(descriptor, *group_set, request_direction)
        || (request.copy_items_size() != 0 && request.copy_direction() != request_direction)) {
        return false;
    }

    CopyItem item;
    auto     sorted_tags = group_set->groupTags();
    std::sort(sorted_tags.begin(), sorted_tags.end());
    for (const auto& tag : sorted_tags) {
        item.add_group_set_tags(tag);
    }
    item.set_is_complete(true);

    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.host_block);
        setDeviceBlocks(descriptor.device_blocks, *group_set, item);
    } else if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        item.set_backing_type(MemoryOperationRequestPB::MEMORY);
        item.set_mem_block(descriptor.host_block);
        setDeviceBlocks(descriptor.device_blocks, *group_set, item);
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
        return false;
    }

    if (request.copy_items_size() == 0) {
        request.set_copy_direction(request_direction);
    }
    request.add_copy_items()->CopyFrom(item);
    return true;
}

bool BlockTransferRequestConverter::decodeTransfer(const MemoryOperationRequestPB&       request,
                                                   int                                   item_index,
                                                   const std::vector<GroupSetPtr>& group_sets,
                                                   TransferDescriptor&                   descriptor) {
    if (item_index < 0 || item_index >= request.copy_items_size()) {
        return false;
    }
    const CopyItem& item            = request.copy_items(item_index);
    const auto      normalized_tags = normalizedTags(item);
    const auto*     group_set = findGroupSet(normalized_tags, group_sets);
    if (group_set == nullptr) {
        RTP_LLM_LOG_WARNING("cannot resolve exact BlockTree GroupSet tag set, item=%d", item_index);
        return false;
    }

    switch (request.copy_direction()) {
        case MemoryOperationRequestPB::D2H:
        case MemoryOperationRequestPB::H2D:
            return decodeDeviceHostTransfer(request, item, *group_set, descriptor);
        case MemoryOperationRequestPB::H2DISK:
        case MemoryOperationRequestPB::DISK2H:
            return decodeHostDiskTransfer(request, item, *group_set, descriptor);
        default:
            return false;
    }
}

}  // namespace rtp_llm
