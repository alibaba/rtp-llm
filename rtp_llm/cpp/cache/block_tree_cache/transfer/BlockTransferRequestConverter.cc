#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BlockTransferRequestConverter::directionFor(const TransferDescriptor&                descriptor,
                                                 MemoryOperationRequestPB::CopyDirection& request_direction) {
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::D2H;
        return true;
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::H2D;
        return true;
    }
    if (descriptor.source_tier == Tier::HOST && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::H2DISK;
        return true;
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::HOST) {
        request_direction = MemoryOperationRequestPB::DISK2H;
        return true;
    }
    if (descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::DISK) {
        request_direction = MemoryOperationRequestPB::D2DISK;
        return true;
    }
    if (descriptor.source_tier == Tier::DISK && descriptor.target_tier == Tier::DEVICE) {
        request_direction = MemoryOperationRequestPB::DISK2D;
        return true;
    }
    return false;
}

bool BlockTransferRequestConverter::decodeDeviceBlocks(const CopyItem&            item,
                                                       const GroupSet&            group_set,
                                                       std::vector<BlockIdxType>& blocks) {
    const auto&                              device_pools = group_set.devicePools();
    const auto& group_ids = group_set.groupIds();
    std::unordered_map<size_t, BlockIdxType> blocks_by_group_id;
    for (const auto& group_block : item.group_blocks()) {
        blocks_by_group_id.emplace(static_cast<size_t>(group_block.group_id()), group_block.block_id());
    }
    blocks.reserve(group_ids.size());
    for (size_t i = 0; i < group_ids.size(); ++i) {
        const auto it = blocks_by_group_id.find(group_ids[i]);
        if (it == blocks_by_group_id.end() || !device_pools[i]->validBlock(it->second)) {
            return false;
        }
        blocks.push_back(it->second);
    }
    return true;
}

bool BlockTransferRequestConverter::encodeTransfer(MemoryOperationRequestPB&              request,
                                                   const std::vector<TransferDescriptor>& descriptors,
                                                   const std::vector<GroupSetPtr>&        group_sets) {
    const TransferDescriptor&               first = descriptors.front();
    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(first, request_direction)) {
        return false;
    }
    request.set_copy_direction(request_direction);

    for (const TransferDescriptor& descriptor : descriptors) {
        if (descriptor.source_tier != first.source_tier || descriptor.target_tier != first.target_tier) {
            return false;
        }
        const GroupSet& group_set = *group_sets[descriptor.group_set_id];
        CopyItem        item;
        item.set_group_set_id(descriptor.group_set_id);

        if (descriptor.source_tier == Tier::HOST || descriptor.target_tier == Tier::HOST) {
            item.set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        }
        if (descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK) {
            item.set_disk_block(descriptor.singleBlockAt(Tier::DISK));
        }
        if (descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE) {
            const auto& blocks    = descriptor.blocksAt(Tier::DEVICE);
            const auto& group_ids = group_set.groupIds();
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto* group_block = item.add_group_blocks();
                group_block->set_group_id(static_cast<int32_t>(group_ids[i]));
                group_block->set_block_id(blocks[i]);
            }
        }
        request.add_copy_items()->CopyFrom(item);
    }
    return true;
}

bool BlockTransferRequestConverter::decodeTransfer(const MemoryOperationRequestPB&  request,
                                                   std::vector<TransferDescriptor>& descriptors,
                                                   const std::vector<GroupSetPtr>&  group_sets) {
    if (request.copy_items_size() == 0) {
        return false;
    }

    for (const CopyItem& item : request.copy_items()) {
        const size_t group_set_id = item.group_set_id();
        if (group_set_id >= group_sets.size()) {
            RTP_LLM_LOG_WARNING("cannot resolve BlockTree GroupSet id=%lu", item.group_set_id());
            return false;
        }
        const GroupSet&    group_set = *group_sets[group_set_id];
        TransferDescriptor descriptor;
        switch (request.copy_direction()) {
            case MemoryOperationRequestPB::D2H: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::deviceToHost(
                    group_set.groupSetId(), std::move(device_blocks), item.mem_block());
                break;
            }
            case MemoryOperationRequestPB::H2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::hostToDevice(
                    group_set.groupSetId(), item.mem_block(), std::move(device_blocks));
                break;
            }
            case MemoryOperationRequestPB::H2DISK:
                if (!group_set.hostPool()->validBlock(item.mem_block())
                    || !group_set.diskPool()->validBlock(item.disk_block())) {
                    return false;
                }
                descriptor =
                    TransferDescriptor::hostToDisk(group_set.groupSetId(), item.mem_block(), item.disk_block());
                break;
            case MemoryOperationRequestPB::DISK2H:
                if (!group_set.hostPool()->validBlock(item.mem_block())
                    || !group_set.diskPool()->validBlock(item.disk_block())) {
                    return false;
                }
                descriptor =
                    TransferDescriptor::diskToHost(group_set.groupSetId(), item.disk_block(), item.mem_block());
                break;
            case MemoryOperationRequestPB::D2DISK: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.diskPool()->validBlock(item.disk_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::deviceToDisk(
                    group_set.groupSetId(), std::move(device_blocks), item.disk_block());
                break;
            }
            case MemoryOperationRequestPB::DISK2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.diskPool()->validBlock(item.disk_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::diskToDevice(
                    group_set.groupSetId(), item.disk_block(), std::move(device_blocks));
                break;
            }
            default:
                return false;
        }
        descriptors.push_back(std::move(descriptor));
    }
    return true;
}

}  // namespace rtp_llm
