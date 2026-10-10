#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

const GroupSet* resolveGroupSet(const MemoryOperationRequestPB::CopyItem& item,
                                const std::vector<GroupSetPtr>&           group_sets) {
    if (item.group_tags_size() == 0) {
        return nullptr;
    }
    std::unordered_set<std::string> requested_tags;
    for (const auto& tag : item.group_tags()) {
        if (tag.empty() || !requested_tags.emplace(tag).second) {
            return nullptr;
        }
    }
    const GroupSet* match = nullptr;
    for (const auto& candidate : group_sets) {
        if (!candidate || candidate->groupTags().size() != requested_tags.size()) {
            continue;
        }
        const bool same_members = std::all_of(
            candidate->groupTags().begin(), candidate->groupTags().end(), [&requested_tags](const std::string& tag) {
                return requested_tags.find(tag) != requested_tags.end();
            });
        if (!same_members) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = candidate.get();
    }
    return match;
}

}  // namespace

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
    const auto&                                   device_pools = group_set.devicePools();
    const auto&                                   group_tags   = group_set.groupTags();
    std::unordered_map<std::string, BlockIdxType> blocks_by_tag;
    for (const auto& group_block : item.group_blocks()) {
        if (group_block.tag().empty() || !blocks_by_tag.emplace(group_block.tag(), group_block.block_id()).second) {
            return false;
        }
    }
    if (blocks_by_tag.size() != group_tags.size() || device_pools.size() != group_tags.size()) {
        return false;
    }
    blocks.reserve(group_tags.size());
    for (size_t i = 0; i < group_tags.size(); ++i) {
        const auto it = blocks_by_tag.find(group_tags[i]);
        if (it == blocks_by_tag.end() || !device_pools[i]->validBlock(it->second)) {
            return false;
        }
        blocks.push_back(it->second);
    }
    return true;
}

bool BlockTransferRequestConverter::encodeTransfer(MemoryOperationRequestPB&       request,
                                                   const TransferTask&             task,
                                                   const std::vector<GroupSetPtr>& group_sets) {
    const auto remaining = task.remainingTimeout();
    if (!remaining) {
        return false;
    }
    request.set_timeout_ms(remaining->count());
    const auto&                             descriptors = task.descriptors();
    const TransferDescriptor&               first       = descriptors.front();
    MemoryOperationRequestPB::CopyDirection request_direction;
    if (!directionFor(first, request_direction)) {
        return false;
    }
    request.set_copy_direction(request_direction);

    for (const TransferDescriptor& descriptor : descriptors) {
        if (descriptor.source_tier != first.source_tier || descriptor.target_tier != first.target_tier) {
            return false;
        }
        if (descriptor.group_set_id >= group_sets.size() || !group_sets[descriptor.group_set_id]) {
            return false;
        }
        const GroupSet& group_set = *group_sets[descriptor.group_set_id];
        CopyItem        item;
        for (const auto& tag : group_set.groupTags()) {
            item.add_group_tags(tag);
        }

        if (descriptor.source_tier == Tier::HOST || descriptor.target_tier == Tier::HOST) {
            item.set_mem_block(descriptor.singleBlockAt(Tier::HOST));
        }
        if (descriptor.source_tier == Tier::DISK || descriptor.target_tier == Tier::DISK) {
            item.set_disk_block(descriptor.singleBlockAt(Tier::DISK));
        }
        if (descriptor.source_tier == Tier::DEVICE || descriptor.target_tier == Tier::DEVICE) {
            const auto& blocks     = descriptor.blocksAt(Tier::DEVICE);
            const auto& group_tags = group_set.groupTags();
            if (blocks.size() != group_tags.size()) {
                return false;
            }
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto* group_block = item.add_group_blocks();
                group_block->set_tag(group_tags[i]);
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

    std::vector<TransferDescriptor> decoded;
    decoded.reserve(static_cast<size_t>(request.copy_items_size()));
    for (const CopyItem& item : request.copy_items()) {
        const GroupSet* group_set_ptr = resolveGroupSet(item, group_sets);
        if (group_set_ptr == nullptr) {
            RTP_LLM_LOG_WARNING("cannot uniquely resolve BlockTree GroupSet membership");
            return false;
        }
        const GroupSet&    group_set = *group_set_ptr;
        TransferDescriptor descriptor;
        switch (request.copy_direction()) {
            case MemoryOperationRequestPB::D2H: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::deviceToHost(
                    group_set.groupSetId(), std::move(device_blocks), item.mem_block());
                break;
            }
            case MemoryOperationRequestPB::H2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::hostToDevice(
                    group_set.groupSetId(), item.mem_block(), std::move(device_blocks));
                break;
            }
            case MemoryOperationRequestPB::H2DISK:
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())) {
                    return false;
                }
                descriptor =
                    TransferDescriptor::hostToDisk(group_set.groupSetId(), item.mem_block(), item.disk_block());
                break;
            case MemoryOperationRequestPB::DISK2H:
                if (!group_set.hostPool() || !group_set.hostPool()->validBlock(item.mem_block())
                    || !group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())) {
                    return false;
                }
                descriptor =
                    TransferDescriptor::diskToHost(group_set.groupSetId(), item.disk_block(), item.mem_block());
                break;
            case MemoryOperationRequestPB::D2DISK: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())
                    || !decodeDeviceBlocks(item, group_set, device_blocks)) {
                    return false;
                }
                descriptor = TransferDescriptor::deviceToDisk(
                    group_set.groupSetId(), std::move(device_blocks), item.disk_block());
                break;
            }
            case MemoryOperationRequestPB::DISK2D: {
                std::vector<BlockIdxType> device_blocks;
                if (!group_set.diskPool() || !group_set.diskPool()->validBlock(item.disk_block())
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
        decoded.push_back(std::move(descriptor));
    }
    descriptors.insert(
        descriptors.end(), std::make_move_iterator(decoded.begin()), std::make_move_iterator(decoded.end()));
    return true;
}

}  // namespace rtp_llm
