#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

#include <algorithm>

namespace rtp_llm {

void GroupSetResource::evictFromTier(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            std::fill(device_blocks.begin(), device_blocks.end(), NULL_BLOCK_IDX);
            break;
        case Tier::HOST:
            host_block = NULL_BLOCK_IDX;
            break;
        case Tier::DISK:
            disk_block = NULL_BLOCK_IDX;
            break;
        default:
            break;
    }
}

std::vector<BlockIdxType> GroupSetResource::getBlocks(Tier tier) const {
    if (!hasTier(tier)) {
        return {};
    }
    switch (tier) {
        case Tier::DEVICE:
            return device_blocks;
        case Tier::HOST:
            return {host_block};
        case Tier::DISK:
            return {disk_block};
        default:
            return {};
    }
}

void GroupSetResource::setBlocks(Tier tier, const std::vector<BlockIdxType>& blocks) {
    switch (tier) {
        case Tier::DEVICE:
            device_blocks = blocks;
            break;
        case Tier::HOST:
            host_block = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        case Tier::DISK:
            disk_block = blocks.empty() ? NULL_BLOCK_IDX : blocks[0];
            break;
        default:
            break;
    }
}

}  // namespace rtp_llm
