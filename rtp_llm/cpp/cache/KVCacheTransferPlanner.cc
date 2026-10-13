#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(size_t         block_num,
                                                   size_t         reuse_block_size,
                                                   bool           use_hybrid,
                                                   CacheGroupType group_type,
                                                   bool           hybrid_full_from_begin) {
    return blockPositionsForCacheTransfer(block_num,
                                          reuse_block_size,
                                          use_hybrid,
                                          /*transfer_tail_blocks=*/group_type != CacheGroupType::FULL,
                                          static_cast<size_t>(defaultCacheGroupPolicy(group_type).active_tail_blocks),
                                          hybrid_full_from_begin);
}

std::vector<size_t> blockPositionsForCacheTransfer(size_t block_num,
                                                   size_t reuse_block_size,
                                                   bool   use_hybrid,
                                                   bool   transfer_tail_blocks,
                                                   size_t tail_block_count,
                                                   bool   hybrid_full_from_begin) {
    std::vector<size_t> block_pos_list;
    block_pos_list.reserve(block_num);
    if (use_hybrid && block_num > 0 && transfer_tail_blocks) {
        const size_t tail_count = std::max<size_t>(1, tail_block_count);
        const size_t start      = block_num > tail_count ? block_num - tail_count : 0;
        for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
            block_pos_list.push_back(block_pos);
        }
        return block_pos_list;
    }
    const size_t start = use_hybrid && hybrid_full_from_begin ? 0 : reuse_block_size;
    for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
        block_pos_list.push_back(block_pos);
    }
    return block_pos_list;
}

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (!tag.empty() && tag != "default") {
        key += "-tag-" + tag;
    }
    return key;
}

}  // namespace rtp_llm
