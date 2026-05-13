#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(size_t         block_num,
                                                   size_t         reuse_block_size,
                                                   bool           use_hybrid,
                                                   CacheGroupType group_type,
                                                   bool           hybrid_full_from_begin) {
    std::vector<size_t> block_pos_list;
    block_pos_list.reserve(block_num);
    if (use_hybrid && block_num > 0 && group_type == CacheGroupType::LINEAR) {
        block_pos_list.push_back(block_num - 1);
        return block_pos_list;
    }
    if (use_hybrid && block_num > 0 && group_type == CacheGroupType::SWA) {
        const size_t start = block_num > 2 ? block_num - 2 : 0;
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

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         reuse_block_size,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size) {
    auto positions = blockPositionsForCacheTransfer(
        total_logical_blocks, reuse_block_size, use_hybrid, group_type, /*hybrid_full_from_begin=*/true);

    std::vector<CacheStoreBlockPair> plan;
    plan.reserve(positions.size());

    const bool sharded = (cp_size > 1) && (group_type == CacheGroupType::FULL);
    if (!sharded) {
        for (auto pos : positions) {
            const int p = static_cast<int>(pos);
            plan.push_back({p, p});
        }
        return plan;
    }
    for (auto pos : positions) {
        const int p = static_cast<int>(pos);
        if (p % cp_size != cp_rank) {
            continue;
        }
        plan.push_back({p, p / cp_size});
    }
    return plan;
}

std::string layerRegionCacheTransferKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (region_name != KVCacheRegionName::DEFAULT) {
        key += "-" + std::to_string(static_cast<int>(region_name));
    }
    return key;
}

}  // namespace rtp_llm
