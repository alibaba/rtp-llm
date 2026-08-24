#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>
#include <stdexcept>

namespace rtp_llm {

std::vector<size_t>
blockPositionsForCacheTransfer(size_t block_num, size_t first_full_block, bool use_hybrid, CacheGroupType group_type) {
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
    const size_t start = std::min(first_full_block, block_num);
    for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
        block_pos_list.push_back(block_pos);
    }
    return block_pos_list;
}

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         first_full_block,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size) {
    std::vector<CacheStoreBlockPair> plan;

    const bool sharded_full        = (cp_size > 1) && (group_type == CacheGroupType::FULL);
    const bool compact_swa_by_cp   = (cp_size > 1) && (group_type == CacheGroupType::SWA);
    if (compact_swa_by_cp) {
        const size_t cp_size_t        = static_cast<size_t>(cp_size);
        const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
        const size_t start            = use_hybrid ? (canonical_blocks > 2 ? canonical_blocks - 2 : 0) :
                                                     std::min(first_full_block, canonical_blocks);
        plan.reserve(canonical_blocks - start);
        for (size_t compact_idx = start; compact_idx < canonical_blocks; ++compact_idx) {
            const size_t key_index = std::min((compact_idx + 1) * cp_size_t - 1, total_logical_blocks - 1);
            plan.push_back({static_cast<int>(key_index), static_cast<int>(compact_idx)});
        }
        return plan;
    }

    auto positions = blockPositionsForCacheTransfer(total_logical_blocks, first_full_block, use_hybrid, group_type);

    plan.reserve(positions.size());

    if (!sharded_full && !compact_swa_by_cp) {
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

std::vector<CacheStoreBlockPair>
buildIncrementalCacheStoreBlockPlan(size_t                        total_logical_blocks,
                                    size_t                        reuse_block_size,
                                    bool                          use_hybrid,
                                    CacheGroupType                group_type,
                                    int                           cp_rank,
                                    int                           cp_size,
                                    const CacheStorePublishRange& publish_range) {
    if (publish_range.begin_block > publish_range.end_block
        || publish_range.end_block > total_logical_blocks) {
        throw std::invalid_argument("incremental cache-store range is outside the logical block table");
    }

    if (group_type == CacheGroupType::LINEAR) {
        if (!publish_range.terminal) {
            return {};
        }
        if (publish_range.end_block != total_logical_blocks) {
            throw std::invalid_argument("terminal LINEAR publication must reach the final logical block");
        }
        return buildCacheStoreBlockPlan(
            total_logical_blocks, reuse_block_size, /*use_hybrid=*/true, group_type, cp_rank, cp_size);
    }
    if (group_type != CacheGroupType::FULL) {
        throw std::invalid_argument("incremental cache-store only supports FULL and LINEAR groups");
    }

    auto plan = buildCacheStoreBlockPlan(
        total_logical_blocks, reuse_block_size, use_hybrid, group_type, cp_rank, cp_size);

    plan.erase(std::remove_if(plan.begin(),
                              plan.end(),
                              [&](const CacheStoreBlockPair& pair) {
                                  const size_t key_index = static_cast<size_t>(pair.key_index);
                                  return key_index < publish_range.begin_block || key_index >= publish_range.end_block;
                              }),
               plan.end());
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
