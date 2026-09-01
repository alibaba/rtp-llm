#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>

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

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t            total_logical_blocks,
                                                          size_t            reuse_block_size,
                                                          bool              use_hybrid,
                                                          CacheGroupType    group_type,
                                                          int               cp_rank,
                                                          int               cp_size,
                                                          KVCacheRegionName region_name) {
    std::vector<CacheStoreBlockPair> plan;

    const bool sharded_full      = (cp_size > 1) && (group_type == CacheGroupType::FULL);
    const bool compact_swa_by_cp = (cp_size > 1) && (group_type == CacheGroupType::SWA);
    if (compact_swa_by_cp) {
        const size_t cp_size_t        = static_cast<size_t>(cp_size);
        const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
        if (canonical_blocks == 0) {
            return plan;
        }
        if (isStateRegion(region_name)) {
            // Recurrent compressor state is a carry, not an SWA token window.
            // Decode needs only the final state row.  Keeping the older row
            // also collides under the request-scoped synthetic state key.
            plan.push_back({static_cast<int>(total_logical_blocks - 1), static_cast<int>(canonical_blocks - 1)});
            return plan;
        }
        const size_t start = use_hybrid ? (canonical_blocks > 2 ? canonical_blocks - 2 : 0) :
                                          std::min(reuse_block_size, canonical_blocks);
        plan.reserve(canonical_blocks - start);
        for (size_t compact_idx = start; compact_idx < canonical_blocks; ++compact_idx) {
            const size_t key_index = std::min((compact_idx + 1) * cp_size_t - 1, total_logical_blocks - 1);
            plan.push_back({static_cast<int>(key_index), static_cast<int>(compact_idx)});
        }
        return plan;
    }

    auto positions = blockPositionsForCacheTransfer(
        total_logical_blocks, reuse_block_size, use_hybrid, group_type, /*hybrid_full_from_begin=*/true);

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

bool needsSegmentedLinearFanIn(bool use_mla, int attn_tp_size, size_t peer_count, bool has_segmented_linear_group) {
    return use_mla && attn_tp_size == 1 && peer_count > 1 && has_segmented_linear_group;
}

std::string cacheTransferTokenKey(const std::string& cache_key, int cp_size, KVCacheRegionName region_name) {
    return cp_size > 1 && isStateRegion(region_name) ? "0" : cache_key;
}

std::string layerRegionCacheTransferKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (region_name != KVCacheRegionName::DEFAULT) {
        key += "-" + std::to_string(static_cast<int>(region_name));
    }
    return key;
}

}  // namespace rtp_llm
