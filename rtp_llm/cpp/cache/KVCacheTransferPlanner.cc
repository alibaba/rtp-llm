#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(size_t         block_num,
                                                   size_t         reuse_block_size,
                                                   bool           use_hybrid,
                                                   CacheGroupType group_type,
                                                   bool           hybrid_full_from_begin) {
    return blockPositionsForCacheTransfer(
        block_num,
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

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t         total_logical_blocks,
                                                          size_t         reuse_block_size,
                                                          bool           use_hybrid,
                                                          CacheGroupType group_type,
                                                          int            cp_rank,
                                                          int            cp_size) {
    const auto policy = defaultCacheGroupPolicy(group_type);
    return buildCacheStoreBlockPlan(total_logical_blocks,
                                    reuse_block_size,
                                    use_hybrid,
                                    /*cp_shardable=*/group_type == CacheGroupType::FULL,
                                    /*cp_compact_tail_blocks=*/group_type == CacheGroupType::SWA,
                                    static_cast<size_t>(policy.active_tail_blocks),
                                    cp_rank,
                                    cp_size);
}

std::vector<CacheStoreBlockPair> buildCacheStoreBlockPlan(size_t                      total_logical_blocks,
                                                          size_t                      reuse_block_size,
                                                          bool                        use_hybrid,
                                                          bool                        cp_shardable,
                                                          bool                        cp_compact_tail_blocks,
                                                          size_t                      tail_block_count,
                                                          int                         cp_rank,
                                                          int                         cp_size) {
    std::vector<CacheStoreBlockPair> plan;

    const bool sharded_full      = (cp_size > 1) && cp_shardable;
    const bool compact_swa_by_cp = (cp_size > 1) && cp_compact_tail_blocks;
    if (compact_swa_by_cp) {
        const size_t cp_size_t        = static_cast<size_t>(cp_size);
        const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
        const size_t tail_count = std::max<size_t>(1, tail_block_count);
        const size_t start = use_hybrid ? (canonical_blocks > tail_count ? canonical_blocks - tail_count : 0) :
                                          std::min(reuse_block_size, canonical_blocks);
        plan.reserve(canonical_blocks - start);
        for (size_t compact_idx = start; compact_idx < canonical_blocks; ++compact_idx) {
            const size_t key_index = std::min((compact_idx + 1) * cp_size_t - 1, total_logical_blocks - 1);
            plan.push_back({static_cast<int>(key_index), static_cast<int>(compact_idx)});
        }
        return plan;
    }

    auto positions = blockPositionsForCacheTransfer(total_logical_blocks,
                                                    reuse_block_size,
                                                    use_hybrid,
                                                    /*transfer_tail_blocks=*/tail_block_count > 0,
                                                    tail_block_count,
                                                    /*hybrid_full_from_begin=*/true);

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

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (!tag.empty() && tag != "default") {
        key += "-tag-" + tag;
    }
    return key;
}

}  // namespace rtp_llm
