#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>
#include <stdexcept>

namespace rtp_llm {

bool isK3PageRRToReplicatedDecode(int prefill_attention_tp,
                                  int decode_attention_tp,
                                  int source_shards,
                                  int peer_count,
                                  int configured_upstream_shards) {
    return source_shards > 1 && source_shards == prefill_attention_tp && peer_count == source_shards
           && source_shards == configured_upstream_shards && decode_attention_tp == 1;
}

K3CacheLoadSourcePlan planK3CacheLoadSource(
    K3CacheLoadSourcePolicy policy, size_t block_position, int peer_index, int peer_count, int decode_dp_rank) {
    if (peer_count <= 0 || peer_index < 0 || peer_index >= peer_count || decode_dp_rank < 0) {
        throw std::invalid_argument("invalid K3 cache-load source coordinates");
    }

    switch (policy) {
        case K3CacheLoadSourcePolicy::PAGE_OWNER:
            return {block_position % static_cast<size_t>(peer_count) == static_cast<size_t>(peer_index), 1, 0};
        case K3CacheLoadSourcePolicy::ALL_PEER_PARTITION:
            return {true, peer_count, peer_index};
        case K3CacheLoadSourcePolicy::SINGLE_REPLICA:
            return {peer_index == decode_dp_rank % peer_count, 1, 0};
    }
    throw std::invalid_argument("unknown K3 cache-load source policy");
}

bool usesVirtualBlockCacheLayout(CacheGroupType group_type,
                                 size_t         physical_page_tokens,
                                 size_t         group_block_tokens,
                                 int            shard_size) {
    if (group_type == CacheGroupType::FULL || physical_page_tokens == 0 || shard_size <= 1) {
        return false;
    }
    return group_block_tokens % physical_page_tokens == 0
           && group_block_tokens / physical_page_tokens == static_cast<size_t>(shard_size);
}

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
                                                          int            cp_size,
                                                          bool           virtual_block_cache_layout) {
    std::vector<CacheStoreBlockPair> plan;
    if (total_logical_blocks == 0) {
        return plan;
    }
    if (cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        throw std::invalid_argument("invalid cache-store CP rank/size");
    }

    const bool sharded_full           = (cp_size > 1) && (group_type == CacheGroupType::FULL);
    const bool compact_virtual_blocks = (cp_size > 1) && virtual_block_cache_layout
                                        && (group_type == CacheGroupType::SWA || group_type == CacheGroupType::LINEAR);
    if (compact_virtual_blocks) {
        const size_t cp_size_t        = static_cast<size_t>(cp_size);
        const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
        const size_t retained_tail    = group_type == CacheGroupType::LINEAR ? 1 : 2;
        const size_t start = use_hybrid ? (canonical_blocks > retained_tail ? canonical_blocks - retained_tail : 0) :
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

    if (!sharded_full) {
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

std::vector<CacheStoreBlockPair> buildIncrementalCacheStoreBlockPlan(size_t                        total_logical_blocks,
                                                                     size_t                        reuse_block_size,
                                                                     bool                          use_hybrid,
                                                                     CacheGroupType                group_type,
                                                                     int                           cp_rank,
                                                                     int                           cp_size,
                                                                     const CacheStorePublishRange& publish_range,
                                                                     bool virtual_block_cache_layout) {
    if (publish_range.begin_block > publish_range.end_block || publish_range.end_block > total_logical_blocks) {
        throw std::invalid_argument("incremental cache-store range is outside the logical block table");
    }

    if (group_type == CacheGroupType::LINEAR || group_type == CacheGroupType::SWA) {
        // Publish retained checkpoints/tail pages at the terminal chunk. The
        // SWA tail can include a page preceding this chunk's begin_block.
        if (!publish_range.terminal) {
            return {};
        }
        if (publish_range.end_block != total_logical_blocks) {
            throw std::invalid_argument("terminal non-FULL publication must reach the final logical block");
        }
        return buildCacheStoreBlockPlan(total_logical_blocks,
                                        reuse_block_size,
                                        /*use_hybrid=*/true,
                                        group_type,
                                        cp_rank,
                                        cp_size,
                                        virtual_block_cache_layout);
    }
    if (group_type != CacheGroupType::FULL) {
        throw std::invalid_argument("incremental cache-store has an unsupported cache group");
    }

    auto plan = buildCacheStoreBlockPlan(
        total_logical_blocks, reuse_block_size, use_hybrid, group_type, cp_rank, cp_size, virtual_block_cache_layout);

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
