#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"

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

NativeTransferSelection projectTokenRangeForGroup(const GroupBase& group,
                                                  size_t           start_token,
                                                  size_t           end_token,
                                                  bool             require_aligned_range,
                                                  int              cp_rank,
                                                  int              cp_size) {
    RTP_LLM_CHECK_WITH_INFO(group.seq_size_per_block > 0, "transfer projector requires a positive group span");
    RTP_LLM_CHECK_WITH_INFO(end_token >= start_token, "transfer projector received an inverted token range");
    RTP_LLM_CHECK_WITH_INFO(cp_size > 0 && cp_rank >= 0 && cp_rank < cp_size,
                            "transfer projector received invalid CP rank/size");
    const size_t span = group.seq_size_per_block;
    if (require_aligned_range) {
        RTP_LLM_CHECK_WITH_INFO(start_token % span == 0 && end_token % span == 0,
                                "prefix transfer range [%zu,%zu) is not aligned to tag=%s span=%zu",
                                start_token,
                                end_token,
                                group.tag.c_str(),
                                span);
    }
    const size_t begin = start_token / span;
    const size_t end   = require_aligned_range ? end_token / span : (end_token + span - 1) / span;

    NativeTransferSelection result;
    result.tag = group.tag;
    if (end <= begin) {
        return result;
    }
    size_t selected_begin = begin;
    if (group.policy.group_type != CacheGroupType::FULL) {
        const size_t tail = std::max<size_t>(1, group.policy.active_tail_blocks);
        selected_begin    = end > tail ? std::max(begin, end - tail) : begin;
    }
    for (size_t ordinal = selected_begin; ordinal < end; ++ordinal) {
        result.global_positions.push_back(ordinal);
        if (cp_size == 1 || group.policy.cp_mapping == CpBlockMappingMode::NONE) {
            result.owned_global_positions.push_back(ordinal);
            result.local_positions.push_back(ordinal);
        } else if (group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
            if (ordinal % static_cast<size_t>(cp_size) == static_cast<size_t>(cp_rank)) {
                result.owned_global_positions.push_back(ordinal);
                result.local_positions.push_back(ordinal / static_cast<size_t>(cp_size));
            }
        } else {
            const size_t compact = ordinal / static_cast<size_t>(cp_size);
            if (result.local_positions.empty() || result.local_positions.back() != compact) {
                result.owned_global_positions.push_back(ordinal);
                result.local_positions.push_back(compact);
            } else {
                result.owned_global_positions.back() = ordinal;
            }
        }
    }
    return result;
}

}  // namespace rtp_llm
