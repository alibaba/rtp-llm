#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);
std::vector<size_t> blockPositionsForCacheTransfer(size_t block_num,
                                                   size_t reuse_block_size,
                                                   bool   use_hybrid,
                                                   bool   transfer_tail_blocks,
                                                   size_t tail_block_count,
                                                   bool   hybrid_full_from_begin);

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag);

struct NativeTransferSelection {
    std::string         tag;
    std::vector<size_t> global_positions;
    std::vector<size_t> owned_global_positions;
    std::vector<size_t> local_positions;
};

using NativeTransferSelections = std::unordered_map<std::string, NativeTransferSelection>;

NativeTransferSelection projectTokenRangeForGroup(const GroupBase& group,
                                                  size_t           start_token,
                                                  size_t           end_token,
                                                  bool             require_aligned_range,
                                                  int              cp_rank = 0,
                                                  int              cp_size = 1);

}  // namespace rtp_llm
