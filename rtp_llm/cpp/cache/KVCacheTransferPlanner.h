#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag);

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);
std::vector<size_t> blockPositionsForCacheTransfer(size_t block_num,
                                                   size_t reuse_block_size,
                                                   bool   use_hybrid,
                                                   bool   transfer_tail_blocks,
                                                   size_t tail_block_count,
                                                   bool   hybrid_full_from_begin);

// The cache_store registration plan (CacheStoreBlockPair + buildCacheStorePlan)
// lives in CacheGroupType.h, keyed on CacheGroupPolicy rather than on a bare
// CacheGroupType, and is reached through CPSlotMapper::buildStorePlan(). Do not
// redeclare either here: this header includes CacheGroupType.h, so a second
// definition of CacheStoreBlockPair in namespace rtp_llm is a redefinition
// error in every translation unit that includes this file.

}  // namespace rtp_llm
