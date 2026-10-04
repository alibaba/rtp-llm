#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

// Page placement and attention-head sharding are independent. A Decode rank
// selects the Prefill owner of each logical page, then reshards KDA state by
// attention TP. DP replicas use the same local attention rank mapping.
struct HeadShardLoadPlan {
    bool selected = false;
    int source_partition_count = 1;
    int source_partition_id = 0;
    int destination_partition_count = 1;
    int destination_partition_id = 0;
};

bool supportsHeadShardTransfer(int source_tp, int destination_tp);
HeadShardLoadPlan planHeadShardLoad(int source_tp, int destination_tp,
                                    int source_rank, int destination_rank);
bool pageOwnedBySource(size_t logical_page, int source_rank, int source_tp);

std::vector<size_t> blockPositionsForCacheTransfer(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);
std::vector<size_t> blockPositionsForCacheTransfer(size_t block_num,
                                                   size_t reuse_block_size,
                                                   bool   use_hybrid,
                                                   bool   transfer_tail_blocks,
                                                   size_t tail_block_count,
                                                   bool   hybrid_full_from_begin);

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag);

// The cache_store registration plan (CacheStoreBlockPair + buildCacheStorePlan)
// lives in CacheGroupType.h, keyed on CacheGroupPolicy rather than on a bare
// CacheGroupType, and is reached through CPSlotMapper::buildStorePlan(). Do not
// redeclare either here: this header includes CacheGroupType.h, so a second
// definition of CacheStoreBlockPair in namespace rtp_llm is a redefinition
// error in every translation unit that includes this file.

}  // namespace rtp_llm
