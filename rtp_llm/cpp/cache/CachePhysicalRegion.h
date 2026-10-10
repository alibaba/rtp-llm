#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

// One selected logical cache block's locally addressable part. The key range
// is half-open; buffer.addr already points at this rank's physical fragment.
struct CacheBlockRegion {
    size_t    key_index       = 0;
    size_t    key_range_begin = 0;
    size_t    key_range_end   = 0;
    size_t    local_slot      = 0;
    size_t    part_index      = 0;
    size_t    content_offset  = 0;
    BlockInfo buffer;
};

// Select one producer peer's contiguous bytes within a complete local decode
// block. This is a physical format query and does not select the peer or plan.
BlockInfo sliceDecodeCacheBlock(
    const GroupBase& group, CpBlockSliceMode slice_mode, BlockInfo full_block, int producer_cp_size, int peer_index);

// The caller selects a logical page and transfer policy. Cache resolves this
// rank's slot, physical block address, and part geometry; a null source block
// returns no regions, matching the existing producer skip behavior.
std::vector<CacheBlockRegion> queryProducerCacheBlockRegions(const CacheConfig&            config,
                                                             int                           layer_id,
                                                             const std::string&            tag,
                                                             size_t                        key_count,
                                                             size_t                        logical_page,
                                                             int                           producer_rank,
                                                             int                           producer_cp_size,
                                                             bool                          opaque_wire,
                                                             const int32_t*                block_ids,
                                                             size_t                        block_id_count,
                                                             const std::vector<BlockInfo>& part_bases);

}  // namespace rtp_llm
