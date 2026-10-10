#include "rtp_llm/cpp/cache/CachePhysicalRegion.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

BlockInfo sliceDecodeCacheBlock(
    const GroupBase& group, CpBlockSliceMode slice_mode, BlockInfo full_block, int producer_cp_size, int peer_index) {
    RTP_LLM_CHECK_WITH_INFO(slice_mode != CpBlockSliceMode::NONE && producer_cp_size > 0 && peer_index >= 0
                                && peer_index < producer_cp_size,
                            "decode cache slice requires valid mode and producer peer");
    RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr && group.kvScaleStrideBytes() == 0,
                            "decode cache slice requires one KV physical part");
    const size_t full_region = slice_mode == CpBlockSliceMode::PAYLOAD_BYTES ? group.spec->k_block_payload_bytes() :
                                                                               group.kvBlockStrideBytes();
    RTP_LLM_CHECK_WITH_INFO(full_region > 0 && full_region % static_cast<size_t>(producer_cp_size) == 0,
                            "decode cache slice does not divide physical region");
    const size_t slice_bytes = full_region / static_cast<size_t>(producer_cp_size);
    const size_t offset      = static_cast<size_t>(peer_index) * slice_bytes;
    RTP_LLM_CHECK_WITH_INFO(full_block.addr != nullptr && offset <= full_block.size_bytes
                                && slice_bytes <= full_block.size_bytes - offset,
                            "decode cache slice exceeds physical destination block");
    full_block.addr       = static_cast<uint8_t*>(full_block.addr) + offset;
    full_block.size_bytes = slice_bytes;
    return full_block;
}

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
                                                             const std::vector<BlockInfo>& part_bases) {
    RTP_LLM_CHECK_WITH_INFO(producer_cp_size > 0 && producer_rank >= 0 && producer_rank < producer_cp_size,
                            "producer cache region has invalid CP topology");
    RTP_LLM_CHECK_WITH_INFO(block_ids != nullptr && !part_bases.empty() && part_bases.size() <= 2,
                            "producer cache region requires block IDs and one or two tensor bases");
    const auto&  group         = config.physicalGroupForLayer(layer_id, tag);
    const size_t scale         = config.groupKeyScale(tag);
    const size_t logical_pages = config.logicalBlockCountForGroup(tag, key_count);
    RTP_LLM_CHECK_WITH_INFO(logical_page < logical_pages, "producer cache logical page is out of range");
    const auto& policy = config.group(tag).policy;
    const bool  rr     = policy.group_type == CacheGroupType::FULL && producer_cp_size > 1
                    && policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    if (rr && logical_page % static_cast<size_t>(producer_cp_size) != static_cast<size_t>(producer_rank)) {
        return {};
    }
    const size_t local_slot = rr ? logical_page / static_cast<size_t>(producer_cp_size) : logical_page;
    RTP_LLM_CHECK_WITH_INFO(local_slot < block_id_count, "producer cache local slot is outside block ID table");
    const int32_t block_id = block_ids[local_slot];
    if (block_id == -1) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(block_id >= 0, "producer cache block ID is negative");

    const size_t key_range_begin = scale == 1 ? logical_page : logical_page * scale;
    const size_t key_range_end   = std::min(key_count, (logical_page + 1) * scale);
    const size_t key_index       = key_range_end - 1;
    const size_t kv_bytes        = group.kvBlockStrideBytes();
    const size_t scale_bytes     = group.kvScaleStrideBytes();
    // A scale allocation in the spec does not require a scale tensor in every
    // producer view. Preserve the writer's existing optional-scale contract.
    RTP_LLM_CHECK_WITH_INFO(kv_bytes > 0 && (part_bases.size() == 1 || scale_bytes > 0),
                            "producer cache tensor bases do not match physical spec");
    RTP_LLM_CHECK_WITH_INFO(opaque_wire || (kv_bytes % 2 == 0 && scale_bytes % 2 == 0),
                            "producer cache K/V parts must split evenly");
    const bool sliced =
        policy.group_type != CacheGroupType::FULL && policy.cp_slice != CpBlockSliceMode::NONE && producer_cp_size > 1;
    const size_t                  part_count = opaque_wire ? part_bases.size() : part_bases.size() * 2;
    std::vector<CacheBlockRegion> regions;
    regions.reserve(part_count);
    for (size_t base_index = 0; base_index < part_bases.size(); ++base_index) {
        const auto&  base   = part_bases[base_index];
        const size_t stride = base_index == 0 ? kv_bytes : scale_bytes;
        RTP_LLM_CHECK_WITH_INFO(base.addr != nullptr && static_cast<size_t>(block_id) <= base.size_bytes / stride
                                    && stride <= base.size_bytes - static_cast<size_t>(block_id) * stride,
                                "producer cache block exceeds tensor base");
        const size_t part_size   = opaque_wire ? stride : stride / 2;
        const size_t split_count = opaque_wire ? 1 : 2;
        for (size_t split = 0; split < split_count; ++split) {
            CacheBlockRegion region;
            region.key_index       = key_index;
            region.key_range_begin = key_range_begin;
            region.key_range_end   = key_range_end;
            region.local_slot      = local_slot;
            region.part_index      = opaque_wire ? base_index : base_index * 2 + split;
            region.content_offset  = sliced ? static_cast<size_t>(producer_rank) * part_size : 0;
            region.buffer          = base;
            region.buffer.addr =
                static_cast<uint8_t*>(base.addr) + static_cast<size_t>(block_id) * stride + split * part_size;
            region.buffer.size_bytes = part_size;
            regions.push_back(region);
        }
    }
    return regions;
}

}  // namespace rtp_llm
