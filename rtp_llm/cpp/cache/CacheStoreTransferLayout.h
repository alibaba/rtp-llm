#pragma once

#include <numeric>
#include <optional>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"

namespace rtp_llm {

struct CacheStoreTransferSegment {
    std::string key_prefix;
    size_t      offset_bytes;
    size_t      size_bytes;
    bool        is_scale;
};

inline bool hasLinearCacheGroup(const CacheTopology& topology) {
    for (const auto& group : topology.groups()) {
        if (group.spec->type == KVCacheSpecType::LinearAttention) {
            return true;
        }
    }
    return false;
}

// The producer and consumer must describe the same logical segments. Physical
// pool padding is not part of the payload, and must never be divided by TP.
inline std::vector<CacheStoreTransferSegment> cacheStoreTransferSegments(const GroupBase& group,
                                                                         bool             has_linear_cache) {
    std::vector<CacheStoreTransferSegment> segments;
    if (group.spec->type == KVCacheSpecType::LinearAttention) {
        const auto* linear = dynamic_cast<const LinearKVCacheSpec*>(group.spec.get());
        RTP_LLM_CHECK_WITH_INFO(linear != nullptr, "linear cache tag=%s has no linear spec", group.tag.c_str());
        size_t offset = 0;
        for (const size_t bytes : linear->cacheStoreSegmentSizes()) {
            RTP_LLM_CHECK_WITH_INFO(bytes > 0, "linear cache tag=%s has an empty segment", group.tag.c_str());
            segments.push_back({"linear_segment_" + std::to_string(segments.size()) + "_", offset, bytes, false});
            offset += bytes;
        }
        RTP_LLM_CHECK_WITH_INFO(offset == group.spec->block_size_bytes() && offset <= group.kv_block_stride_bytes,
                                "linear cache tag=%s segments=%zu exceed or mismatch payload/stride=%zu/%zu",
                                group.tag.c_str(),
                                offset,
                                group.spec->block_size_bytes(),
                                group.kv_block_stride_bytes);
    } else if (group.spec->type == KVCacheSpecType::MultiHeadAttention
               && (has_linear_cache
                   || (group.kernel_seq_size_per_block > 0
                       && group.seq_size_per_block != group.kernel_seq_size_per_block))) {
        // LayerKVCache presents HND kernel pages, each with its own K and V.
        // A physical 2048-token block with 64-token pages contains 32 [K,V]
        // pairs; slicing one opaque physical block cannot preserve these axes.
        // This also applies to a full-only MTP cache with no linear group.
        size_t pages = 1;
        if (group.policy.group_type == CacheGroupType::FULL) {
            RTP_LLM_CHECK_WITH_INFO(group.kernel_seq_size_per_block > 0
                                        && group.seq_size_per_block % group.kernel_seq_size_per_block == 0,
                                    "invalid MHA kernel page size for tag=%s",
                                    group.tag.c_str());
            pages = group.seq_size_per_block / group.kernel_seq_size_per_block;
        }
        auto addPages = [&](size_t total_bytes, bool is_scale) {
            RTP_LLM_CHECK_WITH_INFO(pages > 0 && total_bytes > 0 && total_bytes % (2 * pages) == 0,
                                    "invalid MHA payload bytes=%zu pages=%zu tag=%s",
                                    total_bytes,
                                    pages,
                                    group.tag.c_str());
            const size_t half_page = total_bytes / (2 * pages);
            for (size_t page = 0; page < pages; ++page) {
                const auto suffix =
                    std::string(is_scale ? "_scale_segment_" : "_segment_") + std::to_string(page) + "_";
                segments.push_back({"k" + suffix, 2 * page * half_page, half_page, is_scale});
                segments.push_back({"v" + suffix, (2 * page + 1) * half_page, half_page, is_scale});
            }
        };
        addPages(group.kv_block_stride_bytes, false);
        if (group.kv_scale_stride_bytes > 0) {
            addPages(group.kv_scale_stride_bytes, true);
        }
    }
    return segments;
}

struct CacheStorePeerPartition {
    int count;
    int id;
};

// P>=D: the peer list covers one decode TP rank's attention heads.
// MHA weights use contiguous replica groups (see sp_head / _kv_rank), while
// linear state is sharded across every TP rank. Select one source per MHA
// replica group without dropping the other peers' linear segments.
inline std::optional<CacheStorePeerPartition>
cacheStorePeerPartition(const GroupBase& group, int peer_count, int peer_index) {
    RTP_LLM_CHECK_WITH_INFO(peer_count > 0 && peer_index >= 0 && peer_index < peer_count,
                            "invalid cache-store peer %d/%d",
                            peer_index,
                            peer_count);
    if (group.spec->type == KVCacheSpecType::MultiHeadAttention) {
        RTP_LLM_CHECK_WITH_INFO(group.local_kv_head_num > 0, "MHA cache tag=%s has no heads", group.tag.c_str());
        const int unique_peers = std::gcd(static_cast<uint32_t>(peer_count), group.local_kv_head_num);
        const int replicas     = peer_count / unique_peers;
        if (peer_index % replicas != 0) {
            return std::nullopt;
        }
        return CacheStorePeerPartition{unique_peers, peer_index / replicas};
    }
    return CacheStorePeerPartition{peer_count, peer_index};
}

inline std::vector<BlockInfo> cacheStoreDestinationSegments(const GroupBase&                              group,
                                                            const std::vector<CacheStoreTransferSegment>& segments,
                                                            const std::vector<BlockInfo>&                 blocks,
                                                            int partition_count,
                                                            int partition_id) {
    RTP_LLM_CHECK_WITH_INFO(partition_count > 0 && partition_id >= 0 && partition_id < partition_count,
                            "invalid cache-store partition %d/%d",
                            partition_id,
                            partition_count);
    if (const auto* linear = dynamic_cast<const LinearKVCacheSpec*>(group.spec.get())) {
        linear->checkCacheStorePartition(partition_count);
    } else {
        RTP_LLM_CHECK_WITH_INFO(group.local_kv_head_num > 0 && group.local_kv_head_num % partition_count == 0,
                                "MHA cache tag=%s cannot partition %u heads into %d peers",
                                group.tag.c_str(),
                                group.local_kv_head_num,
                                partition_count);
    }
    std::vector<BlockInfo> result;
    result.reserve(segments.size());
    for (const auto& segment : segments) {
        const size_t index = segment.is_scale ? 1 : 0;
        RTP_LLM_CHECK_WITH_INFO(index < blocks.size(), "missing cache buffer for tag=%s", group.tag.c_str());
        auto block = blocks[index];
        RTP_LLM_CHECK_WITH_INFO(block.addr != nullptr && segment.offset_bytes <= block.size_bytes
                                    && segment.size_bytes <= block.size_bytes - segment.offset_bytes
                                    && segment.size_bytes % partition_count == 0,
                                "cache segment %s is outside tag=%s buffer or not divisible by %d",
                                segment.key_prefix.c_str(),
                                group.tag.c_str(),
                                partition_count);
        const size_t bytes = segment.size_bytes / partition_count;
        block.addr         = static_cast<char*>(block.addr) + segment.offset_bytes + bytes * partition_id;
        block.size_bytes   = bytes;
        result.push_back(block);
    }
    return result;
}

}  // namespace rtp_llm
