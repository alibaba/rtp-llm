#pragma once

#include "rtp_llm/cpp/cache/LinearTpCacheLayout.h"
#include "rtp_llm/cpp/cache/MhaTpCacheLayout.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

namespace rtp_llm {

// LEGACY preserves transport behavior for descriptors that have not opted in.
enum class TpCacheTransferMode : int8_t {
    LEGACY     = 0,
    REPLICATED = 1,
    MHA_HEADS  = 2,
    LINEAR_QKV = 3,
};

struct TpCacheTransferTopology {
    size_t decode_tp;
    size_t peers;
    size_t prefill_cp;
    size_t partitions;

    bool supportsTypedTransfer() const {
        return decode_tp > 0 && peers > 1 && prefill_cp == 1 && partitions == 1;
    }
};

// One source peer's partition of a destination block. The mode resolves whether
// it is read directly as segments or first staged for packed reassembly.
struct TpCacheTransferRoute {
    bool   selected               = true;
    size_t local_partition_count  = 1;
    size_t local_partition_id     = 0;
    size_t remote_partition_count = 1;
    size_t remote_partition_id    = 0;
    bool   reassemble             = false;
    bool   segmented              = false;
};

inline bool tpCachePeerOwnsPayload(TpCacheTransferMode            mode,
                                   size_t                         local_heads,
                                   const TpCacheTransferTopology& topology,
                                   size_t                         peer) {
    if (!topology.supportsTypedTransfer() || peer >= topology.peers) {
        throw std::invalid_argument("unsupported typed TP topology or peer");
    }
    if (mode == TpCacheTransferMode::LINEAR_QKV) {
        return true;
    }
    if (mode == TpCacheTransferMode::MHA_HEADS) {
        if (local_heads == 2 && topology.decode_tp == 1
            && (topology.peers == 2 || topology.peers == 4 || topology.peers == 8)) {
            return mhaTpPeerOwnsHead(topology.peers, 2, peer);
        }
        if (local_heads != 1) {
            throw std::invalid_argument("typed TP MHA handoff requires one local head or D1 with two heads");
        }
    } else if (mode != TpCacheTransferMode::REPLICATED) {
        throw std::invalid_argument("typed TP handoff requires an explicit cache transfer layout");
    }
    return peer == 0;
}

// Pure payload layout. This object owns no memory registration or asynchronous resources.
struct TpCacheTransferPlan {
    size_t                          source_bytes;
    size_t                          destination_bytes;
    size_t                          part_count;
    size_t                          part_id;
    std::vector<LinearTpCacheSlice> slices;

    static TpCacheTransferPlan linear(size_t state_bytes,
                                      size_t q_bytes,
                                      size_t k_bytes,
                                      size_t v_bytes,
                                      size_t history_rows,
                                      size_t peers,
                                      size_t peer,
                                      size_t source_parts   = 1,
                                      size_t source_part_id = 0) {
        auto slices = linearTpCacheSlices(
            state_bytes, q_bytes, k_bytes, v_bytes, history_rows, peers, peer, source_parts, source_part_id);
        const size_t destination_bytes = state_bytes + history_rows * (q_bytes + k_bytes + v_bytes);
        return {destination_bytes / peers * source_parts, destination_bytes, peers, peer, std::move(slices)};
    }

    static TpCacheTransferPlan mha(size_t k_bytes, size_t v_bytes, size_t heads, size_t head, size_t kernel_pages = 1) {
        auto slices = mhaTpCacheSlices(k_bytes, v_bytes, heads, head, kernel_pages);
        return {(k_bytes + v_bytes) / heads, k_bytes + v_bytes, heads, head, std::move(slices)};
    }

    void validateSlices() const {
        if (!source_bytes || !destination_bytes || !part_count || part_id >= part_count || slices.empty()) {
            throw std::invalid_argument("invalid typed TP transfer plan");
        }
        for (const auto& slice : slices) {
            if (!slice.bytes || slice.source_offset > source_bytes || slice.bytes > source_bytes - slice.source_offset
                || slice.destination_offset > destination_bytes
                || slice.bytes > destination_bytes - slice.destination_offset) {
                throw std::invalid_argument("typed TP transfer slice exceeds its payload");
            }
        }
    }
};

// Resolved from a cache descriptor and its local physical geometry at startup.
// No model names, cache tags or model configuration are needed during transfer.
struct TpCacheTransferLayout {
    TpCacheTransferMode mode         = TpCacheTransferMode::LEGACY;
    size_t              q_row_bytes  = 0;
    size_t              k_row_bytes  = 0;
    size_t              v_row_bytes  = 0;
    size_t              history_rows = 0;
    size_t              key_heads    = 0;
    size_t              value_heads  = 0;

    bool needsReassembly(size_t local_heads) const {
        return mode == TpCacheTransferMode::LINEAR_QKV || (mode == TpCacheTransferMode::MHA_HEADS && local_heads > 1);
    }

    TpCacheTransferRoute route(size_t                         local_heads,
                               const TpCacheTransferTopology& topology,
                               size_t                         peer,
                               bool                           typed,
                               TpCacheTransferRoute           fallback,
                               TpCacheTransferRoute           heads) const {
        const bool segmented = !typed && mode == TpCacheTransferMode::LINEAR_QKV;
        auto       result    = segmented ? heads : fallback;
        result.segmented     = segmented;
        result.reassemble    = false;
        if (typed) {
            result.selected   = tpCachePeerOwnsPayload(mode, local_heads, topology, peer);
            result.reassemble = result.selected && needsReassembly(local_heads);
        }
        return result;
    }

    TpCacheTransferPlan plan(size_t k_bytes, size_t v_bytes, const TpCacheTransferRoute& route) const {
        if (mode != TpCacheTransferMode::LINEAR_QKV) {
            throw std::invalid_argument("partitioned cache transfer requires LINEAR_QKV layout");
        }
        auto result = TpCacheTransferPlan::linear(k_bytes,
                                                  q_row_bytes,
                                                  k_row_bytes,
                                                  v_row_bytes,
                                                  history_rows,
                                                  route.local_partition_count,
                                                  route.local_partition_id,
                                                  route.remote_partition_count,
                                                  route.remote_partition_id);
        if (result.destination_bytes - k_bytes != v_bytes) {
            throw std::invalid_argument("typed linear TP history does not match its payload");
        }
        return result;
    }

    TpCacheTransferPlan plan(size_t                         k_bytes,
                             size_t                         v_bytes,
                             size_t                         local_heads,
                             const TpCacheTransferTopology& topology,
                             size_t                         peer,
                             size_t                         kernel_pages = 1) const {
        if (!tpCachePeerOwnsPayload(mode, local_heads, topology, peer)) {
            throw std::invalid_argument("typed TP plan requested from a non-owner peer");
        }
        if (mode == TpCacheTransferMode::LINEAR_QKV) {
            if (!key_heads || !value_heads || key_heads % topology.peers || value_heads % topology.peers) {
                throw std::invalid_argument("typed linear TP heads must divide evenly across producers");
            }
            return plan(k_bytes, v_bytes, TpCacheTransferRoute{true, topology.peers, peer});
        }
        if (mode == TpCacheTransferMode::MHA_HEADS && local_heads == 2) {
            return TpCacheTransferPlan::mha(
                k_bytes, v_bytes, local_heads, peer / (topology.peers / local_heads), kernel_pages);
        }
        throw std::invalid_argument("cache transfer layout does not require reassembly");
    }
};

inline bool useTypedTpCacheLayout(bool declared_layout, bool opaque_store, size_t peers, size_t partitions) {
    return declared_layout && opaque_store && (peers > 1 || partitions > 1);
}

inline size_t registeredCacheBytes(size_t source_bytes) {
    constexpr size_t alignment = 64 * 1024;
    const size_t     max       = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (!source_bytes || source_bytes > max - (alignment - 1)) {
        throw std::overflow_error("typed TP registration size overflow");
    }
    return ((source_bytes + alignment - 1) / alignment) * alignment;
}

inline void validateCompleteTransfer(const std::vector<TpCacheTransferPlan>& plans) {
    if (plans.empty()) {
        throw std::invalid_argument("typed TP transfer has no shards");
    }
    std::vector<bool> seen(plans.front().part_count, false);
    for (const auto& plan : plans) {
        plan.validateSlices();
        if (plan.part_count != seen.size() || plan.destination_bytes != plans.front().destination_bytes
            || seen[plan.part_id]) {
            throw std::invalid_argument("typed TP transfer has inconsistent or duplicate shards");
        }
        seen[plan.part_id] = true;
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool present) { return present; })) {
        throw std::invalid_argument("typed TP transfer is missing a shard");
    }
    // The reassembly scratch is uninitialized, so shard slices must tile
    // [0, destination_bytes) exactly: any gap or overlap would leak garbage
    // bytes into the destination KV cache.
    std::vector<const LinearTpCacheSlice*> coverage;
    for (const auto& plan : plans) {
        for (const auto& slice : plan.slices) {
            coverage.push_back(&slice);
        }
    }
    std::sort(coverage.begin(), coverage.end(), [](const LinearTpCacheSlice* a, const LinearTpCacheSlice* b) {
        return a->destination_offset < b->destination_offset;
    });
    size_t tiled = 0;
    for (const auto* slice : coverage) {
        if (slice->destination_offset < tiled) {
            throw std::invalid_argument("typed TP transfer slices overlap in the destination buffer");
        }
        if (slice->destination_offset > tiled) {
            throw std::invalid_argument("typed TP transfer slices leave a gap in the destination buffer");
        }
        tiled += slice->bytes;
    }
    if (tiled != plans.front().destination_bytes) {
        throw std::invalid_argument("typed TP transfer slices do not cover the destination buffer");
    }
}

}  // namespace rtp_llm
