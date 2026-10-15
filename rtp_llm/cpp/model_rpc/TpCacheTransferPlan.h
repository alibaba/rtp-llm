#pragma once

#include "rtp_llm/cpp/model_rpc/LinearTpCacheLayout.h"
#include "rtp_llm/cpp/model_rpc/MhaTpCacheLayout.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

namespace rtp_llm {

struct TpCacheTransferTopology {
    size_t decode_tp;
    size_t peers;
    size_t prefill_cp;
    size_t partitions;

    bool supportsTypedTransfer() const {
        return decode_tp > 0 && peers > 1 && prefill_cp == 1 && partitions == 1;
    }
};

inline bool replicatedTpCachePeerOwnsTag(
    std::string_view tag, bool mha, size_t local_heads, const TpCacheTransferTopology& topology, size_t peer) {
    if (!topology.supportsTypedTransfer() || peer >= topology.peers) {
        throw std::invalid_argument("unsupported typed TP topology or peer");
    }
    if (tag != "full" && tag != "ple_conv_state" && tag != "ple_ngram_ctx" && tag != "indexer_kv"
        && tag != "indexer_state") {
        throw std::invalid_argument("typed TP handoff rejects an unknown replicated cache tag");
    }
    if (mha) {
        if (local_heads == 2 && topology.decode_tp == 1
            && (topology.peers == 2 || topology.peers == 4 || topology.peers == 8)) {
            return mhaTpPeerOwnsHead(topology.peers, 2, peer);
        }
        if (local_heads != 1) {
            throw std::invalid_argument("typed TP QSA handoff requires one local head or D1 with two heads");
        }
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
                                      size_t peer) {
        auto         slices = linearTpCacheSlices(state_bytes, q_bytes, k_bytes, v_bytes, history_rows, peers, peer);
        const size_t destination_bytes = state_bytes + history_rows * (q_bytes + k_bytes + v_bytes);
        return {destination_bytes / peers, destination_bytes, peers, peer, std::move(slices)};
    }

    static TpCacheTransferPlan mha(size_t k_bytes, size_t v_bytes, size_t heads, size_t head) {
        auto slices = mhaTpCacheSlices(k_bytes, v_bytes, heads, head);
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
}

}  // namespace rtp_llm
