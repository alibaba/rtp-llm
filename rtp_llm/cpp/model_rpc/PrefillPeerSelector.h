#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace rtp_llm {

inline uint64_t mixPrefillPeerRequestId(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline size_t selectPrefillPeerIndex(int64_t request_id, int64_t dp_rank, size_t worker_index, size_t peer_count) {
    if (peer_count == 0) {
        throw std::invalid_argument("cannot select a prefill peer from an empty peer list");
    }
    if (dp_rank < 0) {
        throw std::invalid_argument("decode dp rank must be non-negative");
    }

    const auto request_offset = mixPrefillPeerRequestId(static_cast<uint64_t>(request_id)) % peer_count;
    const auto rank_offset    = static_cast<uint64_t>(dp_rank) % peer_count;
    return static_cast<size_t>((request_offset + rank_offset + worker_index % peer_count) % peer_count);
}

inline size_t selectMlaPrefillPeerIndex(int64_t request_id,
                                        int64_t dp_rank,
                                        size_t  worker_index,
                                        size_t  worker_count,
                                        size_t  peer_count) {
    if (worker_count == 0 || peer_count == 0 || worker_index >= worker_count) {
        throw std::invalid_argument("cannot select an MLA prefill peer from an invalid topology");
    }
    if (worker_count % peer_count == 0) {
        return worker_index / (worker_count / peer_count);
    }
    if (peer_count % worker_count == 0) {
        const auto peer_group_size = peer_count / worker_count;
        const auto peer_group_begin = worker_index * peer_group_size;
        return peer_group_begin
               + selectPrefillPeerIndex(request_id, dp_rank, worker_index, peer_group_size);
    }
    throw std::invalid_argument("MLA prefill and decode worker counts are not divisible");
}

}  // namespace rtp_llm
