#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

struct LinearTpCacheSlice {
    size_t source_offset;
    size_t destination_offset;
    size_t bytes;
};

// LinearCacheConverter stores [V-head, V-dim, K-dim] followed by
// [history, Q-local | K-local | V-local]. Each history row must be reordered
// separately when concatenating multiple prefill TP ranks.
inline std::vector<LinearTpCacheSlice> linearTpCacheSlices(size_t state_bytes,
                                                           size_t q_row_bytes,
                                                           size_t k_row_bytes,
                                                           size_t v_row_bytes,
                                                           size_t history_rows,
                                                           size_t parts,
                                                           size_t part_id) {
    if (parts == 0 || part_id >= parts || history_rows == 0 || state_bytes == 0 || q_row_bytes == 0 || k_row_bytes == 0
        || v_row_bytes == 0 || state_bytes % parts || q_row_bytes % parts || k_row_bytes % parts
        || v_row_bytes % parts) {
        throw std::invalid_argument("invalid linear TP cache layout");
    }
    const auto max = std::numeric_limits<size_t>::max();
    if (q_row_bytes > max - k_row_bytes || q_row_bytes + k_row_bytes > max - v_row_bytes) {
        throw std::overflow_error("linear TP row size overflow");
    }
    const auto row_bytes = q_row_bytes + k_row_bytes + v_row_bytes;
    if (history_rows > (max - state_bytes) / row_bytes) {
        throw std::overflow_error("linear TP cache size overflow");
    }
    const auto                      source_state = state_bytes / parts;
    const auto                      source_q     = q_row_bytes / parts;
    const auto                      source_k     = k_row_bytes / parts;
    const auto                      source_v     = v_row_bytes / parts;
    const auto                      source_row   = row_bytes / parts;
    std::vector<LinearTpCacheSlice> result{{0, part_id * source_state, source_state}};
    for (size_t row = 0; row < history_rows; ++row) {
        const auto source      = source_state + row * source_row;
        const auto destination = state_bytes + row * row_bytes;
        result.push_back({source, destination + part_id * source_q, source_q});
        result.push_back({source + source_q, destination + q_row_bytes + part_id * source_k, source_k});
        result.push_back(
            {source + source_q + source_k, destination + q_row_bytes + k_row_bytes + part_id * source_v, source_v});
    }
    return result;
}

}  // namespace rtp_llm
