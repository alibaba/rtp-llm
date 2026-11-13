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
// [history, Q-local | K-local | V-local]. Partition every history row separately.
// parts and source_parts describe partitions within the destination and source;
// the selected slices have the same byte length.
inline std::vector<LinearTpCacheSlice> linearTpCacheSlices(size_t state_bytes,
                                                           size_t q_row_bytes,
                                                           size_t k_row_bytes,
                                                           size_t v_row_bytes,
                                                           size_t history_rows,
                                                           size_t parts,
                                                           size_t part_id,
                                                           size_t source_parts   = 1,
                                                           size_t source_part_id = 0) {
    if (parts == 0 || part_id >= parts || source_parts == 0 || source_part_id >= source_parts || history_rows == 0
        || state_bytes == 0 || q_row_bytes == 0 || k_row_bytes == 0 || v_row_bytes == 0 || state_bytes % parts
        || q_row_bytes % parts || k_row_bytes % parts || v_row_bytes % parts) {
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
    if (source_parts > 1 && source_parts > max / ((state_bytes + history_rows * row_bytes) / parts)) {
        throw std::overflow_error("linear TP source cache size overflow");
    }
    const auto                      state        = state_bytes / parts;
    const auto                      q            = q_row_bytes / parts;
    const auto                      k            = k_row_bytes / parts;
    const auto                      v            = v_row_bytes / parts;
    const auto                      source_state = state * source_parts;
    const auto                      source_q     = q * source_parts;
    const auto                      source_k     = k * source_parts;
    const auto                      source_row   = (q + k + v) * source_parts;
    std::vector<LinearTpCacheSlice> result{{source_part_id * state, part_id * state, state}};
    for (size_t row = 0; row < history_rows; ++row) {
        const auto source      = source_state + row * source_row;
        const auto destination = state_bytes + row * row_bytes;
        result.push_back({source + source_part_id * q, destination + part_id * q, q});
        result.push_back({source + source_q + source_part_id * k, destination + q_row_bytes + part_id * k, k});
        result.push_back({source + source_q + source_k + source_part_id * v,
                          destination + q_row_bytes + k_row_bytes + part_id * v,
                          v});
    }
    return result;
}

}  // namespace rtp_llm
