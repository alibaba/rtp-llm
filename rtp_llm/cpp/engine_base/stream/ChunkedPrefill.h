#pragma once

#include <cstdint>

namespace rtp_llm {

// Per-stream chunk length. `rows` return sequences share one block-aligned
// window, so the grant is the largest block multiple that fits budget/rows.
inline int64_t computeChunkGrant(
    int64_t budget, int64_t rows, int64_t remaining, int64_t block_size, bool align_final_suffix = false) {
    if (budget <= 0 || rows <= 0 || remaining <= 0 || block_size <= 0) {
        return 0;
    }

    const int64_t max_len = budget / rows;
    if (remaining <= max_len) {
        const int64_t suffix = remaining % block_size;
        if (align_final_suffix && suffix > 0 && remaining > block_size) {
            return remaining - suffix;
        }
        return remaining;
    }
    return (max_len / block_size) * block_size;
}

}  // namespace rtp_llm
