#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// Each row is [current, previous, previous-2, previous-3]. Explicit history
// makes CP boundaries, prefix reuse, PD handover and speculative rollback
// independent of a worker-local token cache.
inline void
fillEngramTokenWindows(const std::vector<int>& history, int start_position, size_t token_count, int32_t* output) {
    if (start_position < 0 || static_cast<size_t>(start_position) + token_count > history.size()) {
        throw std::invalid_argument("Engram token window exceeds complete request history");
    }
    for (size_t row = 0; row < token_count; ++row) {
        for (int lag = 0; lag < 4; ++lag) {
            const int position    = start_position + static_cast<int>(row) - lag;
            output[row * 4 + lag] = position >= 0 ? history[position] : -1;
        }
    }
}

// A verify block begins at the anchor. Its candidate rows override request
// history, including positions beyond the ultimately accepted prefix.
inline void extendEngramVerifyWindows(const int32_t* anchor_window,
                                      const int32_t* verify_tokens,
                                      size_t         verify_width,
                                      int32_t*       output) {
    for (size_t row = 0; row < verify_width; ++row) {
        for (int lag = 0; lag < 4; ++lag) {
            const int source      = static_cast<int>(row) - lag;
            output[row * 4 + lag] = source >= 0 ? verify_tokens[source] : anchor_window[-source];
        }
    }
}

}  // namespace rtp_llm
