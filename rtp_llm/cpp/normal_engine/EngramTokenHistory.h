#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// Current-first windows from committed history keep chunk/CP boundaries and
// prefix restoration independent of worker-local token state.
inline void fillEngramTokenWindows(
    const int32_t* history, size_t history_size, int start_position, size_t token_count, int32_t* output) {
    if (start_position < 0 || static_cast<size_t>(start_position) > history_size
        || token_count > history_size - static_cast<size_t>(start_position)
        || token_count > std::numeric_limits<size_t>::max() / 4 || (token_count && (!history || !output))) {
        throw std::invalid_argument("Engram token window exceeds complete request history or output range");
    }
    for (size_t row = 0; row < token_count; ++row) {
        const size_t position = static_cast<size_t>(start_position) + row;
        for (size_t lag = 0; lag < 4; ++lag) {
            output[row * 4 + lag] = position >= lag ? history[position - lag] : -1;
        }
    }
}

inline void
fillEngramTokenWindows(const std::vector<int>& history, int start_position, size_t token_count, int32_t* output) {
    fillEngramTokenWindows(history.data(), history.size(), start_position, token_count, output);
}

// Candidate rows replace the anchor's current token without mutating committed
// history. Acceptance/rollback remains the speculative scheduler's responsibility.
inline void extendEngramVerifyWindows(const int32_t* anchor_window,
                                      const int32_t* verify_tokens,
                                      size_t         verify_width,
                                      int32_t*       output) {
    if (verify_width > std::numeric_limits<size_t>::max() / 4
        || (verify_width && (!anchor_window || !verify_tokens || !output))) {
        throw std::invalid_argument("Engram verify window has invalid input or output range");
    }
    for (size_t row = 0; row < verify_width; ++row) {
        for (size_t lag = 0; lag < 4; ++lag) {
            output[row * 4 + lag] = row >= lag ? verify_tokens[row - lag] : anchor_window[lag - row];
        }
    }
}

}  // namespace rtp_llm
