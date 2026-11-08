#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

#include <algorithm>
#include <stdexcept>

namespace rtp_llm {

ZigzagTokenLayout makeZigzagTokenLayout(size_t token_count, size_t cp_size) {
    if (cp_size == 0) {
        throw std::invalid_argument("Zigzag cp_size must be positive");
    }

    const size_t alignment          = 2 * cp_size;
    const size_t padded_token_count = (token_count + alignment - 1) / alignment * alignment;
    return {
        padded_token_count,
        padded_token_count - token_count,
        padded_token_count / cp_size,
    };
}

size_t getZigzagPaddedTokenCapacity(size_t token_budget, size_t sequence_budget, size_t cp_size) {
    const auto max_sequence_padding = makeZigzagTokenLayout(1, cp_size).padding_token_count;
    const auto rows = token_budget + std::min(token_budget, sequence_budget) * max_sequence_padding;
    return (rows + cp_size - 1) / cp_size * cp_size;
}

}  // namespace rtp_llm
