#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

#include <limits>
#include <stdexcept>

namespace rtp_llm {

ZigzagTokenLayout makeZigzagTokenLayout(size_t token_count, size_t cp_size, size_t segment_size_alignment) {
    if (cp_size == 0 || segment_size_alignment == 0) {
        throw std::invalid_argument("Zigzag size and alignment must be positive");
    }

    if (cp_size > std::numeric_limits<size_t>::max() / 2 / segment_size_alignment) {
        throw std::overflow_error("Zigzag alignment exceeds size_t range");
    }
    const size_t alignment = 2 * cp_size * segment_size_alignment;
    const size_t remainder = token_count % alignment;
    if (remainder != 0 && token_count > std::numeric_limits<size_t>::max() - (alignment - remainder)) {
        throw std::overflow_error("Zigzag padded token count exceeds size_t range");
    }
    const size_t padded_token_count = remainder == 0 ? token_count : token_count + alignment - remainder;
    return {
        padded_token_count,
        padded_token_count - token_count,
        padded_token_count / cp_size,
    };
}

}  // namespace rtp_llm
