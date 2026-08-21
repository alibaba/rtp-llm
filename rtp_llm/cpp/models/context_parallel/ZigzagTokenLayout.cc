#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

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

}  // namespace rtp_llm
