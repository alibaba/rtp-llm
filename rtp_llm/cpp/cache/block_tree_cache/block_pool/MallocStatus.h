#pragma once

#include <cstdint>

namespace rtp_llm {

// A temporary shortage can be retried after page references are released.
enum class MallocStatus : uint8_t {
    NONE = 0,
    RETRYABLE_RESOURCE_EXHAUSTED,
    PERMANENT_RESOURCE_EXHAUSTED,
    INTERNAL_ERROR,
};

}  // namespace rtp_llm
