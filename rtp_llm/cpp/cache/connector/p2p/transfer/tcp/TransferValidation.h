#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rtp_llm {
namespace transfer {
namespace tcp {

// The wire length is uint32. Check before narrowing or allocating the payload.
inline bool isRepresentableBlockSize(size_t size) {
    return size <= std::numeric_limits<uint32_t>::max();
}

// Both the advertised length and the actual bytes must agree with the local
// allocation. Trusting only the length would permit an out-of-bounds read.
inline bool matchesBlockPayload(size_t expected_size, uint32_t wire_size, size_t payload_size) {
    return isRepresentableBlockSize(expected_size) && expected_size == wire_size && expected_size == payload_size;
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
