#pragma once

#include <cstddef>

namespace rtp_llm {

struct ZigzagTokenLayout {
    size_t padded_token_count   = 0;
    size_t padding_token_count  = 0;
    size_t token_count_per_rank = 0;
};

// Zigzag context parallelism pads every sequence independently to a multiple
// of 2 * cp_size * segment_size_alignment so each rank receives two equally
// sized, segment-aligned chunks.
ZigzagTokenLayout makeZigzagTokenLayout(size_t token_count, size_t cp_size, size_t segment_size_alignment = 1);

}  // namespace rtp_llm
