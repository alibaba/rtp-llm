#pragma once

#include <cstddef>

namespace rtp_llm {

struct ZigzagTokenLayout {
    size_t padded_token_count   = 0;
    size_t padding_token_count  = 0;
    size_t token_count_per_rank = 0;
};

// Zigzag context parallelism pads every sequence independently to a multiple
// of 2 * cp_size so each rank receives two equally sized chunks.
ZigzagTokenLayout makeZigzagTokenLayout(size_t token_count, size_t cp_size);

// Upper bound for a batch described by its logical token and sequence budgets.
size_t getZigzagPaddedTokenCapacity(size_t token_budget, size_t sequence_budget, size_t cp_size);

}  // namespace rtp_llm
