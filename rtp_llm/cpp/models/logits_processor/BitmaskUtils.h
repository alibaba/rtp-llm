#pragma once

#include <cstddef>
#include <cstdint>

#include <dlpack/dlpack.h>

namespace rtp_llm {

// Single-row bitmask view backed by a contiguous int32 buffer of `words` words.
// shape_out must outlive the returned DLTensor (it stores the shape pointer).
DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words, int64_t shape_out[2]);

bool bitmaskAllowsToken(const int32_t* bitmask, size_t words, int32_t token_id);

// Out-of-range token_id is a caller bug; abort instead of returning an all-disabled row.
void forceTokenInBitmask(int32_t* bitmask, size_t words, int64_t token_id);

void clearBitmaskTokenRange(int32_t* bitmask, size_t words, int64_t begin_token, int64_t end_token);

}  // namespace rtp_llm
