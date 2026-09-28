#pragma once

#include <cstdint>

namespace rtp_llm {

// Encoder through L20, decoder/draft live SWA only, and 128-token replay.
// Workers and frontend routing must use the same versioned layout identity.
inline constexpr int64_t DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED = INT64_C(0x4453563431525031);

}  // namespace rtp_llm
