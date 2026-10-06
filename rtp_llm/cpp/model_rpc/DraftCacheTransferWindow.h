#pragma once

#include <cstdint>

namespace rtp_llm {

// Peers must explicitly agree on identical native FULL/page-RR semantics.
// Retain legacy transport whenever D can reuse/publish prefix pages. A cold
// request's tail-only draft must not enter a shared full-history prefix cache.
// Draft-tail validity must first be proven independently of target reuse.
inline uint32_t
negotiateDraftCacheTransferWindow(uint32_t offered_window, uint32_t local_window, bool decode_can_reuse_prefix) {
    return !decode_can_reuse_prefix && offered_window != 0 && offered_window == local_window ? local_window : 0;
}

}  // namespace rtp_llm
