#pragma once

#include <cstdlib>
#include <string>

namespace rtp_llm {

// PD KV transfer switches for CP-replicated prefill (prefill_cp_kv_cache_sharded=0).
//
// RTP_LLM_PD_SERVING_CP_RANK=<r>  (default unset = off)
//   Every CP rank of a pipeline stage holds the same replicated KV, and stage-aware
//   decode reads each stage's layers from every advertised peer of that stage, so the
//   same blocks cross the wire once per CP rank. When set, prefill advertises only CP
//   rank r of each stage to decode, and only that rank publishes cache-store blocks.
//
// RTP_LLM_PD_PUBLISH_AT_END=1  (default 0)
//   Debug/benchmark switch: skip the per-layer cache-store streaming during prefill and
//   rely on the end-of-request publish, so the whole request's KV becomes readable at once
//   (the backlog a prefix-cache hit produces). Used to measure the transport's drain rate.

inline int pdServingCpRank() {
    static const int rank = [] {
        const char* v = std::getenv("RTP_LLM_PD_SERVING_CP_RANK");
        return (v != nullptr && *v != '\0') ? std::atoi(v) : -1;
    }();
    return rank;
}

inline bool pdPublishServingRankOnly() {
    return pdServingCpRank() >= 0;
}

inline bool pdPublishAtEnd() {
    static const bool on = [] {
        const char* v = std::getenv("RTP_LLM_PD_PUBLISH_AT_END");
        return v != nullptr && std::string(v) == "1";
    }();
    return on;
}

}  // namespace rtp_llm
