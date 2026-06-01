#pragma once

namespace rtp_llm {

enum class CacheBlockKind {
    COMPLETE   = 0,
    INCOMPLETE = 1,
};

inline CacheBlockKind blockKindFromComplete(bool is_complete) {
    return is_complete ? CacheBlockKind::COMPLETE : CacheBlockKind::INCOMPLETE;
}

inline const char* cacheBlockKindName(CacheBlockKind kind) {
    return kind == CacheBlockKind::COMPLETE ? "complete" : "incomplete";
}

}  // namespace rtp_llm
