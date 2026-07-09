#pragma once

// TODO(block_tree_cache refactor): This is a temporary stub extracted from
// connector/memory/CacheBlockKind.h to decouple block_tree_cache from the
// connector module during refactoring. Restore the original include path
// once the connector interface is stabilized.

namespace rtp_llm {

enum class CacheBlockKind {
    COMPLETE      = 0,
    INCOMPLETE    = 1,
    COMPRESSED_KV = 2,
    STATE_SWA_KV  = 3,
};

inline CacheBlockKind blockKindFromComplete(bool is_complete) {
    return is_complete ? CacheBlockKind::COMPLETE : CacheBlockKind::INCOMPLETE;
}

inline const char* cacheBlockKindName(CacheBlockKind kind) {
    switch (kind) {
        case CacheBlockKind::COMPLETE:
            return "complete";
        case CacheBlockKind::INCOMPLETE:
            return "incomplete";
        case CacheBlockKind::COMPRESSED_KV:
            return "compressed_kv";
        case CacheBlockKind::STATE_SWA_KV:
            return "state_swa_kv";
    }
    return "unknown";
}

}  // namespace rtp_llm
