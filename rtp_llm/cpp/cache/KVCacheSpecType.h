#pragma once

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        default:
            return "Unknown";
    }
}

}  // namespace rtp_llm
