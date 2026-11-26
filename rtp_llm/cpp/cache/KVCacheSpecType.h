#pragma once

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,        // Per-head K/V pages
    MultiHeadLatentAttention,  // Compressed latent token pages
    LinearAttention,           // Recurrent / convolution state checkpoints
    OpaqueKV,                  // Byte-addressed opaque paged KV pool
    OpaqueState,               // Fixed-allocation opaque state cache
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        case KVCacheSpecType::OpaqueKV:
            return "OpaqueKV";
        case KVCacheSpecType::OpaqueState:
            return "OpaqueState";
        default:
            return "Unknown";
    }
}

}  // namespace rtp_llm
