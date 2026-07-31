#pragma once

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,        // MHAKVCacheSpec: standard multi-head attention KV cache
    MultiHeadLatentAttention,  // MLAKVCacheSpec: MLA compressed latent KV cache
    LinearAttention,           // LinearKVCacheSpec: linear / SSM attention state cache
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
