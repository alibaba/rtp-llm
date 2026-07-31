#pragma once

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,        // MHAKVCacheSpec: standard multi-head attention KV cache
    MultiHeadLatentAttention,  // MLAKVCacheSpec: MLA compressed latent KV cache
    LinearAttention,           // LinearKVCacheSpec: linear / SSM attention state cache
    CompressedKVCache,  // Byte-addressed packed-entry paged KV pool
    // Fixed-allocation state pool. This also covers indexer, CSA and HCA state;
    // the name denotes the allocation behavior rather than a SWA-only payload.
    SWAState,
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        case KVCacheSpecType::CompressedKVCache:
            return "CompressedKVCache";
        case KVCacheSpecType::SWAState:
            return "SWAState";
        default:
            return "Unknown";
    }
}

}  // namespace rtp_llm
