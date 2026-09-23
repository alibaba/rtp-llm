#pragma once

#include <cstddef>
#include <stdexcept>

namespace rtp_llm {

constexpr size_t kIndexerCacheBf16Bytes  = 2;
constexpr size_t kIndexerCacheScaleBytes = sizeof(float);

inline size_t indexerCacheBytesPerToken(size_t indexer_dim, int fp8_mode) {
    // Mode 3 is MiniMax-M3.1 NVFP4: packed E2M1 values and one E4M3 byte for
    // every contiguous group of 16 indexer-K values.
    if (fp8_mode == 3) {
        if (indexer_dim % 16 != 0) {
            throw std::invalid_argument("NVFP4 indexer dimension must be divisible by 16");
        }
        return indexer_dim / 2 + indexer_dim / 16;
    }
    return fp8_mode > 0 ? indexer_dim + kIndexerCacheScaleBytes : indexer_dim * kIndexerCacheBf16Bytes;
}

inline size_t indexerCacheBlockBytes(size_t indexer_dim, int fp8_mode, size_t tokens_per_block) {
    return indexerCacheBytesPerToken(indexer_dim, fp8_mode) * tokens_per_block;
}

}  // namespace rtp_llm
