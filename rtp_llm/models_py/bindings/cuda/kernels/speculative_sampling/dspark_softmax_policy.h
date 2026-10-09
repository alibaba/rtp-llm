// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Adapted cached-cluster host policy; original generated CUDA math is unchanged.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <initializer_list>

namespace rtp_llm {
namespace dspark_softmax {

// Cached-cluster policy extracted from FlashInfer 9aa52b935e55fcfb61fd9390e04bb32e7f3b9860.
// DSpark supplies already temperature-combined logits (upstream parameter_kind=0).
constexpr int kTileElements = 256 * 8;
struct CachedVariant {
    int cluster_ctas;
    int max_tiles;
};

inline bool useCached(int64_t rows, int64_t vocab, int major, int minor, const void* logits, const void* output) {
    return major == 10 && (minor == 0 || minor == 3) && rows > 0 && vocab > 0
           && rows <= std::numeric_limits<int>::max() / vocab && vocab % 8 == 0
           && vocab <= 16 * 8 * kTileElements
           && !(rows > 128 && rows <= 384 && vocab >= 24576 && vocab <= 32000)
           && reinterpret_cast<uintptr_t>(logits) % 32 == 0
           && reinterpret_cast<uintptr_t>(output) % 32 == 0;
}

inline int chunkElements(int64_t vocab, int clusters) {
    return static_cast<int>(((vocab + clusters - 1) / clusters + kTileElements - 1) / kTileElements)
           * kTileElements;
}

inline CachedVariant selectVariant(int64_t rows, int64_t vocab, int sm_count) {
    CachedVariant small{0, 0}, wide{0, 0}, any{0, 0};
    for (int clusters : {1, 2, 4, 8, 16}) {
        for (int tiles : {4, 8}) {
            if (static_cast<int64_t>(clusters) * tiles * kTileElements < vocab) {
                continue;
            }
            if (any.cluster_ctas == 0) any = {clusters, tiles};
            if (tiles == 4 && clusters <= 8 && small.cluster_ctas == 0) small = {clusters, tiles};
            if (tiles == 8 && wide.cluster_ctas == 0) wide = {clusters, tiles};
        }
    }
    CachedVariant selected = small.cluster_ctas != 0 ? small : wide.cluster_ctas != 0 ? wide : any;
    const int64_t useful = std::max<int64_t>(1, (vocab + kTileElements - 1) / kTileElements);
    while (rows * selected.cluster_ctas < 2LL * sm_count && selected.cluster_ctas < 16
           && selected.cluster_ctas * 2 <= useful) {
        selected.cluster_ctas *= 2;
    }
    if (selected.max_tiles == 8 && chunkElements(vocab, selected.cluster_ctas) <= 4 * kTileElements) {
        selected.max_tiles = 4;
    }
    return selected;
}

}  // namespace dspark_softmax
}  // namespace rtp_llm
