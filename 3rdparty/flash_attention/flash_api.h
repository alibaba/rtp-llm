#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include "csrc/flash_attn/src/flash.h"

#define FP16_SWITCH(COND, ...)                                                                                         \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            using elem_type = cutlass::half_t;                                                                         \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            using elem_type = cutlass::bfloat16_t;                                                                     \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define HEADDIM_SWITCH(HEADDIM, ...)                                                                                   \
    [&] {                                                                                                              \
        if (HEADDIM <= 64) {                                                                                           \
            constexpr static int kHeadDim = 64;                                                                        \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 96) {                                                                                    \
            constexpr static int kHeadDim = 96;                                                                        \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 128) {                                                                                   \
            constexpr static int kHeadDim = 128;                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else if (HEADDIM <= 192) {                                                                                   \
            constexpr static int kHeadDim = 192;                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) {
        return 1;
    }
    max_splits                        = std::min({max_splits, num_SMs, num_n_blocks});
    float              max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / std::ceil(n_waves);
            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

inline void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
#ifdef FMHA_SUPPORT_SPLIT
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
#endif
            }
        });
    });
}

#undef FP16_SWITCH
#undef HEADDIM_SWITCH
