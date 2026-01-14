#pragma once

#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/include/gemm_configs.h"

struct ShapeInfo {
    int m, n, k;
    ShapeInfo() = default;
    ShapeInfo(int m, int n, int k): m(m), n(n), k(k) {};
};

inline ShapeInfo get_tile_config_shape_info(CutlassTileConfigSM90 config) {
    switch (config) {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case CutlassTileConfigSM90::CtaShape##M##x##N##x##K##B:                                                            \
        return ShapeInfo(M, N, K);
        SHAPE_CASE(64, 16, 128)
        SHAPE_CASE(64, 32, 128)
        SHAPE_CASE(64, 64, 128)
        SHAPE_CASE(64, 128, 128)
        SHAPE_CASE(64, 256, 128)
        SHAPE_CASE(128, 16, 128)
        SHAPE_CASE(128, 32, 128)
        SHAPE_CASE(128, 64, 128)
        SHAPE_CASE(128, 128, 128)
        SHAPE_CASE(128, 256, 128)
        SHAPE_CASE(256, 128, 128)
        default:
            return ShapeInfo(128, 128, 128);
#undef SHAPE_CASE
    }
}

inline ShapeInfo get_cluster_config_shape_info(ClusterShape config) {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case ClusterShape::ClusterShape_##M##x##N##x##K:                                                                   \
        return ShapeInfo(M, N, K);
    switch (config) {
        SHAPE_CASE(1, 1, 1)
        SHAPE_CASE(2, 1, 1)
        SHAPE_CASE(1, 2, 1)
        SHAPE_CASE(2, 2, 1)
            // SHAPE_CASE(1, 8, 1)
            // SHAPE_CASE(8, 1, 1)
        default:
            return ShapeInfo(1, 1, 1);
#undef SHAPE_CASE
    }
}

inline CutlassTileConfigSM90 tile_shape_converter(int m, int n, int k) {
    if (m == 128 && n == 128 && k == 128) {
        return CutlassTileConfigSM90::CtaShape128x128x128B;
    } else if (m == 64 && n == 16 && k == 128) {
        return CutlassTileConfigSM90::CtaShape64x16x128B;
    } else if (m == 64 && n == 32 && k == 128) {
        return CutlassTileConfigSM90::CtaShape64x32x128B;
    } else if (m == 64 && n == 64 && k == 128) {
        return CutlassTileConfigSM90::CtaShape64x64x128B;
    } else if (m == 64 && n == 128 && k == 128) {
        return CutlassTileConfigSM90::CtaShape64x128x128B;
    } else if (m == 64 && n == 256 && k == 128) {
        return CutlassTileConfigSM90::CtaShape64x256x128B;
    } else if (m == 128 && n == 16 && k == 128) {
        return CutlassTileConfigSM90::CtaShape128x16x128B;
    } else if (m == 128 && n == 32 && k == 128) {
        return CutlassTileConfigSM90::CtaShape128x32x128B;
    } else if (m == 128 && n == 64 && k == 128) {
        return CutlassTileConfigSM90::CtaShape128x64x128B;
    } else if (m == 128 && n == 256 && k == 128) {
        return CutlassTileConfigSM90::CtaShape128x256x128B;
    } else if (m == 256 && n == 128 && k == 128) {
        return CutlassTileConfigSM90::CtaShape256x128x128B;
    } else if (m == 128 && n == 16 && k == 256) {
        return CutlassTileConfigSM90::CtaShape128x16x256B;
    } else {
        printf("[WARN]: unsupported tile shape, failing back to the default tile config.\n");
        return CutlassTileConfigSM90::CtaShape128x128x128B;
    }
}

inline ClusterShape cluster_shape_converter(int m, int n, int k) {
    if (m == 1 && n == 1 && k == 1) {
        return ClusterShape::ClusterShape_1x1x1;
    } else if (m == 2 && n == 1 && k == 1) {
        return ClusterShape::ClusterShape_2x1x1;
    } else if (m == 1 && n == 2 && k == 1) {
        return ClusterShape::ClusterShape_1x2x1;
    } else if (m == 2 && n == 2 && k == 1) {
        return ClusterShape::ClusterShape_2x2x1;
    } else if (m == 1 && n == 8 && k == 1) {
        return ClusterShape::ClusterShape_1x8x1;
    } else if (m == 8 && n == 1 && k == 1) {
        return ClusterShape::ClusterShape_8x1x1;
    } else {
        printf("[WARN]: unsupported cluster shape, failing back to the default cluster config.\n");
        return ClusterShape::ClusterShape_1x1x1;
    }
}
