#pragma once

#include <vector>
#include "cute/tensor.hpp"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/cutlass_heuristic.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/grouped_mm_c3x.cuh"
#include "cutlass/cutlass.h"
#include "cutlass_extensions/compute_occupancy.h"
namespace rtp_llm {

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::cutlass_extensions;

template<typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

struct ShapeInfo {
    int m, n, k;
    ShapeInfo() = default;
    ShapeInfo(int m, int n, int k): m(m), n(n), k(k) {};
};

ShapeInfo get_tile_config_shape_info(tc::CutlassTileConfigSM90 config) {
    switch (config) {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case tc::CutlassTileConfigSM90::CtaShape##M##x##N##x##K##B:                                                        \
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

ShapeInfo get_cluster_config_shape_info(tc::ClusterShape config) {
#define SHAPE_CASE(M, N, K)                                                                                            \
    case tc::ClusterShape::ClusterShape_##M##x##N##x##K:                                                               \
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

tc::CutlassTileConfigSM90 tile_shape_converter(int m, int n, int k) {
    if (m == 128 && n == 128 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape128x128x128B;
    } else if (m == 64 && n == 16 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape64x16x128B;
    } else if (m == 64 && n == 32 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape64x32x128B;
    } else if (m == 64 && n == 64 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape64x64x128B;
    } else if (m == 64 && n == 128 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape64x128x128B;
    } else if (m == 64 && n == 256 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape64x256x128B;
    } else if (m == 128 && n == 16 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape128x16x128B;
    } else if (m == 128 && n == 32 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape128x32x128B;
    } else if (m == 128 && n == 64 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape128x64x128B;
    } else if (m == 128 && n == 256 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape128x256x128B;
    } else if (m == 256 && n == 128 && k == 128) {
        return tc::CutlassTileConfigSM90::CtaShape256x128x128B;
    } else if (m == 128 && n == 16 && k == 256) {
        return tc::CutlassTileConfigSM90::CtaShape128x16x256B;
    } else {
        printf("[WARN]: unsupported tile shape, failing back to the default tile config.\n");
        return tc::CutlassTileConfigSM90::CtaShape128x128x128B;
    }
}

tc::ClusterShape cluster_shape_converter(int m, int n, int k) {
    if (m == 1 && n == 1 && k == 1) {
        return tc::ClusterShape::ClusterShape_1x1x1;
    } else if (m == 2 && n == 1 && k == 1) {
        return tc::ClusterShape::ClusterShape_2x1x1;
    } else if (m == 1 && n == 2 && k == 1) {
        return tc::ClusterShape::ClusterShape_1x2x1;
    } else if (m == 2 && n == 2 && k == 1) {
        return tc::ClusterShape::ClusterShape_2x2x1;
    } else if (m == 1 && n == 8 && k == 1) {
        return tc::ClusterShape::ClusterShape_1x8x1;
    } else if (m == 8 && n == 1 && k == 1) {
        return tc::ClusterShape::ClusterShape_8x1x1;
    } else {
        printf("[WARN]: unsupported cluster shape, failing back to the default cluster config.\n");
        return tc::ClusterShape::ClusterShape_1x1x1;
    }
}

template<typename InType, typename OutType>
tc::CutlassGemmConfig get_best_config_customized_sm90(int m, int n, int k, const int& num_groups, const int num_sms) {
    if (num_sms == 78) {
        auto best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        auto best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x128x128B;
        if (m <= 2) {
            best_tile_shape = tc::CutlassTileConfigSM90::CtaShape128x16x128B;
        } else if (m > 2 && m <= 8) {
            best_tile_shape = tc::CutlassTileConfigSM90::CtaShape64x16x128B;
        } else if (m > 8 && m <= 16) {
            best_tile_shape = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
        } else if (m > 32 && m <= 64) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_2x1x1;
        } else if (m > 64 && m <= 4096) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x64x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        } else {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x128x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_1x2x1;
        }
        tc::CutlassGemmConfig chosen_config(
            best_tile_shape, tc::MainloopScheduleType::AUTO, tc::EpilogueScheduleType::AUTO, best_cluster_shape);
        return chosen_config;
    } else {
        auto best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        auto best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x128x128B;
        if (m <= 1) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x32x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_2x1x1;
        } else if (m > 1 && m <= 2) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x32x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_8x1x1;
        } else if (m > 2 && m <= 4) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_8x1x1;
        } else if (m > 4 && m <= 64) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        } else if (m > 64 && m <= 3072) {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x128x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_1x2x1;
        } else {
            best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x128x128B;
            best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        }
        tc::CutlassGemmConfig chosen_config(
            best_tile_shape, tc::MainloopScheduleType::AUTO, tc::EpilogueScheduleType::AUTO, best_cluster_shape);
        return chosen_config;
    }
}

}  // namespace rtp_llm