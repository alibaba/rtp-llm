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
    } else if (num_sms == 132) {
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
    } else {
        // TODO: this heuristics search strategy is not effective currently.
        auto tma_ws_config_param = static_cast<tc::CutlassGemmConfig::CandidateConfigTypeParam>(
            tc::CutlassGemmConfig::NONE | tc::CutlassGemmConfig::NONE | tc::CutlassGemmConfig::GROUPED_GEMM
            | tc::CutlassGemmConfig::HOPPER | tc::CutlassGemmConfig::FP8_ONLY);

        auto valid_configs =
            tk::cutlass_kernels::get_candidate_configs(90 /*sm*/, 1 /*max_split_k*/, tma_ws_config_param);

        if (valid_configs.empty()) {

            tc::CutlassGemmConfig config(tc::CutlassTileConfigSM90::CtaShape128x128x128B,
                                         tc::MainloopScheduleType::AUTO,
                                         tc::EpilogueScheduleType::AUTO,
                                         tc::ClusterShape::ClusterShape_1x1x1);
            return config;
        }

        std::vector<int> occupancies(valid_configs.size());
        // tileshape k allways equal to 128
        const auto& block_k        = 128;
        int         best_num_waves = INT_MAX, best_last_util = 0;
        // Some util functions
        const auto& get_num_blocks = [=](const int& block_m, const int& block_n) {
            return ceil_div(m, block_m) * ceil_div(n, block_n) * num_groups;
        };

        const auto& get_num_waves = [=](const int& block_m, const int& block_n, const int& occupancy) {
            return ceil_div(get_num_blocks(block_m, block_n), static_cast<int>(num_sms * occupancy));
        };

        const auto& get_last_wave_util = [=](const int& block_m, const int& block_n, const int& occupancy) {
            const auto& num_last_blocks = get_num_blocks(block_m, block_n) % static_cast<int>(num_sms * occupancy);
            return num_last_blocks == 0 ? num_sms * occupancy : num_last_blocks;
        };
        // get best config
        int  best_block_m = 0, best_block_n = 0;
        auto best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
        auto best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x128x128B;
        for (size_t i = 0; i < valid_configs.size(); ++i) {
            auto       tile_shape_info    = get_tile_config_shape_info(valid_configs[i].tile_config_sm90);
            auto       cluster_shape_info = get_cluster_config_shape_info(valid_configs[i].cluster_shape);
            const int& block_m            = tile_shape_info.m;
            const int& block_n            = tile_shape_info.n;

            const int& cluster_m = cluster_shape_info.m;
            bool       is_multicast_on_m =
                (cluster_m > 1 && m >= 512 && (num_sms % cluster_m == 0) && (ceil_div(m, block_m) % cluster_m == 0));
            tc::ClusterShape current_cluster_shape =
                is_multicast_on_m ? tc::ClusterShape::ClusterShape_2x1x1 : tc::ClusterShape::ClusterShape_1x1x1;

            const int&  num_waves = get_num_waves(block_m, block_n, 1);
            const auto& last_util = get_last_wave_util(block_m, block_n, 1);

            bool success = false;
            // use the same strategy as deepgemm
            if (best_block_m == 0 or best_block_n == 0 or num_waves < best_num_waves) {
                success = true;
            } else if (num_waves == best_num_waves) {
                success = last_util > best_last_util;
                if (last_util == best_last_util) {
                    // Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n;
                    // Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m;
                    // Case 3: different for both `block_m` and `block_n`, larger `block_n` is better
                    // NOTES: don't pick `block_m/block_n` larger than shape `m/n` in this case
                    success |= block_m != best_block_m and block_n > best_block_n and block_n <= n and block_m <= m;
                }
            }
            if (success) {
                best_block_m       = block_m;
                best_block_n       = block_n;
                best_num_waves     = num_waves;
                best_last_util     = last_util;
                best_tile_shape    = valid_configs[i].tile_config_sm90;
                best_cluster_shape = current_cluster_shape;
            }
        }
        tc::CutlassGemmConfig chosen_config(
            best_tile_shape, tc::MainloopScheduleType::AUTO, tc::EpilogueScheduleType::AUTO, best_cluster_shape);
        return chosen_config;
    }
}

}  // namespace rtp_llm