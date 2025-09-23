/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/cutlass_heuristic.h"


#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut_utils.h"
#include "rtp_llm/cpp/cuda/trt_utils.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <cuda_runtime_api.h>
#include <vector>
#include <set>

using namespace tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

TileConfig
get_tile_config_from_config(CutlassTileConfig tile_config) {
    switch (tile_config) {
        case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
            return TileConfig{16, 128, 64, 16, 32, 64};
        case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
            return TileConfig{16, 256, 64, 16, 64, 64};
        case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            return TileConfig{32, 128, 64, 32, 32, 64};
        case CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
            return TileConfig{64, 64, 128, 32, 64, 64};
        case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
            return TileConfig{64, 128, 64, 32, 64, 64};
        case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
            return TileConfig{64, 128, 64, 64, 32, 64};
        case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
            return TileConfig{128, 64, 64, 64, 32, 64};
        case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
            return TileConfig{128, 128, 8, 64, 64, 8};
        case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
            return TileConfig{128, 128, 64, 64, 32, 64};
        case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
            return TileConfig{128, 128, 64, 64, 64, 64};
        case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
            return TileConfig{128, 128, 64, 128, 32, 64};
        case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
            return TileConfig{128, 256, 64, 64, 64, 64};
        case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
            return TileConfig{256, 128, 64, 64, 64, 64};
        default:
	    throw std::runtime_error("[TensorRT-LLm Error][get_grid_shape_for_config] Invalid config: " + std::to_string(int(tile_config)));
    }
}

struct TileShape
{
    int m;
    int n;
};

TileShape get_cta_shape_for_config(CutlassTileConfig tile_config)
{
    switch (tile_config)
    {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64: return TileShape{16, 128};
    case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64: return TileShape{16, 256};
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64: return TileShape{32, 128};
    case CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64: return TileShape{64, 64};
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64: return TileShape{64, 128};
    case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64: return TileShape{128, 64};
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64: return TileShape{128, 128};
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64: return TileShape{128, 256};
    case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64: return TileShape{256, 128};
    case CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128: return TileShape{16, 256};
    default: TLLM_THROW("[get_grid_shape_for_config] Invalid config");
    }
}

bool is_valid_split_k_factor(const int64_t m, const int64_t n, const int64_t k, const CutlassGemmConfig gemm_config,
    const size_t workspace_bytes, const bool is_weight_only)
{
    int split_k_factor = gemm_config.split_k_factor;

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;
    TileConfig tile_config = get_tile_config_from_config(gemm_config.tile_config);

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only)
    {
        if ((k % k_tile) != 0)
        {
            return false;
        }

        if ((k % split_k_factor) != 0)
        {
            return false;
        }

        const int k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0)
        {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    const int ctas_in_m_dim = (m + tile_config.block_m - 1) / tile_config.block_m;
    const int ctas_in_n_dim = (n + tile_config.block_n - 1) / tile_config.block_n;
    const int required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > int(workspace_bytes))
    {
        return false;
    }

    return true;
}

bool is_valid_split_k_factor(int64_t const m, int64_t const n, int64_t const k, TileShape const tile_shape,
    int const split_k_factor, size_t const workspace_bytes, bool const is_weight_only)
{

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only)
    {
        if ((k % k_tile) != 0)
        {
            return false;
        }

        if ((k % split_k_factor) != 0)
        {
            return false;
        }

        int const k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0)
        {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
    int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;
    int const required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > workspace_bytes)
    {
        return false;
    }

    return true;
}

std::vector<CutlassTileConfig> get_candidate_tiles(
    const int sm, const bool is_weight_only, const bool simt_configs_only, const bool int8_configs_only)
{
    enum class CutlassGemmType : char
    {
        Default,
        WeightOnly,
        Simt,
        Int8
    };

    CutlassGemmType gemm_type = CutlassGemmType::Default;
    if (simt_configs_only)
    {
        gemm_type = CutlassGemmType::Simt;
    }
    else if (is_weight_only)
    {
        gemm_type = CutlassGemmType::WeightOnly;
    }
    else if (int8_configs_only)
    {
        gemm_type = CutlassGemmType::Int8;
    }

    std::vector<CutlassTileConfig> base_configs{
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64, CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64};
    if (sm >= 75)
    {
        base_configs.push_back(CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64);
    }

    switch (gemm_type)
    {
    case CutlassGemmType::Simt: return {CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};
    case CutlassGemmType::WeightOnly:
        if (sm >= 75)
        {
            return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
                CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64};
        }
        else
        {
            return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64};
        }
    case CutlassGemmType::Int8:
        return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
            CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
            CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
            CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
    default: return base_configs;
    }
}

std::vector<CutlassGemmConfig> get_candidate_configs(int sm, const bool is_weight_only, const bool simt_configs_only,
    const bool int8_configs_only, const int max_split_k)
{
    std::vector<CutlassTileConfig> tiles
        = get_candidate_tiles(sm, is_weight_only, simt_configs_only, int8_configs_only);

    std::vector<CutlassGemmConfig> candidate_configs;
    const int min_stages = int8_configs_only ? (sm >= 80 ? 3 : 2) : 2;
    const int max_stages = int8_configs_only ? (sm >= 80 ? 6 : 2) : (sm >= 80 ? 4 : 2);
    for (const auto& tile_config : tiles)
    {
        for (int stages = min_stages; stages <= max_stages; ++stages)
        {
            CutlassGemmConfig config{tile_config, SplitKStyle::NO_SPLIT_K, 1, stages};
            candidate_configs.push_back(config);
            if (sm >= 70)
            {
                for (int split_k_factor = 2; split_k_factor <= max_split_k; ++split_k_factor)
                {
                    auto config = CutlassGemmConfig{tile_config, SplitKStyle::SPLIT_K_SERIAL, split_k_factor, stages};
                    candidate_configs.push_back(config);
                }
            }
        }
    }

    return candidate_configs;
}

std::vector<CutlassGemmConfig> get_valid_config_from_occupancies(
    const std::vector<CutlassGemmConfig>& candidate_configs, const std::vector<int>& occupancies)
{
    if (occupancies.size() != candidate_configs.size())
    {
        throw std::runtime_error(
            "[TensorRT-LLm Error][estimate_best_config_from_occupancies] occpancies and "
            "candidate configs vectors must have equal length.");
    }

    std::vector<CutlassGemmConfig> valid_configs;

    for (int ii = 0; ii < int(candidate_configs.size()); ++ii)
    {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        int occupancy = occupancies[ii];

        if (occupancy == 0)
        {
            continue;
        }
        valid_configs.push_back(candidate_config);
    }
    return valid_configs;
}

CutlassGemmConfig estimate_best_config_from_occupancies(const std::vector<CutlassGemmConfig>& candidate_configs,
    const std::vector<int>& occupancies, const int64_t m, const int64_t n, const int64_t k,
    const int multi_processor_count)
{

    if (occupancies.size() != candidate_configs.size())
    {
        throw std::runtime_error(
            "[TensorRT-LLm Error][estimate_best_config_from_occupancies] occpancies and "
            "candidate configs vectors must have equal length.");
    }

    CutlassGemmConfig best_config;

    // Score will be [0, 1]. The objective is to minimize this score.
    // It represents the fraction of SM resources unused in the last wave.
    float config_score = 1.0f;
    int config_waves = INT_MAX;
    int current_m_tile = 0;

    for (int ii = 0; ii < int(candidate_configs.size()); ++ii)
    {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        TileConfig tile_config = get_tile_config_from_config(candidate_config.tile_config);
        int occupancy = occupancies[ii];

        if (occupancy == 0)
        {
            continue;
        }

        // Keep small tile sizes when possible.
        if (best_config.tile_config != CutlassTileConfig::ChooseWithHeuristic && m < current_m_tile
            && current_m_tile < tile_config.block_m)
        {
            continue;
        }

        const int ctas_in_m_dim = (m + tile_config.block_m - 1) / tile_config.block_m;
        const int ctas_in_n_dim = (n + tile_config.block_n - 1) / tile_config.block_n;

        const int ctas_per_wave = occupancy * multi_processor_count;
        const int ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * candidate_config.split_k_factor;

        const int num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
        const float num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
        const float current_score = float(num_waves_total) - num_waves_fractional;

        const float score_slack = 0.1f;
        if (current_score < config_score
            || ((config_waves > num_waves_total) && (current_score < config_score + score_slack)))
        {
            config_score = current_score;
            config_waves = num_waves_total;
            best_config = candidate_config;
            current_m_tile = tile_config.block_m;
        }
        else if (current_score == config_score
            && (best_config.stages < candidate_config.stages || candidate_config.split_k_factor < best_config.split_k_factor
                || current_m_tile < tile_config.block_m))
        {
            // Prefer deeper pipeline or smaller split-k
            best_config = candidate_config;
            current_m_tile = tile_config.block_m;
            config_waves = num_waves_total;
        }
    }

    if (best_config.tile_config == CutlassTileConfig::ChooseWithHeuristic)
    {
        throw std::runtime_error("[TensorRT-LLm Error] Heurisitc failed to find a valid config.");
    }

    return best_config;
}

std::vector<CutlassTileConfig> get_candidate_tiles(
    int const sm, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    enum class CutlassGemmType : char
    {
        Default,
        WeightOnly,
        Simt,
        Int8,
        Fp8
    };

    CutlassGemmType gemm_type = CutlassGemmType::Default;
    if (config_type_param & CutlassGemmConfig::SIMT_ONLY)
    {
        gemm_type = CutlassGemmType::Simt;
    }
    else if (config_type_param & CutlassGemmConfig::WEIGHT_ONLY)
    {
        gemm_type = CutlassGemmType::WeightOnly;
    }
    else if (config_type_param & CutlassGemmConfig::INT8_ONLY)
    {
        gemm_type = CutlassGemmType::Int8;
    }
    else if (config_type_param & CutlassGemmConfig::FP8_ONLY)
    {
        gemm_type = CutlassGemmType::Fp8;
    }

    std::vector<CutlassTileConfig> base_configs{
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64, CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64};
    if (sm >= 75)
    {
        base_configs.push_back(CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64);
    }

    switch (gemm_type)
    {
    case CutlassGemmType::Simt: return {CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};
    case CutlassGemmType::WeightOnly:
        if (sm >= 75)
        {
            return {CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
                CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
                CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64};
        }
        else
        {
            return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64};
        }
    case CutlassGemmType::Int8:
        return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
            CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
            CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
            CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
    case CutlassGemmType::Fp8:
        if (config_type_param & CutlassGemmConfig::GROUPED_GEMM)
        {
            if (sm == 89)
            {
                return {CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128,
                    CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                    CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
                    CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
                    CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
            }
            else
            {
                // no valid ampere style fp8 configs for sm90
                return {};
            }
        }
    default: return base_configs;
    }
}

std::vector<CutlassTileConfigSM90> get_candidate_tiles_sm90(
    int const sm, CutlassGemmConfig::CandidateConfigTypeParam const config)
{
#ifdef FAST_BUILD
    // Fast build disables all configs except this one for SM90
    return {CutlassTileConfigSM90::CtaShape128x128x128B};
#else
    if (config & CutlassGemmConfig::GROUPED_GEMM)
    {
        return {CutlassTileConfigSM90::CtaShape128x16x128B, CutlassTileConfigSM90::CtaShape128x32x128B,
            CutlassTileConfigSM90::CtaShape128x64x128B, CutlassTileConfigSM90::CtaShape128x128x128B,
            CutlassTileConfigSM90::CtaShape128x256x128B, CutlassTileConfigSM90::CtaShape256x128x128B};
    }
    else
    {
        return {CutlassTileConfigSM90::CtaShape64x16x128B, CutlassTileConfigSM90::CtaShape64x32x128B,
            CutlassTileConfigSM90::CtaShape64x64x128B, CutlassTileConfigSM90::CtaShape64x128x128B,
            CutlassTileConfigSM90::CtaShape64x256x128B, CutlassTileConfigSM90::CtaShape128x16x128B,
            CutlassTileConfigSM90::CtaShape128x32x128B, CutlassTileConfigSM90::CtaShape128x64x128B,
            CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B};
    }
#endif
}

// We only compile CUTLASS kernels with multi-cast along M if the M tile is >= 128. This is purely to improve
// compilation speed.
bool supports_mcast_along_m(CutlassTileConfigSM90 const tile)
{
    std::set<CutlassTileConfigSM90> valid_tiles{CutlassTileConfigSM90::CtaShape128x16x128B,
        CutlassTileConfigSM90::CtaShape128x32x128B, CutlassTileConfigSM90::CtaShape128x64x128B,
        CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B,
        CutlassTileConfigSM90::CtaShape256x128x128B};
    return valid_tiles.count(tile) == 1;
}

// We only compile CUTLASS kernels with multi-cast along N if the N tile is >= 128. This is purely to improve
// compilation speed.
bool supports_mcast_along_n(CutlassTileConfigSM90 const tile)
{
    std::set<CutlassTileConfigSM90> valid_tiles{CutlassTileConfigSM90::CtaShape64x128x128B,
        CutlassTileConfigSM90::CtaShape64x256x128B, CutlassTileConfigSM90::CtaShape128x128x128B,
        CutlassTileConfigSM90::CtaShape128x256x128B, CutlassTileConfigSM90::CtaShape256x128x128B};
    return valid_tiles.count(tile) == 1;
}

std::vector<CutlassGemmConfig> get_candidate_configs(
    int sm, int const max_split_k, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    if (sm == 90 && (config_type_param & CutlassGemmConfig::HOPPER))
    {
        std::vector<CutlassTileConfigSM90> tiles = get_candidate_tiles_sm90(sm, config_type_param);

        std::vector<CutlassGemmConfig> candidate_configs;
        for (auto const& tile_config : tiles)
        {
            CutlassGemmConfig config(
                tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1);
            candidate_configs.push_back(config);

            bool const has_m_mcast = supports_mcast_along_m(tile_config);
            bool const has_n_mcast = supports_mcast_along_n(tile_config);
            if (has_m_mcast)
            {
                CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_2x1x1);
                candidate_configs.push_back(config);
            }

            if (has_n_mcast)
            {
                CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_1x2x1);
                candidate_configs.push_back(config);
            }

            if (has_m_mcast && has_n_mcast)
            {
                CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_2x2x1);
                candidate_configs.push_back(config);
            }
        }
        return candidate_configs;
    }
    std::vector<CutlassTileConfig> tiles = get_candidate_tiles(sm, config_type_param);

    std::vector<CutlassGemmConfig> candidate_configs;
    bool const int8_configs_only = config_type_param & CutlassGemmConfig::INT8_ONLY;
    int const min_stages = int8_configs_only ? 3 : 2;
    int const max_stages = int8_configs_only ? 6 : (sm >= 80 ? 4 : 2);
    for (auto const& tile_config : tiles)
    {
        for (int stages = min_stages; stages <= max_stages; ++stages)
        {
            CutlassGemmConfig config(tile_config, SplitKStyle::NO_SPLIT_K, 1, stages);
            candidate_configs.push_back(config);
            if (sm >= 75)
            {
                for (int split_k_factor = 2; split_k_factor <= max_split_k; ++split_k_factor)
                {
                    auto config = CutlassGemmConfig{tile_config, SplitKStyle::SPLIT_K_SERIAL, split_k_factor, stages};
                    candidate_configs.push_back(config);
                }
            }
        }
    }

    return candidate_configs;
}

CutlassGemmConfig estimate_best_config_from_occupancies(std::vector<CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const& occupancies, int64_t const m, int64_t const n, int64_t const k, int64_t const num_experts,
    int const split_k_limit, size_t const workspace_bytes, int const multi_processor_count, int const is_weight_only)
{

    if (occupancies.size() != candidate_configs.size())
    {
        TLLM_THROW(
            "[estimate_best_config_from_occupancies] occpancies and "
            "candidate configs vectors must have equal length.");
    }

    CutlassGemmConfig best_config;
    // Score will be [0, 1]. The objective is to minimize this score.
    // It represents the fraction of SM resources unused in the last wave.
    float config_score = 1.0f;
    int config_waves = INT_MAX;
    int current_m_tile = 0;

    int const max_split_k = n >= multi_processor_count * 256 ? 1 : split_k_limit;
    for (int ii = 0; ii < candidate_configs.size(); ++ii)
    {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        TileShape tile_shape = get_cta_shape_for_config(candidate_config.tile_config);
        int occupancy = occupancies[ii];

        if (occupancy == 0)
        {
            continue;
        }

        // Keep small tile sizes when possible.
        if (best_config.tile_config != CutlassTileConfig::ChooseWithHeuristic && m < current_m_tile
            && current_m_tile < tile_shape.m)
        {
            continue;
        }

        int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
        int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;

        for (int split_k_factor = 1; split_k_factor <= max_split_k; ++split_k_factor)
        {
            if (is_valid_split_k_factor(m, n, k, tile_shape, split_k_factor, workspace_bytes, is_weight_only))
            {
                int const ctas_per_wave = occupancy * multi_processor_count;
                int const ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * split_k_factor;

                int const num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                float const num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                float const current_score = float(num_waves_total) - num_waves_fractional;

                float const score_slack = 0.1f;
                if (current_score < config_score
                    || ((config_waves > num_waves_total) && (current_score < config_score + score_slack)))
                {
                    config_score = current_score;
                    config_waves = num_waves_total;
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                }
                else if (current_score == config_score
                    && (best_config.stages < candidate_config.stages || split_k_factor < best_config.split_k_factor
                        || current_m_tile < tile_shape.m))
                {
                    // Prefer deeper pipeline or smaller split-k
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                    config_waves = num_waves_total;
                }
            }
        }
    }

    if (best_config.tile_config == CutlassTileConfig::ChooseWithHeuristic)
    {
        TLLM_THROW("Heurisitc failed to find a valid config.");
    }

    return best_config;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
