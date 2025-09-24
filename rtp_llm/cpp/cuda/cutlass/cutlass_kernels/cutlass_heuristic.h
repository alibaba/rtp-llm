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

#pragma once

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_configs.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> get_candidate_configs(int sm,
    const bool is_weight_only, const bool simt_configs_only, const bool int8_configs_only = false,
    const int max_split_k = 1);

std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> get_valid_config_from_occupancies(
    const std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig>& candidate_configs,
    const std::vector<int>& occupancies);

tensorrt_llm::cutlass_extensions::CutlassGemmConfig estimate_best_config_from_occupancies(
    const std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig>& candidate_configs,
    const std::vector<int>& occupancies, const int64_t m, const int64_t n, const int64_t k,
    const int multi_processor_count);

struct TileConfig{
    int block_m;
    int block_n;
    int block_k;
    int warp_m;
    int warp_n;
    int warp_k;
};

void print_config(tensorrt_llm::cutlass_extensions::CutlassGemmConfig config);
bool is_valid_split_k_factor(const int64_t m, const int64_t n, const int64_t k, const tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config,
    const size_t workspace_bytes, const bool is_weight_only);
TileConfig get_tile_config_from_config(tensorrt_llm::cutlass_extensions::CutlassTileConfig tile_config);

 template <class TileShape, class ClusterShape, class ActivationType>
struct should_filter_sm90_gemm_problem_shape
{
#ifdef FAST_BUILD
    constexpr static int TILE_K = 128 * 8 / cutlass::sizeof_bits<ActivationType>::value;
    using SupportedCtaShape = cute::Shape<cute::_128, cute::_128, cute::Int<TILE_K>>;
    using SupportedCgaShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    constexpr static bool value
        = !cute::is_same_v<SupportedCtaShape, TileShape> || !cute::is_same_v<SupportedCgaShape, ClusterShape>;
#else
    constexpr static bool value = false;
#endif
};
template <class TileShape, class ClusterShape, class ActivationType>
constexpr static bool should_filter_sm90_gemm_problem_shape_v
    = should_filter_sm90_gemm_problem_shape<TileShape, ClusterShape, ActivationType>::value;

std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> get_candidate_configs(
    int sm, int const max_split_k, tensorrt_llm::cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam const);

tensorrt_llm::cutlass_extensions::CutlassGemmConfig estimate_best_config_from_occupancies(
    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const& occupancies, int64_t const m, int64_t const n, int64_t const k, int64_t const num_experts,
    int const split_k_limit, size_t const workspace_bytes, int const multi_processor_count, int const is_weight_only);

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
