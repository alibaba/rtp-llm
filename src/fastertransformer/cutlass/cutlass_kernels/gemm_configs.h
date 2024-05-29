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

namespace tensorrt_llm
{
namespace cutlass_extensions
{
// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig
{
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 64, CTA_K = 256
    CtaShape16x64x256_WarpShape16x16x256,
    CtaShape32x64x256_WarpShape16x32x256,

    // TensorCore configs CTA_N = 32, CTA_K = 256
    CtaShape32x32x256_WarpShape16x16x256,

    // TensorCore configs CTA_N = 128, CTA_K = 128
    CtaShape32x128x128_WarpShape32x32x128,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    CtaShape16x256x64_WarpShape16x64x64,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,
    CtaShape128x256x128_WarpShape64x64x128,

    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,
    CtaShape256x128x128_WarpShape64x64x128
};

enum class SplitKStyle
{
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    // SPLIT_K_PARALLEL // Not supported yet
};

struct CutlassGemmConfig
{
    CutlassTileConfig tile_config = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = -1;
    int stages = -1;

    CutlassGemmConfig() = default;

    CutlassGemmConfig(
        CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages):
        tile_config(tile_config),
        split_k_style(split_k_style),
        split_k_factor(split_k_factor),
        stages(stages){}

    CutlassGemmConfig(
        CutlassTileConfig tile_config, int split_k_factor, int stages):
        tile_config(tile_config),
        split_k_factor(split_k_factor),
        stages(stages), 
        split_k_style(split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K){}

    bool operator==(CutlassGemmConfig const& r) const {
        return tile_config == r.tile_config && split_k_style == r.split_k_style && split_k_factor == r.split_k_factor && stages == r.stages;
    }
};

} // namespace cutlass_extensions
} // namespace tensorrt_llm
