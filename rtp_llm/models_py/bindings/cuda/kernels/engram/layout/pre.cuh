#pragma once

#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>

#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/math.cuh>

namespace rtp_llm::engram::layout::mega_mhc {

// Model geometry
static constexpr uint32_t kNumRoutes    = 4;
static constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);

// FP8 scale-factor layout
static constexpr uint32_t kHiddenPerSF     = 32;
static constexpr uint32_t kHiddenPerSFWord = 128;

// Kernel tile and pipeline geometry
static constexpr uint32_t BLOCK_M           = 64;
static constexpr uint32_t BLOCK_N           = kNumHCOutputs;
static constexpr uint32_t BLOCK_K           = 64;
static constexpr uint32_t kSwizzleMode      = 128;
static constexpr uint32_t kFnAtomK          = kSwizzleMode / sizeof(float);
static constexpr uint32_t kSwizzleAlignment = 8 * kSwizzleMode;
static constexpr uint32_t kNumIOStages      = 4;
static constexpr uint32_t kNumFnStages      = 2;
static constexpr uint32_t kNumAStages       = 4;
static constexpr uint32_t kNumAccumStages   = 2;
static constexpr uint32_t kNumCoeffStages   = 2;
static constexpr uint32_t kNumWarpsPerWG    = 4;
static constexpr uint32_t kNumThreads       = 6 * kNumWarpsPerWG * 32;

// Split-K workspace geometry
static constexpr uint32_t kNumGemmPartialElementsPerTask   = BLOCK_M * kNumHCOutputs;
static constexpr uint32_t kNumSqrSumPartialElementsPerTask = BLOCK_M;
static constexpr uint32_t kDefaultNumSplits                = 16;
static constexpr uint32_t kNumMaxSplits                    = 64;
static constexpr uint32_t kNumSplitBarriers                = 2;
static constexpr uint64_t kSplitBarrierLineBytes           = 128;
static constexpr uint32_t kNumMaxTokens                    = 1u << 20;
static constexpr uint32_t kNumMaxMBlocks                   = math::constexpr_ceil_div(kNumMaxTokens, BLOCK_M);

// Host/device ABI payloads for the two consumer domains. Tensor maps remain
// standalone kernel parameters so TMA descriptors keep direct grid-constant access.
struct MixArgs {
    const float* scales;
    const float* bases;
    float*       new_prev_mix;
    float*       new_post_mix;
    float*       new_comb_res_mix;
    float        hc_norm_eps;
    float        hc_pre_eps;
    float        hc_post_scale;
    float        sinkhorn_eps;
    uint32_t     num_sinkhorn_iters;
};

struct NormArgs {
    uint32_t           num_tokens;
    const nv_bfloat16* weight;
    const nv_bfloat16* new_residual;
    float              eps;
    float              scale;
    nv_bfloat16*       y_bf16;
    __nv_fp8_e4m3*     y_fp8;
    uint32_t*          y_primary_sf;
    int64_t            y_primary_sf_stride_token;
    int64_t            y_primary_sf_stride_word;
    uint32_t*          y_shared_sf;
    int64_t            y_shared_sf_stride_word;
};

}  // namespace rtp_llm::engram::layout::mega_mhc
