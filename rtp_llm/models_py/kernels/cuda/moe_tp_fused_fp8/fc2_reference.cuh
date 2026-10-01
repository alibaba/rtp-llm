#pragma once

// Correctness-first FC2/finalize device reference for the experimental
// pure-TP2 MoE-to-FP8-all-reduce path.  This is deliberately scalar code: it
// defines the reduction and BF16-rounding order for the transport kernel and
// is not an MMA performance implementation.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <stdint.h>

namespace rtp_llm::moe_tp_fused_fp8 {

constexpr int kFc2ReferenceTopK             = 8;
constexpr int kFc2ReferenceExperts          = 256;
constexpr int kFc2ReferenceHiddenSize       = 2048;
constexpr int kFc2ReferenceIntermediateSize = 256;
constexpr int kFc2ReferenceBlockSize        = 128;
constexpr int kFc2ReferenceKBlocks          = kFc2ReferenceIntermediateSize / kFc2ReferenceBlockSize;
constexpr int kFc2ReferenceHBlocks          = kFc2ReferenceHiddenSize / kFc2ReferenceBlockSize;

static_assert(kFc2ReferenceKBlocks == 2);
static_assert(kFc2ReferenceHBlocks == 16);

enum Fc2ReferenceError : uint32_t {
    kFc2ReferenceInvalidCoordinate  = 1U << 0,
    kFc2ReferenceInvalidExpertId    = 1U << 1,
    kFc2ReferenceInvalidExpertCount = 1U << 2,
    kFc2ReferenceNonfiniteOutput    = 1U << 3,
};

// All tensors are contiguous row-major tensors.  Scales are multiplicative
// dequantization scales: `dequant(fp8, scale) = float(fp8) * scale`.
//
//   activation_fp8: [T, 8, 256] E4M3
//   activation_scale: [T, 8, 2] float
//   weight_fp8: [E, 2048, 256] E4M3
//   weight_scale: [E, 16, 2] float
//   route_ids: [T, 8] int32; negative IDs are masked, IDs >= num_experts are
//              reported through error_flags and ignored
//   route_weights: [T, 8] float
//   gated_shared: optional [T, 2048] BF16 local TP partial, already multiplied
//                 by the replicated shared-expert sigmoid gate
//
// The host validates `0 < num_experts <= 256` and all allocation shapes.  The
// device check is retained so malformed route metadata cannot index weights
// out of bounds.  `error_flags` may be null.
struct Fc2ReferenceParams {
    const __nv_fp8_e4m3* activation_fp8;
    const float*         activation_scale;
    const __nv_fp8_e4m3* weight_fp8;
    const float*         weight_scale;
    const int32_t*       route_ids;
    const float*         route_weights;
    const __nv_bfloat16* gated_shared;
    uint32_t*            error_flags;
    int                  tokens;
    int                  num_experts;
};

__device__ __forceinline__ void fc2_reference_set_error(const Fc2ReferenceParams& params, uint32_t error) {
    if (params.error_flags != nullptr) {
        atomicOr(params.error_flags, error);
    }
}

// Computes one complete local pure-TP partial for (token, hidden_index).
// It is independent per call and requires no block-wide synchronization or
// dynamic shared memory.  A transport kernel can therefore invoke it from one
// thread for each token-major output element before it publishes an FP8 packet.
//
// Rounding contract, intentionally fixed for the scalar baseline:
// 1. For each slot in ascending order, form each 128-K FP32 dot in ascending K.
// 2. Multiply each dot by its activation and weight scales, then FP32-accumulate
//    the two K blocks.
// 3. Round that FC2 value to BF16 before multiplying by its FP32 route weight.
// 4. FP32-sum weighted slots in ascending order, round to BF16, then BF16-add
//    the already gated shared partial and round the final output to BF16.
__device__ __forceinline__ __nv_bfloat16 fc2_token_h_reference(const Fc2ReferenceParams& params,
                                                               int                       token,
                                                               int                       hidden_index) {
    if (token < 0 || token >= params.tokens || hidden_index < 0 || hidden_index >= kFc2ReferenceHiddenSize) {
        fc2_reference_set_error(params, kFc2ReferenceInvalidCoordinate);
        return __float2bfloat16_rn(0.0F);
    }
    if (params.num_experts <= 0 || params.num_experts > kFc2ReferenceExperts) {
        fc2_reference_set_error(params, kFc2ReferenceInvalidExpertCount);
        return __float2bfloat16_rn(0.0F);
    }

    float     routed_sum   = 0.0F;
    const int hidden_block = hidden_index / kFc2ReferenceBlockSize;
    for (int slot = 0; slot < kFc2ReferenceTopK; ++slot) {
        const int route_offset = token * kFc2ReferenceTopK + slot;
        const int expert_id    = params.route_ids[route_offset];
        if (expert_id < 0) {
            continue;
        }
        if (expert_id >= params.num_experts) {
            fc2_reference_set_error(params, kFc2ReferenceInvalidExpertId);
            continue;
        }

        float     fc2_sum         = 0.0F;
        const int activation_base = route_offset * kFc2ReferenceIntermediateSize;
        const int weight_base = (expert_id * kFc2ReferenceHiddenSize + hidden_index) * kFc2ReferenceIntermediateSize;
        const int activation_scale_base = route_offset * kFc2ReferenceKBlocks;
        const int weight_scale_base     = (expert_id * kFc2ReferenceHBlocks + hidden_block) * kFc2ReferenceKBlocks;
        for (int k_block = 0; k_block < kFc2ReferenceKBlocks; ++k_block) {
            float     dot     = 0.0F;
            const int k_begin = k_block * kFc2ReferenceBlockSize;
            for (int k = 0; k < kFc2ReferenceBlockSize; ++k) {
                const float activation = static_cast<float>(params.activation_fp8[activation_base + k_begin + k]);
                const float weight     = static_cast<float>(params.weight_fp8[weight_base + k_begin + k]);
                dot                    = fmaf(activation, weight, dot);
            }
            fc2_sum += dot * params.activation_scale[activation_scale_base + k_block]
                       * params.weight_scale[weight_scale_base + k_block];
        }

        const __nv_bfloat16 fc2_bf16 = __float2bfloat16_rn(fc2_sum);
        routed_sum += __bfloat162float(fc2_bf16) * params.route_weights[route_offset];
    }

    __nv_bfloat16 output = __float2bfloat16_rn(routed_sum);
    if (params.gated_shared != nullptr) {
        output = __float2bfloat16_rn(
            __bfloat162float(output)
            + __bfloat162float(params.gated_shared[token * kFc2ReferenceHiddenSize + hidden_index]));
    }
    if (!isfinite(__bfloat162float(output))) {
        fc2_reference_set_error(params, kFc2ReferenceNonfiniteOutput);
    }
    return output;
}

}  // namespace rtp_llm::moe_tp_fused_fp8
