/*
 * Copyright (c) 2026, Alibaba Group. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include "fc2_reference.cuh"
#include "moe_tp_fused_fp8_transport.cuh"

namespace rtp_llm::moe_tp_fused_fp8 {

// PTX m16n8k32 fragment layout: weight rows are the 16 output channels;
// the activation column is replicated eight times. Only column zero is stored.
// This avoids a weight transpose and expert permutation in the protocol prototype.
// The MMA dot accumulation order differs from the scalar reference.
__device__ __forceinline__ void
fp8_mma_16x8x32(float (&acc)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint32_t fp8_load_four(const __nv_fp8_e4m3* ptr) {
    return *reinterpret_cast<const uint32_t*>(ptr);
}

// Called by all 32 lanes of a warp. The tile is aligned to 16 output channels
// and cannot straddle a token or a 128-channel weight-scale block.
__device__ __forceinline__ void
fc2_token_h16_mma(const Fc2ReferenceParams& params, __nv_bfloat16* output, size_t first) {
    const int lane    = threadIdx.x & 31;
    const int row     = lane >> 2;
    const int k_lane  = (lane & 3) * 4;
    const int token   = first / kFc2ReferenceHiddenSize;
    const int h       = first % kFc2ReferenceHiddenSize;
    float     routed0 = 0.f, routed1 = 0.f;
    for (int slot = 0; slot < kFc2ReferenceTopK; ++slot) {
        const int route_offset = token * kFc2ReferenceTopK + slot;
        const int expert       = params.route_ids[route_offset];
        if (expert < 0)
            continue;
        if (expert >= params.num_experts) {
            if (lane == 0)
                fc2_reference_set_error(params, kFc2ReferenceInvalidExpertId);
            continue;
        }
        float       value0 = 0.f, value1 = 0.f;
        const auto* weights =
            params.weight_fp8
            + (static_cast<size_t>(expert) * kFc2ReferenceHiddenSize + h + row) * kFc2ReferenceIntermediateSize;
        const auto* activation = params.activation_fp8 + route_offset * kFc2ReferenceIntermediateSize;
        for (int block = 0; block < kFc2ReferenceKBlocks; ++block) {
            float dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int step = 0; step < kFc2ReferenceBlockSize; step += 32) {
                const int k = block * kFc2ReferenceBlockSize + step + k_lane;
                fp8_mma_16x8x32(dot,
                                fp8_load_four(weights + k),
                                fp8_load_four(weights + 8 * kFc2ReferenceIntermediateSize + k),
                                fp8_load_four(weights + k + 16),
                                fp8_load_four(weights + 8 * kFc2ReferenceIntermediateSize + k + 16),
                                fp8_load_four(activation + k),
                                fp8_load_four(activation + k + 16));
            }
            const float as = params.activation_scale[route_offset * kFc2ReferenceKBlocks + block];
            const float ws =
                params.weight_scale[(expert * kFc2ReferenceHBlocks + h / kFc2ReferenceBlockSize) * kFc2ReferenceKBlocks
                                    + block];
            value0 += dot[0] * as * ws;
            value1 += dot[2] * as * ws;
        }
        const float route = params.route_weights[route_offset];
        routed0 += __bfloat162float(__float2bfloat16_rn(value0)) * route;
        routed1 += __bfloat162float(__float2bfloat16_rn(value1)) * route;
    }
    if ((lane & 3) == 0) {
        __nv_bfloat16 value0 = __float2bfloat16_rn(routed0);
        __nv_bfloat16 value1 = __float2bfloat16_rn(routed1);
        if (params.gated_shared != nullptr) {
            value0 = __float2bfloat16_rn(__bfloat162float(value0) + __bfloat162float(params.gated_shared[first + row]));
            value1 =
                __float2bfloat16_rn(__bfloat162float(value1) + __bfloat162float(params.gated_shared[first + row + 8]));
        }
        if (!isfinite(__bfloat162float(value0)) || !isfinite(__bfloat162float(value1)))
            fc2_reference_set_error(params, kFc2ReferenceNonfiniteOutput);
        output[first + row]     = value0;
        output[first + row + 8] = value1;
    }
}

__device__ __forceinline__ void
fc2_packet_mma(const Fc2ReferenceParams& params, __nv_bfloat16* output, size_t base, size_t numel) {
    static_assert(kPacketValues % 16 == 0 && kFc2ReferenceHiddenSize % 16 == 0);
    const int warp = threadIdx.x / 32;
    for (int item = warp * 16; item < kPacketValues; item += (blockDim.x / 32) * 16) {
        if (base + item < numel)
            fc2_token_h16_mma(params, output, base + item);
    }
}

}  // namespace rtp_llm::moe_tp_fused_fp8
