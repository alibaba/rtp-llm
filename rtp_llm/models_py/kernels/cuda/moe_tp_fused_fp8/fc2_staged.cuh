/*
 * Copyright (c) 2026, Alibaba Group. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Shared-memory, scalar-order-preserving FC2/finalize implementation.
//
// This is deliberately not an MMA implementation.  It changes only where
// raw FP8 bytes are loaded: activation/routing metadata is shared for the two
// tokens a 496-value packet can cover, and each warp stages its own 32x32
// weight microtile.  Each output lane still performs the reference's slot,
// K-block and K-element ascending fmaf sequence, and keeps the BF16 rounding
// points in exactly the same places.

#include "fc2_reference.cuh"
#include "moe_tp_fused_fp8_transport.cuh"

#include <cstddef>
#include <cstdint>

namespace rtp_llm::moe_tp_fused_fp8 {

constexpr int kStagedWarps        = kPacketThreads / 32;
constexpr int kStagedWarpOutputs  = 32;
constexpr int kStagedKTile        = 32;
constexpr int kStagedWordsPerTile = kStagedKTile / 4;
// The ninth word is padding.  A lane reads one row sequentially, while the
// padding prevents the next row from having the same shared-memory bank map.
constexpr int kStagedWeightStrideWords = kStagedWordsPerTile + 1;

static_assert(kPacketThreads == 512, "fc2_packet_staged is specialized for a 512-thread CTA");
static_assert(kPacketValues == 496, "fc2_packet_staged is specialized for 496-value packets");
static_assert(kFc2ReferenceBlockSize % kStagedKTile == 0);

// Constructing the FP8 object from its byte storage retains the source FP8
// value exactly; conversion to float then uses the same type conversion as
// fc2_token_h_reference.
__device__ __forceinline__ float fp8_e4m3_from_packed_word(uint32_t word, int byte_in_word) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>((word >> (byte_in_word * 8)) & 0xffU);
    return static_cast<float>(value);
}

// All threads in the 512-thread CTA must call this routine for every packet.
// `base` is the packet's element offset in the contiguous [T,2048] output.
// It contains block-wide barriers at entry, after input staging, and at exit;
// callers must not place this invocation in a branch taken by only a subset of
// the CTA.  The exit barrier makes it safe to reuse these static shared arrays
// on the next packet.
__device__ __forceinline__ void
fc2_packet_staged(const Fc2ReferenceParams& params, __nv_bfloat16* output, size_t base, size_t numel) {
    // A packet spans at most two output tokens.  Store FP8 as raw words so the
    // staging copy does not perform a conversion or alter numerical behavior.
    __shared__ uint32_t staged_activation[2][kFc2ReferenceTopK][kFc2ReferenceIntermediateSize / 4];
    __shared__ float    staged_activation_scale[2][kFc2ReferenceTopK][kFc2ReferenceKBlocks];
    __shared__ int32_t  staged_route_ids[2][kFc2ReferenceTopK];
    __shared__ float    staged_route_weights[2][kFc2ReferenceTopK];
    // Each warp owns one [32 output channels, 32 K values] tile.  No other
    // warp accesses that slice, so the tile requires only __syncwarp().
    __shared__ uint32_t staged_weight[kStagedWarps][kStagedWarpOutputs][kStagedWeightStrideWords];

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int token0 = static_cast<int>(base / kFc2ReferenceHiddenSize);

    // Ensure a prior packet has finished consuming static shared memory even
    // if a caller's surrounding loop changes in the future.
    __syncthreads();

    // The host validates this before launch.  Retain the device-side reference
    // behavior for malformed direct callers without attempting to stage data.
    const bool valid_expert_count = params.num_experts > 0 && params.num_experts <= kFc2ReferenceExperts;
    if (!valid_expert_count) {
        if (tid < kPacketValues) {
            const size_t index = base + static_cast<size_t>(tid);
            if (index < numel) {
                output[index] =
                    fc2_token_h_reference(params, index / kFc2ReferenceHiddenSize, index % kFc2ReferenceHiddenSize);
            }
        }
        __syncthreads();
        return;
    }

    // Two tokens x eight routes x 256 FP8 bytes = 1024 uint32 words.  Every
    // aligned source route starts at an offset divisible by four.
    constexpr int kActivationWords = 2 * kFc2ReferenceTopK * (kFc2ReferenceIntermediateSize / 4);
    for (int linear = tid; linear < kActivationWords; linear += kPacketThreads) {
        const int token_relative = linear / (kFc2ReferenceTopK * (kFc2ReferenceIntermediateSize / 4));
        const int within_token   = linear % (kFc2ReferenceTopK * (kFc2ReferenceIntermediateSize / 4));
        const int slot           = within_token / (kFc2ReferenceIntermediateSize / 4);
        const int word           = within_token % (kFc2ReferenceIntermediateSize / 4);
        const int token          = token0 + token_relative;
        uint32_t  value          = 0;
        if (token >= 0 && token < params.tokens) {
            const size_t route = static_cast<size_t>(token) * kFc2ReferenceTopK + slot;
            value = *reinterpret_cast<uint32_t const*>(params.activation_fp8 + route * kFc2ReferenceIntermediateSize
                                                       + word * 4);
        }
        staged_activation[token_relative][slot][word] = value;
    }

    // Route metadata and dequantization scales are small, but sharing them
    // avoids 32 lanes rereading the same values for each valid warp.
    for (int linear = tid; linear < 2 * kFc2ReferenceTopK; linear += kPacketThreads) {
        const int token_relative = linear / kFc2ReferenceTopK;
        const int slot           = linear % kFc2ReferenceTopK;
        const int token          = token0 + token_relative;
        if (token >= 0 && token < params.tokens) {
            const size_t route                         = static_cast<size_t>(token) * kFc2ReferenceTopK + slot;
            staged_route_ids[token_relative][slot]     = params.route_ids[route];
            staged_route_weights[token_relative][slot] = params.route_weights[route];
#pragma unroll
            for (int k_block = 0; k_block < kFc2ReferenceKBlocks; ++k_block) {
                staged_activation_scale[token_relative][slot][k_block] =
                    params.activation_scale[route * kFc2ReferenceKBlocks + k_block];
            }
        } else {
            staged_route_ids[token_relative][slot]     = -1;
            staged_route_weights[token_relative][slot] = 0.f;
#pragma unroll
            for (int k_block = 0; k_block < kFc2ReferenceKBlocks; ++k_block)
                staged_activation_scale[token_relative][slot][k_block] = 0.f;
        }
    }
    __syncthreads();

    const size_t first = base + static_cast<size_t>(warp * kStagedWarpOutputs);
    // Warp 15 has only 16 packet positions.  A warp crossing an output-token
    // boundary cannot share one route vector, so all of its lanes uniformly
    // use the scalar reference path.
    const bool full_warp_in_packet = warp * kStagedWarpOutputs + kStagedWarpOutputs <= kPacketValues;
    const bool full_warp_in_output = first + kStagedWarpOutputs <= numel;
    const int  first_token         = static_cast<int>(first / kFc2ReferenceHiddenSize);
    const int  last_token          = static_cast<int>((first + kStagedWarpOutputs - 1) / kFc2ReferenceHiddenSize);
    const bool normal_warp = full_warp_in_packet && full_warp_in_output && first_token == last_token && first_token >= 0
                             && first_token < params.tokens;

    if (!normal_warp) {
        const size_t index = first + static_cast<size_t>(lane);
        if (warp * kStagedWarpOutputs + lane < kPacketValues && index < numel) {
            output[index] =
                fc2_token_h_reference(params, index / kFc2ReferenceHiddenSize, index % kFc2ReferenceHiddenSize);
        }
    } else {
        const int token_relative = first_token - token0;
        const int hidden         = static_cast<int>(first % kFc2ReferenceHiddenSize) + lane;
        float     routed_sum     = 0.f;

        for (int slot = 0; slot < kFc2ReferenceTopK; ++slot) {
            const int expert = staged_route_ids[token_relative][slot];
            if (expert < 0)
                continue;
            if (expert >= params.num_experts) {
                if (lane == 0)
                    fc2_reference_set_error(params, kFc2ReferenceInvalidExpertId);
                continue;
            }

            float fc2_sum = 0.f;
            for (int k_block = 0; k_block < kFc2ReferenceKBlocks; ++k_block) {
                float dot = 0.f;
                for (int k_tile = 0; k_tile < kFc2ReferenceBlockSize / kStagedKTile; ++k_tile) {
                    const int k_offset = k_block * kFc2ReferenceBlockSize + k_tile * kStagedKTile;
                    // Cooperatively load 32 rows x 8 packed words.  Lanes
                    // 0..7 load a contiguous 32-byte segment of one row;
                    // the four segments per iteration avoid the original
                    // lane-strided scalar global access pattern.
                    for (int linear = lane; linear < kStagedWarpOutputs * kStagedWordsPerTile; linear += 32) {
                        const int    row           = linear / kStagedWordsPerTile;
                        const int    word          = linear % kStagedWordsPerTile;
                        const size_t weight_offset = (static_cast<size_t>(expert) * kFc2ReferenceHiddenSize
                                                      + static_cast<size_t>(hidden - lane + row))
                                                         * kFc2ReferenceIntermediateSize
                                                     + k_offset + word * 4;
                        staged_weight[warp][row][word] =
                            *reinterpret_cast<uint32_t const*>(params.weight_fp8 + weight_offset);
                    }
                    __syncwarp();

                    for (int k = 0; k < kStagedKTile; ++k) {
                        const uint32_t activation_word = staged_activation[token_relative][slot][(k_offset + k) / 4];
                        const uint32_t weight_word     = staged_weight[warp][lane][k / 4];
                        const float    activation      = fp8_e4m3_from_packed_word(activation_word, (k_offset + k) & 3);
                        const float    weight          = fp8_e4m3_from_packed_word(weight_word, k & 3);
                        dot                            = fmaf(activation, weight, dot);
                    }
                    __syncwarp();
                }
                fc2_sum += dot * staged_activation_scale[token_relative][slot][k_block]
                           * params.weight_scale[(expert * kFc2ReferenceHBlocks + hidden / kFc2ReferenceBlockSize)
                                                     * kFc2ReferenceKBlocks
                                                 + k_block];
            }

            const __nv_bfloat16 fc2_bf16 = __float2bfloat16_rn(fc2_sum);
            routed_sum += __bfloat162float(fc2_bf16) * staged_route_weights[token_relative][slot];
        }

        __nv_bfloat16 value = __float2bfloat16_rn(routed_sum);
        if (params.gated_shared != nullptr) {
            value = __float2bfloat16_rn(__bfloat162float(value)
                                        + __bfloat162float(params.gated_shared[first + static_cast<size_t>(lane)]));
        }
        if (!isfinite(__bfloat162float(value)))
            fc2_reference_set_error(params, kFc2ReferenceNonfiniteOutput);
        output[first + static_cast<size_t>(lane)] = value;
    }

    // Required before a caller proceeds to packet codec/transport, and before
    // the next packet overwrites static shared activation/metadata storage.
    __syncthreads();
}

}  // namespace rtp_llm::moe_tp_fused_fp8
