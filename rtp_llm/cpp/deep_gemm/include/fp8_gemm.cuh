/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t NUM_WARPS_PER_BLOCK>
static __device__ __forceinline__ void write_result_to_gmem(__nv_bfloat16* gmem_d_this_block, const __nv_bfloat16* smem_d, 
                                                            const uint32_t m_offset, const uint32_t m_boundary,
                                                            const uint32_t n_offset, const uint32_t shape_n,
                                                            const uint32_t ld_output) {
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    constexpr int int4_per_tile_line = BLOCK_N * sizeof(__nv_bfloat16) / sizeof(int4);
    int int4_per_global_line = shape_n * sizeof(__nv_bfloat16) / sizeof(int4);
    constexpr auto num_lines = BLOCK_M;
    constexpr auto num_warps = NUM_WARPS_PER_BLOCK;
    const int4* smem_d_int4 = reinterpret_cast<const int4*>(smem_d);
    bool is_last_n_block = n_offset + BLOCK_N > shape_n;
    int int4_per_line =
        is_last_n_block ? int4_per_global_line % int4_per_tile_line : int4_per_tile_line;

    for (int line_idx = warp_idx; line_idx < num_lines; line_idx += num_warps) {
        if (m_offset + line_idx >= m_boundary) {
            break;
        }
        for (int elem_idx = lane_idx; elem_idx < int4_per_line; elem_idx += 32) {
            uint64_t idx = (uint64_t)line_idx * ld_output + n_offset;
            int4* g_data_addr =
                reinterpret_cast<int4*>(&gmem_d_this_block[idx]) + elem_idx;
            const int4* s_data_addr =
                &smem_d_int4[line_idx * (int4_per_tile_line) + elem_idx];
            *g_data_addr = *s_data_addr;
        }
        __syncwarp();
    }   
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          typename SchedulerType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, typename SchedulerType::Input problem_input,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const auto& func) {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        asm volatile("griddepcontrol.wait;");
#endif
    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A with broadcasting
                        auto& full_barrier = *full_barriers[s];
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_a[s], k_idx, scheduler.get_global_m_idx(m_block_idx));

                        if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset) {
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                       smem_scales_a[s], scheduler.get_global_scales_a_idx(m_block_idx),
                                                       k_idx / BLOCK_K);
                        }
                        else {
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                       smem_scales_a[s], m_block_idx * BLOCK_M,
                                                       scheduler.get_global_scales_a_idx(k_idx / BLOCK_K));
                        }

                        // Issue TMA B without broadcasting
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_n_idx(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_scales_b_idx(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);;
                auto local_scales_b = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32)
                    st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    // Read B scales
                    float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s), scale_b_1 = 1.0f;
                    // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    // Read A scales
                    // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

                    // Commit WGMMA instructions
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_arrive();
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    warpgroup_commit_batch();
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();

                    // Notify barrier arrival
                    empty_barrier_arrive(s);

                    // Promote with scales
                    float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                    float scale_0_1, scale_1_1;
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                        final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                        final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                        final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                        final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            });

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                );
            }

            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset) {
                auto m_global_idx = scheduler.get_global_m_idx(m_block_idx);
                bool cross_boundary = (m_global_idx + BLOCK_M) > scheduler.m_boundary;
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (!cross_boundary) {
                    // Use TMA store to write back to global memory
                    if (threadIdx.x == 0) {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N, m_global_idx);
                        cute::tma_store_arrive();
                        cute::tma_store_wait<0>();
                    }
                } else {
                    __nv_bfloat16* gmem_d_this_block = gmem_d + m_global_idx * SHAPE_N;
                    constexpr int NUM_WARPS =
                        (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                    write_result_to_gmem<BLOCK_M, BLOCK_N, NUM_WARPS>(gmem_d_this_block, smem_d, m_global_idx,
                                                                      scheduler.m_boundary, n_block_idx * BLOCK_N,
                                                                      SHAPE_N, SHAPE_N);
                }
            } else if constexpr (SchedulerType::gemm_type == GemmType::StridedBatched) {
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                __nv_bfloat16* gmem_d_this_block;
                auto m_global_idx = scheduler.get_global_m_idx(m_block_idx);
                gmem_d_this_block = gmem_d + scheduler.curr_group_idx * problem_input.stride_d +
                                    (m_block_idx * BLOCK_M) * problem_input.ld_d;
                constexpr int NUM_WARPS =
                    (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;            
                write_result_to_gmem<BLOCK_M, BLOCK_N, NUM_WARPS>(gmem_d_this_block, smem_d, m_global_idx, scheduler.m_boundary,
                                                                  n_block_idx * BLOCK_N, SHAPE_N, problem_input.ld_d);
            } else {
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                // Use TMA store to write back to global memory
                if (threadIdx.x == 0) {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_d, smem_d, n_block_idx * BLOCK_N,
                        scheduler.get_global_m_idx(m_block_idx));
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
            }

            __syncwarp();
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_M, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          typename SchedulerType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel_swapAB(__nv_bfloat16* gmem_d, float* scales_a, typename SchedulerType::Input problem_input,
                      const __grid_constant__ CUtensorMap tensor_map_a,  // weight (以前是act)
                      const __grid_constant__ CUtensorMap tensor_map_b,  // act (以前是weight)
                      const __grid_constant__ CUtensorMap tensor_map_scales_b,  // act scales (以前是tensor_map_scales_a)
                      const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_M, BLOCK_K) == 1, "Too much A scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    DG_STATIC_ASSERT(BLOCK_K % BLOCK_M == 0, "BLOCK_M should be 64 or 128 and BLOCK_K should be 128");
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_N * BLOCK_M * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE = BLOCK_N * sizeof(float);  // B矩阵(act)的scales
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE_PADDED = ceil_div<uint32_t>(BLOCK_N * sizeof(float), 128)*128;  // B矩阵(act)的scales,128B对齐
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_A_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);  // 重命名为A (weight)

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];  // weight
    __nv_fp8_e4m3* smem_b[kNumStages];  // act
    float* smem_scales_b[kNumStages];   // act scales
    float* smem_scales_a;               // weight scales

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_b[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_B_SIZE_PER_STAGE_PADDED);
    }
    smem_scales_a = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE_PADDED));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_a) + SMEM_SCALES_A_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const auto& func) {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SchedulerType(problem_input);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        asm volatile("griddepcontrol.wait;");
#endif
    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A (weight) now without broadcasting
                        auto& full_barrier = *full_barriers[s];
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_a[s], k_idx, scheduler.get_global_m_idx(SHAPE_M, BLOCK_M, m_block_idx, n_block_idx));

                        // Issue TMA B (act) with broadcasting
                        tma_copy<kNumTMAMulticast>(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                  smem_b[s], k_idx, scheduler.get_global_n_idx(n_block_idx));

                        // Issue TMA scales_b (act scales) for B矩阵
                        if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset) {
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_scales_b[s], scheduler.get_global_scales_b_idx(n_block_idx),
                                                      k_idx / BLOCK_K);
                        }
                        else {
                            tma_copy<kNumTMAMulticast>(&tensor_map_scales_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_scales_b[s], n_block_idx * BLOCK_N,
                                                      scheduler.get_global_scales_b_idx(k_idx / BLOCK_K)); // TODO: should it be altered?
                                                        // k_idx / BLOCK_K);
                        }

                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);

        // 每个线程负责加载连续的2个scale
        const uint32_t scale_offset = (lane_idx % 4) * 2;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // 加载weight scales (scales_a) - 这些与tensor_map_a (weight)相关联
            // Decide the number of scales A to load
            DG_STATIC_ASSERT(SHAPE_M % 8 == 0, "Invalid shape M");
            uint32_t num_scales_a = SHAPE_K_SCALES;

            // Load A scales with math warp-groups (weight scales)
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_scales_a_idx(ceil_div(SHAPE_M, BLOCK_K), 0, 0, n_block_idx);
                auto local_scales_a = scales_a + (num_previous_lines + ((m_block_idx * BLOCK_M) / BLOCK_K)) * SHAPE_K_SCALES;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_a; i += kNumMathThreads - 32)
                    st_shared(smem_scales_a + i, __ldg(local_scales_a + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    // Read weight scales (A scales)
                    float scale_a_0 = ld_shared(smem_scales_a + k_iter * kNumStages + s);

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                     // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    // 每个线程读取连续的两个b scales，每个线程需要读取 WGMMA::N / 4 * 2 个b scales
                    float scale_0_0[WGMMA::kNumAccum / 4], scale_0_1[WGMMA::kNumAccum / 4];
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // 每个线程读取连续的两个b scales，每个线程需要读取 WGMMA::N / 4 * 2 个b scales
                        float scale_b_0 = ld_shared(smem_scales_b[s] + i * 8 + scale_offset);
                        float scale_b_1 = ld_shared(smem_scales_b[s] + i * 8 + scale_offset + 1);
                        scale_0_0[i] = scale_a_0 * scale_b_0, scale_0_1[i] = scale_a_0 * scale_b_1;
                    }

                    // Commit WGMMA instructions
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_arrive();
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    warpgroup_commit_batch();
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();

                    // Notify barrier arrival
                    empty_barrier_arrive(s);

                    // Promote with scales
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // 每个线程读取连续的两个b scales，每个线程需要读取 WGMMA::N / 4 * 2 个b scales
                        // float scale_b_0 = ld_shared(smem_scales_b[s] + i * 8 + scale_offset);
                        // float scale_b_1 = ld_shared(smem_scales_b[s] + i * 8 + scale_offset + 1);
                        // float scale_0_0 = scale_a_0 * scale_b_0, scale_0_1 = scale_a_0 * scale_b_1;
                        final_accum[i * 4 + 0] += scale_0_0[i] * accum[i * 4 + 0];
                        final_accum[i * 4 + 1] += scale_0_1[i] * accum[i * 4 + 1];
                        final_accum[i * 4 + 2] += scale_0_0[i] * accum[i * 4 + 2];
                        final_accum[i * 4 + 3] += scale_0_1[i] * accum[i * 4 + 3];
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            });

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            int tid = 0;
            if(lane_idx<8){
                tid = lane_idx * BLOCK_M;
            }else if(lane_idx<16){
                tid = (lane_idx - 8) * BLOCK_M + 8;
            }else if(lane_idx<24){
                tid = (lane_idx - 8) * BLOCK_M;
            }else{
                tid = (lane_idx - 16) * BLOCK_M + 8;
            }
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_T<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    // smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                    smem_d + warp_idx * 16 + i * 16 * BLOCK_M + tid
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_T<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    // smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                    smem_d + warp_idx * 16 + WGMMA::kNumAccum / 8 * 16 * BLOCK_M + tid
                );
            }

            if constexpr (SchedulerType::gemm_type == GemmType::GroupedWithOffset) {
                auto n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                bool cross_boundary = (n_global_idx + BLOCK_N) > scheduler.n_boundary;
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (!cross_boundary) {
                    // Use TMA store to write back to global memory
                    if (threadIdx.x == 0) {
                        // 调整为M方向优先的存储
                        // cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, m_block_idx * BLOCK_M, n_global_idx);
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, m_block_idx * BLOCK_M, n_global_idx);
                        cute::tma_store_arrive();
                        cute::tma_store_wait<0>();
                    }
                } else {
                     __nv_bfloat16* gmem_d_this_block = gmem_d + n_global_idx * SHAPE_M;
                    constexpr int NUM_WARPS =
                        (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                    write_result_to_gmem<BLOCK_N, BLOCK_M, NUM_WARPS>(gmem_d_this_block, smem_d, n_global_idx,
                                                                      scheduler.n_boundary, m_block_idx * BLOCK_M,
                                                                      SHAPE_M, SHAPE_M);
                }
            } else if constexpr (SchedulerType::gemm_type == GemmType::StridedBatched) {
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                __nv_bfloat16* gmem_d_this_block;
                auto n_global_idx = scheduler.get_global_n_idx(n_block_idx);
                gmem_d_this_block = gmem_d + scheduler.curr_group_idx * problem_input.stride_d +
                                    (n_block_idx * BLOCK_N) * problem_input.ld_d;
                constexpr int NUM_WARPS =
                    (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;            
                // 注意：交换参数顺序
                write_result_to_gmem<BLOCK_N, BLOCK_M, NUM_WARPS>(gmem_d_this_block, smem_d, n_global_idx, scheduler.n_boundary,
                                                                  m_block_idx * BLOCK_M, SHAPE_M, problem_input.ld_d);
            } else {
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                // Use TMA store to write back to global memory
                if (threadIdx.x == 0) {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_d, smem_d, m_block_idx * BLOCK_M,
                        scheduler.get_global_n_idx(n_block_idx));
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
            }

            __syncwarp();
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    // DeepGEMM
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,
                                                         BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation        
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {kNumTMAMulticast, 1, 1};
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;        
        config.attrs = attr;
        config.numAttrs = 2;

        InputType input;
        input.shape_m = shape_m;
        input.grouped_layout = grouped_layout;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
                                         tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // Grouped GEMM with Offset
    // `problem_m_padded_4_offsets` is used for reading scales, to satisfy the alignment requirements of TMA,
    // each problem offset in problem_m_padded_4_offsets must be padded to multiple for 4.
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* problem_m_offsets,
                    LayoutIndexType* problem_m_padded_4_offsets,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,
                                                         BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {kNumTMAMulticast, 1, 1};
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;        
        config.attrs = attr;
        config.numAttrs = 2;

        InputType input;
        input.problem_m_offsets = problem_m_offsets;
        input.problem_m_padded_4_offsets = problem_m_padded_4_offsets;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
                                         tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // Batched Strided GEMM
    static void run(__nv_bfloat16* gmem_d, float* scales_b, uint32_t shape_m,
                    const CUtensorMap& tma_a_desc, const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc, const CUtensorMap& tma_d_desc,
                    uint64_t ld_a, uint64_t stride_a, uint64_t ld_b, uint64_t stride_b,
                    uint64_t ld_d, uint64_t stride_d, cudaStream_t stream, int num_sms, uint32_t smem_size) {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,
                                                         BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {kNumTMAMulticast, 1, 1};
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;        
        config.attrs = attr;
        config.numAttrs = 2;

        InputType input{shape_m, ld_a, stride_a, ld_b, stride_b, ld_d, stride_d};
        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
                                         tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m, uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1),
                                SHAPE_K, BLOCK_M, BLOCK_K, global_stride_in_bytes);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address, uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::ColMajor, SHAPE_K,
                                SHAPE_N * (kGemmType != GemmType::Normal ? kNumGroups : 1), BLOCK_K,
                                BLOCK_N, global_stride_in_bytes);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m, uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N,
                                min(BLOCK_M, shape_m), BLOCK_N, global_stride_in_bytes,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m, uint64_t global_stride_in_bytes = 0) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

        return make_2d_tma_desc(
            global_address, Layout::ColMajor, shape_m,
            ceil_div(SHAPE_K, BLOCK_K) * ((kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::StridedBatched) ? kNumGroups : 1),
            BLOCK_M, 1, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_tma_scales_a_offset_desc(T* global_address, int64_t max_m_padded_total,
                                                     uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::ColMajor, max_m_padded_total,
                                ceil_div(SHAPE_K, BLOCK_K), BLOCK_M, 1, global_stride_in_bytes,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            uint64_t global_stride_in_bytes,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_cols * sizeof(T);
            return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_rows * sizeof(T);
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

template <uint32_t SHAPE_M, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class GemmSwapAB {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    GemmSwapAB() = default;

    // DeepGEMM
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* grouped_layout,
                    uint32_t shape_n,
                    const CUtensorMap& tma_a_desc, // weight
                    const CUtensorMap& tma_b_desc, // act
                    const CUtensorMap& tma_scales_b_desc, // act scales
                    const CUtensorMap& tma_d_desc, // out
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        using SchedulerType = typename SchedulerSelectorSwapAB<kGemmType, SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N,
                                                         BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel_swapAB<SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {kNumTMAMulticast, 1, 1};
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;        
        config.attrs = attr;
        config.numAttrs = 2;

        InputType input;
        input.shape_n = shape_n;
        input.grouped_layout = grouped_layout;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
                                         tma_scales_b_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // Grouped GEMM with Offset
    // `problem_m_padded_4_offsets` is used for reading scales, to satisfy the alignment requirements of TMA,
    // each problem offset in problem_m_padded_4_offsets must be padded to multiple for 4.
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* problem_n_offsets,
                    LayoutIndexType* problem_n_padded_4_offsets,
                    const CUtensorMap& tma_a_desc,  // weight
                    const CUtensorMap& tma_b_desc,  // act
                    const CUtensorMap& tma_scales_a_desc,  // act scales
                    const CUtensorMap& tma_d_desc,  // out
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        using SchedulerType = typename SchedulerSelectorSwapAB<kGemmType, SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N,
                                                         BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel_swapAB<SHAPE_M, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim = {kNumTMAMulticast, 1, 1};
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;        
        config.attrs = attr;
        config.numAttrs = 2;

        InputType input;
        input.problem_n_offsets = problem_n_offsets;
        input.problem_n_padded_4_offsets = problem_n_padded_4_offsets;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
                                         tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // // Batched Strided GEMM
    // static void run(__nv_bfloat16* gmem_d, float* scales_b, uint32_t shape_m,
    //                 const CUtensorMap& tma_a_desc, const CUtensorMap& tma_b_desc,
    //                 const CUtensorMap& tma_scales_a_desc, const CUtensorMap& tma_d_desc,
    //                 uint64_t ld_a, uint64_t stride_a, uint64_t ld_b, uint64_t stride_b,
    //                 uint64_t ld_d, uint64_t stride_d, cudaStream_t stream, int num_sms, uint32_t smem_size) {
    //     using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,
    //                                                      BLOCK_K, kNumGroups, kNumTMAMulticast>::type;
    //     using InputType = typename SchedulerType::Input;
    //     // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
    //     constexpr uint32_t kNumTMAThreads = 128;
    //     constexpr uint32_t kNumMathThreadsPerGroup = 128;
    //     auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
    //                                   kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
    //                                   kNumTMAMulticast, SchedulerType>;
    //     DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    //     // Cluster launch
    //     cudaLaunchConfig_t config;
    //     config.gridDim = num_sms;
    //     config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    //     config.dynamicSmemBytes = smem_size;
    //     config.stream = stream;

    //     // Clusters for TMA multicast
    //     // NOTES: `>= 4` cluster size will cause performance degradation
    //     cudaLaunchAttribute attr;
    //     attr.id = cudaLaunchAttributeClusterDimension;
    //     attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
    //     config.attrs = &attr;
    //     config.numAttrs = 1;

    //     InputType input{shape_m, ld_a, stride_a, ld_b, stride_b, ld_d, stride_d};
    //     // Launch
    //     auto status = cudaLaunchKernelEx(&config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc,
    //                                      tma_scales_a_desc, tma_d_desc);
    //     DG_HOST_ASSERT(status == cudaSuccess);
    // }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                SHAPE_M * (kGemmType != GemmType::Normal ? kNumGroups : 1),
                                SHAPE_K, BLOCK_M, BLOCK_K, global_stride_in_bytes);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address, uint32_t shape_n, uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::ColMajor, SHAPE_K,
                                shape_n * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), BLOCK_K,
                                BLOCK_N, global_stride_in_bytes);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_n, uint64_t global_stride_in_bytes = 0) {
        // return make_2d_tma_desc(global_address, Layout::ColMajor,
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_n * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_M, 
                                min(BLOCK_N, shape_n), BLOCK_M, global_stride_in_bytes,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_b_desc(T* global_address, uint32_t shape_n, uint64_t global_stride_in_bytes = 0) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_n = ceil_div(shape_n, kAlignment) * kAlignment;

        return make_2d_tma_desc(
            global_address, Layout::RowMajor, ceil_div(SHAPE_K, BLOCK_K) * ((kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::StridedBatched) ? kNumGroups : 1),
            shape_n, 1, BLOCK_N, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_tma_scales_b_offset_desc(T* global_address, int64_t max_n_padded_total,
                                                     uint64_t global_stride_in_bytes = 0) {
        return make_2d_tma_desc(global_address, Layout::RowMajor, ceil_div(SHAPE_K, BLOCK_K),
                                max_n_padded_total, 1, BLOCK_N, global_stride_in_bytes,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            uint64_t global_stride_in_bytes,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_cols * sizeof(T);
            return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_rows * sizeof(T);
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
