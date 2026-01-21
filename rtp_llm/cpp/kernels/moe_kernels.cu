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

#include <assert.h>
#include <type_traits>
#include <memory>

#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#if USING_CUDA
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#endif
#include "rtp_llm/cpp/kernels/moe_kernels.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

// Scatter-add operations
template<typename T, int ELEM_PER_THREAD>
__global__ void scatter_add_stable_kernel(T const* src, int N, int K, int32_t const* index, T* out) {
    // 在输出位置上并行,每个线程负责一个输出位置的累加
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    out_idx *= ELEM_PER_THREAD;

    // 计算当前输出元素对应的维度
    const int k     = out_idx % K;
    const int out_n = out_idx / K;

    if (out_n >= N)
        return;

// 对每个输入位置检查,如果它们映射到当前输出位置则累加
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        if (out_idx + i < (size_t)N * K) {
            T sum = out[out_idx + i];
            // 遍历所有输入,找到映射到当前输出位置的元素
            for (int in_n = 0; in_n < N; in_n++) {
                if (index[in_n] == out_n) {
                    sum = sum + src[in_n * K + k + i];
                }
            }
            out[out_idx + i] = sum;
        }
    }
}

template<typename T>
void invokeScatterAddStable(T const* src, int N, int K, int32_t const* index, T* out, cudaStream_t stream) {
    const int  num_threads     = 256;
    const int  elem_per_thread = 4;
    const dim3 block(num_threads);
    RTP_LLM_CHECK(K % (elem_per_thread * 2) == 0);

    auto h_index = std::shared_ptr<int32_t[]>(new int32_t[N], std::default_delete<int32_t[]>());

    check_cuda_value(cudaMemcpy(h_index.get(), index, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int32_t max_out_n = h_index[0];
    for (int i = 1; i < N; i++) {
        max_out_n = max(max_out_n, h_index[i]);
    }
    max_out_n++;

    if constexpr (std::is_same<T, float>::value) {
        const dim3 grid(((size_t)max_out_n * K + num_threads * elem_per_thread - 1) / (num_threads * elem_per_thread));
        scatter_add_stable_kernel<float, elem_per_thread><<<grid, block, 0, stream>>>(src, N, K, index, out);
    } else if (K % 2 == 0) {
        using Tp = typename packed_type_2<T>::type;
        const dim3 grid(((size_t)max_out_n * K / 2 + num_threads * elem_per_thread - 1)
                        / (num_threads * elem_per_thread));
        scatter_add_stable_kernel<Tp, elem_per_thread><<<grid, block, 0, stream>>>((Tp*)src, N, K / 2, index, (Tp*)out);
    } else {
        throw std::invalid_argument("scatter add unsupport type or K [%d]" + std::to_string(K));
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T, int ELEM_PER_THREAD>
__global__ void scatter_add_kernel(T const* src, int N, int K, int32_t const* index, T* out) {
    int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    thread_idx *= ELEM_PER_THREAD;
    // int offset = blockDim.x * gridDim.x;
    int     k       = thread_idx % K;
    int64_t new_idx = (int64_t)index[thread_idx / K] * K;
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (thread_idx + i < (size_t)N * K) {
#if USING_ROCM
#ifdef ENABLE_BF16
            if constexpr (std::is_same<T, __nv_bfloat162>::value) {
                unsafeAtomicAdd(reinterpret_cast<__hip_bfloat162*>(out) + new_idx + k + i,
                                (__hip_bfloat162)src[thread_idx + i]);
            } else {
                unsafeAtomicAdd(out + new_idx + k + i, src[thread_idx + i]);
            }
#else
            unsafeAtomicAdd(out + new_idx + k + i, src[thread_idx + i]);
#endif
#else
            atomicAdd(out + new_idx + k + i, src[thread_idx + i]);
#endif
        }
    }
}

template<typename T>
void invokeScatterAdd(
    T const* src, int N, int K, int32_t const* index, T* out, bool use_stable_scatter_add, cudaStream_t stream) {
    RTP_LLM_CHECK_WITH_INFO(N > 0 && K > 0, "N and K must be greater than 0");
    if (use_stable_scatter_add) {
        invokeScatterAddStable(src, N, K, index, out, stream);
        return;
    }
    const int  num_threads     = 256;
    const int  elem_per_thread = 4;
    const dim3 block(num_threads);
    RTP_LLM_CHECK(K % (elem_per_thread * 2) == 0);

    if constexpr (std::is_same<T, float>::value) {
        const dim3 grid(((size_t)N * K + num_threads * elem_per_thread - 1) / (num_threads * elem_per_thread));
        scatter_add_kernel<float, elem_per_thread><<<grid, block, 0, stream>>>(src, N, K, index, out);
    } else if (K % 2 == 0) {
        using Tp = typename packed_type_2<T>::type;
        const dim3 grid(((size_t)N * K / 2 + num_threads * elem_per_thread - 1) / (num_threads * elem_per_thread));
        scatter_add_kernel<Tp, elem_per_thread><<<grid, block, 0, stream>>>((Tp*)src, N, K / 2, index, (Tp*)out);

    } else {
        throw std::invalid_argument("scatter add unsupport type or K [%d]" + std::to_string(K));
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#define INSTANTIATE_INVOKE_SCATTER_ADD(T)                                                                              \
    template void invokeScatterAdd(                                                                                    \
        T const* src, int N, int K, int32_t const* index, T* out, bool use_stable_scatter_add, cudaStream_t stream)

INSTANTIATE_INVOKE_SCATTER_ADD(half);
INSTANTIATE_INVOKE_SCATTER_ADD(float);

#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SCATTER_ADD(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_SCATTER_ADD

// Slice operations
template<typename T>
__global__ void sliceDim1CopyKernel(T const* src, int dim0, int dim1, int dim1_start, int dim1_size, T* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < dim0 * dim1_size; index += blockDim.x * gridDim.x) {
        const int    col_index = index % dim1_size;
        const size_t batch_id  = index / dim1_size;
        out[index]             = src[batch_id * dim1 + dim1_start + col_index];
    }
}

template<typename T>
void invokeSliceDim1Copy(T const* src, int dim0, int dim1, int dim1_start, int dim1_size, T* out, cudaStream_t stream) {
    if constexpr (std::is_same<uint8_t, T>::value) {
        if (dim1 % 16 == 0 && dim1_start % 16 == 0 && dim1_size % 16 == 0) {
            dim1 /= 16;
            dim1_start /= 16;
            dim1_size /= 16;
            const int grid_size = (int)(ceil((size_t)dim0 * dim1_size / 512.));
            dim3      grid(min(grid_size, 65536));
            dim3      block(512);
            sliceDim1CopyKernel<uint4>
                <<<grid, block, 0, stream>>>((uint4 const*)src, dim0, dim1, dim1_start, dim1_size, (uint4*)out);
        } else if (dim1 % 8 == 0 && dim1_start % 8 == 0 && dim1_size % 8 == 0) {
            dim1 /= 8;
            dim1_start /= 8;
            dim1_size /= 8;
            const int grid_size = (int)(ceil((size_t)dim0 * dim1_size / 512.));
            dim3      grid(min(grid_size, 65536));
            dim3      block(512);
            sliceDim1CopyKernel<uint2>
                <<<grid, block, 0, stream>>>((uint2 const*)src, dim0, dim1, dim1_start, dim1_size, (uint2*)out);
        } else if (dim1 % 4 == 0 && dim1_start % 4 == 0 && dim1_size % 4 == 0) {
            dim1 /= 4;
            dim1_start /= 4;
            dim1_size /= 4;
            const int grid_size = (int)(ceil((size_t)dim0 * dim1_size / 512.));
            dim3      grid(min(grid_size, 65536));
            dim3      block(512);
            sliceDim1CopyKernel<uint>
                <<<grid, block, 0, stream>>>((uint const*)src, dim0, dim1, dim1_start, dim1_size, (uint*)out);
        } else {
            const int grid_size = (int)(ceil((size_t)dim0 * dim1_size / 512.));
            dim3      grid(min(grid_size, 65536));
            dim3      block(512);
            sliceDim1CopyKernel<T><<<grid, block, 0, stream>>>(src, dim0, dim1, dim1_start, dim1_size, out);
        }
    } else {
        const int grid_size = (int)(ceil((size_t)dim0 * dim1_size / 512.));
        dim3      grid(min(grid_size, 65536));
        dim3      block(512);
        sliceDim1CopyKernel<T><<<grid, block, 0, stream>>>(src, dim0, dim1, dim1_start, dim1_size, out);
    }
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#define INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(T)                                                                         \
    template void invokeSliceDim1Copy(                                                                                 \
        T const* src, int dim0, int dim1, int dim1_start, int dim1_size, T* out, cudaStream_t stream)

INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(float);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(half);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(int32_t);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(int8_t);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(uint8_t);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(uint32_t);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(int64_t);
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(uint64_t);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(__nv_bfloat16);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_INVOKE_SlICE_DIM1_COPTY(__nv_fp8_e4m3);
#endif

/*
 * fake_balance_expert: deterministic token->expert ids (for load-balance tests).
 *
 * Global load view:
 *   for dp_rank in [0, dp_size):
 *     for index in [0, size):   # size = token_num * top_k on this dp_rank
 *       expert_counts[expert_id(dp_rank, index)] += 1
 *
 * Intermediate terms:
 *   dest_ep_rank_offset: dp-rank dependent EP "start" offset; rotates which EP ranks
 *     receive the first assignments from this dp_rank.
 *   dest_local_expert_idx_offset: dp-rank dependent local-expert "start" offset; rotates
 *     the local expert index to avoid different dp_ranks landing on the same experts.
 *   dest_ep_rank / dest_local_expert_idx: selected EP partition and local expert index
 *     after applying offsets.
 *   expert_id: global expert id (dest_ep_rank * local_expert_num + dest_local_expert_idx).
 *
 *   dest_ep_rank = (dest_ep_rank_offset + index % ep_size) % ep_size
 *   dest_local_expert_idx = (dest_local_expert_idx_offset + index / ep_size) % local_expert_num
 *   expert_id = dest_ep_rank * local_expert_num + dest_local_expert_idx
 */
template<typename T>
__global__ void fakeBalanceExpertKernel(T*     expert,
                                        float* expert_scales,
                                        int    ep_size,
                                        int    local_expert_num,
                                        int    dest_ep_rank_offset,
                                        int    dest_local_expert_idx_offset,
                                        int    size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int dest_ep_rank          = (dest_ep_rank_offset + index % ep_size) % ep_size;
        int dest_local_expert_idx = (dest_local_expert_idx_offset + index / ep_size) % local_expert_num;
        expert[index]             = dest_ep_rank * local_expert_num + dest_local_expert_idx;
        expert_scales[index]      = 1.0f;
    }
}

void fake_balance_expert(int*         expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream) {

    int local_expert_num    = expert_num / ep_size;
    int dest_ep_rank_offset = ep_size * dp_rank / dp_size;
    int dest_local_expert_idx_offset =
        static_cast<int>(dp_rank * std::max(static_cast<float>(local_expert_num) / dp_size, 1.0f)) % local_expert_num;
    fakeBalanceExpertKernel<int><<<(size + 255) / 256, 256, 0, stream>>>(
        expert, expert_scales, ep_size, local_expert_num, dest_ep_rank_offset, dest_local_expert_idx_offset, size);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

void fake_balance_expert(int64_t*     expert,
                         float*       expert_scales,
                         int          dp_rank,
                         int          dp_size,
                         int          ep_size,
                         int          expert_num,
                         int          size,
                         cudaStream_t stream) {
    int local_expert_num    = expert_num / ep_size;
    int dest_ep_rank_offset = ep_size * dp_rank / dp_size;
    int dest_local_expert_idx_offset =
        static_cast<int>(dp_rank * std::max(static_cast<float>(local_expert_num) / dp_size, 1.0f)) % local_expert_num;
    fakeBalanceExpertKernel<int64_t><<<(size + 255) / 256, 256, 0, stream>>>(
        expert, expert_scales, ep_size, local_expert_num, dest_ep_rank_offset, dest_local_expert_idx_offset, size);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

// Expert indexing operations
__global__ void
genSourceRowKernelRevert(int64_t* expert_rows, int* expert_rows_dst, size_t token_num, size_t top_k, int start_expert) {
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < token_num * top_k) {
        if (expert_rows[idx] >= 0) {
            expert_rows_dst[idx] = static_cast<int>(expert_rows[idx] + start_expert);
        } else {
            expert_rows_dst[idx] = static_cast<int>(expert_rows[idx]);
        }
    }
}

void genSourceRowRevert(
    int64_t* expert_rows, int* expert_rows_dst, size_t token_num, size_t top_k, int start_expert, cudaStream_t stream) {
    size_t const threads = 256;
    size_t const blocks  = token_num * top_k / 256 + 1;

    genSourceRowKernelRevert<<<blocks, threads, 0, stream>>>(
        expert_rows, expert_rows_dst, token_num, top_k, start_expert);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

}  // namespace rtp_llm
