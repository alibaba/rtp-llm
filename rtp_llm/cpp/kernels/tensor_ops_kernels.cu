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
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_fp8_utils.h"
#include "rtp_llm/cpp/utils/math_utils.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#endif
#include "rtp_llm/cpp/kernels/tensor_ops_kernels.h"

#if USING_CUDA
#include "rtp_llm/cpp/kernels/vec_dtypes.cuh"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/kernels/rocm_utils/vec_dtypes_hip.h"
#endif

namespace rtp_llm {

// Transpose operations
template<typename T>
__global__ void
transposeAxis01(T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t elem_num) {
    size_t index = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if (index < elem_num) {
        const size_t input_dim2_index = index % dim2;
        index                         = (index - input_dim2_index) / dim2;
        const size_t input_dim1_index = index % dim1;
        index                         = (index - input_dim1_index) / dim1;
        const size_t input_dim0_index = index % dim0;

        const size_t src_idx = input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index;
        const size_t dst_idx = input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index;
        out[dst_idx]         = in[src_idx];
    }
}

template<typename T>
void invokeTransposeAxis012(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, cudaStream_t stream) {
    const size_t elem_num   = dim0 * dim1 * dim2;
    const size_t thread_num = 512;
    const size_t block_num  = ceil_div<size_t>(elem_num, thread_num);
    dim3         block(thread_num);
    dim3         grid(block_num);
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2, elem_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template<typename T>
__global__ void transposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const size_t dim0, const size_t dim1, const size_t elem_num) {
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    size_t index = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if (index < elem_num) {
        const size_t input_dim1_index = index % dim1;
        index                         = (index - input_dim1_index) / dim1;
        const size_t input_dim0_index = index % dim0;
        const size_t in_offset =
            in_skipping_dim1 == nullptr ? 0 : static_cast<size_t>(in_skipping_dim1[input_dim1_index]) * dim1;

        const size_t src_idx = in_offset + input_dim0_index * dim1 + input_dim1_index;
        const size_t dst_idx = input_dim1_index * dim0 + input_dim0_index;
        out[dst_idx]         = in[src_idx];
    }
}

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const size_t dim0, const size_t dim1, cudaStream_t stream) {
    const size_t elem_num   = dim0 * dim1;
    const size_t thread_num = 512;
    const size_t block_num  = ceil_div<size_t>(elem_num, thread_num);
    dim3         block(thread_num);
    dim3         grid(block_num);
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, nullptr, dim0, dim1, elem_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#define DEFINE_INVOKETRANSPOSE(T)                                                                                      \
    template void invokeTransposeAxis01(T* out, T* in, const size_t dim0, const size_t dim1, cudaStream_t stream);     \
    template void invokeTransposeAxis012(                                                                              \
        T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, cudaStream_t stream)

DEFINE_INVOKETRANSPOSE(int32_t);
DEFINE_INVOKETRANSPOSE(int8_t);
DEFINE_INVOKETRANSPOSE(uint8_t);
DEFINE_INVOKETRANSPOSE(uint32_t);
DEFINE_INVOKETRANSPOSE(int64_t);
DEFINE_INVOKETRANSPOSE(uint64_t);
DEFINE_INVOKETRANSPOSE(float);
DEFINE_INVOKETRANSPOSE(half);
#ifdef ENABLE_BF16
DEFINE_INVOKETRANSPOSE(__nv_bfloat16);
#endif

#ifdef ENABLE_FP8
DEFINE_INVOKETRANSPOSE(__nv_fp8_e4m3);
#endif

template<typename T>
__global__ void transposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3, const size_t elem_num) {
    size_t index = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if (index < elem_num) {
        const size_t input_dim3_index = index % dim3;
        index                         = (index - input_dim3_index) / dim3;
        const size_t input_dim2_index = index % dim2;
        index                         = (index - input_dim2_index) / dim2;
        const size_t input_dim1_index = index % dim1;
        index                         = (index - input_dim1_index) / dim1;
        const size_t input_dim0_index = index % dim0;

        const size_t src_idx = input_dim0_index * dim1 * dim2 * dim3 + input_dim1_index * dim2 * dim3
                               + input_dim2_index * dim3 + input_dim3_index;
        const size_t dst_idx = input_dim0_index * dim1 * dim2 * dim3 + input_dim2_index * dim1 * dim3
                               + input_dim1_index * dim3 + input_dim3_index;
        out[dst_idx] = in[src_idx];
    }
}

template<typename T>
void invokeTransposeAxis12(
    T* out, T* in, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim_3, cudaStream_t stream) {
    const size_t elem_num   = dim0 * dim1 * dim2 * dim_3;
    const size_t thread_num = 512;
    const size_t block_num  = ceil_div<size_t>(elem_num, thread_num);
    dim3         block(thread_num);
    dim3         grid(block_num);
    transposeAxis12<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2, dim_3, elem_num);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

template void invokeTransposeAxis12(float*       out,
                                    float*       in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

template void invokeTransposeAxis12(half*        out,
                                    half*        in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

template void invokeTransposeAxis12(size_t*      out,
                                    size_t*      in,
                                    const size_t dim0,
                                    const size_t dim1,
                                    const size_t dim2,
                                    const size_t dim_3,
                                    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeTransposeAxis12(__nv_bfloat16* out,
                                    __nv_bfloat16* in,
                                    const size_t   dim0,
                                    const size_t   dim1,
                                    const size_t   dim2,
                                    const size_t   dim_3,
                                    cudaStream_t   stream);
#endif

// Sequence operations
template<typename T>
__launch_bounds__(1024, 1) __global__ void lookupHiddenStateOfLastToken(T*           from_tensor,
                                                                        const T*     hidden_state,
                                                                        const int*   input_lengths,
                                                                        const size_t batch_size,
                                                                        const size_t hidden_units,
                                                                        const size_t idx_offset) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)batch_size * hidden_units;
         index += (size_t)blockDim.x * gridDim.x) {
        const size_t col_index = index % hidden_units;
        const size_t batch_id  = index / hidden_units;
        from_tensor[index] = hidden_state[((size_t)input_lengths[batch_id] + idx_offset) * hidden_units + col_index];
    }
}

template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const size_t batch_size,
                                        const size_t hidden_units,
                                        const size_t idx_offset,
                                        cudaStream_t stream) {
    const size_t grid_size = ceil_div<size_t>(batch_size * hidden_units, 1024ul);
    dim3         grid(std::min(grid_size, 65536ul));
    dim3         block(std::min(hidden_units, 1024ul));
    lookupHiddenStateOfLastToken<T>
        <<<grid, block, 0, stream>>>(from_tensor, hidden_state, input_lengths, batch_size, hidden_units, idx_offset);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

#define INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(T)                                                                    \
    template void invokeLookupHiddenStateOfLastToken(T*           from_tensor,                                         \
                                                     const T*     hidden_state,                                        \
                                                     const int*   input_lengths,                                       \
                                                     const size_t batch_size,                                          \
                                                     const size_t hidden_units,                                        \
                                                     const size_t idx_offset,                                          \
                                                     cudaStream_t stream)

INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(float);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(half);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(int32_t);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(int8_t);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(uint8_t);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(uint32_t);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(int64_t);
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(uint64_t);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(__nv_bfloat16);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_INVOKE_LOOKUP_HIDDEN_OF_LAST(__nv_fp8_e4m3);
#endif

template<typename T, size_t vec_size>
__global__ void checkNANKernel(T* input, size_t nums, size_t circle) {
    size_t index =
        (static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x))
        * vec_size * circle;
    vec_t<T, vec_size> inputs;
    size_t             max_index = min(index + vec_size * circle, nums - vec_size);
    for (size_t i = index; i < max_index; i += vec_size) {
        inputs.load(input + i);
        for (size_t j = 0; j < vec_size; ++j)
            if (isnan(float(inputs[j]))) {
                // Fail fast on NaN detection (device-side).
                // This is intended for debugging / validation only.
                __trap();
                break;
            }
    }
}

template<typename T>
void invokeCheckNAN(T* input, size_t nums, cudaStream_t stream) {
    constexpr size_t vec_size = 16 / sizeof(T);
    size_t           circle   = nums / 512 / 65536 + 1;
    dim3             grid(nums / circle / 512);
    dim3             block(512);
    checkNANKernel<T, vec_size><<<grid, block, 0, stream>>>(input, nums, circle);
}

template void invokeCheckNAN(float* input, size_t nums, cudaStream_t stream);
template void invokeCheckNAN(half* input, size_t nums, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeCheckNAN(nv_bfloat16* input, size_t nums, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8
template void invokeCheckNAN(__nv_fp8_e4m3* input, size_t nums, cudaStream_t stream);
#endif

template<typename T>
struct NanInfChecker;

// Float (FP32)
template<>
struct NanInfChecker<float> {
    // Check and reset a 16-byte vector (uint4) in-place.
    __device__ __forceinline__ static bool check_and_reset(uint4* vec_ptr) {
        uint32_t*      p             = reinterpret_cast<uint32_t*>(vec_ptr);
        bool           found_invalid = false;
        const uint32_t exp_mask      = 0x7F800000u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if ((p[i] & exp_mask) == exp_mask) {
                p[i]          = 0u;
                found_invalid = true;
            }
        }
        return found_invalid;
    }

    // Check and reset a single scalar in-place.
    __device__ __forceinline__ static bool check_and_reset(float* val_ptr) {
        float val = *val_ptr;
        if (isnan(val) || isinf(val)) {
            *val_ptr = 0.0f;
            return true;
        }
        return false;
    }
};

// Half (FP16)
template<>
struct NanInfChecker<__half> {
    // Check and reset a 16-byte vector (uint4) in-place.
    __device__ __forceinline__ static bool check_and_reset(uint4* vec_ptr) {
        uint16_t* p             = reinterpret_cast<uint16_t*>(vec_ptr);
        bool      found_invalid = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((p[i] & 0x7C00u) == 0x7C00u) {
                p[i]          = 0u;
                found_invalid = true;
            }
        }
        return found_invalid;
    }

    // Check and reset a single scalar in-place.
    __device__ __forceinline__ static bool check_and_reset(__half* val_ptr) {
        __half val = *val_ptr;
        if (__hisnan(val) || __hisinf(val)) {
            *val_ptr = __half(0.0f);
            return true;
        }
        return false;
    }
};

#ifdef ENABLE_BF16
// BFloat16
template<>
struct NanInfChecker<__nv_bfloat16> {
    // Check and reset a 16-byte vector (uint4) in-place.
    __device__ __forceinline__ static bool check_and_reset(uint4* vec_ptr) {
        uint16_t* p             = reinterpret_cast<uint16_t*>(vec_ptr);
        bool      found_invalid = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((p[i] & 0x7F80u) == 0x7F80u) {
                p[i]          = 0u;
                found_invalid = true;
            }
        }
        return found_invalid;
    }

    // Check and reset a single scalar in-place.
    __device__ __forceinline__ static bool check_and_reset(__nv_bfloat16* val_ptr) {
        uint16_t* bits_ptr = reinterpret_cast<uint16_t*>(val_ptr);
        if ((*bits_ptr & 0x7F80u) == 0x7F80u) {
            *val_ptr = __nv_bfloat16(0.0f);
            return true;
        }
        return false;
    }
};
#endif

#ifdef ENABLE_FP8
// FP8 E4M3
template<>
struct NanInfChecker<__nv_fp8_e4m3> {
    // Check and reset a 16-byte vector (uint4) in-place.
    __device__ __forceinline__ static bool check_and_reset(uint4* vec_ptr) {
        uint8_t* p             = reinterpret_cast<uint8_t*>(vec_ptr);
        bool     found_invalid = false;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (((p[i] >> 3) & 0xFu) == 0xFu) {
                p[i]          = 0u;
                found_invalid = true;
            }
        }
        return found_invalid;
    }

    // Check and reset a single scalar in-place.
    __device__ __forceinline__ static bool check_and_reset(__nv_fp8_e4m3* val_ptr) {
        uint8_t* bits_ptr = reinterpret_cast<uint8_t*>(val_ptr);
        if (((*bits_ptr >> 3) & 0xFu) == 0xFu) {
            *val_ptr = __nv_fp8_e4m3(0.0f);
            return true;
        }
        return false;
    }
};
#endif

// KV cache NaN/Inf check/reset kernel for prefill.
//
// Layout and tensor contracts match invokeCheckAndResetNANKvCachePrefill() in tensor_ops_kernels.h.
template<typename T>
__global__ void check_and_reset_kv_cache_prefill_kernel(const void* const* __restrict__ layer_base_addr,
                                                        const int32_t* __restrict__ kv_cache_block_id,
                                                        const int32_t* __restrict__ prefix_lengths,
                                                        const int32_t* __restrict__ seq_len_cu,
                                                        size_t batch_size,
                                                        size_t layer_num,
                                                        size_t max_blocks_per_batch,
                                                        size_t block_size_bytes,
                                                        size_t seq_size_per_block,
                                                        int32_t* __restrict__ nan_flag) {

    const int batch_id = blockIdx.x;
    const int layer_id = blockIdx.y;

    if (batch_id >= batch_size || layer_id >= layer_num)
        return;

    const void* base_ptr_void = layer_base_addr[layer_id];
    if (!base_ptr_void)
        return;

    char*          base_ptr     = static_cast<char*>(const_cast<void*>(base_ptr_void));
    const int32_t* batch_blocks = kv_cache_block_id + batch_id * max_blocks_per_batch;

    const int prefix_len = prefix_lengths[batch_id];
    const int total_len  = seq_len_cu[batch_id];

    if (prefix_len >= total_len)
        return;

    const size_t token_stride = block_size_bytes / seq_size_per_block;
    const int    full_vecs    = token_stride >> 4;
    const int    remainder    = token_stride & 15;

    int32_t* flag = nan_flag + batch_id;

    // Grid-stride loop
    const int stride      = blockDim.x;
    const int start_token = prefix_len + threadIdx.x;

    for (int token_idx = start_token; token_idx < total_len; token_idx += stride) {
        const int logical_block_idx = token_idx / seq_size_per_block;
        const int offset_in_block   = token_idx % seq_size_per_block;

        // Bounds check on logical block index.
        if (logical_block_idx >= max_blocks_per_batch)
            continue;

        const int physical_block_id = batch_blocks[logical_block_idx];
        if (physical_block_id == -1)
            continue;

        char* data_ptr = base_ptr + physical_block_id * block_size_bytes + offset_in_block * token_stride;

        // Process full uint4 vectors (16 bytes each).
#pragma unroll 4
        for (int i = 0; i < full_vecs; i++) {
            char* vec_addr = data_ptr + (i << 4);
            // Assumes vec_addr is 16-byte aligned for vectorized access.
            uint4* vec_ptr = reinterpret_cast<uint4*>(vec_addr);

            if (NanInfChecker<T>::check_and_reset(vec_ptr)) {
                *flag = 1;
            }
        }

        // Handle remaining bytes (thread 0 only to avoid races).
        if (remainder > 0 && threadIdx.x == 0) {
            char*     remainder_ptr = data_ptr + (full_vecs << 4);
            const int elem_size     = sizeof(T);
            const int num_elems     = remainder / elem_size;
            T*        typed_ptr     = reinterpret_cast<T*>(remainder_ptr);

            for (int i = 0; i < num_elems; i++) {
                if (NanInfChecker<T>::check_and_reset(&typed_ptr[i])) {
                    *flag = 1;
                }
            }
        }
    }
}

// KV cache NaN/Inf check/reset kernel for decode (last token only).
//
// Layout and tensor contracts match invokeCheckAndResetNANKvCacheDecode() in tensor_ops_kernels.h.
template<typename T>
__global__ void check_and_reset_kv_cache_decode_kernel(const void* const* __restrict__ layer_base_addr,
                                                       const int32_t* __restrict__ kv_cache_block_id,
                                                       const int32_t* __restrict__ sequence_lengths,
                                                       size_t batch_size,
                                                       size_t layer_num,
                                                       size_t max_blocks_per_batch,
                                                       size_t block_size_bytes,
                                                       size_t seq_size_per_block,
                                                       int32_t* __restrict__ nan_flag) {

    const int batch_id = blockIdx.x;
    const int layer_id = blockIdx.y;

    if (batch_id >= batch_size || layer_id >= layer_num)
        return;

    const void* base_ptr_void = layer_base_addr[layer_id];
    if (!base_ptr_void)
        return;

    char*     base_ptr = static_cast<char*>(const_cast<void*>(base_ptr_void));
    const int seq_len  = sequence_lengths[batch_id];

    if (seq_len == 0)
        return;

    const int last_token_idx    = seq_len - 1;
    const int logical_block_idx = last_token_idx / seq_size_per_block;
    const int offset_in_block   = last_token_idx % seq_size_per_block;

    // Bounds check on logical block index.
    if (logical_block_idx >= max_blocks_per_batch)
        return;

    const int32_t* batch_blocks      = kv_cache_block_id + batch_id * max_blocks_per_batch;
    const int      physical_block_id = batch_blocks[logical_block_idx];

    if (physical_block_id == -1)
        return;

    const size_t token_stride = block_size_bytes / seq_size_per_block;
    char*        token_data   = base_ptr + physical_block_id * block_size_bytes + offset_in_block * token_stride;

    const int full_vecs = token_stride >> 4;
    const int remainder = token_stride & 15;

    int32_t* flag      = nan_flag + batch_id;
    bool     found_nan = false;

    // Use a single warp (blockDim.x == 32).
    const int vecs_per_thread = (full_vecs + blockDim.x - 1) / blockDim.x;
    const int start_vec       = threadIdx.x * vecs_per_thread;
    const int end_vec         = min(start_vec + vecs_per_thread, full_vecs);

    for (int i = start_vec; i < end_vec; i++) {
        uint4* vec_ptr = reinterpret_cast<uint4*>(token_data + (i << 4));

        // Check and reset only invalid elements, preserving valid values
        if (NanInfChecker<T>::check_and_reset(vec_ptr)) {
            found_nan = true;
        }
    }

    // Collect results within the warp (all threads participate).
    unsigned mask = __ballot_sync(0xFFFFFFFF, found_nan);

    // Thread 0 sets nan_flag and handles remaining bytes.
    if (threadIdx.x == 0) {
        if (mask != 0) {
            *flag = 1;
        }

        // Handle remaining bytes safely.
        if (remainder > 0) {
            char*     remainder_ptr = token_data + (full_vecs << 4);
            const int elem_size     = sizeof(T);
            const int num_elems     = remainder / elem_size;
            T*        typed_ptr     = reinterpret_cast<T*>(remainder_ptr);

            for (int i = 0; i < num_elems; i++) {
                if (NanInfChecker<T>::check_and_reset(&typed_ptr[i])) {
                    *flag = 1;
                }
            }
        }
    }
}

// Wrapper - Prefill
template<typename T>
void invokeCheckAndResetNANKvCachePrefill(const void* const* layer_base_addr,
                                          const int32_t*     kv_cache_block_id,
                                          const int32_t*     prefix_lengths,
                                          const int32_t*     seq_len_cu,
                                          size_t             batch_size,
                                          size_t             layer_num,
                                          size_t             max_blocks_per_batch,
                                          size_t             block_size_bytes,
                                          size_t             seq_size_per_block,
                                          int32_t*           nan_flag,
                                          cudaStream_t       stream) {

    dim3 grid(batch_size, layer_num);
    dim3 block(256);  // 256 threads for token-parallel scanning in prefill

    check_and_reset_kv_cache_prefill_kernel<T><<<grid, block, 0, stream>>>(layer_base_addr,
                                                                           kv_cache_block_id,
                                                                           prefix_lengths,
                                                                           seq_len_cu,
                                                                           batch_size,
                                                                           layer_num,
                                                                           max_blocks_per_batch,
                                                                           block_size_bytes,
                                                                           seq_size_per_block,
                                                                           nan_flag);
}

// Wrapper - Decode
template<typename T>
void invokeCheckAndResetNANKvCacheDecode(const void* const* layer_base_addr,
                                         const int32_t*     kv_cache_block_id,
                                         const int32_t*     sequence_lengths,
                                         size_t             batch_size,
                                         size_t             layer_num,
                                         size_t             max_blocks_per_batch,
                                         size_t             block_size_bytes,
                                         size_t             seq_size_per_block,
                                         int32_t*           nan_flag,
                                         cudaStream_t       stream) {

    dim3 grid(batch_size, layer_num);
    dim3 block(32);  // one warp per (batch, layer): decode checks only the last token

    check_and_reset_kv_cache_decode_kernel<T><<<grid, block, 0, stream>>>(layer_base_addr,
                                                                          kv_cache_block_id,
                                                                          sequence_lengths,
                                                                          batch_size,
                                                                          layer_num,
                                                                          max_blocks_per_batch,
                                                                          block_size_bytes,
                                                                          seq_size_per_block,
                                                                          nan_flag);
}

// Explicit template instantiations
template void invokeCheckAndResetNANKvCachePrefill<float>(const void* const*,
                                                          const int32_t*,
                                                          const int32_t*,
                                                          const int32_t*,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          int32_t*,
                                                          cudaStream_t);
template void invokeCheckAndResetNANKvCacheDecode<float>(
    const void* const*, const int32_t*, const int32_t*, size_t, size_t, size_t, size_t, size_t, int32_t*, cudaStream_t);
template void invokeCheckAndResetNANKvCachePrefill<half>(const void* const*,
                                                         const int32_t*,
                                                         const int32_t*,
                                                         const int32_t*,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         size_t,
                                                         int32_t*,
                                                         cudaStream_t);
template void invokeCheckAndResetNANKvCacheDecode<half>(
    const void* const*, const int32_t*, const int32_t*, size_t, size_t, size_t, size_t, size_t, int32_t*, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeCheckAndResetNANKvCachePrefill<nv_bfloat16>(const void* const*,
                                                                const int32_t*,
                                                                const int32_t*,
                                                                const int32_t*,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                int32_t*,
                                                                cudaStream_t);
template void invokeCheckAndResetNANKvCacheDecode<nv_bfloat16>(
    const void* const*, const int32_t*, const int32_t*, size_t, size_t, size_t, size_t, size_t, int32_t*, cudaStream_t);
#endif
#ifdef ENABLE_FP8
template void invokeCheckAndResetNANKvCachePrefill<__nv_fp8_e4m3>(const void* const*,
                                                                  const int32_t*,
                                                                  const int32_t*,
                                                                  const int32_t*,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  int32_t*,
                                                                  cudaStream_t);
template void invokeCheckAndResetNANKvCacheDecode<__nv_fp8_e4m3>(
    const void* const*, const int32_t*, const int32_t*, size_t, size_t, size_t, size_t, size_t, int32_t*, cudaStream_t);
#endif
}  // namespace rtp_llm