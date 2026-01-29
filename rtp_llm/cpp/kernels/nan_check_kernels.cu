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
#endif
#include "rtp_llm/cpp/kernels/nan_check_kernels.h"

#if USING_CUDA
#include "rtp_llm/cpp/kernels/vec_dtypes.cuh"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/kernels/rocm_utils/vec_dtypes_hip.h"
#endif

namespace rtp_llm {

// General NaN check kernel
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
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
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
// Layout: [layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size]
// Within a block: [2, local_head_num_kv, seq_size_per_block, k_token_size]
// Memory organization: All K first (organized by [head, token, head_dim]), then all V.
//
// For MLA (Multi-head Latent Attention) KV cache:
// - Layout: [layer_num, block_num, seq_size_per_block, k_token_size + v_token_size + layer_out_size]
// - Within a block: K part (kv_lora_rank), V part (rope_head_dim), then layer_out part.
// - Memory organization: All K first, then all V, then all layer_out.
// See invokeCheckAndResetNANKvCachePrefill() in nan_check_kernels.h for detailed documentation.
template<typename T>
__global__ void check_and_reset_kv_cache_prefill_kernel(const void* const* __restrict__ layer_base_addr,
                                                        const int32_t* __restrict__ kv_cache_block_id,
                                                        const int32_t* __restrict__ prefix_lengths,
                                                        const int32_t* __restrict__ input_lengths,
                                                        size_t batch_size,
                                                        size_t layer_num,
                                                        size_t max_blocks_per_batch,
                                                        size_t local_head_num_kv,
                                                        size_t k_token_size,
                                                        size_t v_token_size,
                                                        size_t k_block_size_bytes,
                                                        size_t v_block_size_bytes,
                                                        size_t k_token_bytes,
                                                        size_t v_token_bytes,
                                                        size_t block_size_bytes,
                                                        size_t seq_size_per_block,
                                                        int32_t* __restrict__ nan_flag) {

    const size_t batch_id = blockIdx.x;
    const size_t layer_id = blockIdx.y;

    if (batch_id >= batch_size || layer_id >= layer_num)
        return;

    const void* base_ptr_void = layer_base_addr[layer_id];
    if (!base_ptr_void)
        return;

    char*          base_ptr     = static_cast<char*>(const_cast<void*>(base_ptr_void));
    const int32_t* batch_blocks = kv_cache_block_id + batch_id * max_blocks_per_batch;

    const size_t prefix_len = prefix_lengths[batch_id];
    const size_t total_len  = input_lengths[batch_id];

    if (prefix_len >= total_len)
        return;

    const size_t warp_id   = threadIdx.x >> 5;
    const size_t lane_id   = threadIdx.x & 31;
    const size_t num_warps = blockDim.x >> 5;

    const size_t first_block = prefix_len / seq_size_per_block;
    const size_t last_block  = (total_len - 1) / seq_size_per_block;

    extern __shared__ char shared_mem[];
    int32_t*               shared_block_ids = reinterpret_cast<int32_t*>(shared_mem);
    int32_t*               shared_nan_flags = &shared_block_ids[num_warps];
    int32_t*               shared_nan_flag  = &shared_nan_flags[num_warps];

    if (lane_id == 0) {
        shared_nan_flags[warp_id] = 0;
    }
    if (threadIdx.x == 0) {
        *shared_nan_flag = 0;
    }
    __syncthreads();

    for (size_t block_offset = warp_id; block_offset <= (last_block - first_block); block_offset += num_warps) {
        const size_t logical_block_idx = first_block + block_offset;

        if (logical_block_idx > last_block || logical_block_idx >= max_blocks_per_batch)
            continue;

        int32_t physical_block_id;
        if (lane_id == 0) {
            physical_block_id         = batch_blocks[logical_block_idx];
            shared_block_ids[warp_id] = physical_block_id;
        }
        __syncwarp();
        physical_block_id = shared_block_ids[warp_id];

        if (physical_block_id < 0)
            continue;

        char* block_base = base_ptr + static_cast<size_t>(physical_block_id) * block_size_bytes;

        const size_t block_start_token = logical_block_idx * seq_size_per_block;
        const size_t block_end_token   = min(block_start_token + seq_size_per_block, total_len);
        const size_t start_in_block    = (prefix_len > block_start_token) ? (prefix_len - block_start_token) : 0;
        const size_t end_in_block      = block_end_token - block_start_token;

        // Use local flag per thread to avoid frequent atomic operations
        bool thread_has_nan = false;

        for (size_t token_offset = start_in_block + lane_id; token_offset < end_in_block; token_offset += 32) {
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                char* k_data =
                    block_base + head_id * (seq_size_per_block * k_token_bytes) + token_offset * k_token_bytes;

                const size_t k_vec_count = k_token_bytes >> 4;

#pragma unroll 4
                for (size_t i = 0; i < k_vec_count; i++) {
                    uint4* vec_ptr = reinterpret_cast<uint4*>(k_data + (i << 4));

                    if (NanInfChecker<T>::check_and_reset(vec_ptr)) {
                        thread_has_nan = true;
                    }
                }
            }

            char* v_base = block_base + k_block_size_bytes;
            for (size_t head_id = 0; head_id < local_head_num_kv; head_id++) {
                char* v_data = v_base + head_id * (seq_size_per_block * v_token_bytes) + token_offset * v_token_bytes;

                const size_t v_vec_count = v_token_bytes >> 4;

#pragma unroll 4
                for (size_t i = 0; i < v_vec_count; i++) {
                    uint4* vec_ptr = reinterpret_cast<uint4*>(v_data + (i << 4));

                    if (NanInfChecker<T>::check_and_reset(vec_ptr)) {
                        thread_has_nan = true;
                    }
                }
            }
        }

        // Use warp-level reduction instead of atomic operations
        bool warp_has_nan = __any_sync(0xFFFFFFFF, thread_has_nan);
        if (lane_id == 0 && warp_has_nan) {
            shared_nan_flags[warp_id] = 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        bool any_nan = false;
        for (size_t w = 0; w < num_warps; w++) {
            if (shared_nan_flags[w]) {
                any_nan = true;
                break;
            }
        }
        if (any_nan) {
            nan_flag[batch_id] = 1;
        }
    }
}

// KV cache NaN/Inf check/reset kernel for decode (last token only).
//
// This checks only the last token for each batch, based on sequence_lengths[batch].
//
// Layout: [layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size]
// Within a block: [2, local_head_num_kv, seq_size_per_block, k_token_size]
// Memory organization: All K first (organized by [head, token, head_dim]), then all V.
//
// For MLA (Multi-head Latent Attention) KV cache:
// - Layout: [layer_num, block_num, seq_size_per_block, k_token_size + v_token_size + layer_out_size]
// - Within a block: K part (kv_lora_rank), V part (rope_head_dim), then layer_out part.
// - Memory organization: All K first, then all V, then all layer_out.
// See invokeCheckAndResetNANKvCacheDecode() in nan_check_kernels.h for detailed documentation.
template<typename T>
__global__ void check_and_reset_kv_cache_decode_kernel(const void* const* __restrict__ layer_base_addr,
                                                       const int32_t* __restrict__ kv_cache_block_id,
                                                       const int32_t* __restrict__ sequence_lengths,
                                                       size_t batch_size,
                                                       size_t layer_num,
                                                       size_t max_blocks_per_batch,
                                                       size_t local_head_num_kv,
                                                       size_t k_token_size,
                                                       size_t v_token_size,
                                                       size_t k_block_size_bytes,
                                                       size_t v_block_size_bytes,
                                                       size_t k_token_bytes,
                                                       size_t v_token_bytes,
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

    if (physical_block_id < 0)
        return;

    char* block_base = base_ptr + physical_block_id * block_size_bytes;

    const int tid       = threadIdx.x;
    const int lane_id   = tid % 32;
    const int warp_id   = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    bool thread_has_nan = false;

    const int heads_per_warp = (local_head_num_kv + num_warps - 1) / num_warps;
    const int warp_start     = warp_id * heads_per_warp;
    const int warp_end       = min(warp_start + heads_per_warp, static_cast<int>(local_head_num_kv));

    for (int head_idx = warp_start + lane_id; head_idx < warp_end; head_idx += 32) {
        char* k_data = block_base + head_idx * (seq_size_per_block * k_token_bytes) + offset_in_block * k_token_bytes;

        const int vec4_count = k_token_bytes >> 4;
        uint4*    vec_ptr    = reinterpret_cast<uint4*>(k_data);

#pragma unroll 4
        for (int i = 0; i < vec4_count; i++) {
            if (NanInfChecker<T>::check_and_reset(&vec_ptr[i])) {
                thread_has_nan = true;
            }
        }

        const int remainder = k_token_bytes & 15;
        if (remainder) {
            T*        remainder_ptr   = reinterpret_cast<T*>(k_data + (vec4_count << 4));
            const int remainder_elems = remainder / sizeof(T);

            for (int i = 0; i < remainder_elems; i++) {
                if (NanInfChecker<T>::check_and_reset(&remainder_ptr[i])) {
                    thread_has_nan = true;
                }
            }
        }
    }

    char* v_base = block_base + k_block_size_bytes;

    for (int head_idx = warp_start + lane_id; head_idx < warp_end; head_idx += 32) {
        char* v_data = v_base + head_idx * (seq_size_per_block * v_token_bytes) + offset_in_block * v_token_bytes;

        const int vec4_count = v_token_bytes >> 4;
        uint4*    vec_ptr    = reinterpret_cast<uint4*>(v_data);

#pragma unroll 4
        for (int i = 0; i < vec4_count; i++) {
            if (NanInfChecker<T>::check_and_reset(&vec_ptr[i])) {
                thread_has_nan = true;
            }
        }

        const int remainder = v_token_bytes & 15;
        if (remainder) {
            T*        remainder_ptr   = reinterpret_cast<T*>(v_data + (vec4_count << 4));
            const int remainder_elems = remainder / sizeof(T);

            for (int i = 0; i < remainder_elems; i++) {
                if (NanInfChecker<T>::check_and_reset(&remainder_ptr[i])) {
                    thread_has_nan = true;
                }
            }
        }
    }

    bool warp_has_nan = __any_sync(0xFFFFFFFF, thread_has_nan);
    if (threadIdx.x == 0 && warp_has_nan) {
        nan_flag[batch_id] = 1;
    }
}

// Prefill: check and reset NaN/Inf in KV cache.
// See nan_check_kernels.h for detailed documentation.
template<typename T>
void invokeCheckAndResetNANKvCachePrefill(const void* const* layer_base_addr,
                                          const int32_t*     kv_cache_block_id,
                                          const int32_t*     prefix_lengths,
                                          const int32_t*     input_lengths,
                                          size_t             batch_size,
                                          size_t             layer_num,
                                          size_t             max_blocks_per_batch,
                                          size_t             local_head_num_kv,
                                          size_t             k_token_size,
                                          size_t             v_token_size,
                                          size_t             k_block_size_bytes,
                                          size_t             v_block_size_bytes,
                                          size_t             k_token_bytes,
                                          size_t             v_token_bytes,
                                          size_t             block_size_bytes,
                                          size_t             seq_size_per_block,
                                          int32_t*           nan_flag,
                                          cudaStream_t       stream) {

    const size_t threads_per_block = 256;
    const size_t warps_per_block   = threads_per_block / 32;
    const size_t shared_mem_size   = warps_per_block * sizeof(int32_t) * 2 + sizeof(int32_t);

    dim3 grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(layer_num));
    dim3 block(static_cast<unsigned int>(threads_per_block));

    check_and_reset_kv_cache_prefill_kernel<T><<<grid, block, shared_mem_size, stream>>>(layer_base_addr,
                                                                                         kv_cache_block_id,
                                                                                         prefix_lengths,
                                                                                         input_lengths,
                                                                                         batch_size,
                                                                                         layer_num,
                                                                                         max_blocks_per_batch,
                                                                                         local_head_num_kv,
                                                                                         k_token_size,
                                                                                         v_token_size,
                                                                                         k_block_size_bytes,
                                                                                         v_block_size_bytes,
                                                                                         k_token_bytes,
                                                                                         v_token_bytes,
                                                                                         block_size_bytes,
                                                                                         seq_size_per_block,
                                                                                         nan_flag);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

// Decode: check and reset NaN/Inf in KV cache.
// This checks only the last token for each batch, based on sequence_lengths[batch].
// See nan_check_kernels.h for detailed documentation.
template<typename T>
void invokeCheckAndResetNANKvCacheDecode(const void* const* layer_base_addr,
                                         const int32_t*     kv_cache_block_id,
                                         const int32_t*     sequence_lengths,
                                         size_t             batch_size,
                                         size_t             layer_num,
                                         size_t             max_blocks_per_batch,
                                         size_t             local_head_num_kv,
                                         size_t             k_token_size,
                                         size_t             v_token_size,
                                         size_t             k_block_size_bytes,
                                         size_t             v_block_size_bytes,
                                         size_t             k_token_bytes,
                                         size_t             v_token_bytes,
                                         size_t             block_size_bytes,
                                         size_t             seq_size_per_block,
                                         int32_t*           nan_flag,
                                         cudaStream_t       stream) {

    size_t threads_per_block = 32;

    if (local_head_num_kv > 32) {
        threads_per_block = min(size_t(256), ((local_head_num_kv + 31) / 32) * 32);
        threads_per_block = max(size_t(64), threads_per_block);
    }

    dim3 grid(batch_size, layer_num);
    dim3 block(threads_per_block);

    check_and_reset_kv_cache_decode_kernel<T><<<grid, block, 0, stream>>>(layer_base_addr,
                                                                          kv_cache_block_id,
                                                                          sequence_lengths,
                                                                          batch_size,
                                                                          layer_num,
                                                                          max_blocks_per_batch,
                                                                          local_head_num_kv,
                                                                          k_token_size,
                                                                          v_token_size,
                                                                          k_block_size_bytes,
                                                                          v_block_size_bytes,
                                                                          k_token_bytes,
                                                                          v_token_bytes,
                                                                          block_size_bytes,
                                                                          seq_size_per_block,
                                                                          nan_flag);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
#endif
}

// Explicit template instantiations
#define INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL(T)                                                                 \
    template void invokeCheckAndResetNANKvCachePrefill<T>(const void* const*,                                          \
                                                          const int32_t*,                                              \
                                                          const int32_t*,                                              \
                                                          const int32_t*,                                              \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          size_t,                                                      \
                                                          int32_t*,                                                    \
                                                          cudaStream_t);

#define INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE(T)                                                                  \
    template void invokeCheckAndResetNANKvCacheDecode<T>(const void* const*,                                           \
                                                         const int32_t*,                                               \
                                                         const int32_t*,                                               \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         size_t,                                                       \
                                                         int32_t*,                                                     \
                                                         cudaStream_t);

INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL(float)
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE(float)
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL(half)
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE(half)
#ifdef ENABLE_BF16
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL(nv_bfloat16)
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE(nv_bfloat16)
#endif
#ifdef ENABLE_FP8
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL(__nv_fp8_e4m3)
INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE(__nv_fp8_e4m3)
#endif
#undef INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_PREFILL
#undef INVOKE_CHECK_AND_RESET_NAN_KV_CACHE_DECODE

}  // namespace rtp_llm
