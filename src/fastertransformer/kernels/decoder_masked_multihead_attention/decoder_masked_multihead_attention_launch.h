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

#include "decoder_masked_multihead_attention_template.h"
// #include "driver_types.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include <assert.h>

#if USING_CUDA
#include <cuda_runtime_api.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#endif
#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif

#include <type_traits>

namespace fastertransformer {

template<typename T, int Dh, bool DO_MULTI_BLOCK, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(Multihead_attention_params<T, DO_CROSS_ATTENTION>& params, int threads_per_block)
{
    using Tk = typename kernel_type_t<T>::Type;
    // The amount of shared memory needed to store the Q*K^T values in float.
    const int max_timesteps = DO_CROSS_ATTENTION
        ? params.cyclic_kv_cache_length
        : min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep), params.cyclic_kv_cache_length);
    const auto qk_elts = static_cast<std::size_t>(divUp(max_timesteps + 1, 4));  // explicit cast because of the sign
    const auto qk_sz   = qk_elts * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(Tk) != 4) {
        // TDOD
        logits_sz = qk_elts * 4 * sizeof(Tk);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    auto constexpr threads_per_value_ = threads_per_value<T>(dh_max(Dh));

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value_;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;

    size_t transpose_rotary_size = 0;
    if (params.rotary_embedding_dim > 0) {
        assert(params.rotary_embedding_dim > 0);
        transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(Tk);
    }

    size_t out_oi_sz = 0;
    if (params.multi_block_mode) {
        // The size for partial output reduction computation.
        out_oi_sz = params.max_seq_len_tile * params.hidden_size_per_head * sizeof(T);
    }

    // The max.
    return max(max(max(softmax_sz, red_sz), transpose_rotary_size), out_oi_sz);
}

template<typename T, int Dh, bool DO_MULTI_BLOCK, bool DO_CROSS_ATTENTION>
inline size_t smem_size_for_threads(Multihead_attention_params<T, DO_CROSS_ATTENTION>& params, int threads_per_block)
{
    using Tk = typename kernel_type_t<T>::Type;
    auto constexpr threads_per_value_ = threads_per_value<T>(dh_max(Dh));
    
    size_t red_sz = 0;

    if (DO_MULTI_BLOCK) {
        // The number of partial rows to reduce in the final reduction.
        int rows_per_red = threads_per_block / threads_per_value_;
        // The amount of storage needed to finalize the outputs.
        red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;
    }

    size_t transpose_rotary_size = 0;
    if (params.rotary_embedding_dim > 0) {
        assert(params.rotary_embedding_dim > 0);
        transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(Tk);
    }

    return max(red_sz, transpose_rotary_size);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh, bool DO_CROSS_ATTENTION>
inline void multi_block_grid_setup(dim3&                                              grid,
                                   Multihead_attention_params<T, DO_CROSS_ATTENTION>& params,
                                   int                                                blocks_per_sm,
                                   int                                                block_size,
                                   int                                                tlength,
                                   bool                                               do_multi_block,
                                   int                                                min_seq_len_tile)
{
    if (!do_multi_block) {
        return;
    }

    const int threads_per_value_ = threads_per_value<T>(dh_max(Dh));

    // Make sure high occupancy for device
    const int seq_len_tile_from_occupancy = divUp(params.multi_processor_count * blocks_per_sm, params.batch_size * params.num_heads);

    // Make sure that each block at least processes one loop of kv (unroll size is default at 8).
    const int seq_len_per_kv_loop = divUp(block_size, threads_per_value_) * 8;
    const int max_seq_len_tile    = std::min(divUp(tlength + 1, seq_len_per_kv_loop), params.max_seq_len_tile);

    // force set multi block mode for tlength too large to cause insufficient smem
    params.seq_len_tile = std::max(std::min(seq_len_tile_from_occupancy, max_seq_len_tile), min_seq_len_tile);

    params.timesteps_per_block = divUp(tlength + 1, params.seq_len_tile);

    grid.z = params.seq_len_tile;
}

#define MMHA_LAUNCH_CHECK(DYNAMIC_THDS_PER_BLOCK)                                                                      \
    std::size_t const dynamic_smem_sz{smem_size_for_threads<T, Dh, DO_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};   \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */              \
    if (dynamic_smem_sz >= 46 * 1024) {                                                                                \
        cudaError_t res =                                                                                              \
            cudaFuncSetAttribute((const void*)&masked_multihead_attention_kernel<T,                                    \
                                                                                 T_cache,                              \
                                                                                 KVCacheBuffer,                        \
                                                                                 Dh,                                   \
                                                                                 DYNAMIC_THDS_PER_BLOCK,               \
                                                                                 KernelParamsType::DO_CROSS_ATTENTION, \
                                                                                 HAS_BEAMS,                            \
                                                                                 DO_MULTI_BLOCK>,                      \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,                                          \
                                 dynamic_smem_sz);                                                                     \
        FT_CHECK_WITH_INFO(res == cudaSuccess,                                                                         \
                           "Sequence Length is too long for the MMHA kernel (not enough shared memory).");             \
    }                                                                                                                  \
    check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                    \
        &available_blocks,                                                                                             \
        masked_multihead_attention_kernel<T,                                                                           \
                                          T_cache,                                                                     \
                                          KVCacheBuffer,                                                               \
                                          Dh,                                                                          \
                                          DYNAMIC_THDS_PER_BLOCK,                                                      \
                                          KernelParamsType::DO_CROSS_ATTENTION,                                        \
                                          HAS_BEAMS,                                                                   \
                                          DO_MULTI_BLOCK>,                                                             \
        DYNAMIC_THDS_PER_BLOCK,                                                                                        \
        dynamic_smem_sz));

#define MMHA_KERNEL(DYNAMIC_THDS_PER_BLOCK, ENABLE_MULTI_BLOCK)                                                                                                                                      \
    std::size_t const dynamic_smem_sz{smem_size_in_bytes<T, Dh, ENABLE_MULTI_BLOCK>(params, DYNAMIC_THDS_PER_BLOCK)};                                                                                \
    /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */                                                                                            \
    if (dynamic_smem_sz >= 46 * 1024) {                                                                                                                                                              \
        cudaError_t res =                                                                                                                                                                            \
            cudaFuncSetAttribute((const void*)&masked_multihead_attention_kernel<T,                                                                                                                  \
                                                                                 T_cache,                                                                                                            \
                                                                                 KVCacheBuffer,                                                                                                      \
                                                                                 Dh,                                                                                                                 \
                                                                                 DYNAMIC_THDS_PER_BLOCK,                                                                                             \
                                                                                 KernelParamsType::DO_CROSS_ATTENTION,                                                                               \
                                                                                 HAS_BEAMS,                                                                                                          \
                                                                                 ENABLE_MULTI_BLOCK>,                                                                                                \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,                                                                                                                        \
                                 dynamic_smem_sz);                                                                                                                                                   \
        FT_CHECK_WITH_INFO(                                                                                                                                                                          \
            res == cudaSuccess,                                                                                                                                                                      \
            "Sequence Length is too long for the MMHA kernel (not enough shared memory). batch: %d head: %d seq: %d processor_count: %d smem_required: %d max_device_smem: %d multi_block_mode: %d", \
            params.batch_size,                                                                                                                                                                       \
            params.num_heads,                                                                                                                                                                        \
            tlength,                                                                                                                                                                                 \
            params.multi_processor_count,                                                                                                                                                            \
            dynamic_smem_sz,                                                                                                                                                                         \
            max_dsmem_sz_on_device,                                                                                                                                                                  \
            ENABLE_MULTI_BLOCK);                                                                                                                                                                     \
    }                                                                                                                                                                                                \
    masked_multihead_attention_kernel<T,                                                                                                                                                             \
                                      T_cache,                                                                                                                                                       \
                                      KVCacheBuffer,                                                                                                                                                 \
                                      Dh,                                                                                                                                                            \
                                      DYNAMIC_THDS_PER_BLOCK,                                                                                                                                        \
                                      KernelParamsType::DO_CROSS_ATTENTION,                                                                                                                          \
                                      HAS_BEAMS,                                                                                                                                                     \
                                      ENABLE_MULTI_BLOCK>                                                                                                                                            \
        <<<grid, DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz, stream>>>(params, kv_cache_buffer);

// if resources are not enough to launch 512 threads per block, we will fallback to 256.
#define MMHA_512_BLOCKSIZE_CHECK()                                                                                     \
    MMHA_LAUNCH_CHECK(512);                                                                                            \
    if (available_blocks <= 0) {                                                                                       \
        MMHA_LAUNCH_CHECK(256);                                                                                        \
        dynamic_block_size = 256;                                                                                      \
    }                                                                                                                  \
    else {                                                                                                             \
        dynamic_block_size = 512;                                                                                      \
    }

// if resources are not enough to launch 1024 threads per block, we will fallback to 512.
#define MMHA_1024_BLOCKSIZE_CHECK()                                                                                    \
    MMHA_LAUNCH_CHECK(1024);                                                                                           \
    if (available_blocks > 0) {                                                                                        \
        dynamic_block_size = 1024;                                                                                     \
    }                                                                                                                  \
    else {                                                                                                             \
        MMHA_512_BLOCKSIZE_CHECK();                                                                                    \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T,
         typename T_cache,
         typename KVCacheBuffer,
         typename KernelParamsType,
         int  Dh,
         int  THDS_PER_BLOCK,
         bool HAS_BEAMS,
         bool DO_MULTI_BLOCK>
void mmha_launch_kernel_ex(KernelParamsType&    params,
                           const KVCacheBuffer& kv_cache_buffer,
                           const cudaStream_t&  stream,
                           int                  tlength)
{
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size), 1};

    int min_seq_len_tile = 0;

    // Tune block size based on batchxhead to increase occupancy.
    int num_blocks_per_sm = -1;
    // Set 0 dynamic shared memory size as we need the number of available blocks limited by registers.
    // Dynamic shared memory is fixed for different block size.
    check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        masked_multihead_attention_kernel<T,
                                          T_cache,
                                          KVCacheBuffer,
                                          Dh,
                                          THDS_PER_BLOCK,
                                          KernelParamsType::DO_CROSS_ATTENTION,
                                          HAS_BEAMS,
                                          DO_MULTI_BLOCK>,
        THDS_PER_BLOCK,
        0));

    const int kernel_total_blocks = params.batch_size * params.num_heads;

    int block_size_factor =
        min(divUp(params.multi_processor_count * num_blocks_per_sm, kernel_total_blocks), num_blocks_per_sm);
    // Max block size is 1024.
    int dynamic_block_size = min(THDS_PER_BLOCK * block_size_factor, 1024);

    // MMHA_LAUNCH_CHECK to get max dynamic_block_size. 
    int available_blocks = -1;
    if (dynamic_block_size < 512) {
        MMHA_LAUNCH_CHECK(256);
        dynamic_block_size = 256;
    }
    else if (dynamic_block_size < 1024) {
        MMHA_512_BLOCKSIZE_CHECK();
    }
    else if (dynamic_block_size == 1024) {
        MMHA_1024_BLOCKSIZE_CHECK();
    }

    // check smem to decide if force enable multi block mode
    int device;
    check_cuda_error(cudaGetDevice(&device));
    
    // max smem on device
    int max_dsmem_sz_on_device = -1;
    check_cuda_error(cudaDeviceGetAttribute(
        &max_dsmem_sz_on_device,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device));
    
    // reserve 1KB for static smem.
    // CUDA reserves 1 KB of shared memory per thread block. 
    // Details in https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#unified-shared-memory-l1-texture-cache
    max_dsmem_sz_on_device = max_dsmem_sz_on_device - 1024 * available_blocks - 1024;

    // required smem for current tlength when not using multi block mode
    size_t dsmem_smem_sz_for_kernel{smem_size_in_bytes<T, Dh, false>(params, dynamic_block_size)};
    
    if(dsmem_smem_sz_for_kernel > max_dsmem_sz_on_device){
        if (DO_MULTI_BLOCK){
            min_seq_len_tile = dsmem_smem_sz_for_kernel / max_dsmem_sz_on_device + 1;
        }
        else {
            FT_CHECK_WITH_INFO(false, "Sequence Length is too long for the MMHA kernel (not enough shared memory): %d", tlength);
        }
    }

    // If blocks with larger block size already fill all SMs, then disable the multi blocks mode.
    multi_block_grid_setup<T, Dh>(grid, params, available_blocks, dynamic_block_size, tlength, DO_MULTI_BLOCK, min_seq_len_tile);

    // Launch kernels based on the valid block size.
    switch (dynamic_block_size) {
        case 256:
            if (params.enable_multi_block_mode()) {
                MMHA_KERNEL(256, true);
            }
            else {
                MMHA_KERNEL(256, false);
            }
            break;
        case 512:
            if (params.enable_multi_block_mode()) {
                MMHA_KERNEL(512, true);
            }
            else {
                MMHA_KERNEL(512, false);
            }
            break;
        case 1024:
            if (params.enable_multi_block_mode()) {
                MMHA_KERNEL(1024, true);
            }
            else {
                MMHA_KERNEL(1024, false);
            }
            break;
        default:
            MMHA_KERNEL(1024, false);
    }
}

template<typename T,
         typename KVCacheBuffer,
         typename KernelParamsType,
         int  Dh,
         int  THDS_PER_BLOCK,
         bool HAS_BEAMS,
         bool DO_MULTI_BLOCK>
void mmha_launch_kernel_dispatch_8bits_kv_cache(KernelParamsType&    params,
                                                const KVCacheBuffer& kv_cache_buffer,
                                                const cudaStream_t&  stream,
                                                int                  tlength)
{
    if (params.int8_kv_cache) {
        mmha_launch_kernel_ex<T,
                              int8_t,
                              KVCacheBuffer,
                              KernelParamsType,
                              Dh,
                              THDS_PER_BLOCK,
                              HAS_BEAMS,
                              DO_MULTI_BLOCK>(params, kv_cache_buffer, stream, tlength);
    }
#ifdef ENABLE_FP8
    else if (params.fp8_kv_cache) {
        mmha_launch_kernel_ex<T,
                              __nv_fp8_e4m3,
                              KVCacheBuffer,
                              KernelParamsType,
                              Dh,
                              THDS_PER_BLOCK,
                              HAS_BEAMS,
                              DO_MULTI_BLOCK>(params, kv_cache_buffer, stream, tlength);
    }
#endif  // ENABLE_FP8
    else {
        mmha_launch_kernel_ex<T, T, KVCacheBuffer, KernelParamsType, Dh, THDS_PER_BLOCK, HAS_BEAMS, DO_MULTI_BLOCK>(
            params, kv_cache_buffer, stream, tlength);
    }
}

template<typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh, bool HAS_BEAMS>
void mmha_launch_kernel_dispatch(KernelParamsType&    params,
                                 const KVCacheBuffer& kv_cache_buffer,
                                 const cudaStream_t&  stream)
{
    int const tlength = params.timestep;
    if (params.multi_block_mode) {
        mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, true>(
            params, kv_cache_buffer, stream, tlength);
    }
    else {
        mmha_launch_kernel_dispatch_8bits_kv_cache<T, KVCacheBuffer, KernelParamsType, Dh, 256, HAS_BEAMS, false>(
            params, kv_cache_buffer, stream, tlength);
    }
}

template<typename T, typename KVCacheBuffer, typename KernelParamsType, int Dh>
void mmha_launch_kernel(KernelParamsType& params, const KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    assert(params.rotary_embedding_dim >= 0);
    mmha_launch_kernel_dispatch<T, KVCacheBuffer, KernelParamsType, Dh, false>(params, kv_cache_buffer, stream);
}

#define INSTANTIATE_MMHA_LAUNCHERS(T, Dh)                                                                              \
    template void mmha_launch_kernel<T, KVLinearBuffer, Masked_multihead_attention_params<T>, Dh>(                     \
        Masked_multihead_attention_params<T> & params,                                                                 \
        const KVLinearBuffer& kv_cache_buffer,                                                                         \
        const cudaStream_t&   stream);                                                                                   \
    template void mmha_launch_kernel<T, KVBlockArray, Masked_multihead_attention_params<T>, Dh>(                       \
        Masked_multihead_attention_params<T> & params,                                                                 \
        const KVBlockArray& kv_cache_buffer,                                                                           \
        const cudaStream_t& stream);                                                                                   \
    template void mmha_launch_kernel<T, KVLinearBuffer, Cross_multihead_attention_params<T>, Dh>(                      \
        Cross_multihead_attention_params<T> & params,                                                                  \
        const KVLinearBuffer& kv_cache_buffer,                                                                         \
        const cudaStream_t&   stream);                                                                                   \
    template void mmha_launch_kernel<T, KVBlockArray, Cross_multihead_attention_params<T>, Dh>(                        \
        Cross_multihead_attention_params<T> & params,                                                                  \
        const KVBlockArray& kv_cache_buffer,                                                                           \
        const cudaStream_t& stream);

}  // namespace fastertransformer