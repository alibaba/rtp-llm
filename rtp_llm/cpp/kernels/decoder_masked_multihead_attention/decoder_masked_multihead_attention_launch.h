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
#include "decoder_masked_multihead_attention.h"
#include "rtp_llm/cpp/utils/utils.h"

#if USING_CUDA
#include "driver_types.h"
#include <cuda_runtime_api.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

#include <type_traits>

namespace rtp_llm {

#if USING_ROCM
using namespace rocm;
#endif

template<typename T, int Dh, bool DO_MULTI_BLOCK, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(Multihead_attention_params<T, DO_CROSS_ATTENTION>& params, int threads_per_block) {
    using Tk = typename kernel_type_t<T>::Type;
    // The amount of shared memory needed to store the Q*K^T values in float.
    const int  max_timesteps = DO_CROSS_ATTENTION ? params.cyclic_kv_cache_length :
                                                    min((DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep),
                                                       params.cyclic_kv_cache_length);
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
    if (params.rope_config.dim > 0) {
        transpose_rotary_size = 2 * params.rope_config.dim * sizeof(Tk);
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
inline size_t smem_size_for_threads(Multihead_attention_params<T, DO_CROSS_ATTENTION>& params, int threads_per_block) {
    using Tk                          = typename kernel_type_t<T>::Type;
    auto constexpr threads_per_value_ = threads_per_value<T>(dh_max(Dh));

    size_t red_sz = 0;

    if (DO_MULTI_BLOCK) {
        // The number of partial rows to reduce in the final reduction.
        int rows_per_red = threads_per_block / threads_per_value_;
        // The amount of storage needed to finalize the outputs.
        red_sz = rows_per_red * params.hidden_size_per_head * sizeof(Tk) / 2;
    }

    size_t transpose_rotary_size = 0;
    if (params.rope_config.dim > 0) {
        transpose_rotary_size = 2 * params.rope_config.dim * sizeof(Tk);
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
                                   int                                                min_seq_len_tile) {
    if (!do_multi_block) {
        return;
    }

    const int threads_per_value_ = threads_per_value<T>(dh_max(Dh));

    // Make sure high occupancy for device
    const int seq_len_tile_from_occupancy =
        divUp(params.multi_processor_count * blocks_per_sm, params.batch_size * params.num_heads);

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
                                                                                 DO_MULTI_BLOCK,                       \
                                                                                 ROPE_STYLE>,                          \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,                                          \
                                 dynamic_smem_sz);                                                                     \
        RTP_LLM_CHECK_WITH_INFO(res == cudaSuccess,                                                                    \
                                "Sequence Length is too long for the MMHA kernel (not enough shared memory).");        \
    }                                                                                                                  \
    check_cuda_value(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                    \
        &available_blocks,                                                                                             \
        masked_multihead_attention_kernel<T,                                                                           \
                                          T_cache,                                                                     \
                                          KVCacheBuffer,                                                               \
                                          Dh,                                                                          \
                                          DYNAMIC_THDS_PER_BLOCK,                                                      \
                                          KernelParamsType::DO_CROSS_ATTENTION,                                        \
                                          HAS_BEAMS,                                                                   \
                                          DO_MULTI_BLOCK,                                                              \
                                          ROPE_STYLE>,                                                                 \
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
                                                                                 ENABLE_MULTI_BLOCK,                                                                                                 \
                                                                                 ROPE_STYLE>,                                                                                                        \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,                                                                                                                        \
                                 dynamic_smem_sz);                                                                                                                                                   \
        RTP_LLM_CHECK_WITH_INFO(                                                                                                                                                                     \
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
                                      ENABLE_MULTI_BLOCK,                                                                                                                                            \
                                      ROPE_STYLE>                                                                                                                                                    \
        <<<grid, DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz, stream>>>(params, kv_cache_buffer);

// if resources are not enough to launch 512 threads per block, we will fallback to 256.
#define MMHA_512_BLOCKSIZE_CHECK()                                                                                     \
    MMHA_LAUNCH_CHECK(512);                                                                                            \
    if (available_blocks <= 0) {                                                                                       \
        MMHA_LAUNCH_CHECK(256);                                                                                        \
        dynamic_block_size = 256;                                                                                      \
    } else {                                                                                                           \
        dynamic_block_size = 512;                                                                                      \
    }

// if resources are not enough to launch 1024 threads per block, we will fallback to 512.
#define MMHA_1024_BLOCKSIZE_CHECK()                                                                                    \
    MMHA_LAUNCH_CHECK(1024);                                                                                           \
    if (available_blocks > 0) {                                                                                        \
        dynamic_block_size = 1024;                                                                                     \
    } else {                                                                                                           \
        MMHA_512_BLOCKSIZE_CHECK();                                                                                    \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T,
         typename T_cache,
         typename KVCacheBuffer,
         typename KernelParamsType,
         int       Dh,
         bool      HAS_BEAMS,
         bool      DO_MULTI_BLOCK,
         RopeStyle ROPE_STYLE>
void mmha_launch_kernel_ex(KernelParamsType&    params,
                           const KVCacheBuffer& kv_cache_buffer,
                           const cudaStream_t&  stream,
                           int                  tlength) {
    dim3 grid{static_cast<unsigned>(params.num_heads), static_cast<unsigned>(params.batch_size), 1};

    int min_seq_len_tile   = 0;
    int dynamic_block_size = 1024;

    // MMHA_LAUNCH_CHECK to get max dynamic_block_size.
    int available_blocks = -1;
    if (dynamic_block_size < 512) {
        MMHA_LAUNCH_CHECK(256);
        dynamic_block_size = 256;
    } else if (dynamic_block_size < 1024) {
        MMHA_512_BLOCKSIZE_CHECK();
    } else if (dynamic_block_size == 1024) {
        MMHA_1024_BLOCKSIZE_CHECK();
    }

    int max_dsmem_sz_on_device = getMaxSharedMemoryPerMultiprocessor();

    // reserve 1KB for static smem.
    // CUDA reserves 1 KB of shared memory per thread block.
    // Details in https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#unified-shared-memory-l1-texture-cache
    max_dsmem_sz_on_device = max_dsmem_sz_on_device - 1024 * available_blocks - 1024;

    // required smem for current tlength when not using multi block mode
    size_t dsmem_smem_sz_for_kernel{smem_size_in_bytes<T, Dh, false>(params, dynamic_block_size)};

    if (dsmem_smem_sz_for_kernel > max_dsmem_sz_on_device) {
        if (DO_MULTI_BLOCK) {
            min_seq_len_tile = dsmem_smem_sz_for_kernel / max_dsmem_sz_on_device + 1;
        } else {
            RTP_LLM_FAIL("Sequence Length is too long for the MMHA kernel (not enough shared memory): %d", tlength);
        }
    }

    // If blocks with larger block size already fill all SMs, then disable the multi blocks mode.
    multi_block_grid_setup<T, Dh>(
        grid, params, available_blocks, dynamic_block_size, tlength, DO_MULTI_BLOCK, min_seq_len_tile);

#define FT_BLOCK_SWITCH(COND, ...)                                                                                     \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE(DYNAMIC_BLOCK_SIZE, 256, __VA_ARGS__)                                                   \
            FT_SWITCH_ONE_CASE(DYNAMIC_BLOCK_SIZE, 512, __VA_ARGS__)                                                   \
            FT_SWITCH_ONE_CASE(DYNAMIC_BLOCK_SIZE, 1024, __VA_ARGS__)                                                  \
            FT_SWITCH_DEFAULT_CASE(DYNAMIC_BLOCK_SIZE, 1024, __VA_ARGS__)                                              \
        }                                                                                                              \
    }()

    // Launch kernels based on the valid block size.
    FT_BLOCK_SWITCH(dynamic_block_size, [&] {
        FT_SWITCH(params.enable_multi_block_mode(), MULTI_BLOCK_MODE, [&] {
            MMHA_KERNEL(DYNAMIC_BLOCK_SIZE, MULTI_BLOCK_MODE);
        });
    });
}

}  // namespace rtp_llm
