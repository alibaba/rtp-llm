/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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


#include <fused_multihead_attention_utils.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/hopper/smem_tile.h>
#include <fmha/utils.h>
#include <fmha/hopper/compute_tile.h>

#include <fmha/warpspec/kernel_traits.h>
#include <fmha/warpspec/dma.h>
#include <fmha/warpspec/compute.h>

#include "../fused_multihead_attention_common.h"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off

#if CUDA_VERSION >= 11000

static constexpr int DMA2COMPUTE_DEPTH = 1;

static constexpr bool USE_TMA_STORE = false;


using Attention_mask_type = ContextAttentionMaskType;

using Ktraits = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_bf16_traits,
                64,
                128,
                104,
                0,
                1,
                2,
                NUM_COMPUTE_GROUPS,
                DMA2COMPUTE_DEPTH,
                0,
                false,
                false,
                false,
                1,
                0,
                USE_TMA_STORE,
                false,
                false,
                fmha::bf16_t,
                0,
                0,
                0>;

using Ktraits_causal = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_bf16_traits,
                       64,
                       128,
                       104,
                       0,
                       1,
                       2,
                       NUM_COMPUTE_GROUPS,
                       DMA2COMPUTE_DEPTH,
                       1,
                       false,
                       true,
                       false,
                       1,
                       0,
                       USE_TMA_STORE,
                       false,
                       false,
                       fmha::bf16_t>;

using Ktraits_sliding_or_chunked_causal = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_bf16_traits,
                                      64,
                                      128,
                                      104,
                                      0,
                                      1,
                                      2,
                                      NUM_COMPUTE_GROUPS,
                                      DMA2COMPUTE_DEPTH,
                                      2,
                                      false,
                                      true,
                                      false,
                                      1,
                                      0,
                                      USE_TMA_STORE && false,
                                      false,
                                      false,
                                      fmha::bf16_t>;

using Ktraits_custom_mask = fmha::ws::Kernel_traits<fmha::Hopper_hgmma_bf16_traits,
                            64,
                            128,
                            104,
                            0,
                            1,
                            2,
                            NUM_COMPUTE_GROUPS,
                            DMA2COMPUTE_DEPTH,
                            3,
                            false,
                            true,
                            false,
                            1,
                            0,
                            USE_TMA_STORE && false,
                            false,
                            false,
                            fmha::bf16_t>;

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0 // padding_mask

using Shared = typename Ktraits::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits::THREADS, 1)
void fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared *shared = reinterpret_cast<Shared *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif

        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits::DMA_GROUP_TRANSPOSE_V ) {
            fmha::ws::DMA<Ktraits>::Device dma_device(elect_one);
            dma_device.run_packed_qkv(params, shared);
        } else {
            fmha::ws::DMA<Ktraits>::Device dma_device(elect_one);
            if( tidx < 32 ) {
                dma_device.run_packed_qkv(params, shared);
            }
        }

    } else {  // math


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif


        fmha::ws::Compute<fmha::Hopper_hgmma_bf16_traits, Ktraits> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

#endif // padding mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 1 // causal_mask

using Shared_causal = typename Ktraits_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_causal::THREADS, 1)
void fmha_v2_flash_attention_bf16_64_128_S_qkv_104_causal_alibi_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_causal *shared = reinterpret_cast<Shared_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif

        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_causal::DMA_GROUP_TRANSPOSE_V ) {
            fmha::ws::DMA<Ktraits_causal>::Device dma_device(elect_one);
            dma_device.run_packed_qkv(params, shared);
        } else {
            fmha::ws::DMA<Ktraits_causal>::Device dma_device(elect_one);
            if( tidx < 32 ) {
                dma_device.run_packed_qkv(params, shared);
            }
        }

    } else {  // math


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif


        fmha::ws::Compute<fmha::Hopper_hgmma_bf16_traits, Ktraits_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

#endif // causal mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0 // sliding_or_chunked_causal_mask

using Shared_sliding_or_chunked_causal = typename Ktraits_sliding_or_chunked_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_sliding_or_chunked_causal::THREADS, 1)
void fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sliding_or_chunked_causal_alibi_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_sliding_or_chunked_causal *shared =
        reinterpret_cast<Shared_sliding_or_chunked_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif

        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_sliding_or_chunked_causal::DMA_GROUP_TRANSPOSE_V ) {
            fmha::ws::DMA<Ktraits_sliding_or_chunked_causal>::Device dma_device(elect_one);
            dma_device.run_packed_qkv(params, shared);
        } else {
            fmha::ws::DMA<Ktraits_sliding_or_chunked_causal>::Device dma_device(elect_one);
            if( tidx < 32 ) {
                dma_device.run_packed_qkv(params, shared);
            }
        }

    } else {  // math


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif


        fmha::ws::Compute<fmha::Hopper_hgmma_bf16_traits, Ktraits_sliding_or_chunked_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

#endif // sliding_or_chunked_causal_mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0 // custom_mask

using Shared_custom_mask = typename Ktraits_custom_mask::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_custom_mask::THREADS, 1)
void fmha_v2_flash_attention_bf16_64_128_S_qkv_104_custom_mask_alibi_tma_ws_sm90_kernel(
    const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_custom_mask *shared =
        reinterpret_cast<Shared_custom_mask *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {  // dma + sched


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int DMA_REG_COUNT = 40;
        asm volatile("{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}" ::"n"(DMA_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif

        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_custom_mask::DMA_GROUP_TRANSPOSE_V ) {
            fmha::ws::DMA<Ktraits_custom_mask>::Device dma_device(elect_one);
            dma_device.run_packed_qkv(params, shared);
        } else {
            fmha::ws::DMA<Ktraits_custom_mask>::Device dma_device(elect_one);
            if( tidx < 32 ) {
                dma_device.run_packed_qkv(params, shared);
            }
        }

    } else {  // math


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int COMPUTE_REG_COUNT = 232;
        asm volatile("{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}" ::"n"(COMPUTE_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif


        fmha::ws::Compute<fmha::Hopper_hgmma_bf16_traits, Ktraits_custom_mask> compute;
        compute.run(warp_group, tidx, shared, params);
    }
}

#endif // custom_mask

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90(
    Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params, cudaStream_t stream){


    if( Ktraits::SCHEDULING_MODE > 0 ) {
        FMHA_CHECK_CUDA(cudaMemsetAsync(params.tile_id_counter_ptr, 0, sizeof(uint32_t), stream));
    }

    dim3 block_size;

    if( Ktraits::SCHEDULING_MODE == 0 ) {
        block_size.y = std::min(params.b * params.h, launch_params.multi_processor_count);
        // distribute m steps to multiple blocks (fully utilize SMs)
        // block.x = blocks that handle single head, block.y = blocks that handle different heads
        size_t sms_per_head = (launch_params.multi_processor_count) / block_size.y;
        // Take multiple compute groups into consideration.
        size_t m_steps = size_t((params.s + 64 * NUM_COMPUTE_GROUPS - 1) / (64 * NUM_COMPUTE_GROUPS));

        // 2 * 2 stands for kv cache and 2 bytes per element.
        size_t size_in_bytes = block_size.y * params.s * params.d * 2 * 2;
        if( size_in_bytes <= launch_params.device_l2_cache_size ) {
            // strategy 1: limit to only 1 wave
            block_size.x = std::min(m_steps, sms_per_head);
        } else {
            // strategy 2: fully unroll the q loops (contiguous blocks handle all q loops)
            block_size.x = m_steps;
        }
        params.num_tiles = params.b * params.h;
    } else if( Ktraits::SCHEDULING_MODE == 1 ) {
        // Get the max total M steps
        // Take multiple compute groups into consideration.
        size_t m_steps = size_t((params.s + 64 * NUM_COMPUTE_GROUPS - 1) / (64 * NUM_COMPUTE_GROUPS));
        params.num_tiles_per_head = static_cast<uint32_t>(m_steps);
        params.num_tiles = static_cast<uint32_t>(m_steps * params.b * params.h);
        if (launch_params.attention_mask_type == Attention_mask_type::CAUSAL) {
            // 2 * 2 stands for kv cache and 2 bytes per element.
            size_t size_in_bytes = params.b * params.h * params.s * params.d * 2 * 2;
            params.use_balanced_scheduling = (size_in_bytes <= launch_params.device_l2_cache_size);
        }

        block_size.x = 1;
        block_size.y = std::min(static_cast<int>(params.num_tiles), launch_params.multi_processor_count);
    } else {
        assert(false && "Invalid SCHEDULING_MODE");
    }

    // Reuse the same bytes_per_smem for launching kernels.
    constexpr int SMEM_BYTES = Ktraits::BYTES_PER_SMEM;
    if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {
#if 0 // padding_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_bf16_64_128_S_qkv_104_alibi_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(convertKernelParmas2BertParams(params));
#endif // padding_mask
    } else if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_128_S_qkv_104_causal_alibi_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_bf16_64_128_S_qkv_104_causal_alibi_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(convertKernelParmas2BertParams(params));
#endif // causal mask
    } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 0 // sliding_or_chunked_causal_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sliding_or_chunked_causal_alibi_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_bf16_64_128_S_qkv_104_sliding_or_chunked_causal_alibi_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(convertKernelParmas2BertParams(params));
#endif // sliding_or_chunked_causal_mask
    } else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {
#if 0 // custom_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_128_S_qkv_104_custom_mask_alibi_tma_ws_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        fmha_v2_flash_attention_bf16_64_128_S_qkv_104_custom_mask_alibi_tma_ws_sm90_kernel
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>(convertKernelParmas2BertParams(params));
#endif // custom mask
    }

}

#endif

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm

