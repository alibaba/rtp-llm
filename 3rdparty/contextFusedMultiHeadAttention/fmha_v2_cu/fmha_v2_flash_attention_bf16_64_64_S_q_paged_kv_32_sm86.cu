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


//We can disable the FADD trick for archs with F2IP
#if 0 // disable_fadd_trick
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif // USE_I2F_EMULATION_TRICK

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif // USE_F2I_EMULATION_TRICK
#endif // disable_fadd_trick

#include <cuda.h>

#if CUDA_VERSION >= 11000

#include <fused_multihead_flash_attention_kernel_noloop.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>
#include <fused_multihead_flash_attention_kernel.h>

#include "../fused_multihead_attention_common.h"


namespace tensorrt_llm
{
namespace kernels
{
// clang-format off


using Attention_mask_type = ContextAttentionMaskType;

#if 0 // has_noloop (unconditionally disabled since not maintained & not actively used)
using Kernel_traits = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u>;

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention<Kernel_traits>(params);
}

void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86(
    const bert::Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {
    FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }
  dim3 grid(params.h, params.b);
  fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
}

#endif // has_noloop

#if 1 && !0 // has_noloop && !tiled
using Kernel_traits_nl = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t,
    0,
    0,
    0>;

using Kernel_traits_nl_causal = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

using Kernel_traits_nl_sliding_or_chunked_causal = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

using Kernel_traits_nl_custom_mask = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl>(params);
}

#endif // padding mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_sliding_or_chunked_causal>(params);
}

#endif // sliding_or_chunked_causal_mask

#if 1 // custom_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl<Kernel_traits_nl_custom_mask>(params);
}

#endif // custom_mask

void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl(
    Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){

  // runtime q_loop_iters
  int loop_iters = ( params.s + 64 - 1 )  / 64;
  // dim3 grid(params.h, params.b, loop_iters);
  dim3 grid(loop_iters, params.h, params.b); // better locality
  constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // sliding_or_chunked_causal_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {
#if 1 // padding_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // padding_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {
#if 1 // custom_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // custom mask
  }
}

#endif // has_noloop && !tiled

#if 0 // tiled

using Kernel_traits_nl_tiled = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t,
    0,
    0,
    0>;

using Kernel_traits_nl_tiled_causal = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

using Kernel_traits_nl_tiled_sliding_or_chunked_causal = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

using Kernel_traits_nl_tiled_custom_mask = fmha::Kernel_traits_v2_paged_kv_cache<
    fmha::Ampere_hmma_bf16_traits,
    64,
    32,
    0,
    64,
    4,
    1,
    1,
    0x27u | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    fmha::bf16_t>;

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled>(params);
}

#endif // padding_mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_sliding_or_chunked_causal>(params);
}

#endif // sliding_or_chunked_causal_mask

#if 1 // custom_mask

extern "C"
__global__
void fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_flash_attention_nl_tiled<Kernel_traits_nl_tiled_custom_mask>(params);
}

#endif // custom mask

// Granular tiling
void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl_tiled(
    Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params,
    cudaStream_t stream){
  // runtime q_loop_iters
  using Cta_tile_o = typename Kernel_traits_nl_tiled::Cta_tile_o;
  int ctas_per_o_row = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
  int loop_iters = ( params.s + 64 - 1 )  / 64;
  dim3 grid(loop_iters * ctas_per_o_row, params.h, params.b);
  constexpr int smem_size = Kernel_traits_nl_tiled::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_causal_sm86_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl_tiled,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sliding_or_chunked_causal_sm86_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // sliding_or_chunked_causal_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {
#if 1 // padding_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // padding_mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {
#if 1 // custom_mask
    if( smem_size >= 48*1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }
    fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_custom_mask_sm86_kernel_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // custom mask
  }
}

#endif // tiled

#else // CUDA_VERSION >= 11000

void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

void run_fmha_v2_flash_attention_bf16_64_64_S_q_paged_kv_32_sm86_nl_tiled(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

#endif // CUDA_VERSION >= 11000

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm

