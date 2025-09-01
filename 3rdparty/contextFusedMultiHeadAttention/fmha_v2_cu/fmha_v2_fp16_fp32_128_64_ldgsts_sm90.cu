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
#if 1
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif
#endif

#include <cuda.h>

#if CUDA_VERSION >= 11000

#include <fused_multihead_attention_kernel_4x1_hopper.h>
#if 1
#include <fused_multihead_attention_kernel_4x1_hopper_noloop.h>
#endif

#if 0
// only included if tma is used.
#include <fmha/hopper/tma_descriptor.h>
#endif //use_tma

#include "../fused_multihead_attention_common.h"

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off


using Attention_mask_type = ContextAttentionMaskType;

using Traits_p = fmha::Hopper_hgmma_fp32_traits<64, 128, 16, false, false>;
using Traits_o = fmha::Hopper_hgmma_fp32_traits<64, 64, 16, true, false>;

using Kernel_traits = FMHA_kernel_traits_hopper_v2<
                       Traits_p,
                       Traits_o,
                       128,
                       64,
                       64,
                       4,
                       1,
                       2,
                       0x27u>;

using Kernel_traits_causal = FMHA_kernel_traits_hopper_v2<
                              Traits_p,
                              Traits_o,
                              128,
                              64,
                              64,
                              4,
                              1,
                              3,
                              0x27u>;

using Kernel_traits_sliding_or_chunked_causal = FMHA_kernel_traits_hopper_v2<
                                           Traits_p,
                                           Traits_o,
                                           128,
                                           64,
                                           64,
                                           4,
                                           1,
                                           4,
                                           0x27u>;

#if 0 // use_tma

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_tma<Kernel_traits>(params);
}

#endif // padding_mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_tma<Kernel_traits_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_tma<Kernel_traits_sliding_or_chunked_causal>(params);
}

#endif // sliding_or_chunked_causal_mask

#else

#if 1

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper<Kernel_traits>(params);
}

#endif // padding_mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper<Kernel_traits_causal>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel(const __grid_constant__ bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper<Kernel_traits_sliding_or_chunked_causal>(params);
}
#endif

#endif // sliding_or_chunked_causal_mask

void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90(Fused_multihead_attention_params_v2 &params,
    const Launch_params &launch_params, cudaStream_t stream){
  // setting TMA descriptors if needed.
  // use_tma = 0
#if 0
    // declare TMA desc for Q, K, V
    typename fmha::Multiple_tma_descriptor<3> tma_desc_QKV;

    // GMEM pointers, the offset between each batch is d*3*h*seqlen
    // qkv pointer
    char *qkv_ptr = reinterpret_cast<char*>(params.qkv_ptr);

    // tensor size
    uint32_t tensor_size_qkv[3];
    tensor_size_qkv[2] = 1;
    tensor_size_qkv[1] = params.is_s_padded ? params.s * params.b : launch_params.seqlens[params.b];
    tensor_size_qkv[0] = (params.h + 2 * params.h_kv) * params.d;

    // box size for Q
    uint32_t box_size_q[3];
    box_size_q[2] = 1;
    box_size_q[1] = 64; // STEP size
    box_size_q[0] = 64; // head_size

    // box size for k and v
    uint32_t box_size_kv[3];
    box_size_kv[2] = 1;
    box_size_kv[1] = params.s; // S, should not be actual_s, OOB will be filled with zeros.
    box_size_kv[0] = 64; // head_size

    // stride size
    uint64_t tensor_stride_qkv[2];
    tensor_stride_qkv[0] = tensor_size_qkv[0] * Traits_p::BITS_PER_ELEMENT_A / 8;
    tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];

    // traversal stride
    uint32_t traversal_stride_qkv[3] = {1, 1, 1};

    // OOB fill zeros
    uint32_t oob_fill = 0;

    // FP32 to TF32 conversion disabled
    uint32_t fp32_to_tf32 = 0;

    //setup the descriptors

    //setup the descriptor for Q
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_q,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_q);

    // setup the descriptor for K
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_kv,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_k);

    // setup the descriptor for V
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_kv,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_v);


#endif // use_tma
  dim3 grid(params.h, params.b);
  // Use the same smem_size for all traits.
  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
       FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // sliding_or_chunked_causal_mask
  } else {
#if 1 // padding_mask
    constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }
    fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // padding_mask
  }
}

#if 1


using Kernel_traits_nl = FMHA_kernel_traits_hopper_v2<
                          Traits_p,
                          Traits_o,
                          128,
                          64,
                          64,
                          4,
                          1,
                          2,
                          0x27u>;

using Kernel_traits_causal_nl = FMHA_kernel_traits_hopper_v2<
                                 Traits_p,
                                 Traits_o,
                                 128,
                                 64,
                                 64,
                                 4,
                                 1,
                                 3,
                                 0x27u>;

using Kernel_traits_sliding_or_chunked_causal_nl = FMHA_kernel_traits_hopper_v2<
                                              Traits_p,
                                              Traits_o,
                                              128,
                                              64,
                                              64,
                                              4,
                                              1,
                                              4,
                                              0x27u>;

#if 1 // padding_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_nl<Kernel_traits_nl>(params);
}

#endif // padding_mask

#if 1 // causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_nl<Kernel_traits_causal_nl>(params);
}

#endif // causal mask

#if 1 // sliding_or_chunked_causal_mask

extern "C"
__global__
void fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel_nl(bert::Fused_multihead_attention_params_v2 params){
  fused_multihead_attention::device_4x1_hopper_nl<Kernel_traits_sliding_or_chunked_causal_nl>(params);
}

#endif // sliding_or_chunked_causal_mask

void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_nl(Fused_multihead_attention_params_v2 &params,
    const Launch_params& launch_params, cudaStream_t stream){
  constexpr int loop_iters = 128 / 64;
  static_assert(loop_iters * 64 == 128, "");
  dim3 grid(params.h, params.b, loop_iters);

  // Use the same smem_size for all traits.
  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {
#if 1 // causal_mask
    if( smem_size >= 48*1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }
    fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // causal mask
  } else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {
#if 1 // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }
    fmha_v2_fp16_fp32_128_64_sliding_or_chunked_causal_ldgsts_sm90_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // sliding_or_chunked_causal_mask
  } else {
#if 1 // padding_mask
    if( smem_size >= 48*1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }
    fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(convertKernelParmas2BertParams(params));
#endif // padding_mask
  }
}

#endif

#else

void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

#if 1

void run_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_nl(const bert::Fused_multihead_attention_params_v2 &params, cudaStream_t stream){
    assert(false && "Unsupported CUDA version");
}

#endif

#endif

// clang-format on
} // namespace kernels
} // namespace tensorrt_llm

