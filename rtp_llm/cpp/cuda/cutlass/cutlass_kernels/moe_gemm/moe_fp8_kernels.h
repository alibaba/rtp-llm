#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include "rtp_llm/cpp/model_utils/activation_types.h"

namespace rtp_llm {

template<typename T>
void expandInputRowsKernelLauncherContiguous(T const*       unpermuted_input,
                                             float const*   fp8_scales,
                                             T*             permuted_output,
                                             float*         permuted_output_fp8_scales,
                                             float*         unpermuted_scales,
                                             float*         permuted_scales,
                                             int const*     source_rows,
                                             int const*     permuted_src_row_to_dst,
                                             int *           src_row_to_dst,
                                             int64_t const  num_rows,
                                             int64_t const  dest_num_rows,
                                             int64_t const  cols,
                                             int const      k,
                                             cudaStream_t   stream);

template<typename T>
void expandInputRowsKernelLauncherContiguous_V2(T const*       unpermuted_input,
                                                float const*   fp8_scales,
                                                T*             permuted_output,
                                                float*         permuted_output_fp8_scales,
                                                float*         unpermuted_scales,
                                                float*         permuted_scales,
                                                int const*     source_rows,
                                                int const*     permuted_src_row_to_dst,
                                                int *          src_row_to_dst,
                                                int64_t const* expert_first_token_offset,
                                                size_t         num_experts_per_node,
                                                int64_t const  num_rows,
                                                int64_t const  max_num_rows,
                                                int64_t const  cols,
                                                int const      k,
                                                cudaStream_t   stream);

void computeSrc2Dst(int64_t const* expert_first_token_offset,
                    int*           permuted_src_row_to_dst,
                    int*           masked_m,
                    size_t         num_experts_per_node,
                    size_t         padding_size,
                    cudaStream_t   stream);

template<class GemmOutputType, class ScaleBiasType>
void doActivationContiguous(__nv_fp8_e4m3*        output_fp8,
                            float*                fp8_scale,
                            GemmOutputType const* gemm_result,
                            ScaleBiasType const*  bias,
                            bool                  bias_is_broadcast,
                            int const*            src_row_to_dst,
                            int                   num_rows,
                            int64_t               inter_size,
                            ActivationType        activation_type,
                            int const*            permuted_experts,
                            cudaStream_t          stream);

template<class GemmOutputType, class ScaleBiasType>
void doActivationContiguous_V2(__nv_fp8_e4m3*        output_fp8,
                               float*                fp8_scale,
                               GemmOutputType const* gemm_result,
                               ScaleBiasType const*  bias,
                               bool                  bias_is_broadcast,
                               int const*            src_row_to_dst,
                               int64_t const*        expert_first_token_offset,
                               size_t                num_experts_per_node,
                               int                   max_num_rows,
                               int64_t               inter_size,
                               ActivationType        activation_type,
                               int const*            permuted_experts,
                               cudaStream_t          stream);

template<class GemmOutputType, class ScaleBiasType>
void doActivationMasked(__nv_fp8_e4m3*        output_fp8,
                        float*                fp8_scale,
                        GemmOutputType const* gemm_result,
                        ScaleBiasType const*  bias,
                        bool                  bias_is_broadcast,
                        int                   expert_num,
                        int                   token_num,
                        int64_t               inter_size,
                        ActivationType        activation_type,
                        int const*            masked_m,
                        cudaStream_t          stream);

}  // namespace rtp_llm
