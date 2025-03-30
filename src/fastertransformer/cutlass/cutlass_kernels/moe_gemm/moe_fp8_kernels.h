#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include "src/fastertransformer/utils/activation_types.h"

namespace fastertransformer {

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

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivationContiguous(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
                            bool bias_is_broadcast, int const* src_row_to_dst, int num_rows, int64_t inter_size,
                            ActivationType activation_type, int const* permuted_experts, cudaStream_t stream);

}  // namespace fastertransformer
