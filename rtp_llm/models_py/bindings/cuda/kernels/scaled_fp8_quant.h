#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_fp8_utils.h"

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace rtp_llm {

#ifdef ENABLE_FP8

// Existing function
void per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static);
void per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s);

void invokeComputeFP8Quantize128(__nv_fp8_e4m3*       fp8_output,
                                 float*               quant_ptr,
                                 const __nv_bfloat16* weights,
                                 const int64_t        dim0,
                                 const int64_t        dim1,
                                 const int64_t        size,
                                 bool                 col_major_scale,
                                 cudaStream_t         stream);

#endif  // ENABLE_FP8
}  // namespace rtp_llm
