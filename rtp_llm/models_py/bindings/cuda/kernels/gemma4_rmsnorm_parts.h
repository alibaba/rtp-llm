#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4RmsSquareBf16(const __nv_bfloat16* input,
                               float*               output,
                               int64_t              numel,
                               int32_t              head_count,
                               int64_t              token_stride,
                               int64_t              head_stride,
                               int32_t              hidden_size,
                               cudaStream_t         stream);

void invokeGemma4RmsApplyBf16(const __nv_bfloat16* input,
                              const float*         inv_rms,
                              const float*         weight,
                              __nv_bfloat16*       output,
                              int64_t              numel,
                              int32_t              head_count,
                              int64_t              token_stride,
                              int64_t              head_stride,
                              int32_t              hidden_size,
                              cudaStream_t         stream);

void invokeGemma4RmsMeanFp32(const float* input, float* output, int64_t rows, int32_t hidden_size, cudaStream_t stream);

void invokeGemma4RmsInvFp32(
    const float* input, float* output, int64_t rows, int32_t hidden_size, float eps, cudaStream_t stream);

}  // namespace rtp_llm
