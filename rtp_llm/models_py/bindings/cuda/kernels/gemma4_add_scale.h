#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4AddBf16(
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs, __nv_bfloat16* output, int64_t numel, cudaStream_t stream);

void invokeGemma4ScaleBf16(
    const __nv_bfloat16* input, float scale, __nv_bfloat16* output, int64_t numel, cudaStream_t stream);

void invokeGemma4GatherRowsBf16(const __nv_bfloat16* input,
                                const int32_t*       indices,
                                __nv_bfloat16*       output,
                                int64_t              rows,
                                int32_t              hidden_size,
                                cudaStream_t         stream);

void invokeGemma4LogitSoftcapFp32(const float* input, float* output, int64_t numel, float cap, cudaStream_t stream);

void invokeGemma4AddScaleBf16(const __nv_bfloat16* residual,
                              const __nv_bfloat16* hidden,
                              const __nv_bfloat16* scale,
                              __nv_bfloat16*       output,
                              int64_t              numel,
                              cudaStream_t         stream);

void invokeGemma4RouterScaleBf16(const __nv_bfloat16* input,
                                 const __nv_bfloat16* channel_scale,
                                 float                scalar_scale,
                                 __nv_bfloat16*       output,
                                 int64_t              numel,
                                 int32_t              hidden_size,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
