#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace rtp_llm {

// Weight-only INT8 (symmetric, per-output-row scale) LM head for small-M decode.
// W: [N, K] bf16 row-major. Q: [N, K] int8. S: [N] fp32 with W[n, :] ~= Q[n, :] * S[n].
void invokeQuantizeLmHeadInt8(const __nv_bfloat16* W, int8_t* Q, float* S, int64_t N, int64_t K, cudaStream_t stream);

// Returns true if the shape is supported by invokeInt8LmHeadGemv on the current device
// (M <= 8, K % 256 == 0, K <= 8192 and x[M, K] fits in the per-block shared memory limit).
bool int8LmHeadGemvSupported(int64_t M, int64_t K);

// y[M, N] (fp32) = x[M, K] (bf16) @ (Q * S)^T. Requires int8LmHeadGemvSupported(M, K).
// Returns the CUDA error of the launch setup or the launch itself.
cudaError_t invokeInt8LmHeadGemv(const int8_t*        Q,
                                 const float*         S,
                                 const __nv_bfloat16* x,
                                 float*               y,
                                 int64_t              M,
                                 int64_t              N,
                                 int64_t              K,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
