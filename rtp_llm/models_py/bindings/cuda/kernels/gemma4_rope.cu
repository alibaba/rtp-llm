#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_rope.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

__global__ void gemma4RopeCosSinBf16Kernel(const int32_t* __restrict__ positions,
                                           const float* __restrict__ inv_freq,
                                           __nv_bfloat16* __restrict__ cos,
                                           __nv_bfloat16* __restrict__ sin,
                                           int64_t numel,
                                           int32_t head_dim,
                                           int32_t half_dim) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const int32_t dim   = static_cast<int32_t>(index % head_dim);
    const int64_t token = index / head_dim;
    const float   angle = __fmul_rn(__int2float_rn(positions[token]), inv_freq[dim % half_dim]);
    cos[index]          = __float2bfloat16_rn(cosf(angle));
    sin[index]          = __float2bfloat16_rn(sinf(angle));
}

__device__ __forceinline__ __nv_bfloat16 applyRope(const __nv_bfloat16* input,
                                                   const __nv_bfloat16* cos,
                                                   const __nv_bfloat16* sin,
                                                   int64_t              index,
                                                   int32_t              heads,
                                                   int32_t              head_dim) {
    const int32_t dim         = static_cast<int32_t>(index % head_dim);
    const int64_t row         = index / head_dim;
    const int64_t token       = row / heads;
    const int64_t row_start   = row * head_dim;
    const int32_t rotated_dim = dim < head_dim / 2 ? dim + head_dim / 2 : dim - head_dim / 2;
    float         rotated     = __bfloat162float(input[row_start + rotated_dim]);
    if (dim < head_dim / 2) {
        rotated = -rotated;
    }
    const int64_t       rope_index = token * head_dim + dim;
    const __nv_bfloat16 lhs = __float2bfloat16_rn(__bfloat162float(input[index]) * __bfloat162float(cos[rope_index]));
    const __nv_bfloat16 rhs =
        __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(rotated)) * __bfloat162float(sin[rope_index]));
    return __float2bfloat16_rn(__bfloat162float(lhs) + __bfloat162float(rhs));
}

__global__ void gemma4QkRopeBf16Kernel(const __nv_bfloat16* q,
                                       const __nv_bfloat16* k,
                                       const __nv_bfloat16* cos,
                                       const __nv_bfloat16* sin,
                                       __nv_bfloat16*       q_out,
                                       __nv_bfloat16*       k_out,
                                       int64_t              q_numel,
                                       int64_t              k_numel,
                                       int32_t              query_heads,
                                       int32_t              kv_heads,
                                       int32_t              head_dim) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < q_numel) {
        q_out[index] = applyRope(q, cos, sin, index, query_heads, head_dim);
    }
    if (index < k_numel) {
        k_out[index] = applyRope(k, cos, sin, index, kv_heads, head_dim);
    }
}

}  // namespace

void invokeGemma4RopeCosSinBf16(const int32_t* positions,
                                const float*   inv_freq,
                                __nv_bfloat16* cos,
                                __nv_bfloat16* sin,
                                int64_t        tokens,
                                int32_t        half_dim,
                                cudaStream_t   stream) {
    if (tokens == 0) {
        return;
    }
    constexpr int threads  = 256;
    const int32_t head_dim = half_dim * 2;
    const int64_t numel    = tokens * head_dim;
    const int     blocks   = static_cast<int>((numel + threads - 1) / threads);
    gemma4RopeCosSinBf16Kernel<<<blocks, threads, 0, stream>>>(
        positions, inv_freq, cos, sin, numel, head_dim, half_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4QkRopeBf16(const __nv_bfloat16* q,
                            const __nv_bfloat16* k,
                            const __nv_bfloat16* cos,
                            const __nv_bfloat16* sin,
                            __nv_bfloat16*       q_out,
                            __nv_bfloat16*       k_out,
                            int64_t              tokens,
                            int32_t              query_heads,
                            int32_t              kv_heads,
                            int32_t              head_dim,
                            cudaStream_t         stream) {
    const int64_t q_numel = tokens * query_heads * head_dim;
    const int64_t k_numel = tokens * kv_heads * head_dim;
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((q_numel + threads - 1) / threads);
    gemma4QkRopeBf16Kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos, sin, q_out, k_out, q_numel, k_numel, query_heads, kv_heads, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
