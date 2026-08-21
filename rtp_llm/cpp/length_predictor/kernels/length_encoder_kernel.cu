#include "rtp_llm/cpp/length_predictor/kernels/length_encoder_kernel.h"

#if USING_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace rtp_llm {

namespace {

__device__ __forceinline__ float toFloat(float value) {
    return value;
}
__device__ __forceinline__ float toFloat(__half value) {
    return __half2float(value);
}
__device__ __forceinline__ float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

// erf-form GELU, matching torch.nn.GELU(approximate='none').
__device__ __forceinline__ float geluErf(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

template<int BLOCK_THREADS>
__device__ __forceinline__ float blockReduceSum(float value) {
    static_assert(BLOCK_THREADS % 32 == 0, "block must be a whole number of warps");
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    __shared__ float warp_sums[NUM_WARPS];

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = (threadIdx.x < NUM_WARPS) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;  // valid on thread 0 only
}

// One block per row. Dynamic shared memory holds the fp32 row (hidden_dim
// floats): pass 1 stashes raw values while accumulating sum/sumsq, pass 2
// normalizes in place, pass 3 is a rank-1 accumulation where thread j owns
// output feature j and each iteration reads feature_dim consecutive weights
// (coalesced) while broadcasting one shared-memory value (conflict-free).
template<typename T, int BLOCK_THREADS>
__global__ void lengthEncoderKernel(const T* __restrict__ hidden,
                                    const float* __restrict__ weight_t,
                                    const float* __restrict__ bias,
                                    float* __restrict__ out,
                                    int   hidden_dim,
                                    int   feature_dim,
                                    float eps) {
    extern __shared__ float row[];
    const T* __restrict__ input = hidden + static_cast<size_t>(blockIdx.x) * hidden_dim;

    float sum   = 0.0f;
    float sumsq = 0.0f;
    for (int k = threadIdx.x; k < hidden_dim; k += BLOCK_THREADS) {
        const float value = toFloat(input[k]);
        row[k] = value;
        sum += value;
        sumsq = fmaf(value, value, sumsq);
    }
    __syncthreads();  // protect warp_sums reuse inside the two reductions
    sum = blockReduceSum<BLOCK_THREADS>(sum);
    __syncthreads();
    sumsq = blockReduceSum<BLOCK_THREADS>(sumsq);

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        const float mean     = sum / static_cast<float>(hidden_dim);
        const float variance = fmaxf(sumsq / static_cast<float>(hidden_dim) - mean * mean, 0.0f);
        s_mean = mean;
        s_rstd = rsqrtf(variance + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float rstd = s_rstd;
    for (int k = threadIdx.x; k < hidden_dim; k += BLOCK_THREADS) {
        row[k] = (row[k] - mean) * rstd;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < feature_dim; j += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int k = 0; k < hidden_dim; ++k) {
            acc = fmaf(row[k], weight_t[static_cast<size_t>(k) * feature_dim + j], acc);
        }
        out[static_cast<size_t>(blockIdx.x) * feature_dim + j] = geluErf(acc + bias[j]);
    }
}

}  // namespace

template<typename T>
void invokeLengthEncoderForward(const T* __restrict__ hidden,
                                const float* __restrict__ weight_t,
                                const float* __restrict__ bias,
                                float* __restrict__ out,
                                int          batch,
                                int          hidden_dim,
                                int          feature_dim,
                                float        eps,
                                cudaStream_t stream) {
    constexpr int BLOCK_THREADS = 128;
    if (batch <= 0) {
        return;
    }
    const size_t smem_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    if (smem_bytes > 48 * 1024) {
        // Raising the opt-in limit is idempotent and cheap; do it every call
        // instead of tracking per-device state.
        cudaFuncSetAttribute(lengthEncoderKernel<T, BLOCK_THREADS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }
    lengthEncoderKernel<T, BLOCK_THREADS>
        <<<batch, BLOCK_THREADS, smem_bytes, stream>>>(hidden, weight_t, bias, out, hidden_dim, feature_dim, eps);
}

template void invokeLengthEncoderForward<float>(const float* __restrict__,
                                                const float* __restrict__,
                                                const float* __restrict__,
                                                float* __restrict__,
                                                int,
                                                int,
                                                int,
                                                float,
                                                cudaStream_t);
template void invokeLengthEncoderForward<__half>(const __half* __restrict__,
                                                 const float* __restrict__,
                                                 const float* __restrict__,
                                                 float* __restrict__,
                                                 int,
                                                 int,
                                                 int,
                                                 float,
                                                 cudaStream_t);
template void invokeLengthEncoderForward<__nv_bfloat16>(const __nv_bfloat16* __restrict__,
                                                        const float* __restrict__,
                                                        const float* __restrict__,
                                                        float* __restrict__,
                                                        int,
                                                        int,
                                                        int,
                                                        float,
                                                        cudaStream_t);

}  // namespace rtp_llm

#endif  // USING_CUDA
