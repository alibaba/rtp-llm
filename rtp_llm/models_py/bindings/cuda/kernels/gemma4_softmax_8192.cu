// Derived from PyTorch aten/src/ATen/native/cuda/SoftMax.cu at
// 70d99e998b4955e0049d13a98d77ae1b14db1f45 under the BSD-3-Clause license.
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_softmax_8192.h"

#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

namespace rtp_llm {
namespace {

constexpr int kClasses  = 8192;
constexpr int kThreads  = 1024;
constexpr int kRegCount = 8;
constexpr int kWarpSize = 32;
constexpr int kWarps    = kThreads / kWarpSize;

template<typename T>
struct Add {
    __device__ __forceinline__ T combine(T a, T b) const {
        return a + b;
    }

    __device__ __forceinline__ T warpShflDown(T value, int offset) const {
        return __shfl_down_sync(0xffffffffu, value, offset);
    }
};

template<typename T>
struct Max {
    __device__ __forceinline__ T combine(T a, T b) const {
        return a < b ? b : a;
    }

    __device__ __forceinline__ T warpShflDown(T value, int offset) const {
        return __shfl_down_sync(0xffffffffu, value, offset);
    }
};

template<typename T, typename ReduceOp>
__device__ __forceinline__ T warpReduce(T value, const ReduceOp& op) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = op.combine(value, op.warpShflDown(value, offset));
    }
    return value;
}

template<typename T, typename ReduceOp>
__device__ __forceinline__ T blockReduce(T value, const ReduceOp& op, T identity, T* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value          = warpReduce(value, op);
    __syncthreads();
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < kWarps ? shared[lane] : identity;
    if (warp == 0) {
        value = warpReduce(value, op);
    }
    return value;
}

template<typename T, typename ReduceOp>
__device__ __forceinline__ T blockReduceWarp(T* shared, T value, const ReduceOp& op, T identity) {
    const T result = blockReduce(value, op, identity, shared);
    if (threadIdx.x == 0) {
        shared[0] = result;
    }
    __syncthreads();
    return shared[0];
}

__global__ void gemma4Softmax8192Bf16Kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int32_t query_length, int32_t query_start, int32_t window_left) {
    extern __shared__ unsigned char shared_bytes[];
    auto*                           shared = reinterpret_cast<float*>(shared_bytes);
    input += static_cast<int64_t>(blockIdx.x) * kClasses;
    output += static_cast<int64_t>(blockIdx.x) * kClasses;
    const int32_t query_position = query_start < 0 ? -1 : query_start + static_cast<int32_t>(blockIdx.x) % query_length;

    float values[kRegCount];
    float thread_max = std::numeric_limits<float>::lowest();
#pragma unroll
    for (int i = 0; i < kRegCount; ++i) {
        const int  offset = threadIdx.x + i * blockDim.x;
        const bool masked = query_position >= 0
                            && (offset > query_position || (window_left > 0 && offset <= query_position - window_left));
        values[i]  = masked ? __int_as_float(0xff7f0000) : __bfloat162float(input[offset]);
        thread_max = thread_max < values[i] ? values[i] : thread_max;
    }

    const float max_value = blockReduceWarp(shared, thread_max, Max<float>(), std::numeric_limits<float>::lowest());

    float thread_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegCount; ++i) {
        thread_sum += std::exp(values[i] - max_value);
    }
    const float sum = blockReduceWarp(shared, thread_sum, Add<float>(), 0.0f);

#pragma unroll
    for (int i = 0; i < kRegCount; ++i) {
        const int offset = threadIdx.x + i * blockDim.x;
        output[offset]   = __float2bfloat16(std::exp(values[i] - max_value) / sum);
    }
}

}  // namespace

void invokeGemma4Softmax8192Bf16(const __nv_bfloat16* input,
                                 __nv_bfloat16*       output,
                                 int64_t              rows,
                                 int32_t              query_length,
                                 int32_t              query_start,
                                 int32_t              window_left,
                                 cudaStream_t         stream) {
    if (rows == 0) {
        return;
    }
    gemma4Softmax8192Bf16Kernel<<<static_cast<uint32_t>(rows), kThreads, kWarps * sizeof(float), stream>>>(
        input, output, query_length, query_start, window_left);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
