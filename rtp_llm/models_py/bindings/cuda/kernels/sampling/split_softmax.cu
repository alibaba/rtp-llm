// Local-max/sumexp decomposition follows vLLM's rejection_sampler_utils.py
// (Apache-2.0). Adapted to the existing RTP FP32 probability-output contract.
#include "split_softmax.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace rtp_llm {
namespace {

constexpr int kThreads = 256;
using Reduce           = cub::BlockReduce<float, kThreads>;
struct Maximum {
    __device__ __forceinline__ float operator()(float left, float right) const {
        return fmaxf(left, right);
    }
};

template<int Tile>
__global__ void samplerLocalLogitStats(const float* input, float* maximums, float* totals, int vocab, int parts) {
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float                        maximum;
    const int                               row  = blockIdx.x;
    const int                               part = blockIdx.y;
    float                                   values[Tile / kThreads];
    float                                   local_max = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < Tile / kThreads; ++i) {
        const int col = part * Tile + threadIdx.x + i * kThreads;
        values[i]     = col < vocab ? input[static_cast<int64_t>(row) * vocab + col] : -CUDART_INF_F;
        local_max     = fmaxf(local_max, values[i]);
    }
    const float block_max = Reduce(scratch).Reduce(local_max, Maximum());
    if (threadIdx.x == 0)
        maximum = block_max;
    __syncthreads();
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < Tile / kThreads; ++i) {
        local_sum += __expf(values[i] - maximum);
    }
    const float block_sum = Reduce(scratch).Sum(local_sum);
    if (threadIdx.x == 0) {
        maximums[row * parts + part] = maximum;
        totals[row * parts + part]   = maximum == -CUDART_INF_F ? 0.0f : block_sum;
    }
}

template<int Tile>
__global__ void
samplerWriteProbabilities(float* input, const float* maximums, const float* totals, int vocab, int parts) {
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float                        maximum;
    __shared__ float                        inverse_sum;
    const int                               row       = blockIdx.x;
    const int                               part      = blockIdx.y;
    float                                   local_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < parts; i += kThreads) {
        local_max = fmaxf(local_max, maximums[row * parts + i]);
    }
    const float block_max = Reduce(scratch).Reduce(local_max, Maximum());
    if (threadIdx.x == 0)
        maximum = block_max;
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < parts; i += kThreads) {
        local_sum += totals[row * parts + i] * __expf(maximums[row * parts + i] - maximum);
    }
    const float block_sum = Reduce(scratch).Sum(local_sum);
    if (threadIdx.x == 0)
        inverse_sum = 1.0f / block_sum;
    __syncthreads();
    // The preceding stats launch completed every cross-tile input read.
    // Each CTA now reads and overwrites only its own disjoint tile.
#pragma unroll
    for (int i = 0; i < Tile / kThreads; ++i) {
        const int col = part * Tile + threadIdx.x + i * kThreads;
        if (col < vocab) {
            const int64_t offset = static_cast<int64_t>(row) * vocab + col;
            input[offset]        = __expf(input[offset] - maximum) * inverse_sum;
        }
    }
}

// Parallelize the vocabulary while retaining the installed PyTorch softmax's
// FP32 accumulation order: 1024 logical threads, vec4 loads, scalar tail,
// warp sums then a second warp sum. The two reduction levels may execute in
// different launches without changing their FP32 operands or addition order.
// Reference: PyTorch (BSD-3-Clause), aten/src/ATen/native/cuda/SoftMax.cu,
// cunn_SoftMaxForward<4> and the installed block_reduce.cuh.
__global__ void samplerLocalLogitStatsExact(const float* input, float* maximums, int vocab, int parts) {
    __shared__ typename Reduce::TempStorage scratch;
    const int                               row = blockIdx.x, tile = blockIdx.y;
    float                                   value = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = tile * 1024 + threadIdx.x + j * 256;
        if (col < vocab)
            value = fmaxf(value, input[static_cast<int64_t>(row) * vocab + col]);
    }
    value = Reduce(scratch).Reduce(value, Maximum());
    if (threadIdx.x == 0)
        maximums[row * parts + tile] = value;
}

__global__ void
samplerOrderedWarpSums(const float* input, const float* maximums, float* warp_sums, int vocab, int parts) {
    const int row = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
    float     maximum = -CUDART_INF_F;
    for (int i = lane; i < parts; i += 32)
        maximum = fmaxf(maximum, maximums[row * parts + i]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    maximum                     = __shfl_sync(0xffffffffu, maximum, 0);
    const int    logical_thread = group * 32 + lane;
    const float* row_input      = input + static_cast<int64_t>(row) * vocab;
    const int    vector_end     = vocab - vocab % 4096;
    float        sum            = 0.0f;
    for (int col = logical_thread * 4; col < vector_end; col += 4096) {
        const float4 values = reinterpret_cast<const float4*>(row_input)[col / 4];
        sum                 = sum + expf(values.x - maximum);
        sum                 = sum + expf(values.y - maximum);
        sum                 = sum + expf(values.z - maximum);
        sum                 = sum + expf(values.w - maximum);
    }
    for (int col = vector_end + logical_thread; col < vocab; col += 1024)
        sum = sum + expf(row_input[col] - maximum);
    sum = at::native::cuda_utils::WarpReduceSum(sum);
    if (lane == 0)
        warp_sums[row * 32 + group] = sum;
}

__global__ void
samplerWriteProbabilitiesExact(float* input, const float* maximums, const float* warp_sums, int vocab, int parts) {
    __shared__ float maximum, total;
    const int        row = blockIdx.x, tile = blockIdx.y;
    if (threadIdx.x < 32) {
        float local_maximum = -CUDART_INF_F;
        for (int i = threadIdx.x; i < parts; i += 32)
            local_maximum = fmaxf(local_maximum, maximums[row * parts + i]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_maximum = fmaxf(local_maximum, __shfl_down_sync(0xffffffffu, local_maximum, offset));
        const float sum = at::native::cuda_utils::WarpReduceSum(warp_sums[row * 32 + threadIdx.x]);
        if (threadIdx.x == 0) {
            maximum = local_maximum;
            total   = sum;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = tile * 1024 + threadIdx.x + j * 256;
        if (col < vocab) {
            const int64_t offset = static_cast<int64_t>(row) * vocab + col;
            input[offset]        = expf(input[offset] - maximum) / total;
        }
    }
}

void launchOrderedSoftmax(torch::Tensor logits) {
    const int  rows = logits.size(0), vocab = logits.size(1), parts = (vocab + 1023) / 1024;
    auto       workspace = torch::empty({rows, parts + 32}, logits.options());
    auto*      maximums  = workspace.data_ptr<float>();
    auto*      warp_sums = maximums + rows * parts;
    const auto stream    = at::cuda::getCurrentCUDAStream().stream();
    samplerLocalLogitStatsExact<<<dim3(rows, parts), 256, 0, stream>>>(
        logits.data_ptr<float>(), maximums, vocab, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    samplerOrderedWarpSums<<<dim3(rows, 32), 32, 0, stream>>>(
        logits.data_ptr<float>(), maximums, warp_sums, vocab, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    samplerWriteProbabilitiesExact<<<dim3(rows, parts), 256, 0, stream>>>(
        logits.data_ptr<float>(), maximums, warp_sums, vocab, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Tile>
void launchSplitSoftmax(torch::Tensor logits) {
    const int  rows      = logits.size(0);
    const int  vocab     = logits.size(1);
    const int  parts     = (vocab + Tile - 1) / Tile;
    auto       workspace = torch::empty({2, rows, parts}, logits.options());
    auto*      maximums  = workspace.data_ptr<float>();
    auto*      totals    = maximums + rows * parts;
    const auto stream    = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(rows, parts);
    samplerLocalLogitStats<Tile>
        <<<grid, kThreads, 0, stream>>>(logits.data_ptr<float>(), maximums, totals, vocab, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    samplerWriteProbabilities<Tile>
        <<<grid, kThreads, 0, stream>>>(logits.data_ptr<float>(), maximums, totals, vocab, parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void splitSoftmaxInplace(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
                    && logits.is_contiguous() && logits.size(0) > 0 && logits.size(0) <= 32 && logits.size(1) > 0
                    && logits.size(1) <= 262144,
                "split softmax expects contiguous CUDA FP32 [1..32,1..262144]");
    at::cuda::CUDAGuard guard(logits.device());
    if (logits.size(1) == 248320) {
        if (reinterpret_cast<uintptr_t>(logits.data_ptr<float>()) % 16 == 0) {
            launchOrderedSoftmax(logits);
        } else {
            // The original vec4 alignment prologue has a different reduction
            // schedule. Retain that exact path for unaligned storage views.
            logits.copy_(torch::softmax(logits, -1));
        }
        return;
    }
    if (logits.size(0) <= 8)
        launchSplitSoftmax<1024>(logits);
    else
        launchSplitSoftmax<4096>(logits);
}

}  // namespace rtp_llm
