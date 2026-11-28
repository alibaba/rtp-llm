// Local-max/sumexp decomposition follows vLLM's rejection_sampler_utils.py
// (Apache-2.0). Adapted to the existing RTP FP32 probability-output contract.
#include "split_softmax.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <mutex>
#include <vector>

namespace rtp_llm {
namespace {

constexpr int kThreads = 256;

struct SplitSoftmaxWorkspaceKey {
    int      device_id;
    uint64_t stream_id;

    bool operator==(const SplitSoftmaxWorkspaceKey& other) const {
        return device_id == other.device_id && stream_id == other.stream_id;
    }
};

struct SplitSoftmaxWorkspaceEntry {
    SplitSoftmaxWorkspaceKey key;
    torch::Tensor            buffer;
    uint64_t                 last_used_seq = 0;
};

class SplitSoftmaxWorkspaceCache {
public:
    static constexpr size_t kMaxEntriesPerDevice = 4;

    torch::Tensor
    acquire(int device_id, uint64_t stream_id, int64_t required_elems, const torch::TensorOptions& options) {
        std::lock_guard<std::mutex>    lock(mutex_);
        const SplitSoftmaxWorkspaceKey key{device_id, stream_id};
        const uint64_t                 seq = ++seq_counter_;

        for (auto& entry : entries_) {
            if (entry.key == key) {
                entry.last_used_seq = seq;
                if (entry.buffer.numel() < required_elems) {
                    entry.buffer = torch::empty({required_elems}, options);
                }
                return entry.buffer;
            }
        }

        size_t   device_count = 0;
        size_t   lru_idx      = entries_.size();
        uint64_t lru_seq      = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].key.device_id == device_id) {
                ++device_count;
                if (entries_[i].last_used_seq < lru_seq) {
                    lru_seq = entries_[i].last_used_seq;
                    lru_idx = i;
                }
            }
        }
        if (device_count >= kMaxEntriesPerDevice && lru_idx < entries_.size()) {
            entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(lru_idx));
        }

        auto buffer = torch::empty({required_elems}, options);
        entries_.push_back(SplitSoftmaxWorkspaceEntry{key, buffer, seq});
        return buffer;
    }

private:
    std::mutex                              mutex_;
    std::vector<SplitSoftmaxWorkspaceEntry> entries_;
    uint64_t                                seq_counter_ = 0;
};

torch::Tensor getSplitSoftmaxWorkspace(const c10::Device&          device,
                                       cudaStream_t                stream,
                                       int64_t                     required_elems,
                                       const torch::TensorOptions& options) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long      stream_id      = 0;
    if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess && capture_status == cudaStreamCaptureStatusNone
        && cudaStreamGetId(stream, &stream_id) == cudaSuccess) {
        static auto* cache = new SplitSoftmaxWorkspaceCache();
        return cache->acquire(device.index(), static_cast<uint64_t>(stream_id), required_elems, options);
    }
    return torch::empty({required_elems}, options);
}
using Reduce = cub::BlockReduce<float, kThreads>;
struct Maximum {
    __device__ __forceinline__ float operator()(float left, float right) const {
        return fmaxf(left, right);
    }
};

// Parallelize the vocabulary while retaining the installed PyTorch softmax's
// FP32 accumulation order: 1024 logical threads, vec4 loads, scalar tail,
// warp sums then a second warp sum. The two reduction levels may execute in
// different launches without changing their FP32 operands or addition order.
// Reference: PyTorch (BSD-3-Clause), aten/src/ATen/native/cuda/SoftMax.cu,
// cunn_SoftMaxForward<4> and the installed block_reduce.cuh.
template<int kFloat4PerThread>
__global__ void samplerLocalLogitStatsExact(const float* input, float* maximums, int vocab, int parts) {
    __shared__ typename Reduce::TempStorage scratch;
    const int                               row    = blockIdx.x;
    const int                               tile   = blockIdx.y;
    const int                               vocab4 = vocab / 4;
    const float4* row_v4 = reinterpret_cast<const float4*>(input + static_cast<int64_t>(row) * vocab);
    float         value  = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < kFloat4PerThread; ++j) {
        const int col4 = (tile * kFloat4PerThread + j) * kThreads + threadIdx.x;
        if (col4 < vocab4) {
            const float4 v = row_v4[col4];
            value          = fmaxf(value, v.x);
            value          = fmaxf(value, v.y);
            value          = fmaxf(value, v.z);
            value          = fmaxf(value, v.w);
        }
    }
    value = Reduce(scratch).Reduce(value, Maximum());
    if (threadIdx.x == 0) {
        maximums[row * parts + tile] = value;
    }
}

__global__ void samplerOrderedWarpSums(
    const float* input, const float* maximums, float* row_max, float* warp_sums, int vocab, int parts) {
    __shared__ float warp_maxes[4];
    __shared__ float shared_max;
    const int        row     = blockIdx.x;
    const int        warp_id = threadIdx.x >> 5;
    const int        lane    = threadIdx.x & 31;
    const int        group   = blockIdx.y * 4 + warp_id;

    float maximum = -CUDART_INF_F;
    for (int i = threadIdx.x; i < parts; i += 128) {
        maximum = fmaxf(maximum, maximums[row * parts + i]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = maximum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float m = (threadIdx.x < 4) ? warp_maxes[threadIdx.x] : -CUDART_INF_F;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, offset));
        }
        if (threadIdx.x == 0) {
            shared_max = m;
            if (blockIdx.y == 0) {
                row_max[row] = m;
            }
        }
    }
    __syncthreads();
    maximum = shared_max;

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
    for (int col = vector_end + logical_thread; col < vocab; col += 1024) {
        sum = sum + expf(row_input[col] - maximum);
    }
    sum = at::native::cuda_utils::WarpReduceSum(sum);
    if (lane == 0) {
        warp_sums[row * 32 + group] = sum;
    }
}

template<int kFloat4PerThread>
__global__ void samplerWriteProbabilitiesExact(float* input, const float* row_max, const float* warp_sums, int vocab) {
    __shared__ float maximum, total;
    const int        row = blockIdx.x, tile = blockIdx.y;
    if (threadIdx.x < 32) {
        const float sum = at::native::cuda_utils::WarpReduceSum(warp_sums[row * 32 + threadIdx.x]);
        if (threadIdx.x == 0) {
            maximum = row_max[row];
            total   = sum;
        }
    }
    __syncthreads();
    const int vocab4 = vocab / 4;
    float4*   row_v4 = reinterpret_cast<float4*>(input + static_cast<int64_t>(row) * vocab);
#pragma unroll
    for (int j = 0; j < kFloat4PerThread; ++j) {
        const int col4 = (tile * kFloat4PerThread + j) * kThreads + threadIdx.x;
        if (col4 < vocab4) {
            float4 v     = row_v4[col4];
            v.x          = expf(v.x - maximum) / total;
            v.y          = expf(v.y - maximum) / total;
            v.z          = expf(v.z - maximum) / total;
            v.w          = expf(v.w - maximum) / total;
            row_v4[col4] = v;
        }
    }
}

void launchOrderedSoftmax(torch::Tensor logits) {
    const int rows = logits.size(0), vocab = logits.size(1);
    // Keep > 100 CTAs on small batches (tile=2048) and reduce launch wave overhead on larger batches (tile=4096).
    const bool    large_tile     = rows > 16;
    const int     tile_size      = large_tile ? 4096 : 2048;
    const int     parts          = (vocab + tile_size - 1) / tile_size;
    const auto    stream         = at::cuda::getCurrentCUDAStream().stream();
    const int64_t required_elems = static_cast<int64_t>(rows) * (parts + 33);
    auto          workspace      = getSplitSoftmaxWorkspace(logits.device(), stream, required_elems, logits.options());
    auto*         maximums       = workspace.data_ptr<float>();
    auto*         row_max        = maximums + static_cast<int64_t>(rows) * parts;
    auto*         warp_sums      = row_max + rows;
    if (large_tile) {
        samplerLocalLogitStatsExact<4>
            <<<dim3(rows, parts), 256, 0, stream>>>(logits.data_ptr<float>(), maximums, vocab, parts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        samplerOrderedWarpSums<<<dim3(rows, 8), 128, 0, stream>>>(
            logits.data_ptr<float>(), maximums, row_max, warp_sums, vocab, parts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        samplerWriteProbabilitiesExact<4>
            <<<dim3(rows, parts), 256, 0, stream>>>(logits.data_ptr<float>(), row_max, warp_sums, vocab);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        samplerLocalLogitStatsExact<2>
            <<<dim3(rows, parts), 256, 0, stream>>>(logits.data_ptr<float>(), maximums, vocab, parts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        samplerOrderedWarpSums<<<dim3(rows, 8), 128, 0, stream>>>(
            logits.data_ptr<float>(), maximums, row_max, warp_sums, vocab, parts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        samplerWriteProbabilitiesExact<2>
            <<<dim3(rows, parts), 256, 0, stream>>>(logits.data_ptr<float>(), row_max, warp_sums, vocab);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

void splitSoftmaxInplace(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
                    && logits.is_contiguous() && logits.size(0) > 0 && logits.size(0) <= 256 && logits.size(1) >= 65536
                    && logits.size(1) % 4 == 0,
                "split softmax expects contiguous CUDA FP32 [1..256,vocab>=65536,vocab%4==0]");
    at::cuda::CUDAGuard guard(logits.device());
    if (reinterpret_cast<uintptr_t>(logits.data_ptr<float>()) % 16 == 0) {
        launchOrderedSoftmax(logits);
    } else {
        // The original vec4 alignment prologue has a different reduction
        // schedule. Retain that exact path for unaligned storage views.
        logits.copy_(torch::softmax(logits, -1));
    }
}

}  // namespace rtp_llm
