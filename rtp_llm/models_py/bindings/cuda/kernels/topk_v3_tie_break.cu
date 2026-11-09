#include "rtp_llm/models_py/bindings/cuda/kernels/topk_v3_tie_break.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/topk_v3_tie_break.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace torch_ext {
namespace {

namespace impl = device::topk_tie_break;
namespace base = device::topk;

using base::TopKProblem;
using Register1 = impl::TopKRegister<1>;
using Register2 = impl::TopKRegister<2>;
using Register4 = impl::TopKRegister<4>;
using Streaming = impl::TopKStreaming;

constexpr uint32_t kBlockSize = base::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = base::TopKConfig::kOccupancy;

struct Params {
    const float* __restrict__ scores;
    const int32_t* __restrict__ row_starts;
    const int32_t* __restrict__ row_ends;
    int32_t* __restrict__ output;
    int64_t  stride;
    uint32_t score_width;
    uint32_t rows;
    uint32_t topk;
    uint32_t max_seq_len;

    __device__ __forceinline__ TopKProblem problem(uint32_t row) const {
        const int32_t requested_start = row_starts[row];
        const int32_t requested_end = row_ends[row];
        const uint32_t start = requested_start <= 0
                                   ? 0u
                                   : min(static_cast<uint32_t>(requested_start), score_width);
        uint32_t end = requested_end <= 0 ? 0u : min(static_cast<uint32_t>(requested_end), score_width);
        end = max(end, start);
        const uint32_t seq_len = min(end - start, max_seq_len);
        return TopKProblem{
            .in = scores + static_cast<int64_t>(row) * stride + start,
            .out = output + static_cast<int64_t>(row) * topk,
            .raw_out = nullptr,
            .page_table = nullptr,
            .topk = topk,
            .seq_len = seq_len,
            .page_bits = 0,
        };
    }
};

__device__ __forceinline__ void trivial(const TopKProblem& problem) {
    for (uint32_t i = threadIdx.x; i < problem.topk; i += blockDim.x) {
        problem.out[i] = i < problem.seq_len ? static_cast<int32_t>(i) : -1;
    }
}

template <typename Implementation>
__device__ __forceinline__ void run_implementation(const TopKProblem& problem, void* smem) {
    const bool aligned = (reinterpret_cast<uintptr_t>(problem.in) & 0xfu) == 0u;
    if (aligned) {
        Implementation::template forward<true>(problem, smem);
    } else {
        Implementation::template forward<false>(problem, smem);
    }
}

template <int Level>
__global__ __launch_bounds__(kBlockSize, kOccupancy)
void main_kernel(const __grid_constant__ Params params) {
    device::enable_smem_spilling();
    const TopKProblem problem = params.problem(blockIdx.x);
    if (problem.seq_len <= problem.topk) {
        trivial(problem);
        return;
    }
    __shared__ base::MaxSmem<Register1::Smem, Register2::Smem, Register4::Smem, Streaming::Smem> smem;
    if constexpr (Level == -1) {
        run_implementation<Register1>(problem, &smem);
    } else if constexpr (Level == 0) {
        run_implementation<Register2>(problem, &smem);
    } else if constexpr (Level == 1) {
        run_implementation<Register4>(problem, &smem);
    } else {
        run_implementation<Streaming>(problem, &smem);
    }
}

cudaError_t launch_topk(const Params& params, int64_t k, int64_t max_seq_len, cudaStream_t stream) {
    if (max_seq_len <= Register1::kMaxSeqLen && (params.rows <= 32 || k <= 1024)) {
        main_kernel<-1><<<params.rows, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= Register2::kMaxSeqLen) {
        main_kernel<0><<<params.rows, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= Register4::kMaxSeqLen) {
        main_kernel<1><<<params.rows, kBlockSize, 0, stream>>>(params);
    } else {
        main_kernel<2><<<params.rows, kBlockSize, 0, stream>>>(params);
    }
    return cudaGetLastError();
}

}  // namespace

void topk_v3_tie_break(const torch::Tensor& scores,
                       const torch::Tensor& row_starts,
                       const torch::Tensor& row_ends,
                       torch::Tensor&       output,
                       int64_t              k,
                       int64_t              max_seq_len) {
    TORCH_CHECK(scores.is_cuda() && row_starts.is_cuda() && row_ends.is_cuda() && output.is_cuda(),
                "scores, row_starts, row_ends, and output must be CUDA tensors");
    TORCH_CHECK(row_starts.device() == scores.device() && row_ends.device() == scores.device() &&
                    output.device() == scores.device(),
                "scores, row_starts, row_ends, and output must be on the same CUDA device");
    TORCH_CHECK(scores.dtype() == torch::kFloat32 && row_starts.dtype() == torch::kInt32 &&
                    row_ends.dtype() == torch::kInt32 && output.dtype() == torch::kInt32,
                "expected float32 scores and int32 row_starts/row_ends/output");
    TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1,
                "scores must be a dense-last-dimension 2D tensor");
    TORCH_CHECK(row_starts.is_contiguous() && row_ends.is_contiguous() &&
                    row_starts.numel() == scores.size(0) && row_ends.numel() == scores.size(0),
                "row_starts/row_ends shape mismatch");
    TORCH_CHECK(output.dim() == 2 && output.is_contiguous() && output.size(0) == scores.size(0) &&
                    output.size(1) == k,
                "output shape mismatch");
    TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
                "topk_v3_tie_break supports K=512, 1024, or 2048");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= scores.size(1), "invalid max_seq_len");
    TORCH_CHECK(scores.size(1) <= std::numeric_limits<int32_t>::max(),
                "score width exceeds int32 row-bound range");
    TORCH_CHECK(scores.size(0) <= std::numeric_limits<uint32_t>::max(),
                "too many score rows for the CUDA dispatcher");

    if (scores.size(0) == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(scores.device());
    const Params params{
        .scores = scores.data_ptr<float>(),
        .row_starts = row_starts.data_ptr<int32_t>(),
        .row_ends = row_ends.data_ptr<int32_t>(),
        .output = output.data_ptr<int32_t>(),
        .stride = scores.stride(0),
        .score_width = static_cast<uint32_t>(scores.size(1)),
        .rows = static_cast<uint32_t>(scores.size(0)),
        .topk = static_cast<uint32_t>(k),
        .max_seq_len = static_cast<uint32_t>(max_seq_len),
    };
    const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
    const cudaError_t error = launch_topk(params, k, max_seq_len, stream);
    TORCH_CHECK(error == cudaSuccess,
                "topk_v3_tie_break launch failed: ", cudaGetErrorString(error));
}

}  // namespace torch_ext
