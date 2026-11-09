#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <torch/library.h>

namespace {
constexpr int kWidth   = 16384;
constexpr int kLanes   = 32;
constexpr int kWarps   = 16;
constexpr int kThreads = kLanes * kWarps;
struct alignas(8) Input4 {
    c10::BFloat16 x[4];
};

__global__ void
cast_square_sum(const Input4* __restrict__ input, float4* __restrict__ converted, float* __restrict__ squares) {
    const int row     = blockIdx.x;
    const int thread  = threadIdx.y * kLanes + threadIdx.x;
    float     sums[4] = {};
#pragma unroll
    for (int i = 0; i < kWidth / (4 * kThreads); ++i) {
        const int   offset = row * (kWidth / 4) + i * kThreads + thread;
        const auto  v      = input[offset];
        const float x0     = static_cast<float>(v.x[0]);
        const float x1     = static_cast<float>(v.x[1]);
        const float x2     = static_cast<float>(v.x[2]);
        const float x3     = static_cast<float>(v.x[3]);
        converted[offset]  = make_float4(x0, x1, x2, x3);
        sums[0]            = __fadd_rn(sums[0], __fmul_rn(x0, x0));
        sums[1]            = __fadd_rn(sums[1], __fmul_rn(x1, x1));
        sums[2]            = __fadd_rn(sums[2], __fmul_rn(x2, x2));
        sums[3]            = __fadd_rn(sums[3], __fmul_rn(x3, x3));
    }
    // Torch's four input-vector accumulators, then X and Y trees.
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    for (int delta = kLanes / 2; delta; delta /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
    }
    __shared__ float warp_sums[kWarps];
    if (threadIdx.x == 0) {
        warp_sums[threadIdx.y] = sum;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        sum = threadIdx.x < kWarps ? warp_sums[threadIdx.x] : 0.0f;
        for (int delta = kWarps / 2; delta; delta /= 2) {
            sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
        }
        if (threadIdx.x == 0) {
            squares[row] = sum;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> hc_prepare(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2 && input.scalar_type() == at::kBFloat16
                    && input.size(0) >= 1024 && input.size(0) <= 32768 && input.size(1) == kWidth
                    && reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(Input4) == 0,
                "Expected aligned BF16 [M,16384], 1024 <= M <= 32768");
    const c10::cuda::CUDAGuard guard(input.device());
    auto                       converted   = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    auto                       squares     = at::empty({input.size(0)}, converted.options());
    auto                       square_rows = squares.view({input.size(0), 1});
    auto                       iter        = at::TensorIterator::reduce_op(square_rows, converted);
    auto                       config      = at::native::setReduceConfig<float, float, 4, 4>(iter);
    TORCH_CHECK(iter.can_use_32bit_indexing() && config.vectorize_input && config.output_vec_size == 1
                    && config.num_inputs == kWidth && config.num_outputs == input.size(0)
                    && config.block_width == kLanes && config.block_height == kWarps && config.step_input == kThreads
                    && config.step_output == 1 && config.input_mult[0] == 1 && config.input_mult[1] == kLanes
                    && config.ctas_per_output == 1 && !config.should_global_reduce(),
                "Unexpected live Torch reduction geometry");
    cast_square_sum<<<input.size(0), dim3(kLanes, kWarps), 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const Input4*>(input.data_ptr<c10::BFloat16>()),
        reinterpret_cast<float4*>(converted.data_ptr<float>()),
        squares.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {converted, squares};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(rtp_llm_ppu, m) {
    m.def("hc_prepare(Tensor input) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(rtp_llm_ppu, CUDA, m) {
    m.impl("hc_prepare", hc_prepare);
}
