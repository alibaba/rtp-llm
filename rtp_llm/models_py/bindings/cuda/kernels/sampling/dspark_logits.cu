#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/dspark_logits.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

template<typename Bias>
__global__ void prepareDSparkLogitsKernel(const float* __restrict__ base,
                                          const Bias* __restrict__ bias,
                                          const float* __restrict__ temperature,
                                          float* __restrict__ output,
                                          int64_t vocab,
                                          int64_t base_row_stride,
                                          int64_t bias_row_stride) {
    const int64_t row = blockIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < vocab) {
        // Preserve the separate FP32 add and tensor division, including the
        // BF16 GEMM's already-rounded bias. Do not multiply by reciprocal(T).
        const float sum =
            __fadd_rn(base[row * base_row_stride + col], static_cast<float>(bias[row * bias_row_stride + col]));
        output[row * vocab + col] = __fdiv_rn(sum, temperature[row]);
    }
}

}  // namespace

torch::Tensor tryPrepareDSparkLogits(const torch::Tensor& base_logits,
                                     const torch::Tensor& markov_bias,
                                     const torch::Tensor& temperature) {
    if (!base_logits.defined() || !markov_bias.defined() || !temperature.defined() || !base_logits.is_cuda()
        || markov_bias.device() != base_logits.device() || temperature.device() != base_logits.device()
        || base_logits.scalar_type() != torch::kFloat32
        || (markov_bias.scalar_type() != torch::kFloat32 && markov_bias.scalar_type() != torch::kBFloat16)
        || temperature.scalar_type() != torch::kFloat32 || base_logits.dim() != 2 || markov_bias.dim() != 2
        || base_logits.sizes() != markov_bias.sizes() || temperature.dim() != 1
        || temperature.numel() != base_logits.size(0) || !temperature.is_contiguous() || base_logits.stride(1) != 1
        || markov_bias.stride(1) != 1 || base_logits.stride(0) < base_logits.size(1)
        || markov_bias.stride(0) < markov_bias.size(1) || base_logits.size(0) > 65535) {
        return {};
    }
    at::cuda::CUDAGuard guard(base_logits.device());
    auto                output = torch::empty(base_logits.sizes(), base_logits.options());
    if (output.numel() == 0) {
        return output;
    }
    constexpr int threads = 256;
    const dim3    grid((base_logits.size(1) + threads - 1) / threads, base_logits.size(0));
    const auto    stream = at::cuda::getCurrentCUDAStream().stream();
    if (markov_bias.scalar_type() == torch::kBFloat16) {
        prepareDSparkLogitsKernel<<<grid, threads, 0, stream>>>(base_logits.data_ptr<float>(),
                                                                markov_bias.data_ptr<at::BFloat16>(),
                                                                temperature.data_ptr<float>(),
                                                                output.data_ptr<float>(),
                                                                base_logits.size(1),
                                                                base_logits.stride(0),
                                                                markov_bias.stride(0));
    } else {
        prepareDSparkLogitsKernel<<<grid, threads, 0, stream>>>(base_logits.data_ptr<float>(),
                                                                markov_bias.data_ptr<float>(),
                                                                temperature.data_ptr<float>(),
                                                                output.data_ptr<float>(),
                                                                base_logits.size(1),
                                                                base_logits.stride(0),
                                                                markov_bias.stride(0));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace rtp_llm
