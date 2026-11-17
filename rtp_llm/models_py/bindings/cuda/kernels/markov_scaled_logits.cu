#include "rtp_llm/models_py/bindings/cuda/kernels/markov_scaled_logits.h"

#include <cuda_runtime.h>

#if !defined(USE_ROCM) && CUDART_VERSION >= 12090
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

namespace torch_ext {

#if !defined(USE_ROCM) && CUDART_VERSION >= 12090
namespace {
template<typename BiasT>
__global__ void markovScaledLogitsKernel(const float* __restrict__ logits,
                                         int64_t logits_row_stride,
                                         const BiasT* __restrict__ bias,
                                         const float* __restrict__ temperature,
                                         float* __restrict__ output,
                                         int vocab_size) {
    const int     row           = blockIdx.y;
    const int     col           = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int64_t offset        = static_cast<int64_t>(row) * vocab_size + col;
    const int64_t logits_offset = static_cast<int64_t>(row) * logits_row_stride + col;
    const float   scale         = temperature[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (col + i < vocab_size) {
            output[offset + i] = (logits[logits_offset + i] + static_cast<float>(bias[offset + i])) / scale;
        }
    }
}
}  // namespace
#endif

torch::Tensor
markov_scaled_logits(const torch::Tensor& logits, const torch::Tensor& markov_bias, const torch::Tensor& temperature) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
                    && logits.stride(1) == 1 && markov_bias.is_cuda() && markov_bias.is_contiguous()
                    && markov_bias.sizes() == logits.sizes()
                    && (markov_bias.scalar_type() == torch::kFloat32 || markov_bias.scalar_type() == torch::kBFloat16
                        || markov_bias.scalar_type() == torch::kFloat16)
                    && temperature.is_cuda() && temperature.scalar_type() == torch::kFloat32
                    && temperature.is_contiguous() && temperature.dim() == 1 && temperature.size(0) == logits.size(0)
                    && logits.device() == markov_bias.device() && logits.device() == temperature.device(),
                "DSpARK scaled logits require same-device FP32 logits [B,V], contiguous bias [B,V]"
                " in FP32/BF16/FP16, and contiguous FP32 temperature [B]");
#if !defined(USE_ROCM) && CUDART_VERSION >= 12090
    auto output = torch::empty(logits.sizes(), logits.options());
    if (logits.numel() == 0) {
        return output;
    }
    const c10::cuda::CUDAGuard guard(logits.device());
    constexpr int              threads = 256;
    const dim3                 grid((logits.size(1) + threads * 4 - 1) / (threads * 4), logits.size(0));
    auto                       stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    if (markov_bias.scalar_type() == torch::kFloat32) {
        markovScaledLogitsKernel<float><<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                                                      logits.stride(0),
                                                                      markov_bias.data_ptr<float>(),
                                                                      temperature.data_ptr<float>(),
                                                                      output.data_ptr<float>(),
                                                                      logits.size(1));
    } else if (markov_bias.scalar_type() == torch::kBFloat16) {
        markovScaledLogitsKernel<__nv_bfloat16>
            <<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                           logits.stride(0),
                                           reinterpret_cast<const __nv_bfloat16*>(markov_bias.data_ptr()),
                                           temperature.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           logits.size(1));
    } else {
        markovScaledLogitsKernel<__half>
            <<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                           logits.stride(0),
                                           reinterpret_cast<const __half*>(markov_bias.data_ptr()),
                                           temperature.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           logits.size(1));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#else
    return (logits + markov_bias.to(torch::kFloat32)).div(temperature.unsqueeze(1));
#endif
}

}  // namespace torch_ext
