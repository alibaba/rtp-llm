#include "rtp_llm/models_py/bindings/cuda/Gemma4AddScaleOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_add_scale.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace torch_ext {

at::Tensor gemma4_add_bf16(const at::Tensor& lhs, const at::Tensor& rhs) {
    TORCH_CHECK(lhs.is_cuda() && rhs.is_cuda(), "gemma4_add_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(lhs.scalar_type() == at::kBFloat16 && rhs.scalar_type() == at::kBFloat16,
                "gemma4_add_bf16: inputs must be bfloat16");
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous(), "gemma4_add_bf16: inputs must be contiguous");
    TORCH_CHECK(lhs.sizes() == rhs.sizes(), "gemma4_add_bf16: tensor shapes must match");
    TORCH_CHECK(lhs.numel() % 8 == 0, "gemma4_add_bf16: element count must be divisible by 8");
    TORCH_CHECK(lhs.get_device() == rhs.get_device(), "gemma4_add_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(lhs.device());
    auto                       output = at::empty_like(lhs);
    rtp_llm::invokeGemma4AddBf16(reinterpret_cast<const __nv_bfloat16*>(lhs.const_data_ptr<at::BFloat16>()),
                                 reinterpret_cast<const __nv_bfloat16*>(rhs.const_data_ptr<at::BFloat16>()),
                                 reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                 lhs.numel(),
                                 at::cuda::getCurrentCUDAStream(lhs.get_device()).stream());
    return output;
}

at::Tensor gemma4_scale_bf16(const at::Tensor& input, double scale) {
    TORCH_CHECK(input.is_cuda(), "gemma4_scale_bf16: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_scale_bf16: input must be bfloat16");
    TORCH_CHECK(input.is_contiguous(), "gemma4_scale_bf16: input must be contiguous");
    TORCH_CHECK(input.numel() % 8 == 0, "gemma4_scale_bf16: element count must be divisible by 8");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty_like(input);
    rtp_llm::invokeGemma4ScaleBf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                   static_cast<float>(scale),
                                   reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                   input.numel(),
                                   at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_add_scale_bf16(const at::Tensor& residual, const at::Tensor& hidden, const at::Tensor& scale) {
    TORCH_CHECK(residual.is_cuda() && hidden.is_cuda() && scale.is_cuda(),
                "gemma4_add_scale_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && hidden.scalar_type() == at::kBFloat16
                    && scale.scalar_type() == at::kBFloat16,
                "gemma4_add_scale_bf16: inputs must be bfloat16");
    TORCH_CHECK(residual.is_contiguous() && hidden.is_contiguous() && scale.is_contiguous(),
                "gemma4_add_scale_bf16: inputs must be contiguous");
    TORCH_CHECK(residual.sizes() == hidden.sizes(), "gemma4_add_scale_bf16: tensor shapes must match");
    TORCH_CHECK(residual.numel() % 8 == 0, "gemma4_add_scale_bf16: element count must be divisible by 8");
    TORCH_CHECK(scale.numel() == 1, "gemma4_add_scale_bf16: scale must contain one value");
    TORCH_CHECK(residual.get_device() == hidden.get_device() && residual.get_device() == scale.get_device(),
                "gemma4_add_scale_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(residual.device());
    auto                       output = at::empty_like(residual);
    rtp_llm::invokeGemma4AddScaleBf16(reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr<at::BFloat16>()),
                                      reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr<at::BFloat16>()),
                                      reinterpret_cast<const __nv_bfloat16*>(scale.const_data_ptr<at::BFloat16>()),
                                      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                      residual.numel(),
                                      at::cuda::getCurrentCUDAStream(residual.get_device()).stream());
    return output;
}

at::Tensor gemma4_router_scale_bf16(const at::Tensor& input, const at::Tensor& channel_scale, double scalar_scale) {
    TORCH_CHECK(input.is_cuda() && channel_scale.is_cuda(), "gemma4_router_scale_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && channel_scale.scalar_type() == at::kBFloat16,
                "gemma4_router_scale_bf16: inputs must be bfloat16");
    TORCH_CHECK(input.is_contiguous() && channel_scale.is_contiguous(),
                "gemma4_router_scale_bf16: inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && channel_scale.dim() == 1 && input.size(1) == channel_scale.numel(),
                "gemma4_router_scale_bf16: expected input [tokens,hidden] and scale [hidden]");
    TORCH_CHECK(input.size(1) % 8 == 0, "gemma4_router_scale_bf16: hidden size must be divisible by 8");
    TORCH_CHECK(input.size(1) <= std::numeric_limits<int32_t>::max(),
                "gemma4_router_scale_bf16: hidden size exceeds int32 limit");
    TORCH_CHECK(input.get_device() == channel_scale.get_device(),
                "gemma4_router_scale_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty_like(input);
    rtp_llm::invokeGemma4RouterScaleBf16(
        reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(channel_scale.const_data_ptr<at::BFloat16>()),
        static_cast<float>(scalar_scale),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
        input.numel(),
        static_cast<int32_t>(input.size(1)),
        at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

}  // namespace torch_ext
