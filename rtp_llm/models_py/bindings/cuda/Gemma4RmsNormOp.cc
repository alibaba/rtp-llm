#include "rtp_llm/models_py/bindings/cuda/Gemma4RmsNormOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_rmsnorm_parts.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace torch_ext {
namespace {

void checkInput(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Gemma4 RMSNorm input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "Gemma4 RMSNorm input must be bfloat16");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "Gemma4 RMSNorm input must be 2-D or 3-D");
    TORCH_CHECK(input.size(-1) > 0 && input.stride(-1) == 1,
                "Gemma4 RMSNorm input must have a contiguous final dimension");
    TORCH_CHECK(input.size(-1) <= std::numeric_limits<int32_t>::max(),
                "Gemma4 RMSNorm hidden size exceeds int32 limit");
}

at::Tensor apply(const at::Tensor& input, const at::Tensor& inv_rms, const float* weight) {
    checkInput(input);
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.scalar_type() == at::kFloat,
                "Gemma4 RMSNorm inv_rms must be a CUDA float32 tensor");
    TORCH_CHECK(inv_rms.is_contiguous(), "Gemma4 RMSNorm inv_rms must be contiguous");
    TORCH_CHECK(inv_rms.get_device() == input.get_device(), "Gemma4 RMSNorm tensors must share a device");
    TORCH_CHECK(inv_rms.numel() == input.numel() / input.size(-1),
                "Gemma4 RMSNorm inv_rms must contain one value per row");

    const int32_t              head_count  = input.dim() == 3 ? static_cast<int32_t>(input.size(1)) : 1;
    const int64_t              head_stride = input.dim() == 3 ? input.stride(1) : input.size(-1);
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty(input.sizes(), input.options());
    rtp_llm::invokeGemma4RmsApplyBf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                      inv_rms.const_data_ptr<float>(),
                                      weight,
                                      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                      input.numel(),
                                      head_count,
                                      input.stride(0),
                                      head_stride,
                                      static_cast<int32_t>(input.size(-1)),
                                      at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

}  // namespace

at::Tensor gemma4_rms_square_bf16(const at::Tensor& input) {
    checkInput(input);
    const int32_t              head_count  = input.dim() == 3 ? static_cast<int32_t>(input.size(1)) : 1;
    const int64_t              head_stride = input.dim() == 3 ? input.stride(1) : input.size(-1);
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    rtp_llm::invokeGemma4RmsSquareBf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                       output.mutable_data_ptr<float>(),
                                       input.numel(),
                                       head_count,
                                       input.stride(0),
                                       head_stride,
                                       static_cast<int32_t>(input.size(-1)),
                                       at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_rms_mean_fp32(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "gemma4_rms_mean_fp32: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "gemma4_rms_mean_fp32: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "gemma4_rms_mean_fp32: input must be contiguous");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "gemma4_rms_mean_fp32: input must be 2-D or 3-D");
    TORCH_CHECK(input.size(-1) == 256 || input.size(-1) == 512 || input.size(-1) == 2816,
                "gemma4_rms_mean_fp32: unsupported final dimension");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output_sizes = input.sizes().vec();
    output_sizes.back()                     = 1;
    auto output                             = at::empty(output_sizes, input.options());
    rtp_llm::invokeGemma4RmsMeanFp32(input.const_data_ptr<float>(),
                                     output.mutable_data_ptr<float>(),
                                     input.numel() / input.size(-1),
                                     static_cast<int32_t>(input.size(-1)),
                                     at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_rms_inv_fp32(const at::Tensor& input, double eps) {
    TORCH_CHECK(input.is_cuda(), "gemma4_rms_inv_fp32: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "gemma4_rms_inv_fp32: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "gemma4_rms_inv_fp32: input must be contiguous");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "gemma4_rms_inv_fp32: input must be 2-D or 3-D");
    TORCH_CHECK(input.size(-1) == 256 || input.size(-1) == 512 || input.size(-1) == 2816,
                "gemma4_rms_inv_fp32: unsupported final dimension");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output_sizes = input.sizes().vec();
    output_sizes.back()                     = 1;
    auto output                             = at::empty(output_sizes, input.options());
    rtp_llm::invokeGemma4RmsInvFp32(input.const_data_ptr<float>(),
                                    output.mutable_data_ptr<float>(),
                                    input.numel() / input.size(-1),
                                    static_cast<int32_t>(input.size(-1)),
                                    static_cast<float>(eps),
                                    at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_rms_apply_bf16(const at::Tensor& input, const at::Tensor& inv_rms, const at::Tensor& weight) {
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat,
                "Gemma4 RMSNorm weight must be a CUDA float32 tensor");
    TORCH_CHECK(weight.is_contiguous(), "Gemma4 RMSNorm weight must be contiguous");
    TORCH_CHECK(weight.get_device() == input.get_device(), "Gemma4 RMSNorm tensors must share a device");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1),
                "Gemma4 RMSNorm weight must match the input's final dimension");
    return apply(input, inv_rms, weight.const_data_ptr<float>());
}

at::Tensor gemma4_rms_apply_unweighted_bf16(const at::Tensor& input, const at::Tensor& inv_rms) {
    return apply(input, inv_rms, nullptr);
}

}  // namespace torch_ext
