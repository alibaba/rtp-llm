#include "rtp_llm/models_py/bindings/cuda/Gemma4RopeOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_rope.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_rope_cos_sin_bf16(const at::Tensor& positions, const at::Tensor& inv_freq) {
    TORCH_CHECK(positions.is_cuda() && inv_freq.is_cuda(), "gemma4_rope_cos_sin_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(positions.scalar_type() == at::kInt, "gemma4_rope_cos_sin_bf16: positions must be int32");
    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "gemma4_rope_cos_sin_bf16: inv_freq must be float32");
    TORCH_CHECK(positions.is_contiguous() && inv_freq.is_contiguous(),
                "gemma4_rope_cos_sin_bf16: inputs must be contiguous");
    TORCH_CHECK(positions.dim() == 1 && inv_freq.dim() == 1 && inv_freq.numel() > 0,
                "gemma4_rope_cos_sin_bf16: expected positions [tokens] and inv_freq [head_dim/2]");
    TORCH_CHECK(inv_freq.numel() <= std::numeric_limits<int32_t>::max() / 2,
                "gemma4_rope_cos_sin_bf16: head dimension exceeds int32 limit");
    TORCH_CHECK(positions.get_device() == inv_freq.get_device(),
                "gemma4_rope_cos_sin_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(positions.device());
    auto                       output_shape   = std::vector<int64_t>{positions.numel(), inv_freq.numel() * 2};
    auto                       output_options = inv_freq.options().dtype(at::kBFloat16);
    auto                       cos            = at::empty(output_shape, output_options);
    auto                       sin            = at::empty(output_shape, output_options);
    rtp_llm::invokeGemma4RopeCosSinBf16(positions.const_data_ptr<int32_t>(),
                                        inv_freq.const_data_ptr<float>(),
                                        reinterpret_cast<__nv_bfloat16*>(cos.mutable_data_ptr<at::BFloat16>()),
                                        reinterpret_cast<__nv_bfloat16*>(sin.mutable_data_ptr<at::BFloat16>()),
                                        positions.numel(),
                                        static_cast<int32_t>(inv_freq.numel()),
                                        at::cuda::getCurrentCUDAStream(positions.get_device()).stream());
    return {cos, sin};
}

std::tuple<at::Tensor, at::Tensor>
gemma4_qk_rope_bf16(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cos, const at::Tensor& sin) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda(),
                "gemma4_qk_rope_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16
                    && cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16,
                "gemma4_qk_rope_bf16: inputs must be bfloat16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
                "gemma4_qk_rope_bf16: inputs must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && cos.dim() == 2 && sin.dim() == 2,
                "gemma4_qk_rope_bf16: q/k must be [T,H,D] and cos/sin [T,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2),
                "gemma4_qk_rope_bf16: q/k token and head dimensions must match");
    TORCH_CHECK(cos.sizes() == sin.sizes() && cos.size(0) == q.size(0) && cos.size(1) == q.size(2),
                "gemma4_qk_rope_bf16: cos/sin shape must be [T,D]");
    TORCH_CHECK(q.size(2) % 2 == 0, "gemma4_qk_rope_bf16: head dimension must be even");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int64_t>::max() / q.size(1) / q.size(2),
                "gemma4_qk_rope_bf16: q size overflow");
    TORCH_CHECK(q.size(1) <= std::numeric_limits<int32_t>::max() && k.size(1) <= std::numeric_limits<int32_t>::max()
                    && q.size(2) <= std::numeric_limits<int32_t>::max(),
                "gemma4_qk_rope_bf16: dimensions exceed int32 limits");
    TORCH_CHECK(q.get_device() == k.get_device() && q.get_device() == cos.get_device()
                    && q.get_device() == sin.get_device(),
                "gemma4_qk_rope_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(q.device());
    auto                       q_out = at::empty_like(q);
    auto                       k_out = at::empty_like(k);
    rtp_llm::invokeGemma4QkRopeBf16(reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr<at::BFloat16>()),
                                    reinterpret_cast<const __nv_bfloat16*>(k.const_data_ptr<at::BFloat16>()),
                                    reinterpret_cast<const __nv_bfloat16*>(cos.const_data_ptr<at::BFloat16>()),
                                    reinterpret_cast<const __nv_bfloat16*>(sin.const_data_ptr<at::BFloat16>()),
                                    reinterpret_cast<__nv_bfloat16*>(q_out.mutable_data_ptr<at::BFloat16>()),
                                    reinterpret_cast<__nv_bfloat16*>(k_out.mutable_data_ptr<at::BFloat16>()),
                                    q.size(0),
                                    static_cast<int32_t>(q.size(1)),
                                    static_cast<int32_t>(k.size(1)),
                                    static_cast<int32_t>(q.size(2)),
                                    at::cuda::getCurrentCUDAStream(q.get_device()).stream());
    return {q_out, k_out};
}

}  // namespace torch_ext
