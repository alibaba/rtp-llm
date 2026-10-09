#include "rtp_llm/models_py/bindings/cuda/Gemma4ExpandKvOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_expand_kv.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_expand_kv_heads_8_bf16(const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "gemma4_expand_kv_heads_8_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "gemma4_expand_kv_heads_8_bf16: inputs must be bfloat16");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "gemma4_expand_kv_heads_8_bf16: inputs must be contiguous");
    TORCH_CHECK(k.dim() == 3 && k.size(1) == 2 && k.size(2) == 512,
                "gemma4_expand_kv_heads_8_bf16: k must be [T,2,512]");
    TORCH_CHECK(v.sizes() == k.sizes(), "gemma4_expand_kv_heads_8_bf16: v shape must match k");
    TORCH_CHECK(k.get_device() == v.get_device(), "gemma4_expand_kv_heads_8_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(k.device());
    auto                       expanded_k = at::empty({k.size(0), 16, 512}, k.options());
    auto                       expanded_v = at::empty({v.size(0), 16, 512}, v.options());
    rtp_llm::invokeGemma4ExpandKvHeads8Bf16(
        reinterpret_cast<const __nv_bfloat16*>(k.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(expanded_k.mutable_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(expanded_v.mutable_data_ptr<at::BFloat16>()),
        k.size(0),
        at::cuda::getCurrentCUDAStream(k.get_device()).stream());
    return {expanded_k, expanded_v};
}

std::tuple<at::Tensor, at::Tensor> gemma4_expand_kv_heads_2_bf16(const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "gemma4_expand_kv_heads_2_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "gemma4_expand_kv_heads_2_bf16: inputs must be bfloat16");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "gemma4_expand_kv_heads_2_bf16: inputs must be contiguous");
    TORCH_CHECK(k.dim() == 3 && k.size(1) == 8 && k.size(2) == 256,
                "gemma4_expand_kv_heads_2_bf16: k must be [T,8,256]");
    TORCH_CHECK(v.sizes() == k.sizes(), "gemma4_expand_kv_heads_2_bf16: v shape must match k");
    TORCH_CHECK(k.get_device() == v.get_device(), "gemma4_expand_kv_heads_2_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(k.device());
    auto                       expanded_k = at::empty({k.size(0), 16, 256}, k.options());
    auto                       expanded_v = at::empty({v.size(0), 16, 256}, v.options());
    rtp_llm::invokeGemma4ExpandKvHeads2Bf16(
        reinterpret_cast<const __nv_bfloat16*>(k.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(expanded_k.mutable_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(expanded_v.mutable_data_ptr<at::BFloat16>()),
        k.size(0),
        at::cuda::getCurrentCUDAStream(k.get_device()).stream());
    return {expanded_k, expanded_v};
}

}  // namespace torch_ext
