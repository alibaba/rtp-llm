// Copyright 2026 Tencent
// HY4 iHC AOT binding adapted from hpc-ops (MIT).

#include "rtp_llm/models_py/bindings/cuda/Hy4IhcOp.h"

#include "rtp_llm/models_py/bindings/cuda/kernels/hy4_ihc.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int kHy4HcMult = 4;

void check_cuda_bf16_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

void check_cuda_f32_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
}

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as channels");
}

void check_hy4_geometry(const at::Tensor& channels) {
    TORCH_CHECK(channels.dim() == 3, "channels must be [M, 4, H]");
    TORCH_CHECK(channels.size(1) == kHy4HcMult, "HY4 iHC requires four residual channels");
    TORCH_CHECK(channels.size(2) == 4096 || channels.size(2) == 6144,
                "HY4 iHC AOT kernel supports hidden size 4096 or 6144, got ", channels.size(2));
}

}  // namespace

namespace torch_ext {

at::Tensor fuse_hy4_ihc_head(const at::Tensor& channels,
                             const at::Tensor& fn_weight,
                             const at::Tensor& scale,
                             const at::Tensor& base,
                             double norm_eps,
                             double hc_eps) {
    check_cuda_bf16_contiguous(channels, "channels");
    check_cuda_f32_contiguous(fn_weight, "fn_weight");
    check_cuda_f32_contiguous(scale, "scale");
    check_cuda_f32_contiguous(base, "base");
    check_same_device(channels, fn_weight, "fn_weight");
    check_same_device(channels, scale, "scale");
    check_same_device(channels, base, "base");
    check_hy4_geometry(channels);

    const int64_t m = channels.size(0);
    const int64_t hidden = channels.size(2);
    TORCH_CHECK(fn_weight.sizes() == at::IntArrayRef({kHy4HcMult, kHy4HcMult * hidden}),
                "head fn_weight must be [4, 4 * H]");
    TORCH_CHECK(scale.numel() == 1, "head scale must contain one value");
    TORCH_CHECK(base.numel() == kHy4HcMult, "head base must contain four values");

    auto output = torch::empty({m, hidden}, channels.options());
    if (m == 0) {
        return output;
    }

    at::cuda::CUDAGuard device_guard(channels.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(channels.get_device()).stream();
    rtp_llm::hy4_ihc::fuse_ihc_head_async(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(channels.const_data_ptr()),
        fn_weight.const_data_ptr<float>(),
        scale.const_data_ptr<float>(),
        base.const_data_ptr<float>(),
        static_cast<int>(m),
        kHy4HcMult,
        static_cast<int>(hidden),
        static_cast<float>(norm_eps),
        static_cast<float>(hc_eps),
        stream,
        nullptr,
        0.0F,
        false,
        true);
    return output;
}

}  // namespace torch_ext
