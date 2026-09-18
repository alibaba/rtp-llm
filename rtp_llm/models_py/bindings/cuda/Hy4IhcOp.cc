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

std::tuple<at::Tensor, at::Tensor, at::Tensor> fuse_hy4_ihc_post_pre(
    const at::Tensor& block_output,
    const at::Tensor& residual,
    const at::Tensor& post_gate,
    const at::Tensor& next_fn_weight,
    const at::Tensor& next_scale,
    const at::Tensor& next_base,
    double ihc_norm_eps,
    double hc_eps,
    double magnitude,
    const at::Tensor& rms_weight,
    double rms_eps,
    bool cast_bfloat_for_norm) {
    check_cuda_bf16_contiguous(block_output, "block_output");
    check_cuda_bf16_contiguous(residual, "residual");
    check_cuda_f32_contiguous(post_gate, "post_gate");
    check_cuda_f32_contiguous(next_fn_weight, "next_fn_weight");
    check_cuda_f32_contiguous(next_scale, "next_scale");
    check_cuda_f32_contiguous(next_base, "next_base");
    check_cuda_bf16_contiguous(rms_weight, "rms_weight");
    check_same_device(block_output, residual, "residual");
    check_same_device(block_output, post_gate, "post_gate");
    check_same_device(block_output, next_fn_weight, "next_fn_weight");
    check_same_device(block_output, next_scale, "next_scale");
    check_same_device(block_output, next_base, "next_base");
    check_same_device(block_output, rms_weight, "rms_weight");

    TORCH_CHECK(block_output.dim() == 2, "block_output must be [M, H]");
    check_hy4_geometry(residual);
    const int64_t m = residual.size(0);
    const int64_t hidden = residual.size(2);
    TORCH_CHECK(block_output.sizes() == at::IntArrayRef({m, hidden}),
                "block_output must match residual batch and hidden dimensions");
    TORCH_CHECK(post_gate.sizes() == at::IntArrayRef({m, kHy4HcMult}),
                "post_gate must be [M, 4]");
    TORCH_CHECK(next_fn_weight.sizes() == at::IntArrayRef({2 * kHy4HcMult, kHy4HcMult * hidden}),
                "next_fn_weight must be [8, 4 * H]");
    TORCH_CHECK(next_scale.numel() == 2, "next_scale must contain two values");
    TORCH_CHECK(next_base.numel() == 2 * kHy4HcMult, "next_base must contain eight values");
    TORCH_CHECK(rms_weight.dim() == 1 && rms_weight.size(0) == hidden,
                "rms_weight must be [H]");

    auto output_channels = torch::empty({m, kHy4HcMult, hidden}, residual.options());
    auto next_input = torch::empty({m, hidden}, block_output.options());
    auto next_post_gate = torch::empty({m, kHy4HcMult}, post_gate.options());
    if (m == 0) {
        return std::make_tuple(output_channels, next_input, next_post_gate);
    }

    const size_t scratch_floats = rtp_llm::hy4_ihc::ihc_post_pre_scratch_floats(
        static_cast<int>(m), kHy4HcMult, static_cast<int>(hidden));
    auto scratch = torch::empty({static_cast<int64_t>(scratch_floats)}, post_gate.options());
    auto* scratch_ptr = scratch_floats == 0 ? nullptr : scratch.data_ptr<float>();

    at::cuda::CUDAGuard device_guard(block_output.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(block_output.get_device()).stream();
    rtp_llm::hy4_ihc::fuse_ihc_post_pre_async(
        reinterpret_cast<__nv_bfloat16*>(output_channels.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(next_input.data_ptr()),
        next_post_gate.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(block_output.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
        post_gate.const_data_ptr<float>(),
        next_fn_weight.const_data_ptr<float>(),
        next_scale.const_data_ptr<float>(),
        next_base.const_data_ptr<float>(),
        static_cast<int>(m),
        kHy4HcMult,
        static_cast<int>(hidden),
        static_cast<float>(ihc_norm_eps),
        static_cast<float>(hc_eps),
        static_cast<float>(magnitude),
        scratch_ptr,
        stream,
        reinterpret_cast<const __nv_bfloat16*>(rms_weight.const_data_ptr()),
        static_cast<float>(rms_eps),
        cast_bfloat_for_norm,
        true);
    return std::make_tuple(output_channels, next_input, next_post_gate);
}

}  // namespace torch_ext
