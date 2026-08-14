#include "rtp_llm/models_py/bindings/cuda/GroupTopKOp.h"

#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {

GroupTopKOp::GroupTopKOp() {}

void GroupTopKOp::forward(torch::Tensor&       topk_values,
                          torch::Tensor&       topk_indices,
                          torch::Tensor const& scores,
                          torch::Tensor const& scores_with_bias,
                          int64_t              n_group,
                          int64_t              topk_group,
                          int64_t              topk,
                          bool                 renormalize,
                          double               routed_scaling_factor) {
    auto    data_type   = scores_with_bias.scalar_type();
    auto    input_size  = scores_with_bias.sizes();
    int64_t num_tokens  = input_size[0];
    int64_t num_experts = input_size[1];

    torch::Tensor group_scores =
        torch::empty({num_tokens, n_group}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto stream = c10::cuda::getCurrentCUDAStream(scores_with_bias.get_device());

    switch (topk_indices.scalar_type()) {
        case torch::kInt64:
            invokeNoAuxTc<float, int64_t>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
                                          reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                                          reinterpret_cast<int64_t*>(topk_indices.mutable_data_ptr()),
                                          reinterpret_cast<float*>(scores_with_bias.data_ptr()),
                                          num_tokens,
                                          num_experts,
                                          n_group,
                                          topk_group,
                                          topk,
                                          renormalize,
                                          routed_scaling_factor,
                                          stream);
            break;
        case torch::kInt32:
            invokeNoAuxTc<float, int32_t>(reinterpret_cast<float*>(scores.mutable_data_ptr()),
                                          reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                                          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
                                          reinterpret_cast<float*>(scores_with_bias.data_ptr()),
                                          num_tokens,
                                          num_experts,
                                          n_group,
                                          topk_group,
                                          topk,
                                          renormalize,
                                          routed_scaling_factor,
                                          stream);
            break;
        default:
            // Handle other data types
            throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
            break;
    }
    return;
}

void GroupTopKOp::forwardFused(torch::Tensor&       topk_values,
                               torch::Tensor&       topk_indices,
                               torch::Tensor const& logits,
                               torch::Tensor const& correction_bias,
                               int64_t              n_group,
                               int64_t              topk_group,
                               int64_t              topk,
                               bool                 renormalize,
                               double               routed_scaling_factor) {
    TORCH_CHECK(logits.is_cuda() && correction_bias.is_cuda(), "GroupTopK fused inputs must be CUDA tensors");
    TORCH_CHECK(topk_values.is_cuda() && topk_indices.is_cuda(), "GroupTopK fused outputs must be CUDA tensors");
    TORCH_CHECK(logits.is_contiguous() && correction_bias.is_contiguous(), "GroupTopK fused inputs must be contiguous");
    TORCH_CHECK(topk_values.is_contiguous() && topk_indices.is_contiguous(),
                "GroupTopK fused outputs must be contiguous");
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == 256,
                "GroupTopK fused path requires logits shaped [num_tokens, 256]");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16,
                "GroupTopK fused path requires BF16 logits; unsupported dtypes must use the legacy path");
    TORCH_CHECK(correction_bias.dim() == 1 && correction_bias.numel() == 256
                    && correction_bias.scalar_type() == torch::kFloat32,
                "GroupTopK fused path requires a contiguous FP32 correction bias with 256 elements");
    TORCH_CHECK((n_group == 8 && topk_group == 4 && topk == 8)
                    || (n_group == 1 && topk_group == 1 && topk == 8),
                "GroupTopK fused path requires (n_group, topk_group, topk)=(8,4,8) or (1,1,8)");
    TORCH_CHECK(topk_values.scalar_type() == torch::kFloat32 && topk_values.dim() == 2
                    && topk_values.size(0) == logits.size(0) && topk_values.size(1) == topk,
                "GroupTopK fused weights must be contiguous FP32 [num_tokens, topk]");
    TORCH_CHECK(topk_indices.dim() == 2 && topk_indices.size(0) == logits.size(0) && topk_indices.size(1) == topk,
                "GroupTopK fused indices must be contiguous [num_tokens, topk]");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32 || topk_indices.scalar_type() == torch::kInt64,
                "GroupTopK fused indices must use int32 or int64");
    TORCH_CHECK(logits.get_device() == correction_bias.get_device() && logits.get_device() == topk_values.get_device()
                    && logits.get_device() == topk_indices.get_device(),
                "GroupTopK fused inputs and outputs must be on the same CUDA device");

    const int64_t num_tokens = logits.size(0);
    if (num_tokens == 0) {
        return;
    }
#ifdef ENABLE_BF16
    c10::cuda::CUDAGuard device_guard(logits.device());
    auto                 stream     = c10::cuda::getCurrentCUDAStream(logits.get_device());
    const auto*          logits_ptr = reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>());
    const float*         bias_ptr   = correction_bias.data_ptr<float>();
    float*               values_ptr = topk_values.data_ptr<float>();

    switch (topk_indices.scalar_type()) {
        case torch::kInt64:
            if (n_group == 1) {
                invokeFusedNoAuxTcSingleGroup<__nv_bfloat16, int64_t>(logits_ptr,
                                                                      bias_ptr,
                                                                      values_ptr,
                                                                      topk_indices.data_ptr<int64_t>(),
                                                                      num_tokens,
                                                                      renormalize,
                                                                      routed_scaling_factor,
                                                                      stream);
            } else {
                invokeFusedNoAuxTc<__nv_bfloat16, int64_t>(logits_ptr,
                                                           bias_ptr,
                                                           values_ptr,
                                                           topk_indices.data_ptr<int64_t>(),
                                                           num_tokens,
                                                           renormalize,
                                                           routed_scaling_factor,
                                                           stream);
            }
            break;
        case torch::kInt32:
            if (n_group == 1) {
                invokeFusedNoAuxTcSingleGroup<__nv_bfloat16, int32_t>(logits_ptr,
                                                                      bias_ptr,
                                                                      values_ptr,
                                                                      topk_indices.data_ptr<int32_t>(),
                                                                      num_tokens,
                                                                      renormalize,
                                                                      routed_scaling_factor,
                                                                      stream);
            } else {
                invokeFusedNoAuxTc<__nv_bfloat16, int32_t>(logits_ptr,
                                                           bias_ptr,
                                                           values_ptr,
                                                           topk_indices.data_ptr<int32_t>(),
                                                           num_tokens,
                                                           renormalize,
                                                           routed_scaling_factor,
                                                           stream);
            }
            break;
        default:
            TORCH_CHECK(false, "GroupTopK fused indices must use int32 or int64");
    }
#else
    TORCH_CHECK(false, "GroupTopK fused path requires ENABLE_BF16");
#endif
}

void registerGroupTopKOp(const pybind11::module& m) {
    pybind11::class_<GroupTopKOp>(m, "GroupTopKOp")
        .def(pybind11::init<>())
        .def("forward",
             &GroupTopKOp::forward,
             pybind11::arg("topk_values"),
             pybind11::arg("topk_indices"),
             pybind11::arg("scores"),
             pybind11::arg("scores_with_bias"),
             pybind11::arg("n_group"),
             pybind11::arg("topk_group"),
             pybind11::arg("topk"),
             pybind11::arg("renormalize"),
             pybind11::arg("routed_scaling_factor"))
        .def("forward_fused",
             &GroupTopKOp::forwardFused,
             pybind11::arg("topk_values"),
             pybind11::arg("topk_indices"),
             pybind11::arg("logits"),
             pybind11::arg("correction_bias"),
             pybind11::arg("n_group"),
             pybind11::arg("topk_group"),
             pybind11::arg("topk"),
             pybind11::arg("renormalize"),
             pybind11::arg("routed_scaling_factor"));
}
}  // namespace rtp_llm
