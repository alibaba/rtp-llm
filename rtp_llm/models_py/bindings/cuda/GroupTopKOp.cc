#include "rtp_llm/models_py/bindings/cuda/GroupTopKOp.h"

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

void GroupTopKOp::forwardFusedSigmoid(torch::Tensor&       topk_values,
                                      torch::Tensor&       topk_indices,
                                      torch::Tensor const& router_logits,
                                      torch::Tensor const& correction_bias,
                                      int64_t              topk,
                                      bool                 renormalize,
                                      double               routed_scaling_factor) {
    TORCH_CHECK(router_logits.is_cuda() && router_logits.is_contiguous(),
                "fused sigmoid top-k requires contiguous CUDA router logits");
    TORCH_CHECK(router_logits.scalar_type() == torch::kFloat32,
                "fused sigmoid top-k requires FP32 router logits");
    TORCH_CHECK(router_logits.dim() == 2 && router_logits.size(1) == 896 && topk == 16,
                "fused sigmoid top-k only supports [tokens, 896] logits and topk=16");
    TORCH_CHECK(router_logits.size(0) > 0, "fused sigmoid top-k requires at least one token");
    TORCH_CHECK(correction_bias.is_cuda() && correction_bias.is_contiguous()
                    && correction_bias.scalar_type() == torch::kFloat32 && correction_bias.numel() == 896,
                "fused sigmoid top-k requires a contiguous FP32 [896] correction bias");
    TORCH_CHECK(correction_bias.get_device() == router_logits.get_device(),
                "fused sigmoid top-k logits and correction bias must use the same device");
    TORCH_CHECK(topk_values.is_cuda() && topk_values.is_contiguous()
                    && topk_values.scalar_type() == torch::kFloat32
                    && topk_values.dim() == 2 && topk_values.size(0) == router_logits.size(0)
                    && topk_values.size(1) == topk,
                "fused sigmoid top-k values must be contiguous CUDA FP32");
    TORCH_CHECK(topk_indices.is_cuda() && topk_indices.is_contiguous(),
                "fused sigmoid top-k indices must be contiguous CUDA tensors");
    TORCH_CHECK(topk_indices.sizes() == topk_values.sizes()
                    && topk_indices.get_device() == router_logits.get_device()
                    && topk_values.get_device() == router_logits.get_device(),
                "fused sigmoid top-k outputs must match [tokens, topk] on the input device");

    int64_t const num_tokens = router_logits.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream(router_logits.get_device());
    switch (topk_indices.scalar_type()) {
        case torch::kInt64:
            invokeFusedSigmoidTopk<int64_t>(reinterpret_cast<float const*>(router_logits.data_ptr()),
                                            reinterpret_cast<float const*>(correction_bias.data_ptr()),
                                            topk_values.mutable_data_ptr<float>(),
                                            topk_indices.mutable_data_ptr<int64_t>(),
                                            num_tokens,
                                            896,
                                            topk,
                                            renormalize,
                                            routed_scaling_factor,
                                            stream);
            break;
        case torch::kInt32:
            invokeFusedSigmoidTopk<int32_t>(reinterpret_cast<float const*>(router_logits.data_ptr()),
                                            reinterpret_cast<float const*>(correction_bias.data_ptr()),
                                            topk_values.mutable_data_ptr<float>(),
                                            topk_indices.mutable_data_ptr<int32_t>(),
                                            num_tokens,
                                            896,
                                            topk,
                                            renormalize,
                                            routed_scaling_factor,
                                            stream);
            break;
        default:
            TORCH_CHECK(false, "fused sigmoid top-k indices must be int32 or int64");
    }
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
        .def("forward_fused_sigmoid",
             &GroupTopKOp::forwardFusedSigmoid,
             pybind11::arg("topk_values"),
             pybind11::arg("topk_indices"),
             pybind11::arg("router_logits"),
             pybind11::arg("correction_bias"),
             pybind11::arg("topk"),
             pybind11::arg("renormalize"),
             pybind11::arg("routed_scaling_factor"));
}
}  // namespace rtp_llm
