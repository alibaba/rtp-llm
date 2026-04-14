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
            throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
            break;
    }
    return;
}

void GroupTopKOp::forward_fused(torch::Tensor&       topk_values,
                                torch::Tensor&       topk_indices,
                                torch::Tensor&       raw_logits,
                                torch::Tensor const& bias,
                                torch::Tensor&       scores_out,
                                int64_t              n_group,
                                int64_t              topk_group,
                                int64_t              topk,
                                bool                 renormalize,
                                double               routed_scaling_factor) {
    auto    input_size  = raw_logits.sizes();
    int64_t num_tokens  = input_size[0];
    int64_t num_experts = input_size[1];

    torch::Tensor group_scores =
        torch::empty({num_tokens, n_group}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto stream = c10::cuda::getCurrentCUDAStream(raw_logits.get_device());

    switch (topk_indices.scalar_type()) {
        case torch::kInt64:
            invokeNoAuxTcFused<float, int64_t>(reinterpret_cast<float*>(raw_logits.mutable_data_ptr()),
                                               reinterpret_cast<float const*>(bias.data_ptr()),
                                               reinterpret_cast<float*>(scores_out.mutable_data_ptr()),
                                               reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                               reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                                               reinterpret_cast<int64_t*>(topk_indices.mutable_data_ptr()),
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
            invokeNoAuxTcFused<float, int32_t>(reinterpret_cast<float*>(raw_logits.mutable_data_ptr()),
                                               reinterpret_cast<float const*>(bias.data_ptr()),
                                               reinterpret_cast<float*>(scores_out.mutable_data_ptr()),
                                               reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
                                               reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
                                               reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
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
            throw std::invalid_argument("Invalid dtype, only supports int32 and int64");
            break;
    }
    return;
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
             &GroupTopKOp::forward_fused,
             pybind11::arg("topk_values"),
             pybind11::arg("topk_indices"),
             pybind11::arg("raw_logits"),
             pybind11::arg("bias"),
             pybind11::arg("scores_out"),
             pybind11::arg("n_group"),
             pybind11::arg("topk_group"),
             pybind11::arg("topk"),
             pybind11::arg("renormalize"),
             pybind11::arg("routed_scaling_factor"));
}
}  // namespace rtp_llm
