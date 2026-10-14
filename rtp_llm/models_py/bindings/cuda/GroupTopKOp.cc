#include "rtp_llm/models_py/bindings/cuda/GroupTopKOp.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace rtp_llm {

namespace {

#define DISPATCH_GROUP_TOPK(LOGITS_CTYPE, BIAS_CTYPE)                                                            \
    invokeNoAuxTc<LOGITS_CTYPE, BIAS_CTYPE, IdxT>(reinterpret_cast<const LOGITS_CTYPE*>(router_logits.data_ptr()), \
                                                  reinterpret_cast<float*>(group_scores.mutable_data_ptr()),      \
                                                  reinterpret_cast<float*>(topk_values.mutable_data_ptr()),        \
                                                  reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),        \
                                                  reinterpret_cast<const BIAS_CTYPE*>(correction_bias.data_ptr()), \
                                                  num_tokens,                                                    \
                                                  num_experts,                                                   \
                                                  n_group,                                                       \
                                                  topk_group,                                                    \
                                                  topk,                                                          \
                                                  renormalize,                                                   \
                                                  routed_scaling_factor,                                         \
                                                  stream)

template<typename IdxT>
void invokeGroupTopKByLogitDtype(torch::Tensor const& router_logits,
                                 torch::Tensor const& correction_bias,
                                 torch::Tensor&       group_scores,
                                 torch::Tensor&       topk_values,
                                 torch::Tensor&       topk_indices,
                                 int64_t              num_tokens,
                                 int64_t              num_experts,
                                 int64_t              n_group,
                                 int64_t              topk_group,
                                 int64_t              topk,
                                 bool                 renormalize,
                                 double               routed_scaling_factor) {
    auto stream = c10::cuda::getCurrentCUDAStream(router_logits.get_device());
    switch (router_logits.scalar_type()) {
        case torch::kFloat32:
            TORCH_CHECK(correction_bias.scalar_type() == torch::kFloat32,
                        "GroupTopKOp supports a float32 correction bias for float32 router_logits, got ",
                        correction_bias.scalar_type());
            DISPATCH_GROUP_TOPK(float, float);
            break;
        case torch::kHalf:
            TORCH_CHECK(
                correction_bias.scalar_type() == torch::kFloat32 || correction_bias.scalar_type() == torch::kHalf,
                "GroupTopKOp supports fp32/fp16 correction bias for fp16 router_logits, got ",
                correction_bias.scalar_type());
            if (correction_bias.scalar_type() == torch::kFloat32) {
                DISPATCH_GROUP_TOPK(half, float);
            } else {
                DISPATCH_GROUP_TOPK(half, half);
            }
            break;
        case torch::kBFloat16:
            TORCH_CHECK(
                correction_bias.scalar_type() == torch::kFloat32 || correction_bias.scalar_type() == torch::kBFloat16,
                "GroupTopKOp supports fp32/bf16 correction bias for bf16 router_logits, got ",
                correction_bias.scalar_type());
            if (correction_bias.scalar_type() == torch::kFloat32) {
                DISPATCH_GROUP_TOPK(__nv_bfloat16, float);
            } else {
                DISPATCH_GROUP_TOPK(__nv_bfloat16, __nv_bfloat16);
            }
            break;
        default:
            // Handle other data types
            throw std::invalid_argument("Invalid router_logits dtype, only supports float32, float16, and bfloat16");
            break;
    }
#undef DISPATCH_GROUP_TOPK
}

}  // namespace

GroupTopKOp::GroupTopKOp() {}

void GroupTopKOp::forward(torch::Tensor&       topk_values,
                          torch::Tensor&       topk_indices,
                          torch::Tensor const& router_logits,
                          torch::Tensor const& correction_bias,
                          int64_t              n_group,
                          int64_t              topk_group,
                          int64_t              topk,
                          bool                 renormalize,
                          double               routed_scaling_factor) {
    auto    input_size  = router_logits.sizes();
    int64_t num_tokens  = input_size[0];
    int64_t num_experts = input_size[1];

    // The kernels read the bias with raw per-expert indexing; keep the tensor
    // shape contract explicit instead of reading out of bounds.
    TORCH_CHECK(correction_bias.numel() == num_experts,
                "GroupTopKOp correction_bias must have ",
                num_experts,
                " elements, got ",
                correction_bias.numel());

    // No-op aten-wise for the contiguous gate output / checkpoint bias that
    // every real caller passes; matches SelectTopkOp's convention and keeps
    // raw-pointer reads correct for exotic strided inputs.
    torch::Tensor logits = router_logits.contiguous();
    torch::Tensor bias   = correction_bias.contiguous();

    torch::Tensor group_scores =
        torch::empty({num_tokens, n_group}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    switch (topk_indices.scalar_type()) {
        case torch::kInt64:
            invokeGroupTopKByLogitDtype<int64_t>(logits,
                                                 bias,
                                                 group_scores,
                                                 topk_values,
                                                 topk_indices,
                                                 num_tokens,
                                                 num_experts,
                                                 n_group,
                                                 topk_group,
                                                 topk,
                                                 renormalize,
                                                 routed_scaling_factor);
            break;
        case torch::kInt32:
            invokeGroupTopKByLogitDtype<int32_t>(logits,
                                                 bias,
                                                 group_scores,
                                                 topk_values,
                                                 topk_indices,
                                                 num_tokens,
                                                 num_experts,
                                                 n_group,
                                                 topk_group,
                                                 topk,
                                                 renormalize,
                                                 routed_scaling_factor);
            break;
        default:
            // Handle other data types
            throw std::invalid_argument("Invalid dtype, only supports int32 and int64 topk_indices");
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
             pybind11::arg("router_logits"),
             pybind11::arg("correction_bias"),
             pybind11::arg("n_group"),
             pybind11::arg("topk_group"),
             pybind11::arg("topk"),
             pybind11::arg("renormalize"),
             pybind11::arg("routed_scaling_factor"));
}
}  // namespace rtp_llm
