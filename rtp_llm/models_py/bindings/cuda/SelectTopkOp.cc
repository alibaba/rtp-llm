#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

SelectTopkOp::SelectTopkOp(const ModelConfig& model_config, bool use_fused_512):
    expert_num_(model_config.expert_num),
    moe_k_(model_config.moe_k),
    has_moe_norm_(model_config.has_moe_norm),
    use_fused_512_(use_fused_512 && model_config.expert_num == 512) {}

void SelectTopkOp::forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales) {
    if (router_logits.scalar_type() == torch::kBFloat16) {
        TORCH_CHECK(use_fused_512_ && expert_num_ == 512 && moe_k_ == 10 && has_moe_norm_,
                    "BF16 routing requires fused E=512, K=10 and normalization");
        TORCH_CHECK(router_logits.is_cuda() && router_logits.dim() == 2 && router_logits.size(1) == 512
                        && router_logits.is_contiguous(),
                    "BF16 router_logits must be contiguous CUDA [T, 512]");
        const auto rows = router_logits.size(0);
        TORCH_CHECK(expert_ids.device() == router_logits.device() && expert_scales.device() == router_logits.device(),
                    "Routing outputs must be on the input device");
        TORCH_CHECK(expert_ids.dim() == 2 && expert_ids.size(0) == rows && expert_ids.size(1) == 10
                        && expert_scales.sizes() == expert_ids.sizes() && expert_ids.is_contiguous()
                        && expert_scales.is_contiguous(),
                    "Routing outputs must be contiguous [T, 10]");
        TORCH_CHECK(expert_scales.scalar_type() == torch::kFloat32, "Routing weights must be float32");
        TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32 || expert_ids.scalar_type() == torch::kInt64,
                    "Routing IDs must be int32 or int64");
        StreamType current_stream = GET_CURRENT_STREAM();
        if (expert_ids.scalar_type() == torch::kInt64) {
            tensorrt_llm::kernels::invokeSelectExpertsForTokensBf16<int64_t>(router_logits.data_ptr(),
                                                                             expert_scales.data_ptr<float>(),
                                                                             expert_ids.data_ptr<int64_t>(),
                                                                             rows,
                                                                             current_stream);
        } else {
            tensorrt_llm::kernels::invokeSelectExpertsForTokensBf16<int32_t>(router_logits.data_ptr(),
                                                                             expert_scales.data_ptr<float>(),
                                                                             expert_ids.data_ptr<int32_t>(),
                                                                             rows,
                                                                             current_stream);
        }
        return;
    }
    const auto token_num = router_logits.sizes()[0];
    if (token_num == 0) {
        return;
    }
    const auto num_expert     = expert_num_;
    const auto top_k          = moe_k_;
    auto normalization_mode   = has_moe_norm_ ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                                tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    auto topk_t               = expert_ids.dtype();
    const auto softmax_out    = use_fused_512_ ?
                                    torch::Tensor() :
                                    torch::empty({token_num, num_expert}, router_logits.options().dtype(torch::kFloat32));
    const auto source_rows    = torch::empty({token_num, top_k}, router_logits.options().dtype(torch::kInt32));
    StreamType current_stream = GET_CURRENT_STREAM();
    router_logits             = router_logits.contiguous();
    if (topk_t == torch::kInt64) {
        tensorrt_llm::kernels::invokeSelectExpertsForTokens<int64_t>(router_logits.data_ptr<float>(),
                                                                     router_logits.data_ptr<float>(),
                                                                     expert_scales.data_ptr<float>(),
                                                                     nullptr,  // sparse_mixer_out
                                                                     use_fused_512_ ? nullptr :
                                                                                      softmax_out.data_ptr<float>(),
                                                                     expert_ids.data_ptr<int64_t>(),
                                                                     source_rows.data_ptr<int32_t>(),
                                                                     token_num,
                                                                     num_expert,
                                                                     top_k,
                                                                     0,
                                                                     num_expert,
                                                                     0,
                                                                     normalization_mode,
                                                                     current_stream,
                                                                     use_fused_512_);
    } else if (topk_t == torch::kInt32) {
        tensorrt_llm::kernels::invokeSelectExpertsForTokens<int32_t>(router_logits.data_ptr<float>(),
                                                                     router_logits.data_ptr<float>(),
                                                                     expert_scales.data_ptr<float>(),
                                                                     nullptr,  // sparse_mixer_out
                                                                     use_fused_512_ ? nullptr :
                                                                                      softmax_out.data_ptr<float>(),
                                                                     expert_ids.data_ptr<int32_t>(),
                                                                     source_rows.data_ptr<int32_t>(),
                                                                     token_num,
                                                                     num_expert,
                                                                     top_k,
                                                                     0,
                                                                     num_expert,
                                                                     0,
                                                                     normalization_mode,
                                                                     current_stream,
                                                                     use_fused_512_);
    } else {
        throw std::runtime_error("Unimplemented dtype for SelectTopkOp: " + std::string(topk_t.name()));
    }
}

void registerSelectTopkOp(const py::module& m) {
    pybind11::class_<SelectTopkOp>(m, "SelectTopkOp")
        .def(pybind11::init<const ModelConfig&, bool>(), py::arg("model_config"), py::arg("use_fused_512") = false)
        .def("forward",
             &SelectTopkOp::forward,
             py::arg("router_logits"),
             py::arg("expert_ids"),
             py::arg("expert_scales"));
}

}  // namespace rtp_llm