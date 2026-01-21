#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"

namespace rtp_llm {

SelectTopkOp::SelectTopkOp(const ModelConfig& model_config, bool fake_balance_expert, int64_t dp_rank):
    expert_num_(model_config.expert_num),
    moe_k_(model_config.moe_k),
    has_moe_norm_(model_config.has_moe_norm),
    fake_balance_expert_(fake_balance_expert),
    dp_rank_(dp_rank),
    moe_plugin_(std::make_unique<trt_plugins::MixtureOfExpertsPlugin>()) {}

void SelectTopkOp::forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales) {
    const auto token_num        = router_logits.sizes()[0];
    const auto num_expert       = expert_num_;
    const auto top_k            = moe_k_;
    auto normalization_mode     = has_moe_norm_ ? tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                                  tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    auto topk_t                 = expert_ids.dtype();
    const auto   softmax_out    = torch::empty({token_num, num_expert}, router_logits.options().dtype(torch::kFloat32));
    const auto   source_rows    = torch::empty({token_num, top_k}, router_logits.options().dtype(torch::kInt32));
    StreamType current_stream = GET_CURRENT_STREAM();
    router_logits = router_logits.contiguous();
    if (topk_t == torch::kInt64) {
        moe_plugin_->selectExpertsForTokens<int64_t>(router_logits.data_ptr<float>(),
                                                     router_logits.data_ptr<float>(),
                                                     expert_scales.data_ptr<float>(),
                                                     nullptr,  // sparse_mixer_out
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
                                                     current_stream);
    } else if (topk_t == torch::kInt32) {
        moe_plugin_->selectExpertsForTokens<int32_t>(router_logits.data_ptr<float>(),
                                                     router_logits.data_ptr<float>(),
                                                     expert_scales.data_ptr<float>(),
                                                     nullptr,  // sparse_mixer_out
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
                                                     current_stream);

    } else {
        throw std::runtime_error("Unimplemented dtype for SelectTopkOp: " + std::string(topk_t.name()));
    }

    if (fake_balance_expert_) {
        if (expert_ids.dtype() == torch::kInt64) {
            fake_balance_expert(expert_ids.data_ptr<int64_t>(),
                                expert_scales.data_ptr<float>(),
                                dp_rank_,
                                num_expert,
                                token_num * top_k,
                                current_stream);
        } else if (expert_ids.dtype() == torch::kInt32) {
            fake_balance_expert(expert_ids.data_ptr<int32_t>(),
                                expert_scales.data_ptr<float>(),
                                dp_rank_,
                                num_expert,
                                token_num * top_k,
                                current_stream);
        }
    }
}

void registerSelectTopkOp(const py::module& m) {
    pybind11::class_<SelectTopkOp>(m, "SelectTopkOp")
        .def(pybind11::init<const ModelConfig&, bool, int64_t>(),
             py::arg("model_config"),
             py::arg("fake_balance_expert"),
             py::arg("dp_rank"))
        .def("forward",
             &SelectTopkOp::forward,
             py::arg("router_logits"),
             py::arg("expert_ids"),
             py::arg("expert_scales"));
}

}  // namespace rtp_llm