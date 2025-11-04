#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

SelectTopkOp::SelectTopkOp(const GptInitParameter& gpt_init_parameter):
    configs_(gpt_init_parameter), moe_plugin_(std::make_unique<trt_plugins::MixtureOfExpertsPlugin>()) {}

void SelectTopkOp::forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales) {
    const auto   token_num          = router_logits.sizes()[0];
    const auto   num_expert         = configs_.expert_num_;
    const auto   top_k              = configs_.moe_k_;
    auto         normalization_mode = configs_.has_moe_norm_ ?
                                          tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE :
                                          tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    auto         topk_t             = expert_ids.dtype();
    const auto   softmax_out    = torch::zeros({token_num, num_expert}, router_logits.options().dtype(torch::kFloat32));
    const auto   source_rows    = torch::zeros({token_num, top_k}, router_logits.options().dtype(torch::kInt32));
    cudaStream_t current_stream = 0;
    if (DeviceFactory::isAlreadyInit()) {
        current_stream = dynamic_cast<CudaDevice*>(DeviceFactory::getDefaultDevice())->getStream();
    }
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

    if (configs_.moe_config.fake_balance_expert) {
        if (expert_ids.dtype() == torch::kInt64) {
            fake_balance_expert(expert_ids.data_ptr<int64_t>(),
                                expert_scales.data_ptr<float>(),
                                configs_.dp_rank_,
                                num_expert,
                                token_num * top_k,
                                current_stream);
        } else if (expert_ids.dtype() == torch::kInt32) {
            fake_balance_expert(expert_ids.data_ptr<int32_t>(),
                                expert_scales.data_ptr<float>(),
                                configs_.dp_rank_,
                                num_expert,
                                token_num * top_k,
                                current_stream);
        }
    }
}

void registerSelectTopkOp(const py::module& m) {
    pybind11::class_<SelectTopkOp>(m, "SelectTopkOp")
        .def(pybind11::init<GptInitParameter>(), py::arg("gpt_init_parameter"))
        .def("forward",
             &SelectTopkOp::forward,
             py::arg("router_logits"),
             py::arg("expert_ids"),
             py::arg("expert_scales"));
}

}  // namespace rtp_llm