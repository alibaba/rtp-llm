#pragma once
#include "rtp_llm/cpp/th_op/GptInitParameter.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

namespace trt_plugins = tensorrt_llm::plugins;

namespace rtp_llm {

class SelectTopkOp {
public:
    SelectTopkOp(const GptInitParameter& gpt_init_parameter);
    void forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales);

private:
    GptInitParameter                                     configs_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;
};

void registerSelectTopkOp(const py::module& m);
}  // namespace rtp_llm