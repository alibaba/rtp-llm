#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe_kernels.h"

namespace trt_plugins = tensorrt_llm::plugins;

namespace rtp_llm {

class SelectTopkOp {
public:
    SelectTopkOp(const ModelConfig& model_config);
    void forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales);

private:
    int64_t                                              expert_num_;
    int64_t                                              moe_k_;
    bool                                                 has_moe_norm_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;
};

void registerSelectTopkOp(const pybind11::module& m);
}  // namespace rtp_llm