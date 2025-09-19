#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

namespace trt_plugins = tensorrt_llm::plugins;

namespace rtp_llm {

nvinfer1::DataType nvinfer1DtypeConvert(at::ScalarType dtype);

class FusedMoEOp {
public:
    FusedMoEOp(const GptInitParameter& gpt_init_parameter);
    void forward(torch::Tensor hidden_states,
                 torch::Tensor up_proj,
                 torch::Tensor down_proj,
                 torch::Tensor expert_scales,
                 torch::Tensor expert_ids,
                 torch::Tensor outputs);

private:
    GptInitParameter                                     configs_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;
};

void registerFusedMoEOp(const pybind11::module& m);
}  // namespace rtp_llm