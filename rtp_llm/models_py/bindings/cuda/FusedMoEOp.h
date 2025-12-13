#pragma once

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

namespace trt_plugins = tensorrt_llm::plugins;
namespace py          = pybind11;

namespace rtp_llm {

nvinfer1::DataType nvinfer1DtypeConvert(at::ScalarType dtype);

class FusedMoEOp {
public:
    FusedMoEOp(const ModelConfig& model_config, const ParallelismConfig& parallelism_config);
    void forward(torch::Tensor hidden_states,
                 torch::Tensor up_proj,
                 torch::Tensor down_proj,
                 torch::Tensor expert_scales,
                 torch::Tensor expert_ids,
                 torch::Tensor outputs);

private:
    int64_t                                              expert_num_;
    int64_t                                              moe_k_;
    bool                                                 moe_normalize_expert_scale_;
    bool                                                 has_moe_norm_;
    ActivationType                                       activation_type_;
    int64_t                                              ep_size_;
    int64_t                                              ep_rank_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;
};

void registerFusedMoEOp(const py::module& m);
}  // namespace rtp_llm
