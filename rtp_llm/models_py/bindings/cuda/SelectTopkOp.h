#pragma once

#include <torch/torch.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"
#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"

namespace trt_plugins = tensorrt_llm::plugins;

namespace rtp_llm {

class SelectTopkOp {
public:
    SelectTopkOp(const ModelConfig& model_config, bool fake_balance_expert, int64_t dp_rank);
    void forward(torch::Tensor router_logits,
                 torch::Tensor expert_ids,
                 torch::Tensor expert_scales,
                 torch::Tensor log2phy          = torch::Tensor(),
                 torch::Tensor logic_expert_cnt = torch::Tensor(),
                 int64_t       phy_exp_num      = 0,
                 int64_t       ep_rank          = 0);

private:
    int64_t                                              expert_num_;
    int64_t                                              moe_k_;
    bool                                                 has_moe_norm_;
    bool                                                 fake_balance_expert_;
    int64_t                                              dp_rank_;
    std::unique_ptr<trt_plugins::MixtureOfExpertsPlugin> moe_plugin_;
};

void registerSelectTopkOp(const pybind11::module& m);
}  // namespace rtp_llm