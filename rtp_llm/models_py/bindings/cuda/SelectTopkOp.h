#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class SelectTopkOp {
public:
    SelectTopkOp(const ModelConfig& model_config, bool use_fused_512 = false);
    void forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales);

private:
    int64_t expert_num_;
    int64_t moe_k_;
    bool    has_moe_norm_;
    bool    use_fused_512_;
};

void registerSelectTopkOp(const pybind11::module& m);
}  // namespace rtp_llm