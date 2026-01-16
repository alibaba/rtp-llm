#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <memory>
#include <cstdlib>

namespace rtp_llm {

class Executor {
public:
    Executor(rtp_llm::DeviceBase* device): device_(device) {};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams) = 0;

    static GptModelDescription genModelDescription(const ModelConfig&       model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   const EPLBConfig&        eplb_config,
                                                   const MoeConfig&         moe_config) {
        AttentionConfigs attention_config = model_config.getAttentionConfigs(parallelism_config.tp_size);
        // TP在init的时候处理，认为每个MOE Plugin只看到一个TP rank；EP在MOE Plugin中处理；
        auto moe_configs =
            model_config.moe_style ?
                (std::optional<rtp_llm::MoeConfigs>)rtp_llm::MoeConfigs(
                    {(size_t)model_config.expert_num,
                     (size_t)(eplb_config.phy_exp_num(model_config.expert_num) - model_config.expert_num),
                     (size_t)model_config.moe_k,
                     model_config.moe_normalize_expert_scale,
                     model_config.has_moe_norm,
                     moe_config.use_all_gather,
                     (size_t)parallelism_config.ep_rank,
                     (size_t)parallelism_config.ep_size,
                     (size_t)parallelism_config.tp_rank,
                     (size_t)parallelism_config.tp_size,
                     (size_t)parallelism_config.dp_rank,
                     (size_t)parallelism_config.dp_size,
                     (int)model_config.scoring_func,
                     (int)model_config.moe_topk_group,
                     (int)model_config.moe_n_group,
                     model_config.routed_scaling_factor,
                     eplb_config.enable_eplb()}) :
                std::nullopt;
        rtp_llm::FfnConfigs ffn_config{
            model_config.activation_type,
            std::move(moe_configs),
        };
        rtp_llm::QScheme act_qscheme = rtp_llm::QScheme::NoQuantize;
        if (model_config.quant_algo.isPerTensorQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerTensor;
        } else if (model_config.quant_algo.isSmoothQuant() || model_config.quant_algo.isOmniQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerToken;
        } else if (model_config.quant_algo.isFp8() && !model_config.quant_algo.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTensor;
        } else if (model_config.quant_algo.isFp8() && model_config.quant_algo.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTokenBlock;
        } else if (model_config.quant_algo.isFp8PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        } else if (model_config.quant_algo.isW4a8Int4PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        }

        return {attention_config,
                ffn_config,
                model_config.norm_type,
                model_config.data_type,
                act_qscheme,
                model_config.data_type,
                model_config.layernorm_eps,
                (size_t)model_config.vocab_size,
                model_config.layernorm_type == rtp_llm::LayerNormType::post_layernorm,
                model_config.input_embedding_scalar,
                model_config.residual_scalar,
                model_config.reverse_e_h_norm};
    }

    virtual ~Executor() {};

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }

public:
    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
