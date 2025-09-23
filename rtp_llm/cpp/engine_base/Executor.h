#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include <memory>

namespace rtp_llm {

class Executor {
public:
    Executor(rtp_llm::DeviceBase* device): device_(device) {};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams) = 0;

    static GptModelDescription genModelDescription(const rtp_llm::GptInitParameter& params) {
        AttentionConfigs attention_config = params.getAttentionConfigs();
        int              moe_tp_size      = params.tp_size_ * params.dp_size_ / params.ep_size_;
        // TP在init的时候处理，认为每个MOE Plugin只看到一个TP rank；EP在MOE Plugin中处理；
        auto                moe_configs = params.moe_style_ ? (std::optional<rtp_llm::MoeConfigs>)rtp_llm::MoeConfigs(
                                                   {(size_t)params.expert_num_,
                                                                   (size_t)(params.phy_exp_num_ - params.expert_num_),
                                                                   (size_t)params.moe_k_,
                                                                   params.moe_normalize_expert_scale_,
                                                                   params.moe_inter_padding_size_ / moe_tp_size,
                                                                   params.has_moe_norm_,
                                                                   params.use_all_gather_,
                                                                   (size_t)params.ep_rank_,
                                                                   (size_t)params.ep_size_,
                                                                   (size_t)params.tp_rank_,
                                                                   (size_t)params.tp_size_,
                                                                   (size_t)params.dp_rank_,
                                                                   (size_t)params.dp_size_,
                                                                   (int)params.scoring_func_,
                                                                   (int)params.moe_topk_group_,
                                                                   (int)params.moe_n_group_,
                                                                   params.routed_scaling_factor_,
                                                                   params.enable_eplb_}) :
                                                              std::nullopt;
        rtp_llm::FfnConfigs ffn_config{
            rtp_llm::getActivationType(params.activation_type_str_),
            std::move(moe_configs),
        };
        rtp_llm::QScheme act_qscheme = rtp_llm::QScheme::NoQuantize;
        if (params.quant_algo_.isPerTensorQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerTensor;
        } else if (params.quant_algo_.isSmoothQuant() || params.quant_algo_.isOmniQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerToken;
        } else if (params.quant_algo_.isFp8() && !params.quant_algo_.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTensor;
        } else if (params.quant_algo_.isFp8() && params.quant_algo_.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTokenBlock;
        } else if (params.quant_algo_.isFp8PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        }

        return {attention_config,
                ffn_config,
                rtp_llm::getNormType(params.norm_type_str_),
                params.data_type_,
                act_qscheme,
                params.data_type_,
                params.layernorm_eps_,
                (size_t)params.vocab_size_,
                params.layernorm_type_ == rtp_llm::LayerNormType::post_layernorm,
                params.input_embedding_scalar_,
                params.residual_scalar_,
                params.reverse_e_h_norm_};
    }

    virtual ~Executor() {};

    virtual bool updateEplbConfig(const EplbConfig& config) {
        return false;
    }

public:
    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
