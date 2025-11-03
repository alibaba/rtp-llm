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

    static GptModelDescription genModelDescription(const ModelConfig& model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   const EPLBConfig& eplb_config) {
        AttentionConfigs attention_config = model_config.getAttentionConfigs(
            parallelism_config.tp_size,
            model_config.seq_size_per_block_,
            model_config.is_causal_,
            model_config.use_kvcache_);
        int              moe_tp_size      = parallelism_config.tp_size * parallelism_config.dp_size / parallelism_config.ep_size;
        // TP在init的时候处理，认为每个MOE Plugin只看到一个TP rank；EP在MOE Plugin中处理；
        auto                moe_configs = model_config.moe_style_ ? (std::optional<rtp_llm::MoeConfigs>)rtp_llm::MoeConfigs(
                                                   {(size_t)model_config.expert_num_,
                                                                   (size_t)(eplb_config.phy_exp_num - model_config.expert_num_),
                                                                   (size_t)model_config.moe_k_,
                                                                   model_config.moe_normalize_expert_scale_,
                                                                   model_config.moe_inter_padding_size_ / moe_tp_size,
                                                                   model_config.has_moe_norm_,
                                                                   parallelism_config.use_all_gather,
                                                                   (size_t)parallelism_config.ep_rank,
                                                                   (size_t)parallelism_config.ep_size,
                                                                   (size_t)parallelism_config.tp_rank,
                                                                   (size_t)parallelism_config.tp_size,
                                                                   (size_t)parallelism_config.dp_rank,
                                                                   (size_t)parallelism_config.dp_size,
                                                                   (int)model_config.scoring_func_,
                                                                   (int)model_config.moe_topk_group_,
                                                                   (int)model_config.moe_n_group_,
                                                                   model_config.routed_scaling_factor_,
                                                                   eplb_config.enable_eplb}) :
                                                              std::nullopt;
        rtp_llm::FfnConfigs ffn_config{
            rtp_llm::getActivationType(model_config.activation_type_str_),
            std::move(moe_configs),
        };
        rtp_llm::QScheme act_qscheme = rtp_llm::QScheme::NoQuantize;
        if (model_config.quant_algo_.isPerTensorQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerTensor;
        } else if (model_config.quant_algo_.isSmoothQuant() || model_config.quant_algo_.isOmniQuant()) {
            act_qscheme = rtp_llm::QScheme::Qint8PerToken;
        } else if (model_config.quant_algo_.isFp8() && !model_config.quant_algo_.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTensor;
        } else if (model_config.quant_algo_.isFp8() && model_config.quant_algo_.isGroupwise()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerTokenBlock;
        } else if (model_config.quant_algo_.isFp8PTPC()) {
            act_qscheme = rtp_llm::QScheme::Qfp8PerToken;
        }

        return {attention_config,
                ffn_config,
                rtp_llm::getNormType(model_config.norm_type_str_),
                model_config.data_type_,
                act_qscheme,
                model_config.data_type_,
                model_config.layernorm_eps_,
                (size_t)model_config.vocab_size_,
                model_config.layernorm_type_ == rtp_llm::LayerNormType::post_layernorm,
                model_config.input_embedding_scalar_,
                model_config.residual_scalar_,
                model_config.reverse_e_h_norm_};
    }

    virtual ~Executor() {};

    virtual bool updateEplbConfig(const EplbConfig& config) {
        return false;
    }

public:
    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
