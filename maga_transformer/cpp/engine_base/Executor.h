#pragma once

#include "absl/status/statusor.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include <memory>

namespace rtp_llm {

class Executor {
public:
    Executor(ft::DeviceBase* device): device_(device){};
    virtual absl::Status process(const std::list<GenerateStreamPtr>& streams) = 0;

    static GptModelDescription genModelDescription(const ft::GptInitParameter& params) {
        ft::RopeConfig rope_config = params.getRopeConfig();
        int moe_tp_size = params.tp_size_ * params.dp_size_ / params.ep_size_;
        KvCacheDataType      kv_cache_dtype = loadKvCacheDataTypeFromDataType(params.kv_cache_data_type_);
        ft::AttentionConfigs attention_config{
            params.head_num_ > 1 ? (size_t)params.head_num_ / params.tp_size_ : 1,
            params.head_num_kv_ > 1 ? (size_t)params.head_num_kv_ / params.tp_size_ : 1,
            (size_t)params.size_per_head_,
            (size_t)params.hidden_size_,
            rope_config,
            (size_t)params.seq_size_per_block_,
            params.is_causal_ ? fastertransformer::AttentionMaskType::causalMask :
                                fastertransformer::AttentionMaskType::noMask,
            1.0,
            // if qk_norm or use embedding model, fuse add bias in gemm
            params.qk_norm_ || (params.rotary_embedding_style_ == 0 && !params.use_kvcache_) ? false : true,
            false,
            params.use_mla_,
            (size_t)params.q_lora_rank_,
            (size_t)params.kv_lora_rank_,
            (size_t)params.nope_head_dim_,
            (size_t)params.rope_head_dim_,
            (size_t)params.v_head_dim_,
            params.softmax_extra_scale_,
            kv_cache_dtype};
        // TP在init的时候处理，认为每个MOE Plugin只看到一个TP rank；EP在MOE Plugin中处理；
        auto moe_configs = params.moe_style_ ?
            (std::optional<ft::MoeConfigs>)ft::MoeConfigs({
                (size_t)params.expert_num_,
                (size_t)(params.phy_exp_num_-params.expert_num_),
                (size_t)params.moe_k_,
                params.moe_normalize_expert_scale_,
                params.moe_inter_padding_size_ / moe_tp_size,
                params.has_moe_norm_,
                (size_t)params.ep_rank_,
                (size_t)params.ep_size_,
                (size_t)params.tp_rank_,
                (size_t)params.tp_size_,
                (size_t)params.dp_rank_,
                (size_t)params.dp_size_,
                (int)params.scoring_func_,
                (int)params.moe_topk_group_,
                (int)params.moe_n_group_,
                params.enable_eplb_
            }) : std::nullopt;
        ft::FfnConfigs ffn_config{
            ft::getActivationType(params.activation_type_str_),
            move(moe_configs),
        };
        ft::QScheme act_qscheme = ft::QScheme::NoQuantize;
        if (params.quant_algo_.isPerTensorQuant()) {
            act_qscheme = ft::QScheme::Qint8PerTensor;
        } else if (params.quant_algo_.isSmoothQuant() || params.quant_algo_.isOmniQuant()) {
            act_qscheme = ft::QScheme::Qint8PerToken;
        } else if (params.quant_algo_.isFp8() && !params.quant_algo_.isGroupwise()) {
            act_qscheme = ft::QScheme::Qfp8PerTensor;
        } else if (params.quant_algo_.isFp8() && params.quant_algo_.isGroupwise()) {
            act_qscheme = ft::QScheme::Qfp8PerTokenBlock;
        }

        return {attention_config,
                ffn_config,
                ft::getNormType(params.norm_type_str_),
                act_qscheme,
                params.layernorm_eps_,
                (size_t)params.vocab_size_,
                params.layernorm_type_ == ft::LayerNormType::post_layernorm,
                params.input_embedding_scalar_,
                params.residual_scalar_,
                params.reverse_e_h_norm_};
    }

    virtual ~Executor(){};

public:
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
