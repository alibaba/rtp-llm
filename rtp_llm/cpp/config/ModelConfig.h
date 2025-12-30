#pragma once

#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/quantization.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/SpecialTokens.h"
#include <vector>
#include <string>
#include <map>

namespace rtp_llm {

enum TaskType {
    DENSE_EMBEDDING    = 0,
    ALL_EMBEDDING      = 1,
    SPARSE_EMBEDDING   = 2,
    COLBERT_EMBEDDING  = 3,
    LANGUAGE_MODEL     = 4,
    SEQ_CLASSIFICATION = 5,
    RERANKER           = 6,
    LINEAR_SOFTMAX     = 7,
    BGE_M3             = 8
};

class MMModelConfig {
public:
    bool                              is_multimodal         = false;
    std::vector<std::vector<int64_t>> mm_sep_tokens         = {};
    bool                              include_sep_tokens    = false;
    int64_t                           mm_position_ids_style = 0;  // 0 for default; 1 for chatglm4v; 2 for qwen2 vl
};

class ModelConfig {
public:
    // model variant params used in ft
    int64_t num_layers  = 0;
    int64_t hidden_size = 0;

    // Attention configuration - contains all attention-related params
    AttentionConfigs      attn_config;
    LinearAttentionConfig linear_attention_config;
    HybridAttentionConfig hybrid_attention_config;

    // mla ops type (not in AttentionConfigs)
    MlaOpsType mla_ops_type = MlaOpsType::AUTO;

    // rope config for deepseek
    double deepseek_rope_mscale    = 1.0;
    double deepseek_mscale_all_dim = 1.0;

    // deepseek moe extra params
    int64_t moe_n_group    = 1;
    int64_t moe_topk_group = 1;

    double routed_scaling_factor = 1.0;  // used in deepseek v2 and glm4 moe

    double         layernorm_eps   = 1e-5;
    LayerNormType  layernorm_type  = LayerNormType::pre_layernorm;
    NormType       norm_type       = NormType::rmsnorm;
    TaskType       task_type       = TaskType::LANGUAGE_MODEL;
    ActivationType activation_type = ActivationType::Swiglu;
    DataType       data_type       = DataType::TYPE_FP16;

    int64_t position_ids_style    = 0;
    double  partial_rotary_factor = 1.0;
    // for Gemma, hidden_states = hidden_states * (hidden_size**0.5)
    double input_embedding_scalar = 1;
    double residual_scalar        = 1;

    bool qk_norm = false;

    bool use_norm_input_residual    = false;
    bool use_norm_attn_out_residual = false;

    int64_t max_seq_len                = 0;
    int64_t vocab_size                 = 0;
    int64_t input_vocab_size           = 0;  // 0 if not set
    int64_t type_vocab_size            = 0;
    int64_t embedding_size             = 0;
    int64_t expert_num                 = 0;
    int64_t moe_k                      = 0;
    bool    moe_normalize_expert_scale = false;
    // 0 for no moe; 1 for all layer moe; 2 for partial layer moe
    int64_t moe_style = 0;
    // 0 for softmax; 1 for sigmoid
    int64_t              scoring_func    = 0;
    std::vector<int64_t> moe_layer_index = {};

    bool   has_positional_encoding    = false;
    bool   has_pre_decoder_layernorm  = false;
    bool   has_post_decoder_layernorm = true;
    bool   has_lm_head                = true;
    bool   use_attention_linear_bias  = false;
    bool   use_fp32_to_compute_logit  = false;
    bool   add_bias_linear            = false;
    bool   has_moe_norm               = false;
    double logit_scale                = 1.0;
    bool   use_kvcache                = true;

    int64_t pre_seq_len       = 0;
    bool    prefix_projection = false;

    bool reverse_e_h_norm = false;

    // Model loading and quantization
    std::string                        tokenizer_path = "";
    std::string                        ckpt_path      = "";
    std::map<std::string, std::string> lora_infos     = {};  // Map of lora name to path
    SpecialTokens                      special_tokens;
    QuantAlgo                          quant_algo;

    // EPLB configuration
    EPLBConfig eplb_config;

    // Multimodal model configuration
    MMModelConfig mm_model_config;

    // Fields merged from PyModelConfig
    std::string extra_data_path       = "";
    std::string local_extra_data_path = "";
    std::string model_type            = "";
    std::string ptuning_path          = "";

    ModelConfig() = default;

    // Setter methods with validation (throw exception on invalid input)
    void             set_layer_norm_type(std::string layernorm_type_str);
    void             set_norm_type(std::string norm_type_str);
    void             set_task_type(std::string task);
    void             set_activation_type(std::string activation_type_str);
    void             set_data_type(std::string data_type_str);
    void             set_mla_ops_type(std::string mla_ops_type_str);
    bool             isGatedActivation() const;
    AttentionConfigs getAttentionConfigs(int64_t tp_size) const;
    bool             isKvCacheQuant() const;
    std::string      to_string() const;
};

}  // namespace rtp_llm
