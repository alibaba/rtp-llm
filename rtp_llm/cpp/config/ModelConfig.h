#pragma once

#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/model_utils/quantization.h"
#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include "rtp_llm/cpp/model_utils/MlaConfig.h"
#include "rtp_llm/cpp/model_utils/QuantInfo.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/core/Types.h"
#include <vector>
#include <string>

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

struct RoleSpecialTokens {
public:
    std::vector<int64_t> token_ids_;
    std::vector<int64_t> eos_token_ids_;
};

struct SpecialTokens {
public:
    SpecialTokens();
    int64_t                           bos_token_id_           = -1;
    int64_t                           eos_token_id_           = 0;
    int64_t                           pad_token_id_           = 0;
    int64_t                           decoder_start_token_id_ = -1;
    RoleSpecialTokens                 user_;
    RoleSpecialTokens                 assistant_;
    RoleSpecialTokens                 system_;
    std::vector<std::vector<int64_t>> stop_words_id_list_;
    std::vector<std::string>          stop_words_str_list_;
};

class ModelConfig {
public:
    // model variant params used in ft
    int64_t head_num_               = 0;
    int64_t head_num_kv_            = -1;
    int64_t size_per_head_          = 0;
    int64_t inter_size_             = 0;
    int64_t inter_padding_size_     = -1;
    int64_t moe_inter_padding_size_ = -1;
    int64_t num_layers_             = 0;
    int64_t hidden_size_            = 0;

    // mla extra params
    bool       use_mla_       = false;
    int64_t    q_lora_rank_   = 0;
    int64_t    kv_lora_rank_  = 0;
    int64_t    nope_head_dim_ = 0;
    int64_t    rope_head_dim_ = 0;
    int64_t    v_head_dim_    = 0;
    MlaOpsType mla_ops_type_  = MlaOpsType::AUTO;

    // rope config for deepseek
    double deepseek_rope_mscale_    = 1.0;
    double deepseek_mscale_all_dim_ = 1.0;

    // deepseek moe extra params
    int64_t moe_n_group_    = 1;
    int64_t moe_topk_group_ = 1;

    double routed_scaling_factor_ = 1.0;  // used in deepseek v2 and glm4 moe

    double         layernorm_eps_          = 1e-5;
    LayerNormType  layernorm_type_         = LayerNormType::pre_layernorm;
    NormType       norm_type_              = NormType::layernorm;
    TaskType       task_type_              = TaskType::LANGUAGE_MODEL;
    ActivationType activation_type_        = ActivationType::Gelu;
    DataType       data_type_              = DataType::TYPE_FP16;
    DataType       kv_cache_data_type_     = DataType::TYPE_FP16;

    RopeConfig     rope_config_;
    int64_t        position_ids_style_     = 0;
    double         partial_rotary_factor_  = 1.0;
    // for Gemma, hidden_states = hidden_states * (hidden_size**0.5)
    double               input_embedding_scalar_ = 1;
    double               residual_scalar_        = 1;
    float                softmax_extra_scale_    = 1.0f;

    bool   use_logn_attn_ = false;
    double q_scaling_     = 1;
    bool   qk_norm_       = false;

    bool use_norm_input_residual_    = false;
    bool use_norm_attn_out_residual_ = false;

    int64_t max_seq_len_                = 0;
    int64_t vocab_size_                 = 0;
    int64_t input_vocab_size_           = 0;  // 0 if not set
    int64_t type_vocab_size_            = 0;
    int64_t embedding_size_             = 0;
    int64_t seq_size_per_block_         = 8;
    int64_t expert_num_                 = 0;
    int64_t moe_k_                      = 0;
    bool    moe_normalize_expert_scale_ = false;
    // 0 for no moe; 1 for all layer moe; 2 for partial layer moe
    int64_t moe_style_ = 0;
    // 0 for softmax; 1 for sigmoid
    int64_t              scoring_func_    = 0;
    std::vector<int64_t> moe_layer_index_ = {};

    bool   has_positional_encoding_    = false;
    bool   has_pre_decoder_layernorm_  = false;
    bool   has_post_decoder_layernorm_ = false;
    bool   has_lm_head_                = true;
    bool   use_attention_linear_bias_  = false;
    bool   use_fp32_to_compute_logit_  = false;
    bool   add_bias_linear_            = false;
    bool   has_moe_norm_               = false;
    double logit_scale_                = 1.0;
    bool   is_causal_                  = true;
    bool   use_kvcache_                = true;

    int64_t pre_seq_len_       = 0;
    bool    prefix_projection_ = false;

    bool reverse_e_h_norm_ = false;

    // Model loading and quantization
    std::string tokenizer_path_    = "";
    std::string ckpt_path_         = "";
    SpecialTokens special_tokens_;
    QuantAlgo     quant_algo_;

    ModelConfig() = default;

    // Getter methods that return string
    std::string      get_layer_norm_type() const;
    std::string      get_norm_type() const;
    std::string      get_activation_type() const;
    std::string      get_data_type() const;
    std::string      get_kv_cache_data_type() const;
    std::string      get_task_type() const;
    
    // Setter methods with validation (throw exception on invalid input)
    void             set_layer_norm_type(std::string layernorm_type_str);
    void             set_norm_type(std::string norm_type_str);
    void             set_task_type(std::string task);
    void             set_activation_type(std::string activation_type_str);
    void             set_data_type(std::string data_type_str);
    void             set_kv_cache_data_type(std::string kv_cache_data_type_str);
    bool             isGatedActivation() const;
    AttentionConfigs getAttentionConfigs(int64_t tp_size,
                                         int64_t seq_size_per_block,
                                         bool    is_causal,
                                         bool    use_kvcache) const;
    bool             isKvCacheQuant() const;
};

class MMModelConfig {
public:
    bool                              is_multimodal_          = false;
    std::vector<std::vector<int64_t>> mm_sep_tokens_          = {};
    bool                              include_sep_tokens_     = false;
    int64_t                           mm_position_ids_style_  = 0;  // 0 for default; 1 for chatglm4v; 2 for qwen2 vl
    int64_t                           position_id_len_factor_ = 1;
};

}  // namespace rtp_llm

