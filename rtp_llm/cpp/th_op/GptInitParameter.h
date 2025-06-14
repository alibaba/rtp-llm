#pragma once

#include "rtp_llm/cpp/utils/layernorm_types.h"
#include "rtp_llm/cpp/utils/activation_types.h"
#include "rtp_llm/cpp/utils/quantization.h"
#include "rtp_llm/cpp/utils/RopeConfig.h"
#include "rtp_llm/cpp/utils/MlaConfig.h"
#include "rtp_llm/cpp/utils/EplbConfig.h"
#include "rtp_llm/cpp/utils/QuantInfo.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/th_op/GptInitParameterRegister.h"

#include <vector>
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

struct RoleSpecialTokens {
public:
    std::vector<int64_t> token_ids_;
    std::vector<int64_t> eos_token_ids_;
};

struct SpecialTokens {
public:
    SpecialTokens();
    int64_t                               bos_token_id_ = -1;
    int64_t                               eos_token_id_ = 0;
    int64_t                               pad_token_id_ = 0;
    int64_t                               decoder_start_token_id_ = -1;
    RoleSpecialTokens user_;
    RoleSpecialTokens assistant_;
    RoleSpecialTokens system_;
    std::vector<std::vector<int64_t>>     stop_words_id_list_;
    std::vector<std::string>              stop_words_str_list_;
};

class GptInitParameter {
public:
    // model variant params used in ft
    int64_t head_num_               = 0;
    int64_t head_num_kv_            = -1;
    int64_t size_per_head_          = 0;
    int64_t inter_size_             = 0;
    int64_t inter_padding_size_     = -1;
    int64_t moe_inter_padding_size_ = -1;
    int64_t num_layers_             = 0;
    int64_t num_valid_layer_        = 0;
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
    int64_t moe_n_group_          = 1;
    int64_t moe_topk_group_       = 1;

    // in sparse, those params might vary among layers
    bool                 is_sparse_head_           = false;
    std::vector<int64_t> layer_head_num_           = {};
    std::vector<int64_t> layer_head_num_kv_        = {};
    std::vector<int64_t> layer_inter_size_         = {};
    std::vector<int64_t> layer_inter_padding_size_ = {};

    double             layernorm_eps_          = 1e-5;
    std::string        layernorm_type_str_     = "pre_layernorm";
    std::string        norm_type_str_          = "layernorm";
    std::string        activation_type_str_    = "Gelu";
    std::string        kv_cache_data_type_str_ = "fp16";
    LayerNormType  layernorm_type_      = LayerNormType::pre_layernorm;
    NormType       norm_type_           = NormType::layernorm;
    TaskType       task_type_           = TaskType::LANGUAGE_MODEL;
    ActivationType activation_type_     = ActivationType::Gelu;
    DataType kv_cache_data_type_ = DataType::TYPE_FP16;

    int64_t rotary_embedding_dim_    = 0;
    int64_t rotary_embedding_style_  = 0;
    int64_t position_ids_style_      = 0;
    float   rotary_embedding_base_   = 10000.f;
    double  rotary_embedding_scale_  = 1.0;
    double  rotary_factor1_          = 0;
    double  rotary_factor2_          = 0;
    int64_t org_embedding_max_pos_   = 0;
    double  rotary_embedding_mscale_ = 1.0;
    int64_t rotary_embedding_offset_ = 0;
    // for Gemma, hidden_states = hidden_states * (hidden_size**0.5)
    double  input_embedding_scalar_    = 1;
    double  residual_scalar_    = 1;
    float   softmax_extra_scale_       = 1.0f;
    std::vector<int64_t> mrope_section_ = {};

    bool    use_logn_attn_ = false;
    double  q_scaling_ = 1;
    bool    qk_norm_ = false;

    bool    use_cross_attn_ = false;
    int64_t cross_attn_input_len_ = 0;

    bool use_norm_input_residual_    = false;
    bool use_norm_attn_out_residual_ = false;

    std::string data_type_         = "fp16";
    int64_t     local_rank_        = 0;

    int64_t              max_seq_len_                = 0;
    int64_t              vocab_size_                 = 0;
    int64_t              input_vocab_size_           = 0; // 0 if not set
    int64_t              type_vocab_size_            = 0;
    int64_t              embedding_size_             = 0;
    int64_t              expert_num_                 = 0;
    int64_t              moe_k_                      = 0;
    bool                 moe_normalize_expert_scale_ = false;
    // 0 for no moe; 1 for all layer moe; 2 for partial layer moe
    int64_t              moe_style_                  = 0;
    // 0 for softmax; 1 for sigmoid
    int64_t              scoring_func_               = 0;
    std::vector<int64_t> moe_layer_index_            = {};

    // EPLB
    bool       enable_eplb_      = false;
    int64_t    phy_exp_num_      = 0;  // number of physical experts
    int64_t    eplb_update_time_ = 5000;
    EplbMode   eplb_mode_        = EplbMode::NONE;
    py::object py_eplb_;

    bool has_positional_encoding_    = false;
    bool has_pre_decoder_layernorm_  = false;
    bool has_post_decoder_layernorm_ = false;
    bool has_lm_head_                = true;
    bool use_attention_linear_bias_  = false;
    bool use_fp32_to_compute_logit_  = false;
    bool add_bias_linear_            = false;
    bool has_moe_norm_               = false;
    double logit_scale_              = 1.0;
    bool is_causal_                  = true;
    bool use_kvcache_                = true;

    std::string tokenizer_path_    = "";
    std::string ckpt_path_         = "";
    int64_t     pre_seq_len_       = 0;
    bool        prefix_projection_ = false;
    bool        using_hf_sampling_ = false;

    SpecialTokens special_tokens_;
    QuantAlgo quant_algo_;

    // async mode config
    int64_t max_generate_batch_size_ = 1;
    int64_t max_context_batch_size_  = 1;
    int64_t gen_num_per_circle_      = 1;

    bool                 is_multimodal_ = false;
    std::vector<std::vector<int64_t>> mm_sep_tokens_ = {};
    bool                 include_sep_tokens_ = false;
    int64_t              mm_position_ids_style_ = 0; // 0 for default; 1 for chatglm4v; 2 for qwen2 vl
    int64_t              position_id_len_factor_ = 1;

    bool     pre_allocate_op_mem_ = true;
    int64_t  seq_size_per_block_  = 8;

    int64_t  block_nums_                       = 0;
    int64_t  scheduler_reserve_resource_ratio_ = 5;
    int64_t  reserve_runtime_mem_mb_           = 0;
    int64_t  kv_cache_mem_mb_                  = 0;
    bool     reuse_cache_                      = false;
    bool     enable_partial_fallback_          = false;
    bool     enable_fast_gen_                  = false;
    bool     warm_up_                          = false;
    bool     warm_up_with_loss_                = false;
    int64_t  fast_gen_max_context_len_         = 0;
    bool     reverse_e_h_norm_                 = false;
    bool use_expert_attention_ = false; // true for CogVLM2, false for other models

    std::string nccl_ip_          = "";
    bool        use_all_gather_ = false;
    int64_t     tp_nccl_port_     = 0;
    int64_t     dp_tp_nccl_port_  = 0;
    int64_t     ffn_tp_nccl_port_ = 0;
    int64_t     http_port_        = 0;
    int64_t     model_rpc_port_   = 0;
    int64_t     tp_size_          = 1;
    int64_t     tp_rank_          = 0;
    int64_t     ep_size_          = 1;
    int64_t     ep_rank_          = 0;
    int64_t     dp_size_          = 1;
    int64_t     dp_rank_          = 0;
    int64_t     ffn_tp_size_      = 1;
    int64_t     ffn_tp_rank_      = 0;
    bool        enable_sp_        = false;

    int64_t     world_size_     = 1;

    // pd speration
    bool        pd_separation_                      = false;
    bool        use_cache_store_                    = false;
    bool        cache_store_rdma_mode_              = true;
    int64_t     cache_store_listen_port_            = 0;
    int64_t     cache_store_connect_port_           = 0;
    int64_t     cache_store_rdma_listen_port_       = 0;
    int64_t     cache_store_rdma_connect_port_      = 0;
    int64_t     remote_rpc_server_port_             = 0;
    int64_t     prefill_retry_times_                = 0;
    int64_t     prefill_retry_timeout_ms_           = 0;
    int64_t     prefill_max_wait_timeout_ms_        = 0;
    int64_t     decode_retry_times_                 = 0;
    int64_t     decode_retry_timeout_ms_            = 0;
    int64_t     decode_polling_kv_cache_step_ms_    = 0;
    bool        decode_use_async_load_cache_        = true;
    int64_t     rdma_connect_retry_times_           = 0;
    bool        pd_sep_enable_fallback_             = false;
    std::string load_balance_policy_name_           = "";
    int64_t     sync_status_interval_ms_            = 0;
    int64_t     load_cache_timeout_ms_              = 0;
    int64_t     max_rpc_timeout_ms_                 = 0;
    int64_t     worker_port_offset_                 = 0;

    std::map<std::string, std::vector<int>> multi_task_prompt_tokens_;

    // 0 for no sep, 1 for server, 2 for client
    int64_t     vit_separation_                     = 0;
    bool        enable_speculative_decoding_        = false;


    std::string model_name_ = "";

    //multi machine
    std::vector<std::string> worker_addrs_;
    std::vector<std::string> worker_grpc_addrs_;

    GptInitParameter();

    GptInitParameter(
        int64_t head_num, int64_t size_per_head, int64_t num_layers, int64_t max_seq_len,
        int64_t vocab_size, int64_t hidden_size);

    void insertMultiTaskPromptTokens(std::string task_id, std::vector<int64_t> tokens_id);
    void setLayerNormType();
    void setNormType();
    void setTaskType(std::string task);
    void setActivationType();
    void setKvCacheDataType();
    bool isGatedActivation() const;
    RopeConfig getRopeConfig() const;
    bool isKvCacheQuant() const;

    // is not pd-sep
    bool isPDFusion() const;
    // is prefill in p-d sep
    bool isPrefillRole() const;
    // is decode in p-d sep
    bool isDecodeRole() const;
};

}
