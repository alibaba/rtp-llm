#pragma once
#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "torch/extension.h"
#include <torch/custom_class.h>
#include <torch/script.h>
#include "src/fastertransformer/utils/quantization.h"
#include <vector>

namespace ft = fastertransformer;
namespace th = torch;

struct RoleSpecialTokens: public th::jit::CustomClassHolder {
public:
    std::vector<int64_t> token_ids_;
    std::vector<int64_t> eos_token_ids_;
};

struct QuantAlgo: public th::jit::CustomClassHolder {
public:
    bool    int8_mode_              = false;
    bool    int4_mode_              = false;
    bool    has_zeros_              = false;
    int64_t weight_only_group_size_ = 0;
    bool    is_gptq_                = false;
    bool    is_awq_                 = false;
};

struct SpecialTokens: public th::jit::CustomClassHolder {
public:
    SpecialTokens();
    int64_t                               bos_token_id_ = -1;
    int64_t                               eos_token_id_ = 0;
    int64_t                               pad_token_id_ = 0;
    c10::intrusive_ptr<RoleSpecialTokens> user_;
    c10::intrusive_ptr<RoleSpecialTokens> assistant_;
    c10::intrusive_ptr<RoleSpecialTokens> system_;
    std::vector<std::vector<int64_t>>     stop_words_list_;
    std::vector<std::string>              stop_words_str_;
};

class GptInitParameter: public th::jit::CustomClassHolder {
public:
    // model variant params used in ft
    int64_t head_num_           = 0;
    int64_t head_num_kv_        = -1;
    int64_t size_per_head_      = 0;
    int64_t inter_size_         = 0;
    int64_t inter_padding_size_ = -1;
    int64_t num_layers_         = 0;
    int64_t num_valid_layer_    = 0;
    int64_t hidden_size_        = 0;

    // in sparse, those params might vary among layers
    bool                 is_sparse_head_           = false;
    std::vector<int64_t> layer_head_num_           = {};
    std::vector<int64_t> layer_head_num_kv_        = {};
    std::vector<int64_t> layer_inter_size_         = {};
    std::vector<int64_t> layer_inter_padding_size_ = {};

    double             layernorm_eps_       = 1e-5;
    std::string        layernorm_type_str_  = "pre_layernorm";
    std::string        norm_type_str_       = "layernorm";
    std::string        activation_type_str_ = "Gelu";
    ft::LayerNormType  layernorm_type_      = ft::LayerNormType::pre_layernorm;
    ft::NormType       norm_type_           = ft::NormType::layernorm;
    ft::ActivationType activation_type_     = ft::ActivationType::Gelu;

    int64_t rotary_embedding_dim_      = 0;
    int64_t rotary_embedding_style_    = 0;
    int64_t rotary_embedding_base_     = 10000;
    double  dynamic_embedding_scalar_  = 0.0;
    int64_t dynamic_embedding_max_pos_ = 0;
    int64_t position_embeddings_scale_ = 1;
    int64_t base_scale_                = 1;
    double  input_embedding_scalar_    = 1; // for Gemma, hidden_states = hidden_states * (hidden_size**0.5)

    bool    use_logn_attn_ = false;
    int64_t logn_seq_len_  = 2048;

    bool use_norm_input_residual_    = false;
    bool use_norm_attn_out_residual_ = false;

    std::string weights_data_type_ = "fp16";
    std::string data_type_         = "fp16";

    int64_t              max_seq_len_     = 0;
    int64_t              vocab_size_      = 0;
    int64_t              type_vocab_size_ = 0;
    int64_t              expert_num_      = 0;
    int64_t              moe_k_           = 0;
    std::vector<int64_t> moe_layer_index_ = {};

    bool has_positional_encoding_    = false;
    bool has_pre_decoder_layernorm_  = false;
    bool has_post_decoder_layernorm_ = false;
    bool has_lm_head_                = true;        
    bool use_attention_linear_bias_  = false;
    bool use_fp32_to_compute_logit_  = false;
    bool add_bias_linear_            = false;
    bool has_moe_norm_               = false;            

    bool is_causal_                  = true;

    std::string tokenizer_path_    = "";
    std::string ckpt_path_         = "";
    int64_t     pre_seq_len_       = 0;
    bool        prefix_projection_ = false;
    bool        using_hf_sampling_ = false;

    c10::intrusive_ptr<SpecialTokens> special_tokens_;
    c10::intrusive_ptr<QuantAlgo> quant_algo_;

    // async mode config
    int64_t max_generate_batch_size_ = 1;
    int64_t max_context_batch_size_  = 1;
    int64_t gen_num_per_circle_      = 1;

    bool    is_multimodal_       = false;
    bool    int8_kv_cache_       = false;
    bool    pre_allocate_op_mem_ = true;
    int64_t seq_size_per_block_  = 8;

    bool use_medusa_ = false;

    GptInitParameter();

    GptInitParameter(
        int64_t head_num, int64_t size_per_head, int64_t num_layers, int64_t max_seq_len,
        int64_t vocab_size, int64_t hidden_size);

    void setLayerNormType();
    void setNormType();
    void setActivationType();
    bool isGatedActivation();
};
