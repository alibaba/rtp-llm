#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace th = torch;
namespace ft = fastertransformer;

SpecialTokens::SpecialTokens():
    user_(c10::intrusive_ptr<RoleSpecialTokens>::reclaim(new RoleSpecialTokens)),
    assistant_(c10::intrusive_ptr<RoleSpecialTokens>::reclaim(new RoleSpecialTokens)),
    system_(c10::intrusive_ptr<RoleSpecialTokens>::reclaim(new RoleSpecialTokens)) {}

GptInitParameter::GptInitParameter():
    special_tokens_(c10::intrusive_ptr<SpecialTokens>::reclaim(new SpecialTokens)),
    quant_algo_(c10::intrusive_ptr<QuantAlgo>::reclaim(new QuantAlgo)) {}

GptInitParameter::GptInitParameter(
    int64_t head_num, int64_t size_per_head, int64_t num_layers,
    int64_t max_seq_len, int64_t vocab_size, int64_t hidden_size):
    head_num_(head_num),
    size_per_head_(size_per_head),
    num_layers_(num_layers),
    max_seq_len_(max_seq_len),
    vocab_size_(vocab_size),
    hidden_size_(hidden_size),
    special_tokens_(c10::intrusive_ptr<SpecialTokens>::reclaim(new SpecialTokens)),
    quant_algo_(c10::intrusive_ptr<QuantAlgo>::reclaim(new QuantAlgo)) {}

void GptInitParameter::setLayerNormType() {
    layernorm_type_ = ft::getLayerNormType(layernorm_type_str_);
}

void GptInitParameter::setNormType() {
    norm_type_ = ft::getNormType(norm_type_str_);
}

void GptInitParameter::setActivationType() {
    activation_type_ = ft::getActivationType(activation_type_str_);
}

bool GptInitParameter::isGatedActivation() {
    return ft::isGatedActivation(activation_type_);
}

// refister propertys for quant_algo
#define DEF_PROPERTY(name) .def_readwrite(#name, &QuantAlgo::name##_)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(int8_mode)                                                                                            \
    DEF_PROPERTY(int4_mode)                                                                                            \
    DEF_PROPERTY(has_zeros)                                                                                            \
    DEF_PROPERTY(weight_only_group_size)                                                                               \
    DEF_PROPERTY(is_gptq)                                                                                              \
    DEF_PROPERTY(is_awq)

static auto quantAlgoTHS =
#ifdef LEGACY_THS
    torch::jit::class_<QuantAlgo>("FasterTransformerQuantAlgo")
#else
    torch::jit::class_<QuantAlgo>("FasterTransformer", "QuantAlgo")
#endif
        .def(torch::jit::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

// refister propertys for role_special_tokens
#define DEF_PROPERTY(name) .def_readwrite(#name, &RoleSpecialTokens::name##_)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(token_ids)                                                                                            \
    DEF_PROPERTY(eos_token_ids)

static auto roleSpecialTokensTHS =
#ifdef LEGACY_THS
    torch::jit::class_<RoleSpecialTokens>("FasterTransformerRoleSpecialTokens")
#else
    torch::jit::class_<RoleSpecialTokens>("FasterTransformer", "RoleSpecialTokens")
#endif
        .def(torch::jit::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

// refister propertys for special_tokens
#define DEF_PROPERTY(name) .def_readwrite(#name, &SpecialTokens::name##_)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(bos_token_id)                                                                                         \
    DEF_PROPERTY(eos_token_id)                                                                                         \
    DEF_PROPERTY(user)                                                                                                 \
    DEF_PROPERTY(assistant)                                                                                            \
    DEF_PROPERTY(system)                                                                                               \
    DEF_PROPERTY(stop_words_list)                                                                                      \
    DEF_PROPERTY(stop_words_str)                                                                                       \
    DEF_PROPERTY(pad_token_id)

static auto specialTokensTHS =
#ifdef LEGACY_THS
    torch::jit::class_<SpecialTokens>("FasterTransformerSpecialTokens")
#else
    torch::jit::class_<SpecialTokens>("FasterTransformer", "SpecialTokens")
#endif
        .def(torch::jit::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

#define DEF_PROPERTY(name, member) .def_readwrite(#name, &GptInitParameter::member)

#define REGISTER_PROPERTYS                                                                                             \
    DEF_PROPERTY(head_num, head_num_)                                                                                  \
    DEF_PROPERTY(head_num_kv, head_num_kv_)                                                                            \
    DEF_PROPERTY(size_per_head, size_per_head_)                                                                        \
    DEF_PROPERTY(max_seq_len, max_seq_len_)                                                                            \
    DEF_PROPERTY(vocab_size, vocab_size_)                                                                              \
    DEF_PROPERTY(hidden_size, hidden_size_)                                                                            \
    DEF_PROPERTY(type_vocab_size, type_vocab_size_)                                                                    \
    DEF_PROPERTY(gen_num_per_circle, gen_num_per_circle_)                                                              \
    DEF_PROPERTY(inter_size, inter_size_)                                                                              \
    DEF_PROPERTY(inter_padding_size, inter_padding_size_)                                                              \
    DEF_PROPERTY(is_sparse_head, is_sparse_head_)                                                                      \
    DEF_PROPERTY(layer_head_num, layer_head_num_)                                                                      \
    DEF_PROPERTY(layer_head_num_kv, layer_head_num_kv_)                                                                \
    DEF_PROPERTY(layer_inter_size, layer_inter_size_)                                                                  \
    DEF_PROPERTY(layer_inter_padding_size, layer_inter_padding_size_)                                                  \
    DEF_PROPERTY(num_layers, num_layers_)                                                                              \
    DEF_PROPERTY(layer_num, num_layers_)                                                                               \
    DEF_PROPERTY(num_valid_layer, num_valid_layer_)                                                                    \
    DEF_PROPERTY(expert_num, expert_num_)                                                                              \
    DEF_PROPERTY(moe_k, moe_k_)                                                                                        \
    DEF_PROPERTY(moe_layer_index, moe_layer_index_)                                                                    \
    DEF_PROPERTY(layernorm_eps, layernorm_eps_)                                                                        \
    /* In python, the following types use strings for branch condition */                                              \
    /* Everytime type changes, corresponding set type function should  */                                              \
    /* be called.                                                      */                                              \
    DEF_PROPERTY(layernorm_type, layernorm_type_str_)                                                                  \
    DEF_PROPERTY(norm_type, norm_type_str_)                                                                            \
    DEF_PROPERTY(activation_type, activation_type_str_)                                                                \
    DEF_PROPERTY(rotary_embedding_dim, rotary_embedding_dim_)                                                          \
    DEF_PROPERTY(rotary_embedding_style, rotary_embedding_style_)                                                      \
    DEF_PROPERTY(rotary_embedding_base, rotary_embedding_base_)                                                        \
    DEF_PROPERTY(dynamic_embedding_scalar, dynamic_embedding_scalar_)                                                  \
    DEF_PROPERTY(dynamic_embedding_max_pos, dynamic_embedding_max_pos_)                                                \
    DEF_PROPERTY(position_embeddings_scale, position_embeddings_scale_)                                                \
    DEF_PROPERTY(base_scale, base_scale_)                                                                              \
    DEF_PROPERTY(input_embedding_scalar, input_embedding_scalar_)                                                      \
    DEF_PROPERTY(use_norm_input_residual, use_norm_input_residual_)                                                    \
    DEF_PROPERTY(use_norm_attn_out_residual, use_norm_attn_out_residual_)                                              \
    DEF_PROPERTY(weights_data_type, weights_data_type_)                                                                \
    DEF_PROPERTY(data_type, data_type_)                                                                                \
    DEF_PROPERTY(has_positional_encoding, has_positional_encoding_)                                                    \
    DEF_PROPERTY(has_pre_decoder_layernorm, has_pre_decoder_layernorm_)                                                \
    DEF_PROPERTY(has_post_decoder_layernorm, has_post_decoder_layernorm_)                                              \
    DEF_PROPERTY(has_moe_norm, has_moe_norm_)                                                                          \
    DEF_PROPERTY(has_lm_head, has_lm_head_)                                                                            \
    DEF_PROPERTY(use_attention_linear_bias, use_attention_linear_bias_)                                                \
    DEF_PROPERTY(use_fp32_to_compute_logit, use_fp32_to_compute_logit_)                                                \
    DEF_PROPERTY(add_bias_linear, add_bias_linear_)                                                                    \
    DEF_PROPERTY(tokenizer_path, tokenizer_path_)                                                                      \
    DEF_PROPERTY(ckpt_path, ckpt_path_)                                                                                \
    DEF_PROPERTY(pre_seq_len, pre_seq_len_)                                                                            \
    DEF_PROPERTY(prefix_projection, prefix_projection_)                                                                \
    DEF_PROPERTY(using_hf_sampling, using_hf_sampling_)                                                                \
    DEF_PROPERTY(max_generate_batch_size, max_generate_batch_size_)                                                    \
    DEF_PROPERTY(max_context_batch_size, max_context_batch_size_)                                                      \
    DEF_PROPERTY(special_tokens, special_tokens_)                                                                      \
    DEF_PROPERTY(quant_algo, quant_algo_)                                                                              \
    DEF_PROPERTY(use_logn_attn, use_logn_attn_)                                                                        \
    DEF_PROPERTY(logn_seq_len, logn_seq_len_)                                                                          \
    DEF_PROPERTY(is_multimodal, is_multimodal_)                                                                        \
    DEF_PROPERTY(pre_allocate_op_mem, pre_allocate_op_mem_)                                                            \
    DEF_PROPERTY(seq_size_per_block, seq_size_per_block_)                                                              \
    DEF_PROPERTY(int8_kv_cache, int8_kv_cache_)                                                                        \
    DEF_PROPERTY(is_causal, is_causal_)                                                                                \
    DEF_PROPERTY(use_medusa, use_medusa_)

static auto fasterTransformerGptInitParameterTHS =
#ifdef LEGACY_THS
    torch::jit::class_<GptInitParameter>("FasterTransformerGptInitParameter")
#else
    torch::jit::class_<GptInitParameter>("FasterTransformer", "GptInitParameter")
#endif
        .def(torch::jit::init<int64_t,     // head_num
                              int64_t,     // size_per_head
                              int64_t,     // num_layers
                              int64_t,     // max_seq_len
                              int64_t,     // vocab_size
                              int64_t      // hidden_size
        >())
        .def("setLayerNormType", &GptInitParameter::setLayerNormType)
        .def("setNormType", &GptInitParameter::setNormType)
        .def("setActivationType", &GptInitParameter::setActivationType)
        .def("isGatedActivation", &GptInitParameter::isGatedActivation)  REGISTER_PROPERTYS;

