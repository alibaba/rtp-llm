#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "c10/util/intrusive_ptr.h"
#include "pybind11/detail/common.h"

namespace th = torch;

namespace fastertransformer {

SpecialTokens::SpecialTokens() {}

GptInitParameter::GptInitParameter() {}

GptInitParameter::GptInitParameter(int64_t head_num,
                                   int64_t size_per_head,
                                   int64_t num_layers,
                                   int64_t max_seq_len,
                                   int64_t vocab_size,
                                   int64_t hidden_size):
    head_num_(head_num),
    size_per_head_(size_per_head),
    num_layers_(num_layers),
    max_seq_len_(max_seq_len),
    vocab_size_(vocab_size),
    hidden_size_(hidden_size) {
}

void GptInitParameter::insertMultiTaskPromptTokens(int64_t task_id, std::vector<int64_t> tokens_id) {
    std::vector<int> new_tokens_id; // to convert tokens of type int64_t to type int32_t
    for (auto token_id : tokens_id) {
        new_tokens_id.push_back(token_id);
    }
    multi_task_prompt_tokens_[(int)task_id] = new_tokens_id;
}

void GptInitParameter::setLayerNormType() {
    layernorm_type_ = getLayerNormType(layernorm_type_str_);
}

void GptInitParameter::setNormType() {
    norm_type_ = getNormType(norm_type_str_);
}

void GptInitParameter::setActivationType() {
    activation_type_ = getActivationType(activation_type_str_);
}

bool GptInitParameter::isGatedActivation() {
    return fastertransformer::isGatedActivation(activation_type_);
}

void QuantAlgo::setQuantAlgo(const std::string &quant_method, int64_t bits, int64_t group_size) {
    if (quant_method == "gptq") {
        quant_method_ = GptQ;
        weight_bits_ = bits;
        group_size_ = group_size;
    } else if (quant_method == "awq") {
        quant_method_ = Awq;
        weight_bits_ = bits;
        group_size_ = group_size;
    } else if (quant_method == "weight_only_per_col") {
        quant_method_ = WeightOnlyPerCol;
        weight_bits_ = bits;
        if (weight_bits_ != 8) {
            throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
        }
    } else if (quant_method == "smooth_quant") {
        quant_method_ = SmoothQuant;
        weight_bits_ = 8;
    } else if (quant_method == "omni_quant") {
        quant_method_ = OmniQuant;
        weight_bits_ = 8;
    } else {
        throw std::invalid_argument("unknown quant_method: " + quant_method);
    }
    if (weight_bits_ != 4 && weight_bits_ != 8) {
        throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
    }
    if (group_size_ != 0 && group_size_ != 64 && group_size_ != 128) {
        throw std::invalid_argument("invalid group_size: " + std::to_string(group_size_));
    }
}

void registerGptInitParameter(py::module m) {
#define DEF_PROPERTY(name) .def_readwrite(#name, &SpecialTokens::name##_)

#define REGISTER_PROPERTYS                      \
    DEF_PROPERTY(bos_token_id)                  \
    DEF_PROPERTY(eos_token_id)                  \
    DEF_PROPERTY(user)                          \
    DEF_PROPERTY(assistant)                     \
    DEF_PROPERTY(system)                        \
    DEF_PROPERTY(stop_words_list)               \
    DEF_PROPERTY(stop_words_str)                \
    DEF_PROPERTY(pad_token_id)

    pybind11::class_<SpecialTokens>(m, "SpecialTokens")
    .def(pybind11::init<>()) REGISTER_PROPERTYS;

#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS
  
#define DEF_PROPERTY(name) .def_readwrite(#name, &RoleSpecialTokens::name##_)

#define REGISTER_PROPERTYS                      \
    DEF_PROPERTY(token_ids)                     \
    DEF_PROPERTY(eos_token_ids)

    pybind11::class_<RoleSpecialTokens>(m, "RoleSpecialTokens")
    .def(pybind11::init<>()) REGISTER_PROPERTYS;


#undef DEF_PROPERTY
#undef REGISTER_PROPERTYS

    pybind11::class_<QuantAlgo>(m, "QuantAlgo")
    .def(pybind11::init<>())  // quant_pre_scales
    .def("setQuantAlgo", &QuantAlgo::setQuantAlgo)
    .def("isWeightOnlyPerCol", &QuantAlgo::isWeightOnlyPerCol)
    .def("isGptq", &QuantAlgo::isGptq)
    .def("isAwq", &QuantAlgo::isAwq)
    .def("isSmoothQuant", &QuantAlgo::isSmoothQuant)
    .def("isOmniQuant", &QuantAlgo::isOmniQuant)
    .def("isQuant", &QuantAlgo::isQuant)
    .def("isGroupwise", &QuantAlgo::isGroupwise)
    .def("getGroupSize", &QuantAlgo::getGroupSize)
    .def("getWeightBits", &QuantAlgo::getWeightBits)
    .def(py::pickle(
        [](const QuantAlgo& quant_algo) {
            return py::make_tuple(int(quant_algo.getQuantMethod()),
                    int(quant_algo.getWeightBits()), int(quant_algo.getGroupSize()));
        }
        , [](py::tuple t){
            return QuantAlgo(QuantMethod(t[0].cast<int>()), t[1].cast<int>(), t[2].cast<int>());
        }));


#define DEF_PROPERTY(name, member) .def_readwrite(#name, &GptInitParameter::member)

#define REGISTER_PROPERTYS                                              \
    DEF_PROPERTY(head_num, head_num_)                                   \
    DEF_PROPERTY(head_num_kv, head_num_kv_)                             \
    DEF_PROPERTY(size_per_head, size_per_head_)                         \
    DEF_PROPERTY(max_seq_len, max_seq_len_)                             \
    DEF_PROPERTY(vocab_size, vocab_size_)                               \
    DEF_PROPERTY(hidden_size, hidden_size_)                             \
    DEF_PROPERTY(type_vocab_size, type_vocab_size_)                     \
    DEF_PROPERTY(gen_num_per_circle, gen_num_per_circle_)               \
    DEF_PROPERTY(inter_size, inter_size_)                               \
    DEF_PROPERTY(inter_padding_size, inter_padding_size_)               \
    DEF_PROPERTY(moe_inter_padding_size, moe_inter_padding_size_)       \
    DEF_PROPERTY(is_sparse_head, is_sparse_head_)                       \
    DEF_PROPERTY(layer_head_num, layer_head_num_)                       \
    DEF_PROPERTY(layer_head_num_kv, layer_head_num_kv_)                 \
    DEF_PROPERTY(layer_inter_size, layer_inter_size_)                   \
    DEF_PROPERTY(layer_inter_padding_size, layer_inter_padding_size_)   \
    DEF_PROPERTY(num_layers, num_layers_)                               \
    DEF_PROPERTY(layer_num, num_layers_)                                \
    DEF_PROPERTY(num_valid_layer, num_valid_layer_)                     \
    DEF_PROPERTY(expert_num, expert_num_)                               \
    DEF_PROPERTY(moe_k, moe_k_)                                         \
    DEF_PROPERTY(moe_normalize_expert_scale, moe_normalize_expert_scale_) \
    DEF_PROPERTY(moe_style, moe_style_)                                 \
    DEF_PROPERTY(moe_layer_index, moe_layer_index_)                     \
    DEF_PROPERTY(layernorm_eps, layernorm_eps_)                         \
    /* In python, the following types use strings for branch condition */ \
    /* Everytime type changes, corresponding set type function should  */ \
    /* be called.                                                      */ \
    DEF_PROPERTY(layernorm_type, layernorm_type_str_)                   \
    DEF_PROPERTY(norm_type, norm_type_str_)                             \
    DEF_PROPERTY(activation_type, activation_type_str_)                 \
    DEF_PROPERTY(rotary_embedding_dim, rotary_embedding_dim_)           \
    DEF_PROPERTY(rotary_embedding_style, rotary_embedding_style_)       \
    DEF_PROPERTY(position_ids_style, position_ids_style_)               \
    DEF_PROPERTY(rotary_embedding_base, rotary_embedding_base_)         \
    DEF_PROPERTY(rotary_embedding_scale, rotary_embedding_scale_)       \
    DEF_PROPERTY(dynamic_embedding_max_pos, dynamic_embedding_max_pos_) \
    DEF_PROPERTY(base_scale, base_scale_)                               \
    DEF_PROPERTY(input_embedding_scalar, input_embedding_scalar_)       \
    DEF_PROPERTY(use_norm_input_residual, use_norm_input_residual_)     \
    DEF_PROPERTY(use_norm_attn_out_residual, use_norm_attn_out_residual_) \
    DEF_PROPERTY(weights_data_type, weights_data_type_)                 \
    DEF_PROPERTY(data_type, data_type_)                                 \
    DEF_PROPERTY(has_positional_encoding, has_positional_encoding_)     \
    DEF_PROPERTY(has_pre_decoder_layernorm, has_pre_decoder_layernorm_) \
    DEF_PROPERTY(has_post_decoder_layernorm, has_post_decoder_layernorm_) \
    DEF_PROPERTY(has_moe_norm, has_moe_norm_)                           \
    DEF_PROPERTY(logit_scale, logit_scale_)                             \
    DEF_PROPERTY(has_lm_head, has_lm_head_)                             \
    DEF_PROPERTY(use_attention_linear_bias, use_attention_linear_bias_) \
    DEF_PROPERTY(use_fp32_to_compute_logit, use_fp32_to_compute_logit_) \
    DEF_PROPERTY(add_bias_linear, add_bias_linear_)                     \
    DEF_PROPERTY(tokenizer_path, tokenizer_path_)                       \
    DEF_PROPERTY(ckpt_path, ckpt_path_)                                 \
    DEF_PROPERTY(pre_seq_len, pre_seq_len_)                             \
    DEF_PROPERTY(prefix_projection, prefix_projection_)                 \
    DEF_PROPERTY(using_hf_sampling, using_hf_sampling_)                 \
    DEF_PROPERTY(max_generate_batch_size, max_generate_batch_size_)     \
    DEF_PROPERTY(max_context_batch_size, max_context_batch_size_)       \
    DEF_PROPERTY(special_tokens, special_tokens_)                       \
    DEF_PROPERTY(quant_algo, quant_algo_)                               \
    DEF_PROPERTY(use_logn_attn, use_logn_attn_)                         \
    DEF_PROPERTY(logn_seq_len, logn_seq_len_)                           \
    DEF_PROPERTY(q_scaling, q_scaling_)                                 \
    DEF_PROPERTY(qk_norm, qk_norm_)                                     \
    DEF_PROPERTY(is_multimodal, is_multimodal_)                         \
    DEF_PROPERTY(pre_allocate_op_mem, pre_allocate_op_mem_)             \
    DEF_PROPERTY(seq_size_per_block, seq_size_per_block_)               \
    DEF_PROPERTY(block_nums, block_nums_)                               \
    DEF_PROPERTY(scheduler_reserve_resource_ratio, scheduler_reserve_resource_ratio_) \
    DEF_PROPERTY(kv_cache_mem_mb, kv_cache_mem_mb_)                     \
    DEF_PROPERTY(reserve_runtime_mem_mb, reserve_runtime_mem_mb_)       \
    DEF_PROPERTY(reuse_cache, reuse_cache_)                             \
    DEF_PROPERTY(int8_kv_cache, int8_kv_cache_)                         \
    DEF_PROPERTY(is_causal, is_causal_)                                 \
    DEF_PROPERTY(use_medusa, use_medusa_)                               \
    DEF_PROPERTY(nccl_ip, nccl_ip_)                                     \
    DEF_PROPERTY(nccl_port, nccl_port_)                                 \
    DEF_PROPERTY(model_rpc_port, model_rpc_port_)                       \
    DEF_PROPERTY(tp_size, tp_size_)                                     \
    DEF_PROPERTY(tp_rank, tp_rank_)                                     \
    DEF_PROPERTY(use_rpc, use_rpc_)                                     \
    DEF_PROPERTY(use_kvcache, use_kvcache_)

    pybind11::class_<GptInitParameter>(m, "GptInitParameter")
    .def(pybind11::init<int64_t,     // head_num
         int64_t,     // size_per_head
         int64_t,     // num_layers
         int64_t,     // max_seq_len
         int64_t,     // vocab_size
         int64_t      // hidden_size
         >())
    .def("insertMultiTaskPromptTokens", &GptInitParameter::insertMultiTaskPromptTokens)
    .def("setLayerNormType", &GptInitParameter::setLayerNormType)
    .def("setNormType", &GptInitParameter::setNormType)
    .def("setActivationType", &GptInitParameter::setActivationType)
    .def("isGatedActivation", &GptInitParameter::isGatedActivation)  REGISTER_PROPERTYS;
}

}
