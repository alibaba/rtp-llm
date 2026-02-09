#include "rtp_llm/cpp/config/ModelConfig.h"
#include "autil/Log.h"
#include "rtp_llm/cpp/model_utils/layernorm_types.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <sstream>
#include <string>

namespace rtp_llm {

// Helper functions to convert enums to strings
static std::string taskTypeToString(TaskType task_type) {
    switch (task_type) {
        case TaskType::DENSE_EMBEDDING:
            return "DENSE_EMBEDDING";
        case TaskType::ALL_EMBEDDING:
            return "ALL_EMBEDDING";
        case TaskType::SPARSE_EMBEDDING:
            return "SPARSE_EMBEDDING";
        case TaskType::COLBERT_EMBEDDING:
            return "COLBERT_EMBEDDING";
        case TaskType::LANGUAGE_MODEL:
            return "LANGUAGE_MODEL";
        case TaskType::SEQ_CLASSIFICATION:
            return "SEQ_CLASSIFICATION";
        case TaskType::RERANKER:
            return "RERANKER";
        case TaskType::LINEAR_SOFTMAX:
            return "LINEAR_SOFTMAX";
        case TaskType::BGE_M3:
            return "BGE_M3";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(task_type)) + ")";
    }
}

static std::string mlaOpsTypeToString(MlaOpsType mla_ops_type) {
    switch (mla_ops_type) {
        case MlaOpsType::AUTO:
            return "AUTO";
        case MlaOpsType::MHA:
            return "MHA";
        case MlaOpsType::FLASH_INFER:
            return "FLASH_INFER";
        case MlaOpsType::FLASH_MLA:
            return "FLASH_MLA";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(mla_ops_type)) + ")";
    }
}

static std::string quantMethodToString(QuantMethod quant_method) {
    switch (quant_method) {
        case QuantMethod::None:
            return "None";
        case QuantMethod::WeightOnlyPerCol:
            return "WeightOnlyPerCol";
        case QuantMethod::GptQ:
            return "GptQ";
        case QuantMethod::Awq:
            return "Awq";
        case QuantMethod::SmoothQuant:
            return "SmoothQuant";
        case QuantMethod::OmniQuant:
            return "OmniQuant";
        case QuantMethod::PerTensorQuant:
            return "PerTensorQuant";
        case QuantMethod::FP8Quant:
            return "FP8Quant";
        case QuantMethod::FP8PTPC:
            return "FP8PTPC";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(quant_method)) + ")";
    }
}

// Setter methods with validation (throw exception on invalid input)
void ModelConfig::set_layer_norm_type(std::string layernorm_type_str) {
    try {
        layernorm_type = getLayerNormType(layernorm_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid layernorm_type: " + layernorm_type_str);
    }
}

void ModelConfig::set_norm_type(std::string norm_type_str) {
    try {
        norm_type = getNormType(norm_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid norm_type: " + norm_type_str);
    }
}

void ModelConfig::set_task_type(std::string task) {
    if (task == "DENSE_EMBEDDING") {
        task_type = TaskType::DENSE_EMBEDDING;
    } else if (task == "ALL_EMBEDDING") {
        task_type = TaskType::ALL_EMBEDDING;
    } else if (task == "SPARSE_EMBEDDING") {
        task_type = TaskType::SPARSE_EMBEDDING;
    } else if (task == "COLBERT_EMBEDDING") {
        task_type = TaskType::COLBERT_EMBEDDING;
    } else if (task == "LANGUAGE_MODEL") {
        task_type = TaskType::LANGUAGE_MODEL;
    } else if (task == "SEQ_CLASSIFICATION") {
        task_type = TaskType::SEQ_CLASSIFICATION;
    } else if (task == "RERANKER") {
        task_type = TaskType::RERANKER;
    } else if (task == "LINEAR_SOFTMAX") {
        task_type = TaskType::LINEAR_SOFTMAX;
    } else if (task == "BGE_M3") {
        task_type = TaskType::BGE_M3;
    } else {
        throw std::runtime_error("Invalid task_type: " + task);
    }
}

void ModelConfig::set_activation_type(std::string activation_type_str) {
    try {
        activation_type = getActivationType(activation_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid activation_type: " + activation_type_str);
    }
}

void ModelConfig::set_data_type(std::string data_type_str) {
    try {
        data_type = getDataType(data_type_str);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("Invalid data_type: " + data_type_str + " - " + e.what());
    }
}

void ModelConfig::set_mla_ops_type(std::string mla_ops_type_str) {
    try {
        mla_ops_type = getMlaOpsType(mla_ops_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid mla_ops_type: " + mla_ops_type_str);
    }
}

bool ModelConfig::isGatedActivation() const {
    return rtp_llm::isGatedActivation(activation_type);
}

bool ModelConfig::isKvCacheQuant() const {
    return attn_config.kv_cache_dtype == KvCacheDataType::FP8 || attn_config.kv_cache_dtype == KvCacheDataType::INT8;
}

AttentionConfigs ModelConfig::getAttentionConfigs(int64_t tp_size) const {
    AttentionConfigs config = attn_config;

    config.head_num    = config.head_num / tp_size;
    config.kv_head_num = config.kv_head_num / tp_size;

    // if qk_norm or use embedding model, fuse add bias in gemm
    config.fuse_qkv_add_bias = qk_norm || (config.rope_config.style == RopeStyle::No && !use_kvcache) ? false : true;

    // Set dtype from model data type
    config.dtype = dataTypeToTorchType(data_type);

    // Set max_seq_len for RoPE cache generation
    config.max_seq_len = max_seq_len;

    return config;
}

std::string ModelConfig::to_string() const {
    std::ostringstream oss;

    // Model variant params
    oss << "num_layers: " << num_layers << "\n"
        << "hidden_size: " << hidden_size << "\n"
        << "attn_config: {\n"
        << attn_config.DebugAttentionConfigStr() << "\n}\n"
        << "mla_ops_type: " << mlaOpsTypeToString(mla_ops_type) << "\n"
        << "deepseek_rope_mscale: " << deepseek_rope_mscale << "\n"
        << "deepseek_mscale_all_dim: " << deepseek_mscale_all_dim << "\n"
        << "moe_n_group: " << moe_n_group << "\n"
        << "moe_topk_group: " << moe_topk_group << "\n"
        << "routed_scaling_factor: " << routed_scaling_factor << "\n"
        << "layernorm_eps: " << layernorm_eps << "\n"
        << "layernorm_type: " << getLayerNormTypeStr(layernorm_type) << "\n"
        << "norm_type: " << getNormTypeStr(norm_type) << "\n"
        << "task_type: " << taskTypeToString(task_type) << "\n"
        << "activation_type: " << getActivationTypeStr(activation_type) << "\n"
        << "data_type: " << getDataTypeStr(data_type) << "\n"
        << "position_ids_style: " << position_ids_style << "\n"
        << "partial_rotary_factor: " << partial_rotary_factor << "\n"
        << "input_embedding_scalar: " << input_embedding_scalar << "\n"
        << "residual_scalar: " << residual_scalar << "\n"
        << "qk_norm: " << qk_norm << "\n"
        << "use_norm_input_residual: " << use_norm_input_residual << "\n"
        << "use_norm_attn_out_residual: " << use_norm_attn_out_residual << "\n"
        << "max_seq_len: " << max_seq_len << "\n"
        << "vocab_size: " << vocab_size << "\n"
        << "input_vocab_size: " << input_vocab_size << "\n"
        << "type_vocab_size: " << type_vocab_size << "\n"
        << "embedding_size: " << embedding_size << "\n"
        << "expert_num: " << expert_num << "\n"
        << "moe_k: " << moe_k << "\n"
        << "moe_normalize_expert_scale: " << moe_normalize_expert_scale << "\n"
        << "moe_style: " << moe_style << "\n"
        << "scoring_func: " << scoring_func << "\n"
        << "moe_layer_index: [";
    for (size_t i = 0; i < moe_layer_index.size(); ++i) {
        oss << moe_layer_index[i];
        if (i < moe_layer_index.size() - 1)
            oss << ", ";
    }
    oss << "]\n"
        << "has_positional_encoding: " << has_positional_encoding << "\n"
        << "has_pre_decoder_layernorm: " << has_pre_decoder_layernorm << "\n"
        << "has_post_decoder_layernorm: " << has_post_decoder_layernorm << "\n"
        << "has_lm_head: " << has_lm_head << "\n"
        << "use_attention_linear_bias: " << use_attention_linear_bias << "\n"
        << "use_fp32_to_compute_logit: " << use_fp32_to_compute_logit << "\n"
        << "add_bias_linear: " << add_bias_linear << "\n"
        << "has_moe_norm: " << has_moe_norm << "\n"
        << "logit_scale: " << logit_scale << "\n"
        << "use_kvcache: " << use_kvcache << "\n"
        << "pre_seq_len: " << pre_seq_len << "\n"
        << "prefix_projection: " << prefix_projection << "\n"
        << "reverse_e_h_norm: " << reverse_e_h_norm << "\n"
        << "tokenizer_path: " << tokenizer_path << "\n"
        << "ckpt_path: " << ckpt_path << "\n"
        << "lora_infos: {";
    bool first = true;
    for (const auto& pair : lora_infos) {
        if (!first)
            oss << ", ";
        oss << pair.first << ": " << pair.second;
        first = false;
    }
    oss << "}\n"
        << "special_tokens: {\n"
        << "  bos_token_id: " << special_tokens.bos_token_id << "\n"
        << "  eos_token_id: " << special_tokens.eos_token_id << "\n"
        << "  pad_token_id: " << special_tokens.pad_token_id << "\n"
        << "  decoder_start_token_id: " << special_tokens.decoder_start_token_id << "\n"
        << "  stop_words_id_list: [";
    for (size_t i = 0; i < special_tokens.stop_words_id_list.size(); ++i) {
        oss << "[";
        for (size_t j = 0; j < special_tokens.stop_words_id_list[i].size(); ++j) {
            oss << special_tokens.stop_words_id_list[i][j];
            if (j < special_tokens.stop_words_id_list[i].size() - 1)
                oss << ", ";
        }
        oss << "]";
        if (i < special_tokens.stop_words_id_list.size() - 1)
            oss << ", ";
    }
    oss << "]\n"
        << "  stop_words_str_list: [";
    for (size_t i = 0; i < special_tokens.stop_words_str_list.size(); ++i) {
        oss << "\"" << special_tokens.stop_words_str_list[i] << "\"";
        if (i < special_tokens.stop_words_str_list.size() - 1)
            oss << ", ";
    }
    oss << "]\n}\n"
        << "quant_algo: {\n"
        << "  method: " << quantMethodToString(quant_algo.getQuantMethod()) << "\n"
        << "  bits: " << quant_algo.getWeightBits() << "\n"
        << "  group_size: " << quant_algo.getGroupSize() << "\n}\n"
        << "eplb_config: {\n"
        << eplb_config.to_string() << "\n}\n"
        << "mm_model_config: {\n"
        << "  is_multimodal: " << (mm_model_config.is_multimodal ? "true" : "false") << "\n"
        << "  include_sep_tokens: " << (mm_model_config.include_sep_tokens ? "true" : "false") << "\n"
        << "  mm_position_ids_style: " << mm_model_config.mm_position_ids_style << "\n"
        << "  mm_sep_tokens: [";
    for (size_t i = 0; i < mm_model_config.mm_sep_tokens.size(); ++i) {
        oss << "[";
        for (size_t j = 0; j < mm_model_config.mm_sep_tokens[i].size(); ++j) {
            oss << mm_model_config.mm_sep_tokens[i][j];
            if (j < mm_model_config.mm_sep_tokens[i].size() - 1)
                oss << ", ";
        }
        oss << "]";
        if (i < mm_model_config.mm_sep_tokens.size() - 1)
            oss << ", ";
    }
    oss << "]\n}\n"
        << "extra_data_path: " << extra_data_path << "\n"
        << "local_extra_data_path: " << local_extra_data_path
        << "\n"
        //<< "act_type: " << act_type << "\n"
        << "model_type: " << model_type << "\n"
        << "ptuning_path: " << ptuning_path;

    return oss.str();
}

}  // namespace rtp_llm
