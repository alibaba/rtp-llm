#include "rtp_llm/cpp/config/ModelConfig.h"
#include "autil/Log.h"

namespace rtp_llm {

SpecialTokens::SpecialTokens() {}

// Getter methods that return string
std::string ModelConfig::get_layer_norm_type() const {
    return getLayerNormTypeStr(layernorm_type_);
}

std::string ModelConfig::get_norm_type() const {
    return getNormTypeStr(norm_type_);
}

std::string ModelConfig::get_activation_type() const {
    return getActivationTypeStr(activation_type_);
}

std::string ModelConfig::get_data_type() const {
    return getDataTypeStr(data_type_);
}

std::string ModelConfig::get_kv_cache_data_type() const {
    return getDataTypeStr(kv_cache_data_type_);
}

std::string ModelConfig::get_task_type() const {
    switch (task_type_) {
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
            throw std::runtime_error("Invalid TaskType: " + std::to_string(static_cast<int>(task_type_)));
    }
}

// Setter methods with validation (throw exception on invalid input)
void ModelConfig::set_layer_norm_type(std::string layernorm_type_str) {
    try {
        layernorm_type_ = getLayerNormType(layernorm_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid layernorm_type: " + layernorm_type_str);
    }
}

void ModelConfig::set_norm_type(std::string norm_type_str) {
    try {
        norm_type_ = getNormType(norm_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid norm_type: " + norm_type_str);
    }
}

void ModelConfig::set_task_type(std::string task) {
    if (task == "DENSE_EMBEDDING") {
        task_type_ = TaskType::DENSE_EMBEDDING;
    } else if (task == "ALL_EMBEDDING") {
        task_type_ = TaskType::ALL_EMBEDDING;
    } else if (task == "SPARSE_EMBEDDING") {
        task_type_ = TaskType::SPARSE_EMBEDDING;
    } else if (task == "COLBERT_EMBEDDING") {
        task_type_ = TaskType::COLBERT_EMBEDDING;
    } else if (task == "LANGUAGE_MODEL") {
        task_type_ = TaskType::LANGUAGE_MODEL;
    } else if (task == "SEQ_CLASSIFICATION") {
        task_type_ = TaskType::SEQ_CLASSIFICATION;
    } else if (task == "RERANKER") {
        task_type_ = TaskType::RERANKER;
    } else if (task == "LINEAR_SOFTMAX") {
        task_type_ = TaskType::LINEAR_SOFTMAX;
    } else if (task == "BGE_M3") {
        task_type_ = TaskType::BGE_M3;
    } else {
        throw std::runtime_error("Invalid task_type: " + task);
    }
}

void ModelConfig::set_activation_type(std::string activation_type_str) {
    try {
        activation_type_ = getActivationType(activation_type_str);
    } catch (...) {
        throw std::runtime_error("Invalid activation_type: " + activation_type_str);
    }
}

void ModelConfig::set_data_type(std::string data_type_str) {
    try {
        data_type_ = getDataType(data_type_str);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("Invalid data_type: " + data_type_str + " - " + e.what());
    }
}

void ModelConfig::set_kv_cache_data_type(std::string kv_cache_data_type_str) {
    try {
        kv_cache_data_type_ = getDataType(kv_cache_data_type_str);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("Invalid kv_cache_data_type: " + kv_cache_data_type_str + " - " + e.what());
    }
}

bool ModelConfig::isGatedActivation() const {
    return rtp_llm::isGatedActivation(activation_type_);
}

bool ModelConfig::isKvCacheQuant() const {
    return kv_cache_data_type_ == DataType::TYPE_FP8_E4M3 || kv_cache_data_type_ == DataType::TYPE_INT8;
}

KvCacheDataType loadKvCacheDataTypeFromDataType(rtp_llm::DataType type) {
    if (type == rtp_llm::DataType::TYPE_INT8) {
        return KvCacheDataType::INT8;
    } else if (type == rtp_llm::DataType::TYPE_FP8_E4M3) {
        return KvCacheDataType::FP8;
    } else {
        return KvCacheDataType::BASE;
    }
}

AttentionConfigs ModelConfig::getAttentionConfigs(int64_t tp_size,
                                                   int64_t seq_size_per_block,
                                                   bool    is_causal,
                                                   bool    use_kvcache) const {
    AttentionConfigs attention_config{head_num_ > 1 ? (size_t)head_num_ / tp_size : 1,
                                      head_num_kv_ > 1 ? (size_t)head_num_kv_ / tp_size : 1,
                                      (size_t)size_per_head_,
                                      (size_t)hidden_size_,
                                      rope_config_,
                                      (size_t)seq_size_per_block,
                                      is_causal ? rtp_llm::AttentionMaskType::causalMask :
                                                   rtp_llm::AttentionMaskType::noMask,
                                      1.0,
                                      // if qk_norm or use embedding model, fuse add bias in gemm
                                      qk_norm_ || (rope_config_.style == RopeStyle::No && !use_kvcache) ? false : true,
                                      false,
                                      use_mla_,
                                      (size_t)q_lora_rank_,
                                      (size_t)kv_lora_rank_,
                                      (size_t)nope_head_dim_,
                                      (size_t)rope_head_dim_,
                                      (size_t)v_head_dim_,
                                      softmax_extra_scale_,
                                      loadKvCacheDataTypeFromDataType(kv_cache_data_type_)};
    return attention_config;
}

}  // namespace rtp_llm

