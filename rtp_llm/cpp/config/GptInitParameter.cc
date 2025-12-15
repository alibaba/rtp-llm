#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "autil/Log.h"

namespace rtp_llm {

namespace py = pybind11;

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
    hidden_size_(hidden_size),
    max_seq_len_(max_seq_len),
    vocab_size_(vocab_size) {}

void GptInitParameter::insertMultiTaskPromptTokens(std::string task_id, std::vector<int64_t> tokens_id) {
    std::vector<int> new_tokens_id;  // to convert tokens of type int64_t to type int32_t
    for (auto token_id : tokens_id) {
        new_tokens_id.push_back(token_id);
    }
    multi_task_prompt_tokens_[task_id] = new_tokens_id;
}

void GptInitParameter::setLayerNormType() {
    layernorm_type_ = getLayerNormType(layernorm_type_str_);
}

void GptInitParameter::setNormType() {
    norm_type_ = getNormType(norm_type_str_);
}

void GptInitParameter::setTaskType(std::string task) {
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
        RTP_LLM_CHECK_WITH_INFO(false, "unkown task type: " + task);
    }
}

void GptInitParameter::setActivationType() {
    activation_type_ = getActivationType(activation_type_str_);
}

void GptInitParameter::setDataType() {
    data_type_ = getDataType(data_type_str_);
}

void GptInitParameter::setKvCacheDataType() {
    kv_cache_data_type_ = getDataType(kv_cache_data_type_str_);
}

bool GptInitParameter::isGatedActivation() const {
    return rtp_llm::isGatedActivation(activation_type_);
}

bool GptInitParameter::isKvCacheQuant() const {
    return kv_cache_data_type_ == DataType::TYPE_FP8_E4M3 || kv_cache_data_type_ == DataType::TYPE_INT8;
}

void QuantAlgo::setQuantAlgo(const std::string& quant_method, int64_t bits, int64_t group_size) {
    if (quant_method == "gptq") {
        quant_method_ = GptQ;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (quant_method == "awq") {
        quant_method_ = Awq;
        weight_bits_  = bits;
        group_size_   = group_size;
    } else if (quant_method == "weight_only_per_col") {
        quant_method_ = WeightOnlyPerCol;
        weight_bits_  = bits;
        if (weight_bits_ != 8) {
            throw std::invalid_argument("invalid weight_bits: " + std::to_string(weight_bits_));
        }
    } else if (quant_method == "smooth_quant") {
        quant_method_ = SmoothQuant;
        weight_bits_  = 8;
    } else if (quant_method == "omni_quant") {
        quant_method_ = OmniQuant;
        weight_bits_  = 8;
    } else if (quant_method == "pertensor_quant") {
        quant_method_ = PerTensorQuant;
        weight_bits_  = 8;
    } else if (quant_method == "fp8" || quant_method == "fp8_dynamic_per_tensor") {
        quant_method_ = FP8Quant;
        weight_bits_  = 8;
        group_size_   = group_size;
    } else if (quant_method == "fp8-perchannel-compressed-tensors") {
        quant_method_ = FP8PTPC;
        weight_bits_  = 8;
    } else if (quant_method == "fp8-perchannel-quark") {
        quant_method_ = FP8PTPC;
        weight_bits_  = 8;
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

void GptInitParameter::showDebugInfo() const {
    std::ostringstream oss;
    oss << "\n========== ParallelismDistributedConfig ==========\n"
        << parallelism_distributed_config.to_string() << "\n"
        << "========== ConcurrencyConfig ==========\n"
        << concurrency_config.to_string() << "\n"
        << "========== FMHAConfig ==========\n"
        << fmha_config.to_string() << "\n"
        << "========== KVCacheConfig ==========\n"
        << kv_cache_config.to_string() << "\n"
        << "========== ProfilingDebugLoggingConfig ==========\n"
        << profiling_debug_logging_config.to_string() << "\n"
        << "========== HWKernelConfig ==========\n"
        << hw_kernel_config.to_string() << "\n"
        << "========== DeviceResourceConfig ==========\n"
        << device_resource_config.to_string() << "\n"
        << "========== SamplerConfig ==========\n"
        << sampler_config.to_string() << "\n"
        << "========== MoeConfig ==========\n"
        << moe_config.to_string() << "\n"
        << "========== ModelSpecificConfig ==========\n"
        << model_specific_config.to_string() << "\n"
        << "========== SpeculativeExecutionConfig ==========\n"
        << sp_config.to_string() << "\n"
        << "========== ServiceDiscoveryConfig ==========\n"
        << service_discovery_config.to_string() << "\n"
        << "========== CacheStoreConfig ==========\n"
        << cache_store_config.to_string() << "\n"
        << "========== SchedulerConfig ==========\n"
        << scheduler_config.to_string() << "\n"
        << "========== BatchDecodeSchedulerConfig ==========\n"
        << batch_decode_scheduler_config.to_string() << "\n"
        << "========== FIFOSchedulerConfig ==========\n"
        << fifo_scheduler_config.to_string() << "\n"
        << "========== MiscellaneousConfig ==========\n"
        << misc_config.to_string() << "\n"
        << "========== ArpcConfig ==========\n"
        << arpc_config.to_string() << "\n"
        << "========== GrpcConfig ==========\n"
        << grpc_config.to_string() << "\n";
    if (ffn_disaggregate_config.enable_ffn_disaggregate) {
        oss << "========== FfnDisAggregateConfig ==========\n" << ffn_disaggregate_config.to_string() << "\n";
    }
    if (linear_attention_config.linear_conv_kernel_dim > 0) {
        oss << "========== LinearAttentionConfig ==========\n" << linear_attention_config.to_string() << "\n";
    }
    RTP_LLM_LOG_INFO(oss.str());
}

RopeConfig GptInitParameter::getRopeConfig() const {
    RopeConfig rope_config;
    rope_config.style                = (RopeStyle)rotary_embedding_style_;
    rope_config.dim                  = rotary_embedding_dim_;
    rope_config.base                 = rotary_embedding_base_;
    rope_config.scale                = rotary_embedding_scale_;
    rope_config.max_pos              = org_embedding_max_pos_;
    rope_config.factor1              = rotary_factor1_;
    rope_config.factor2              = rotary_factor2_;
    rope_config.mscale               = rotary_embedding_mscale_;
    rope_config.offset               = rotary_embedding_offset_;
    rope_config.index_factor         = position_id_len_factor_;
    rope_config.extrapolation_factor = rotary_embedding_extrapolation_factor_;
    if (rope_config.style == RopeStyle::Mrope) {
        rope_config.mrope_dim1 = mrope_section_[0];
        rope_config.mrope_dim2 = mrope_section_[1];
        rope_config.mrope_dim3 = mrope_section_[2];
    }
    return rope_config;
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

AttentionConfigs GptInitParameter::getAttentionConfigs() const {
    AttentionConfigs attention_config{head_num_ > 1 ? (size_t)head_num_ / tp_size_ : 1,
                                      head_num_kv_ > 1 ? (size_t)head_num_kv_ / tp_size_ : 1,
                                      (size_t)size_per_head_,
                                      (size_t)hidden_size_,
                                      getRopeConfig(),
                                      (size_t)seq_size_per_block_,
                                      is_causal_ ? rtp_llm::AttentionMaskType::causalMask :
                                                   rtp_llm::AttentionMaskType::noMask,
                                      1.0,
                                      // if qk_norm or use embedding model, fuse add bias in gemm
                                      qk_norm_ || (rotary_embedding_style_ == 0 && !use_kvcache_) ? false : true,
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
