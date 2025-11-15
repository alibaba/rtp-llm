#include "rtp_llm/cpp/cache_new/CacheConfigCreator.h"

#include <algorithm>
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// create cache config for models with one full attention
CacheConfig CacheConfigCreator::createBasicConfig(const rtp_llm::GptInitParameter& param, bool is_mtp) {
    int local_head_num_kv = (param.head_num_kv_ > 1) ? static_cast<int>(param.head_num_kv_ / param.tp_size_) :
                                                       static_cast<int>(param.head_num_kv_);

    const auto device_prop = rtp_llm::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto       dtype       = param.kv_cache_data_type_;
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype = param.kv_cache_data_type_ == rtp_llm::DataType::TYPE_INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }

    int layer_num = static_cast<int>(param.num_layers_);
    if (is_mtp) {
        layer_num = 1;
    }

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    CacheConfig config;
    config.layer_type_num     = 1;
    config.layer_num          = layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<size_t>(param.seq_size_per_block_);

    // for adaption
    config.use_mla = param.use_mla_;

    if (param.use_mla_ && param.mla_ops_type_ != rtp_llm::MlaOpsType::MHA) {
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->seq_size_per_block = static_cast<uint>(param.seq_size_per_block_);
        spec->layer_num          = static_cast<uint>(layer_num);
        spec->block_nums         = 0;
        spec->kv_lora_rank       = static_cast<uint>(param.kv_lora_rank_);
        spec->rope_head_dim      = static_cast<uint>(param.rope_head_dim_);
        spec->local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        spec->layer_ids_         = all_layer_ids;

        config.layer_type_params.push_back(spec);
        config.block_size = static_cast<int>(spec->block_size() * spec->layer_num);

        // Set block strides for backward compatibility
        config.k_block_stride        = spec->k_block_size();
        config.v_block_stride        = spec->v_block_size();
        config.kv_scale_block_stride = 0;  // MLA typically doesn't use scale
    } else {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadAttention;
        spec->dtype              = dtype;
        spec->seq_size_per_block = static_cast<uint>(param.seq_size_per_block_);
        spec->layer_num          = static_cast<uint>(layer_num);
        spec->block_nums         = 0;
        spec->local_head_num_kv  = static_cast<uint>(std::max(1, local_head_num_kv));
        spec->size_per_head      = static_cast<uint>(param.size_per_head_);
        spec->layer_ids_         = all_layer_ids;

        config.layer_type_params.push_back(spec);
        config.block_size = static_cast<int>(spec->block_size() * spec->layer_num);

        // Set block strides for backward compatibility
        config.k_block_stride = spec->k_block_size();
        config.v_block_stride = spec->v_block_size();
        // Calculate scale block stride for INT8/FP8 KV cache
        if (dtype == rtp_llm::DataType::TYPE_INT8 || dtype == rtp_llm::DataType::TYPE_FP8_E4M3) {
            config.kv_scale_block_stride =
                local_head_num_kv * 4 * param.seq_size_per_block_;  // 4 bytes per scale value
        } else {
            config.kv_scale_block_stride = 0;
        }
    }

    config.layer_ids.push_back(all_layer_ids);

    return config;
}

size_t CacheConfigCreator::getDefaultRuntimeMemorySize(const rtp_llm::GptInitParameter& params) {
    auto reserve_runtime_mem_bytes = params.reserve_runtime_mem_mb_ * 1024 * 1024;
    RTP_LLM_LOG_INFO("GptInitParameter has reserve_runtime_mem_mb_=%ld", params.reserve_runtime_mem_mb_);

    const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::max(4, 8 / (int)params.tp_size_);
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        RTP_LLM_LOG_INFO("tp_size %d needs at least %d MiB memory for runtime by default, "
                         "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                         params.tp_size_,
                         minimal_runtime_bytes / 1024 / 1024,
                         reserve_runtime_mem_bytes / 1024 / 1024);
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
    }

    if (params.is_multimodal_) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("multimodal needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    if (params.enable_speculative_decoding_) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("speculative decoding  needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    return reserve_runtime_mem_bytes;
}

size_t CacheConfigCreator::getKVCacheMemorySize(const rtp_llm::GptInitParameter&   params,
                                                const std::optional<WarmUpResult>& warm_up_result) {
    const auto device                       = rtp_llm::DeviceFactory::getDefaultDevice();
    size_t     device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t     runtime_required_bytes       = 0;

    if (params.kv_cache_mem_mb_ > 0) {
        RTP_LLM_LOG_INFO("GptInitParameter explicitly specified kv cache memory size %ld MiB", params.kv_cache_mem_mb_);
        return params.kv_cache_mem_mb_ * 1024 * 1024;
    }

    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        size_t env_runtime_required_bytes = getDefaultRuntimeMemorySize(params);
        runtime_required_bytes            = std::max(env_runtime_required_bytes, warm_up_result->max_used_memory);

        RTP_LLM_LOG_INFO(
            "devices reserved %ld MiB memory, warm up consumed %ld MiB max memory, env runtime memory %ld MiB, final runtime memory %ld MiB",
            device_reserved_memory_bytes / 1024 / 1024,
            warm_up_result->max_used_memory / 1024 / 1024,
            env_runtime_required_bytes / 1024 / 1024,
            runtime_required_bytes / 1024 / 1024);
    } else {
        runtime_required_bytes = getDefaultRuntimeMemorySize(params);
        RTP_LLM_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                         runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem =
        (size_t)params.max_generate_batch_size_ * params.vocab_size_ * 4 * 8;  // just estimated value
    RTP_LLM_LOG_INFO("sampler needs %ld MiB memory, model runtime needs %ld MiB memory, take max value.",
                     sample_need_mem / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024);
    runtime_required_bytes = std::max(sample_need_mem, runtime_required_bytes);

    RTP_LLM_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes,
                            "device reserved memory %ld  MiB is less than runtime required memory %ld MiB",
                            device_reserved_memory_bytes / 1024 / 1024,
                            runtime_required_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %ld MiB", kv_cache_mem_size / 1024 / 1024);
    return kv_cache_mem_size;
}

CacheConfig CacheConfigCreator::createConfig(const rtp_llm::GptInitParameter&   param,
                                             const std::optional<WarmUpResult>& warm_up_result) {
    CacheConfig config     = CacheConfigCreator::createBasicConfig(param);
    uint32_t    block_nums = 0;

    if (param.block_nums_ > 0) {
        RTP_LLM_LOG_INFO("GptInitParameter explicitly specified kv cache block num %d", param.block_nums_);
        block_nums = static_cast<uint32_t>(param.block_nums_);
    } else {
        const auto kv_cache_mem_size = getKVCacheMemorySize(param, warm_up_result);
        block_nums = static_cast<uint32_t>(kv_cache_mem_size / static_cast<size_t>(config.block_size));
    }
    RTP_LLM_CHECK_WITH_INFO(block_nums > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_nums,
                            static_cast<long>(config.block_size / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_nums) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_nums);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_nums, kv_cache_seq_len);
    if (kv_cache_seq_len < static_cast<size_t>(param.max_seq_len_)) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_nums,
                            kv_cache_seq_len,
                            param.max_seq_len_);
    }
    return config;
}

std::tuple<CacheConfig, CacheConfig>
CacheConfigCreator::createSpConfig(const rtp_llm::GptInitParameter&   score_param,
                                   const rtp_llm::GptInitParameter&   propose_param,
                                   const std::optional<WarmUpResult>& warm_up_result,
                                   bool                               is_mtp,
                                   bool                               is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_param);

    CacheConfig propose_config = CacheConfigCreator::createBasicConfig(propose_param, is_mtp);
    size_t      block_nums     = 0;
    if (score_param.block_nums_ > 0) {
        block_nums = static_cast<size_t>(score_param.block_nums_);
    } else {
        const auto kv_cache_mem_size = CacheConfigCreator::getKVCacheMemorySize(score_param, warm_up_result);
        if (is_mtp) {
            auto cache_num = propose_param.gen_num_per_circle_;
            if (is_eagle) {
                cache_num = 1;
            }

            block_nums = kv_cache_mem_size
                         / (static_cast<size_t>(score_config.block_size)
                            + static_cast<size_t>(propose_config.block_size) * static_cast<size_t>(cache_num));
        } else {
            block_nums =
                kv_cache_mem_size
                / (static_cast<size_t>(score_config.block_size) + static_cast<size_t>(propose_config.block_size));
        }
    }
    RTP_LLM_CHECK_WITH_INFO(block_nums > 0, "kv cache needs at least 1 block but %ld", block_nums);

    score_config.block_num   = static_cast<int>(block_nums);
    propose_config.block_num = static_cast<int>(block_nums);
    RTP_LLM_LOG_INFO("kv cache block nums is %u", static_cast<unsigned>(block_nums));
    return std::make_tuple(score_config, propose_config);
}

}  // namespace rtp_llm
