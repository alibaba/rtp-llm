#include "rtp_llm/cpp/cache_new/CacheConfigCreator.h"

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig& model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool is_mtp) {
    int        local_head_num_kv = (model_config.attn_config.kv_head_num > 1) ? model_config.attn_config.kv_head_num / parallelism_config.tp_size : model_config.attn_config.kv_head_num;
    const auto device_prop       = rtp_llm::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto       dtype             = model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ? rtp_llm::DataType::TYPE_INT8 : (model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 : model_config.data_type);
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype = model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }
    auto layer_num = model_config.num_layers;
    if (is_mtp) {
        layer_num = 1;
    }

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    CacheConfig config;
    config.layer_num          = layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<size_t>(param.seq_size_per_block_);

    config.use_mla = param.use_mla_;

    if (param.use_mla_ && param.mla_ops_type_ != rtp_llm::MlaOpsType::MHA) {
        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadLatentAttention;
        spec->dtype              = dtype;
        spec->seq_size_per_block = static_cast<uint>(model_config.attn_config.tokens_per_block);
        spec->layer_num          = static_cast<uint>(layer_num);
        spec->block_nums         = 0;
        spec->kv_lora_rank       = static_cast<uint>(model_config.attn_config.kv_lora_rank,);
        spec->rope_head_dim      = static_cast<uint>(model_config.attn_config.rope_head_dim);
        spec->local_head_num_kv  = 1;  // mla set local_head_num_kv to 1

        config.cache_specs.push_back(spec);
        config.block_size   = static_cast<int>(spec->block_size() * spec->layer_num);
        config.block_stride = spec->block_size();

        // TODO(chanyin) set k_block_stride as block size for compatibility, fix in future
        config.k_block_stride  = spec->block_size();
        config.v_block_stride  = 0;
        config.kv_block_stride = spec->block_size();
    } else {
        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->type               = KVCacheType::MultiHeadAttention;
        spec->dtype              = dtype;
        spec->seq_size_per_block = static_cast<uint>(model_config.attn_config.tokens_per_block);
        spec->layer_num          = static_cast<uint>(layer_num);
        spec->block_nums         = 0;
        spec->local_head_num_kv  = static_cast<uint>(std::max(1, local_head_num_kv));
        spec->size_per_head      = static_cast<uint>(model_config.attn_config.size_per_head);

        config.cache_specs.push_back(spec);
        config.block_size   = static_cast<int>(spec->block_size() * spec->layer_num);
        config.block_stride = spec->block_size();
        // Set block strides for backward compatibility, TODO, fix this
        config.k_block_stride  = spec->block_size();
        config.v_block_stride  = 0;
        config.kv_block_stride = spec->block_size();
    }

    config.layer_ids.push_back(all_layer_ids);

    return config;
}

size_t CacheConfigCreator::getDefaultRuntimeMemorySize(const RuntimeConfig& runtime_config,
                                                       const ParallelismConfig& parallelism_config,
                                                       const ModelConfig& model_config,
                                                       const std::optional<SpeculativeExecutionConfig>& sp_config) {
    auto reserve_runtime_mem_bytes = runtime_config.reserve_runtime_mem_mb * 1024 * 1024;
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

    const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::max(4, 8 / (int)parallelism_config.tp_size);
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        RTP_LLM_LOG_INFO("tp_size %d needs at least %d MiB memory for runtime by default, "
                         "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                         parallelism_config.tp_size,
                         minimal_runtime_bytes / 1024 / 1024,
                         reserve_runtime_mem_bytes / 1024 / 1024);
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
    }

    if (model_config.mm_model_config.is_multimodal) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("multimodal needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    if (sp_config && sp_config->type != SP_TYPE_NONE) {
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

size_t CacheConfigCreator::getKVCacheMemorySize(const RuntimeConfig& runtime_config,
                                                const KVCacheConfig& kv_cache_config,
                                                const ModelConfig& model_config,
                                                const ParallelismConfig& parallelism_config,
                                                const std::optional<WarmUpResult>& warm_up_result,
                                                const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto device                       = rtp_llm::DeviceFactory::getDefaultDevice();
    size_t     device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t     runtime_required_bytes       = 0;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB", kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    // Unified call to getDefaultRuntimeMemorySize
    size_t env_runtime_required_bytes = getDefaultRuntimeMemorySize(runtime_config, parallelism_config, model_config, sp_config);

    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        runtime_required_bytes = std::max(env_runtime_required_bytes, warm_up_result->max_used_memory);

        RTP_LLM_LOG_INFO(
            "devices reserved %ld MiB memory, warm up consumed %ld MiB max memory, env runtime memory %ld MiB, final runtime memory %ld MiB",
            device_reserved_memory_bytes / 1024 / 1024,
            warm_up_result->max_used_memory / 1024 / 1024,
            env_runtime_required_bytes / 1024 / 1024,
            runtime_required_bytes / 1024 / 1024);
    } else {
        runtime_required_bytes = env_runtime_required_bytes;
        RTP_LLM_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                         runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
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

CacheConfig CacheConfigCreator::createConfig(const ModelConfig& model_config,
                                             const ParallelismConfig& parallelism_config,
                                             const RuntimeConfig& runtime_config,
                                             const KVCacheConfig& kv_cache_config,
                                             const std::optional<WarmUpResult>& warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config     = CacheConfigCreator::createBasicConfig(model_config, parallelism_config);
    uint32_t    block_nums = 0;

    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_nums = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = getKVCacheMemorySize(runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        block_nums                   = kv_cache_mem_size / config.block_size;
    }
    RTP_LLM_CHECK_WITH_INFO(block_nums > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_nums,
                            static_cast<long>(config.block_size / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_nums) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_nums);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_nums, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_nums,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

std::tuple<CacheConfig, CacheConfig>
CacheConfigCreator::createSpConfig(const ModelConfig& score_model_config,
                                   const ModelConfig& propose_model_config,
                                   const ParallelismConfig& parallelism_config,
                                   const RuntimeConfig& runtime_config,
                                   const KVCacheConfig& kv_cache_config,
                                   const SpeculativeExecutionConfig& sp_config,
                                   const std::optional<WarmUpResult>& warm_up_result,
                                   bool is_mtp,
                                   bool is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_model_config, parallelism_config);
    CacheConfig propose_config = CacheConfigCreator::createBasicConfig(propose_model_config, parallelism_config, is_mtp);
    size_t      block_nums     = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_nums = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = CacheConfigCreator::getKVCacheMemorySize(runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);
        if (is_mtp) {
            auto cache_num = sp_config.gen_num_per_cycle;
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
