#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig& model_config,
                                                   const ParallelismConfig& parallelism_config,
                                                   bool is_mtp) {
    int        local_head_num_kv = (model_config.head_num_kv_ > 1) ? model_config.head_num_kv_ / parallelism_config.tp_size : model_config.head_num_kv_;
    const auto device_prop       = rtp_llm::DeviceFactory::getDefaultDevice()->getDeviceProperties();
    auto       dtype             = model_config.kv_cache_data_type_;
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype = model_config.kv_cache_data_type_ == rtp_llm::DataType::TYPE_INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }
    auto layer_num = model_config.num_layers_;
    if (is_mtp) {
        layer_num = 1;
    }

    if (model_config.use_mla_ && model_config.mla_ops_type_ != rtp_llm::MlaOpsType::MHA) {
        return CacheConfig(MlaCacheParam{(uint)model_config.num_layers_,
                                         (uint)0,
                                         (uint)(model_config.kv_lora_rank_ + model_config.rope_head_dim_),
                                         (uint)0,
                                         (uint)model_config.seq_size_per_block_,
                                         dtype});
    }

    return CacheConfig(KVCacheParam{(uint)layer_num,
                                    (uint)0,
                                    (uint)local_head_num_kv,
                                    (uint)model_config.size_per_head_,
                                    (uint)model_config.seq_size_per_block_,
                                    dtype});
}

size_t CacheConfigCreator::getDefaultRuntimeMemorySize(const RuntimeConfig& runtime_config,
                                                       const ParallelismConfig& parallelism_config,
                                                       const MMModelConfig& mm_model_config) {
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

    if (mm_model_config.is_multimodal_) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        if (reserve_runtime_mem_bytes < minimal_runtime_required) {
            reserve_runtime_mem_bytes = minimal_runtime_required;
            RTP_LLM_LOG_INFO("multimodal needs at least %ld MiB memory for runtime by default, "
                             "but only %ld MiB memory reserved. adjust to minimal value.",
                             minimal_runtime_required / 1024 / 1024,
                             reserve_runtime_mem_bytes / 1024 / 1024);
        }
    }

    if (runtime_config.enable_speculative_decoding) {
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
                                                const ModelConfig& model_config,
                                                const ParallelismConfig& parallelism_config,
                                                const MMModelConfig& mm_model_config,
                                                const std::optional<WarmUpResult>& warm_up_result) {
    const auto device                       = rtp_llm::DeviceFactory::getDefaultDevice();
    size_t     device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t     runtime_required_bytes       = 0;

    if (runtime_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("RuntimeConfig explicitly specified kv cache memory size %ld MiB", runtime_config.kv_cache_mem_mb);
        return runtime_config.kv_cache_mem_mb * 1024 * 1024;
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

        size_t env_runtime_required_bytes = getDefaultRuntimeMemorySize(runtime_config, parallelism_config, mm_model_config);
        runtime_required_bytes            = std::max(env_runtime_required_bytes, warm_up_result->max_used_memory);

        RTP_LLM_LOG_INFO(
            "devices reserved %ld MiB memory, warm up consumed %ld MiB max memory, env runtime memory %ld MiB, final runtime memory %ld MiB",
            device_reserved_memory_bytes / 1024 / 1024,
            warm_up_result->max_used_memory / 1024 / 1024,
            env_runtime_required_bytes / 1024 / 1024,
            runtime_required_bytes / 1024 / 1024);
    } else {
        runtime_required_bytes = getDefaultRuntimeMemorySize(runtime_config, parallelism_config, mm_model_config);
        RTP_LLM_LOG_INFO("warm up result not available, use default runtime memory size %ld MiB",
                         runtime_required_bytes / 1024 / 1024);
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size_ * 4 * 8;  // just estimated value
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
                                             const MMModelConfig& mm_model_config,
                                             const ParallelismConfig& parallelism_config,
                                             const RuntimeConfig& runtime_config,
                                             const std::optional<WarmUpResult>& warm_up_result) {
    CacheConfig config     = CacheConfigCreator::createBasicConfig(model_config, parallelism_config);
    uint32_t    block_nums = 0;

    if (runtime_config.block_nums > 0) {
        RTP_LLM_LOG_INFO("RuntimeConfig explicitly specified kv cache block num %d", runtime_config.block_nums);
        block_nums = runtime_config.block_nums;
    } else {
        const auto kv_cache_mem_size = getKVCacheMemorySize(runtime_config, model_config, parallelism_config, mm_model_config, warm_up_result);
        block_nums                   = kv_cache_mem_size / config.block_size;
    }
    RTP_LLM_CHECK_WITH_INFO(block_nums > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_nums,
                            config.block_size / 1024 / 1024);

    const auto kv_cache_seq_len = block_nums * config.seq_size_per_block;
    config.block_nums           = block_nums;
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_nums, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len_) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_nums,
                            kv_cache_seq_len,
                            model_config.max_seq_len_);
    }
    return config;
}

std::tuple<CacheConfig, CacheConfig>
CacheConfigCreator::createSpConfig(const ModelConfig& score_model_config,
                                   const MMModelConfig& score_mm_model_config,
                                   const ParallelismConfig& score_parallelism_config,
                                   const RuntimeConfig& score_runtime_config,
                                   const ModelConfig& propose_model_config,
                                   const MMModelConfig& propose_mm_model_config,
                                   const ParallelismConfig& propose_parallelism_config,
                                   const RuntimeConfig& propose_runtime_config,
                                   const std::optional<WarmUpResult>& warm_up_result,
                                   bool is_mtp,
                                   bool is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_model_config, score_parallelism_config);

    CacheConfig propose_config = CacheConfigCreator::createBasicConfig(propose_model_config, propose_parallelism_config, is_mtp);
    size_t      block_nums     = 0;
    if (score_runtime_config.block_nums > 0) {
        block_nums = score_runtime_config.block_nums;
    } else {
        const auto kv_cache_mem_size = CacheConfigCreator::getKVCacheMemorySize(score_runtime_config, score_model_config, score_parallelism_config, score_mm_model_config, warm_up_result);
        if (is_mtp) {
            auto cache_num = propose_runtime_config.gen_num_per_circle;
            if (is_eagle) {
                cache_num = 1;
            }

            block_nums = kv_cache_mem_size / (score_config.block_size + propose_config.block_size * cache_num);
        } else {
            block_nums = kv_cache_mem_size / (score_config.block_size + propose_config.block_size);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(block_nums > 0, "kv cache needs at least 1 block but %ld", block_nums);

    score_config.block_nums   = block_nums;
    propose_config.block_nums = block_nums;
    RTP_LLM_LOG_INFO("kv cache block nums is %u", block_nums);
    return std::make_tuple(score_config, propose_config);
}

}  // namespace rtp_llm
