#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

#include <numeric>

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Helper function to update memory size if below minimum requirement
void MemoryEvaluationHelper::updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario) {
    if (current_size < min_required) {
        current_size = min_required;
        RTP_LLM_LOG_INFO("%s needs at least %ld MiB memory for runtime by default, "
                         "but only %ld MiB memory reserved. adjust to minimal value.",
                         scenario,
                         min_required / 1024 / 1024,
                         current_size / 1024 / 1024);
    }
}

// Helper function to determine data type based on model configuration and device properties
rtp_llm::DataType MemoryEvaluationHelper::getDataTypeForCache(const ModelConfig&               model_config,
                                                              const rtp_llm::DeviceProperties& device_prop) {
    auto dtype =
        model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ?
            rtp_llm::DataType::TYPE_INT8 :
            (model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 :
                                                                               model_config.data_type);
    if (device_prop.type == rtp_llm::DeviceType::ArmCpu) {
        // Arm attention operator support FP32 data type only
        dtype =
            model_config.attn_config.kv_cache_dtype == KvCacheDataType::INT8 ? rtp_llm::TYPE_INT8 : rtp_llm::TYPE_FP32;
    }
    return dtype;
}

size_t MemoryEvaluationHelper::getDefaultRuntimeMemorySize(const RuntimeConfig&     runtime_config,
                                                           const ParallelismConfig& parallelism_config,
                                                           const ModelConfig&       model_config,
                                                           const std::optional<SpeculativeExecutionConfig>& sp_config) {
    size_t reserve_runtime_mem_bytes = runtime_config.reserve_runtime_mem_mb * 1024 * 1024;
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

    const auto minimal_runtime_bytes = 256L * 1024 * 1024 * std::max(4, 8 / (int)parallelism_config.get_attn_tp_size());
    if (reserve_runtime_mem_bytes < minimal_runtime_bytes) {
        RTP_LLM_LOG_INFO("tp_size %d needs at least %d MiB memory for runtime by default, "
                         "but only %ld MiB reserved memory set by config. adjust to minimal value.",
                         parallelism_config.get_attn_tp_size(),
                         minimal_runtime_bytes / 1024 / 1024,
                         reserve_runtime_mem_bytes / 1024 / 1024);
        reserve_runtime_mem_bytes = minimal_runtime_bytes;
    }

    if (model_config.mm_model_config.is_multimodal) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        updateMemoryIfNeeded(reserve_runtime_mem_bytes, minimal_runtime_required, "multimodal");
    }

    if (sp_config && sp_config->type != SP_TYPE_NONE) {
        const auto minimal_runtime_required = 2L * 1024 * 1024 * 1024;  // 2 GiB
        updateMemoryIfNeeded(reserve_runtime_mem_bytes, minimal_runtime_required, "speculative decoding");
    }

    return reserve_runtime_mem_bytes;
}

size_t MemoryEvaluationHelper::getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                                    const KVCacheConfig&                             kv_cache_config,
                                                    const ModelConfig&                               model_config,
                                                    const ParallelismConfig&                         parallelism_config,
                                                    const std::optional<WarmUpResult>&               warm_up_result,
                                                    const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto device                       = rtp_llm::DeviceFactory::getDefaultDevice();
    size_t     device_reserved_memory_bytes = device->getDeviceStatus().device_memory_status.preserved_bytes;
    size_t     runtime_required_bytes       = 0;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    size_t env_runtime_required_bytes = MemoryEvaluationHelper::getDefaultRuntimeMemorySize(
        runtime_config, parallelism_config, model_config, sp_config);

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

}  // namespace rtp_llm