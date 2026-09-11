#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

#include <cstdint>
#include <exception>
#include <limits>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
constexpr size_t kBytesPerMiB        = 1024 * 1024;
constexpr size_t kNoWarmupFloorBytes = 2048ULL * kBytesPerMiB;

size_t checkedMiBToBytes(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value >= 0, "%s must be non-negative, got %ld", name, value);
    RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(value) <= std::numeric_limits<size_t>::max() / kBytesPerMiB,
                            "%s is too large: %ld MiB",
                            name,
                            value);
    return static_cast<size_t>(value) * kBytesPerMiB;
}

}  // namespace

// Helper function to update memory size if below minimum requirement
void MemoryEvaluationHelper::updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario) {
    if (current_size < min_required) {
        const size_t original_size = current_size;
        current_size               = min_required;
        RTP_LLM_LOG_INFO("%s runtime memory reserve adjusted from %ld MiB to %ld MiB",
                         scenario,
                         original_size / 1024 / 1024,
                         min_required / 1024 / 1024);
    }
}

rtp_llm::DataType MemoryEvaluationHelper::getDataTypeForCache(const ModelConfig& model_config) {
#if defined(BUILDING_ARM_ONLY)
    auto dtype = rtp_llm::TYPE_FP32;
#else
    auto dtype = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8 ? rtp_llm::DataType::TYPE_FP8_E4M3 :
                                                                                   model_config.data_type;
#endif
    return dtype;
}

size_t
MemoryEvaluationHelper::getConfiguredRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                                       const ModelConfig&                               model_config,
                                                       const std::optional<SpeculativeExecutionConfig>& sp_config) {
    // The C++ field drops the trailing "r" of the CLI flag; operator-facing text carries both
    // spellings so either one greps.
    static constexpr const char* kReserveKnobName = "reserve_runtime_mem_mb (--reserver_runtime_mem_mb)";
    size_t reserve_runtime_mem_bytes = checkedMiBToBytes(runtime_config.reserve_runtime_mem_mb, kReserveKnobName);
    RTP_LLM_LOG_INFO("RuntimeConfig has %s=%ld", kReserveKnobName, runtime_config.reserve_runtime_mem_mb);

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
                                                    const MemoryStatus&                              gpu_memory_status,
                                                    const std::optional<WarmUpResult>&               warm_up_result,
                                                    const std::optional<SpeculativeExecutionConfig>& sp_config) {
    const auto&  gpu_mem               = gpu_memory_status;
    const size_t free_gpu_memory_bytes = gpu_mem.available_bytes;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return checkedMiBToBytes(kv_cache_config.kv_cache_mem_mb, "kv_cache_mem_mb");
    }

    size_t configured_reserve_bytes =
        MemoryEvaluationHelper::getConfiguredRuntimeMemorySize(runtime_config, model_config, sp_config);

    size_t transient_peak_headroom_bytes = 0;
    size_t cuda_graph_memory_bytes       = 0;
    bool   has_trusted_measurement       = false;
    if (warm_up_result) {
        if (warm_up_result->forward_measurement_trusted) {
            transient_peak_headroom_bytes = warm_up_result->transient_peak_headroom_bytes;
            has_trusted_measurement       = true;
        }
        if (warm_up_result->cuda_graph_measurement_trusted) {
            cuda_graph_memory_bytes = warm_up_result->cuda_graph_memory_bytes;
            has_trusted_measurement = true;
        }

        if (has_trusted_measurement) {
            const size_t init_free_memory_bytes = warm_up_result->init_free_memory_bytes;
            RTP_LLM_CHECK_WITH_INFO(init_free_memory_bytes >= free_gpu_memory_bytes,
                                    "Error in memory profiling: initial free memory %zu MiB is less than current free "
                                    "memory %zu MiB. This indicates that another process released GPU memory during "
                                    "profiling.",
                                    init_free_memory_bytes / kBytesPerMiB,
                                    free_gpu_memory_bytes / kBytesPerMiB);
            const size_t total_consumed_bytes = init_free_memory_bytes - free_gpu_memory_bytes;
            RTP_LLM_LOG_INFO("memory profiling diagnostic: base %zu MiB, latest free memory %zu MiB, "
                             "total consumed (persistent) %zu MiB",
                             init_free_memory_bytes / kBytesPerMiB,
                             free_gpu_memory_bytes / kBytesPerMiB,
                             total_consumed_bytes / kBytesPerMiB);
        }
    }

    const double             safety_ratio = kv_cache_config.runtime_mem_safety_ratio;
    RuntimeMemorySizingInput sizing_input;
    sizing_input.has_memory_profile            = has_trusted_measurement;
    sizing_input.configured_reserve_bytes      = configured_reserve_bytes;
    sizing_input.transient_peak_headroom_bytes = transient_peak_headroom_bytes;
    sizing_input.cuda_graph_memory_bytes       = cuda_graph_memory_bytes;
    sizing_input.total_gpu_bytes               = gpu_mem.total_bytes;
    sizing_input.safety_ratio                  = safety_ratio;
    sizing_input.no_warmup_floor_bytes         = kNoWarmupFloorBytes;
    size_t runtime_headroom_bytes              = 0;
    try {
        runtime_headroom_bytes = calculateRuntimeMemorySizing(sizing_input);
    } catch (const std::exception& e) { RTP_LLM_FAIL("%s", e.what()); }

    RTP_LLM_CHECK_WITH_INFO(free_gpu_memory_bytes > runtime_headroom_bytes,
                            "current free memory %zu MiB is less than runtime headroom %zu MiB",
                            free_gpu_memory_bytes / 1024 / 1024,
                            runtime_headroom_bytes / 1024 / 1024);

    auto kv_cache_mem_size = free_gpu_memory_bytes - runtime_headroom_bytes;

    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %zu MiB", kv_cache_mem_size / 1024 / 1024);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm
