#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input) {
    if (!std::isfinite(input.safety_ratio) || input.safety_ratio < 0.0 || input.safety_ratio >= 1.0) {
        throw std::invalid_argument("runtime memory safety ratio must be finite and in [0, 1)");
    }

    const size_t safety_headroom_bytes = static_cast<size_t>(input.total_gpu_bytes * input.safety_ratio);
    if (!input.has_warmup) {
        return {safety_headroom_bytes,
                std::max({input.configured_reserve_bytes,
                          input.sampler_required_bytes,
                          input.no_warmup_floor_bytes,
                          safety_headroom_bytes})};
    }

    const size_t base_required_bytes =
        std::max({input.configured_reserve_bytes, input.warmup_required_bytes, input.sampler_required_bytes});
    // This is intentionally additive: warmup reserves max(configured, measured, sampler) plus
    // headroom for runtime variation that is not represented by the warmup measurement.
    if (base_required_bytes > std::numeric_limits<size_t>::max() - safety_headroom_bytes) {
        throw std::overflow_error("runtime memory sizing overflow");
    }
    return {safety_headroom_bytes, base_required_bytes + safety_headroom_bytes};
}

std::optional<double> parseRuntimeMemorySafetyRatio(std::string_view value) {
    if (value.empty()) {
        return std::nullopt;
    }
    const std::string owned_value(value);
    char*             end = nullptr;
    errno                 = 0;
    const double ratio    = std::strtod(owned_value.c_str(), &end);
    if (errno == ERANGE || end != owned_value.c_str() + owned_value.size() || !std::isfinite(ratio)
        || ratio < 0.0 || ratio >= 1.0) {
        return std::nullopt;
    }
    return ratio;
}

std::optional<int64_t> parseRuntimeMemoryNoWarmupFloorMiB(std::string_view value) {
    if (value.empty()) {
        return std::nullopt;
    }
    const std::string owned_value(value);
    char*             end = nullptr;
    errno                 = 0;
    const long long floor = std::strtoll(owned_value.c_str(), &end, 10);
    if (errno == ERANGE || end != owned_value.c_str() + owned_value.size() || floor < 0
        || static_cast<uint64_t>(floor) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
    }
    return static_cast<int64_t>(floor);
}

namespace {
constexpr size_t kBytesPerMiB = 1024 * 1024;

size_t checkedMiBToBytes(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value >= 0, "%s must be non-negative, got %ld", name, value);
    RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(value) <= std::numeric_limits<size_t>::max() / kBytesPerMiB,
                            "%s is too large: %ld MiB",
                            name,
                            value);
    return static_cast<size_t>(value) * kBytesPerMiB;
}

double getRuntimeMemorySafetyRatio() {
    const char* raw_value = std::getenv("RUNTIME_MEM_SAFETY_RATIO");
    if (raw_value == nullptr) {
        return kDefaultRuntimeMemorySafetyRatio;
    }
    const auto ratio = parseRuntimeMemorySafetyRatio(raw_value);
    RTP_LLM_CHECK_WITH_INFO(ratio.has_value(),
                            "RUNTIME_MEM_SAFETY_RATIO must be a finite number in [0, 1), got '%s'",
                            raw_value);
    return *ratio;
}

int64_t getRuntimeMemoryNoWarmupFloorMiB() {
    const char* raw_value = std::getenv("RUNTIME_MEM_NO_WARMUP_FLOOR_MB");
    if (raw_value == nullptr) {
        return kDefaultRuntimeNoWarmupFloorMiB;
    }
    const auto floor = parseRuntimeMemoryNoWarmupFloorMiB(raw_value);
    RTP_LLM_CHECK_WITH_INFO(floor.has_value(),
                            "RUNTIME_MEM_NO_WARMUP_FLOOR_MB must be a non-negative integer, got '%s'",
                            raw_value);
    return *floor;
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

size_t MemoryEvaluationHelper::getDefaultRuntimeMemorySize(const RuntimeConfig&     runtime_config,
                                                           const ParallelismConfig& parallelism_config,
                                                           const ModelConfig&       model_config,
                                                           const std::optional<SpeculativeExecutionConfig>& sp_config) {
    size_t reserve_runtime_mem_bytes =
        checkedMiBToBytes(runtime_config.reserve_runtime_mem_mb, "reserve_runtime_mem_mb");
    RTP_LLM_LOG_INFO("RuntimeConfig has reserve_runtime_mem_mb=%ld", runtime_config.reserve_runtime_mem_mb);

    (void)parallelism_config;

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
    const auto   gpu_mem                      = getGpuExecStatus().device_memory_status;
    size_t       device_reserved_memory_bytes = gpu_mem.available_bytes;
    const size_t total_gpu_bytes              = gpu_mem.used_bytes + gpu_mem.free_bytes;

    if (kv_cache_config.kv_cache_mem_mb > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache memory size %ld MiB",
                         kv_cache_config.kv_cache_mem_mb);
        return kv_cache_config.kv_cache_mem_mb * 1024 * 1024;
    }

    size_t env_runtime_required_bytes = MemoryEvaluationHelper::getDefaultRuntimeMemorySize(
        runtime_config, parallelism_config, model_config, sp_config);

    size_t warmup_required_bytes = 0;
    if (warm_up_result) {
        if (device_reserved_memory_bytes != warm_up_result->device_reserved_bytes) {
            RTP_LLM_LOG_WARNING("device reserved memory bytes %ld when create config does not equal to "
                                "the amount when warm up %ld. take min value.",
                                device_reserved_memory_bytes,
                                warm_up_result->device_reserved_bytes);
            device_reserved_memory_bytes =
                std::min(device_reserved_memory_bytes, warm_up_result->device_reserved_bytes);
        }

        warmup_required_bytes = warm_up_result->max_used_memory;
    }

    size_t sample_need_mem =
        (size_t)runtime_config.max_generate_batch_size * model_config.vocab_size * 4 * 8;  // just estimated value
    const double  safety_ratio = getRuntimeMemorySafetyRatio();
    const int64_t no_warmup_floor_mb = getRuntimeMemoryNoWarmupFloorMiB();
    const size_t no_warmup_floor_bytes  = checkedMiBToBytes(no_warmup_floor_mb, "RUNTIME_MEM_NO_WARMUP_FLOOR_MB");
    const auto   sizing                 = calculateRuntimeMemorySizing({warm_up_result.has_value(),
                                                      env_runtime_required_bytes,
                                                      warmup_required_bytes,
                                                      sample_need_mem,
                                                      total_gpu_bytes,
                                                      safety_ratio,
                                                      no_warmup_floor_bytes});
    const size_t runtime_required_bytes = sizing.runtime_required_bytes;

    RTP_LLM_LOG_INFO("sampler needs %ld MiB memory, final runtime needs %ld MiB memory.",
                     sample_need_mem / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024);

    RTP_LLM_CHECK_WITH_INFO(device_reserved_memory_bytes > runtime_required_bytes,
                            "device reserved memory %ld  MiB is less than runtime required memory %ld MiB",
                            device_reserved_memory_bytes / 1024 / 1024,
                            runtime_required_bytes / 1024 / 1024);

    const auto kv_cache_mem_size = device_reserved_memory_bytes - runtime_required_bytes;
    RTP_LLM_LOG_INFO("cache config final decided kv cache memory size %ld MiB", kv_cache_mem_size / 1024 / 1024);
    RTP_LLM_LOG_INFO("[KV_ALLOC] warm_up=%d device_reserved=%ld MiB | runtime_required: torch_peak=%ld MiB "
                     "non_torch_delta=%ld MiB safety_%.0f%%=%ld MiB total=%ld MiB | "
                     "kv_cache_free=%ld MiB (%.2f GiB)",
                     warm_up_result.has_value(),
                     device_reserved_memory_bytes / 1024 / 1024,
                     warm_up_result ? warm_up_result->torch_peak_increase / 1024 / 1024 : 0,
                     warm_up_result ? warm_up_result->non_torch_increase / 1024 / 1024 : 0,
                     safety_ratio * 100,
                     sizing.safety_headroom_bytes / 1024 / 1024,
                     runtime_required_bytes / 1024 / 1024,
                     kv_cache_mem_size / 1024 / 1024,
                     kv_cache_mem_size / 1024.0 / 1024.0 / 1024.0);
    return kv_cache_mem_size;
}

}  // namespace rtp_llm
