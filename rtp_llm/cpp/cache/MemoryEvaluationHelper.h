#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {

class MemoryEvaluationHelper {
public:
    // The operator-configured reserve (reserve_runtime_mem_mb), raised to a scenario floor for
    // multimodal and speculative deployments. Named "configured" rather than "default" because
    // it is only one term of the sizing max() in getKVCacheMemorySize, not a fallback used when
    // nothing else applies.
    static size_t
                  getConfiguredRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                                 const ModelConfig&                               model_config,
                                                 const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static size_t getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                       const KVCacheConfig&                             kv_cache_config,
                                       const ModelConfig&                               model_config,
                                       const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);

    // Testable core of the overload above: takes the device memory sample as an argument instead
    // of reading the GPU via getGpuExecStatus(), so unit tests can drive the sizing decisions
    // (warmup-vs-no-warmup formula selection, measurement degradation, the device_reserved min
    // reduction, knob validation) without a device. The sampling overload is a thin wrapper.
    static size_t getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                       const KVCacheConfig&                             kv_cache_config,
                                       const ModelConfig&                               model_config,
                                       const MemoryStatus&                              gpu_memory_status,
                                       const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);

    // Helper function to update memory size if below minimum requirement
    static void updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario);

    static rtp_llm::DataType getDataTypeForCache(const ModelConfig& model_config);
};

}  // namespace rtp_llm
