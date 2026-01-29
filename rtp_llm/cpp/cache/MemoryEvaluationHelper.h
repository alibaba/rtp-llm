#pragma once

#include <memory>
#include <optional>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class MemoryEvaluationHelper {
public:
    static size_t
                  getDefaultRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                              const ParallelismConfig&                         parallelism_config,
                                              const ModelConfig&                               model_config,
                                              const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static size_t getKVCacheMemorySize(const RuntimeConfig&                             runtime_config,
                                       const KVCacheConfig&                             kv_cache_config,
                                       const ModelConfig&                               model_config,
                                       const ParallelismConfig&                         parallelism_config,
                                       const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);

    // Helper function to update memory size if below minimum requirement
    static void updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario);

    // Helper function to determine data type based on model configuration and device properties
    static rtp_llm::DataType getDataTypeForCache(const ModelConfig&               model_config,
                                                 const rtp_llm::DeviceProperties& device_prop);
};

}  // namespace rtp_llm