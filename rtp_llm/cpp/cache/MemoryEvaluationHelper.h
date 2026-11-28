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
                                       const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt,
                                       size_t                                           transfer_memory_bytes = 0);
    static size_t availableCacheMemory(const RuntimeConfig&                             runtime_config,
                                       const ModelConfig&                               model_config,
                                       const ParallelismConfig&                         parallelism_config,
                                       const std::optional<WarmUpResult>&               warm_up_result,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config,
                                       const KVCacheConfig*                             kv_cache_config = nullptr);

    // Helper function to update memory size if below minimum requirement
    static void updateMemoryIfNeeded(size_t& current_size, size_t min_required, const char* scenario);

    static rtp_llm::DataType getDataTypeForCache(const ModelConfig& model_config);

private:
    // Keep sampled device/default-runtime budgets separate from the deterministic sizing policy.
    static size_t getKVCacheMemorySizeForBudget(const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const ModelConfig&                 model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               size_t                             device_reserved_memory_bytes,
                                               size_t                             env_runtime_required_bytes,
                                               const std::optional<WarmUpResult>& warm_up_result);
};

}  // namespace rtp_llm
