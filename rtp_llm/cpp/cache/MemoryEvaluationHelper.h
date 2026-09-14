#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/MemoryStatus.h"

namespace rtp_llm {

class MemoryEvaluationHelper {
public:
    // Returns the configured runtime reserve after applying scenario-specific floors.
    static size_t
                  getConfiguredRuntimeMemorySize(const RuntimeConfig&                             runtime_config,
                                                 const ModelConfig&                               model_config,
                                                 const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    // The device wrapper supplies the latest GPU memory sample to this sizing implementation.
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
