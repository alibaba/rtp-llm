#pragma once

#include <memory>
#include <optional>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         bool                     is_mtp = false);
    static CacheConfig createConfig(const ModelConfig&                               model_config,
                                    const ParallelismConfig&                         parallelism_config,
                                    const RuntimeConfig&                             runtime_config,
                                    const KVCacheConfig&                             kv_cache_config,
                                    const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                    const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt);
    static CacheConfig createSpConfig(const ModelConfig&                 score_model_config,
                                      const ModelConfig&                 propose_model_config,
                                      const ParallelismConfig&           parallelism_config,
                                      const RuntimeConfig&               runtime_config,
                                      const KVCacheConfig&               kv_cache_config,
                                      const SpeculativeExecutionConfig&  sp_config,
                                      const std::optional<WarmUpResult>& warm_up_result,
                                      bool                               is_mtp,
                                      bool                               is_eagle);

private:
    // Removed functions moved to MemoryEvaluationHelper:
    // getDefaultRuntimeMemorySize
    // getKVCacheMemorySize

    // Removed functions moved to dedicated creators:
    // createSingleConfig
    // createHybridConfig
    // splitIntoGroups (moved to HybridConfigCreator)
};

}  // namespace rtp_llm
