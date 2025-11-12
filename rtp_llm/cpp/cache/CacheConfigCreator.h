#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const ModelConfig& model_config,
                                         const ParallelismConfig& parallelism_config,
                                         bool is_mtp = false);
    static CacheConfig createConfig(const ModelConfig& model_config,
                                    const MMModelConfig& mm_model_config,
                                    const ParallelismConfig& parallelism_config,
                                    const RuntimeConfig& runtime_config,
                                    const KVCacheConfig& kv_cache_config,
                                    const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
    static std::tuple<CacheConfig, CacheConfig> createSpConfig(const ModelConfig& score_model_config,
                                                               const MMModelConfig& score_mm_model_config,
                                                               const ParallelismConfig& score_parallelism_config,
                                                               const RuntimeConfig& score_runtime_config,
                                                               const KVCacheConfig& score_kv_cache_config,
                                                               const SpeculativeExecutionConfig& sp_config,
                                                               const ModelConfig& propose_model_config,
                                                               const MMModelConfig& propose_mm_model_config,
                                                               const ParallelismConfig& propose_parallelism_config,
                                                               const RuntimeConfig& propose_runtime_config,
                                                               const KVCacheConfig& propose_kv_cache_config,
                                                               const std::optional<WarmUpResult>& warm_up_result,
                                                               bool is_mtp,
                                                               bool is_eagle);

private:
    static size_t getDefaultRuntimeMemorySize(const RuntimeConfig& runtime_config,
                                               const ParallelismConfig& parallelism_config,
                                               const MMModelConfig& mm_model_config,
                                               const SpeculativeExecutionConfig* sp_config = nullptr);
    static size_t getKVCacheMemorySize(const RuntimeConfig& runtime_config,
                                       const KVCacheConfig& kv_cache_config,
                                       const ModelConfig& model_config,
                                       const ParallelismConfig& parallelism_config,
                                       const MMModelConfig& mm_model_config,
                                       const std::optional<WarmUpResult>& warm_up_result = std::nullopt);
};

}  // namespace rtp_llm
