#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

// Returns the largest joint token capacity whose complete set of tag-local
// pools fits in total_budget_bytes. LOGICAL groups use ceil(tokens / physical
// span) blocks. FIXED groups use their descriptor count and cap the result.
uint64_t maxKVCacheTokenCapacityForBudget(size_t total_budget_bytes, const CacheConfig& config);

class CacheConfigCreator {
public:
    static CacheConfig createBasicConfig(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         bool                     is_mtp,
                                         int                      gen_num_per_cycle);
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

    // Unified desc->spec conversion. Callers provide the runtime build context;
    // descs remain read-only.
    static LayerKVCacheSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                      const SpecBuildContext&      ctx,
                                                      int64_t                      expected_layer_num);

private:
    // Removed functions moved to MemoryEvaluationHelper:
    // getDefaultRuntimeMemorySize
    // getKVCacheMemorySize

    // Removed functions moved to dedicated creators:
    // createSingleConfig
    // createHybridConfig
};

}  // namespace rtp_llm
