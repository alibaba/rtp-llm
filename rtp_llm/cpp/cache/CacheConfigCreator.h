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

struct KVCacheBlockBudget {
    size_t explicit_pool_reserve_bytes = 0;
    size_t paged_block_bytes           = 0;
    size_t swa_block_bytes             = 0;
};

// Returns the largest global block count whose independent-pool backing fits
// in total_budget_bytes:
//   explicit reserve + N * paged bytes + ceil(N / linear_step) * SWA bytes.
uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step);

class CacheConfigCreator {
public:
    static CacheConfig createWarmupConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          int                      gen_num_per_cycle = 0);
    static CacheConfig createWarmupConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          const KVCacheConfig&     kv_cache_config,
                                          int                      gen_num_per_cycle = 0);
    // Returns layout and a candidate baseline; KVCacheManager confirms group capacity.
    static CacheConfig createConfig(const ModelConfig&                               model_config,
                                    const ParallelismConfig&                         parallelism_config,
                                    const RuntimeConfig&                             runtime_config,
                                    const KVCacheConfig&                             kv_cache_config,
                                    const std::optional<WarmUpResult>&               warm_up_result     = std::nullopt,
                                    const std::optional<SpeculativeExecutionConfig>& sp_config          = std::nullopt,
                                    const ModelConfig*                               draft_model_config = nullptr,
                                    bool                                             is_mtp             = false,
                                    bool                                             is_eagle           = false);

    // Unified desc->spec conversion. Callers provide the runtime build context;
    // descs remain read-only.
    static LayerKVCacheSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                      const SpecBuildContext&      ctx,
                                                      int64_t                      expected_layer_num);

private:
    static CacheConfig createBasicConfig(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         const KVCacheConfig&     kv_cache_config,
                                         int                      gen_num_per_cycle);
};

}  // namespace rtp_llm
