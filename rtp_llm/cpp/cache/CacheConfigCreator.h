#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

struct KVCacheBlockBudget {
    struct ProjectedPool {
        size_t   bytes_per_block;
        uint32_t block_count_divisor;
    };

    size_t                     explicit_pool_reserve_bytes = 0;
    size_t                     paged_block_bytes           = 0;
    size_t                     swa_block_bytes             = 0;
    std::vector<ProjectedPool> projected_pools;
};

// Returns the largest global block count whose independent-pool backing fits
// in total_budget_bytes:
//   explicit reserve + N * (paged bytes + SWA bytes) + projected state pools.
// Fully explicit layouts use a baseline of one when their reservation fits.
uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget);

class CacheConfigCreator {
public:
    static CacheConfig createWarmupConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          int                      gen_num_per_cycle = 0);
    static CacheConfig createWarmupConfig(const ModelConfig&       model_config,
                                          const ParallelismConfig& parallelism_config,
                                          const KVCacheConfig&     kv_cache_config,
                                          int                      gen_num_per_cycle = 0);
    // Builds main and draft layouts. Capacity is computed and finalized separately.
    static CacheConfig createConfig(const ModelConfig&                               model_config,
                                    const ParallelismConfig&                         parallelism_config,
                                    const KVCacheConfig&                             kv_cache_config,
                                    const std::optional<SpeculativeExecutionConfig>& sp_config          = std::nullopt,
                                    const ModelConfig*                               draft_model_config = nullptr,
                                    bool                                             is_mtp             = false,
                                    bool                                             is_eagle           = false);

    static uint32_t computeLocalBlockNum(const CacheConfig&                               config,
                                         const ModelConfig&                               model_config,
                                         const RuntimeConfig&                             runtime_config,
                                         const KVCacheConfig&                             kv_cache_config,
                                         const ParallelismConfig&                         parallelism_config,
                                         const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                                         const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static uint32_t synchronizeBlockNum(uint32_t local_block_num, const ParallelismConfig& parallelism_config);
    static uint32_t selectConfirmedBlockNum(const int* candidates, size_t count);

    // Shared hybrid pools agree on bytes before deriving tag-local slot ids.
    // A block count is not comparable across heterogeneous cache groups.
    static bool     useSharedBlockPool(const CacheConfig&       config,
                                       const ModelConfig&       model_config,
                                       const ParallelismConfig& parallelism_config,
                                       const KVCacheConfig&     kv_cache_config);
    static uint64_t computeLocalCacheBytes(const CacheConfig&                 config,
                                           const ModelConfig&                 model_config,
                                           const RuntimeConfig&               runtime_config,
                                           const KVCacheConfig&               kv_cache_config,
                                           const ParallelismConfig&           parallelism_config,
                                           const std::optional<WarmUpResult>& warm_up_result          = std::nullopt,
                                           const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static uint64_t synchronizeCacheBytes(uint64_t local_bytes, const ParallelismConfig& parallelism_config);
    static uint64_t selectConfirmedCacheBytes(const int64_t* candidates, size_t count);

    // Unified desc->spec conversion. Callers provide the runtime build context;
    // descs remain read-only.
    static LayerKVCacheSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                      const SpecBuildContext&      ctx,
                                                      int64_t                      expected_layer_num);

    static ModelConfig stageScopedModelConfig(const ModelConfig&       model_config,
                                              const ParallelismConfig& parallelism_config,
                                              bool                     is_draft_model = false);

private:
    static CacheConfig createBasicConfig(const ModelConfig&       model_config,
                                         const ParallelismConfig& parallelism_config,
                                         const KVCacheConfig&     kv_cache_config,
                                         int                      gen_num_per_cycle,
                                         uint32_t                 global_layer_begin);
};

}  // namespace rtp_llm
